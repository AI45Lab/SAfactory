from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

import yaml
from fastapi.testclient import TestClient

from env.patcheval import claudecode_runner, generate_full_config
from env.patcheval.claude_adapter import conversion
from env.patcheval.claude_adapter.app import create_app as create_adapter_app
from evaluator.rule_evaluator import discover_rule_eval_spec
from gateway import app as gateway_app
from gateway.app import _anthropic_response_from_sse
from gateway.config import GatewayConfig, LLMRouteConfig
from gateway.inference_forwarder import (
    ForwardResult,
    InferenceForwarder,
    StreamForwardContext,
)
from gateway.llm_router import LLMRouteTarget
from gateway.provider_trace import ProviderTraceWriter
from gateway.storage import GatewayStorage


def test_shared_adapter_health(monkeypatch) -> None:
    monkeypatch.setenv(
        "CLAUDE_ADAPTER_GATEWAY_SESSION_BASE_URL",
        "http://127.0.0.1:18000/v1/sessions",
    )
    monkeypatch.setenv("CLAUDE_ADAPTER_ROUTE_MODEL", "route/model")
    with TestClient(create_adapter_app()) as client:
        assert client.get("/readyz").json() == {"status": "ready"}


def test_anthropic_tool_messages_convert_to_openai() -> None:
    payload = {
        "model": "claude",
        "max_tokens": 4096,
        "system": [{"type": "text", "text": "system prompt"}],
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "inspect the repository"}]},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "Read",
                        "input": {"path": "a.py"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_1",
                        "content": "file contents",
                    }
                ],
            },
        ],
        "tools": [
            {
                "name": "Read",
                "description": "read a file",
                "input_schema": {
                    "$schema": "http://json-schema.org/draft-07/schema#",
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                },
            }
        ],
    }

    converted = conversion.anthropic_to_openai(payload, "route/model")

    assert converted["model"] == "route/model"
    assert converted["stream"] is False
    assert converted["messages"][0] == {"role": "system", "content": "system prompt"}
    assert converted["messages"][2]["tool_calls"][0]["function"]["name"] == "Read"
    assert converted["messages"][3]["role"] == "tool"
    assert "$schema" not in converted["tools"][0]["function"]["parameters"]


def test_openai_tool_response_converts_to_anthropic_stream() -> None:
    response = {
        "id": "chatcmpl_1",
        "model": "route/model",
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": "I will inspect it.",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "Read", "arguments": '{"path":"a.py"}'},
                        }
                    ],
                },
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }

    converted = conversion.openai_to_anthropic(response, {"model": "claude"})
    stream = conversion.anthropic_sse(converted)

    assert converted["stop_reason"] == "tool_use"
    assert converted["content"][1]["input"] == {"path": "a.py"}
    assert "event: message_start" in stream
    assert "event: content_block_delta" in stream
    assert "event: message_stop" in stream


def test_tool_use_count_reads_claude_stream_event() -> None:
    line = json.dumps(
        {
            "type": "assistant",
            "message": {
                "content": [
                    {"type": "text", "text": "checking"},
                    {"type": "tool_use", "name": "Read"},
                    {"type": "tool_use", "name": "Grep"},
                ]
            },
        }
    )
    assert claudecode_runner._count_tool_uses(line) == 2


def test_find_claude_code_in_explicit_npm_prefix(tmp_path) -> None:
    executable = tmp_path / "node_modules" / ".bin" / "claude"
    executable.parent.mkdir(parents=True)
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    executable.chmod(executable.stat().st_mode | os.X_OK)

    assert claudecode_runner._find_claude_code(tmp_path) == str(executable.resolve())


def test_generate_claudecode_exp1_config(tmp_path, monkeypatch) -> None:
    runtime = tmp_path / "runtime"
    template = runtime / "exp_agent" / "claudecode" / "templates" / "default.md"
    template.parent.mkdir(parents=True)
    template.write_text("official template", encoding="utf-8")
    output = tmp_path / "generated"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_full_config.py",
            "--output-dir",
            str(output),
            "--official-runtime-dir",
            str(runtime),
            "--baseline",
            "claudecode",
            "--agent-experiment",
            "exp1",
            "--claude-gateway-base-url",
            "http://gateway:18000/v1/sessions",
            "--claude-model",
            "claude-route",
            "--claude-max-thinking-tokens",
            "1024",
            "--limit",
            "1",
        ],
    )

    generate_full_config.main()

    start = yaml.safe_load((output / "patcheval_start.yaml").read_text(encoding="utf-8"))
    env_name, agent_config = next(iter(start["agents"].items()))
    agent = agent_config["container"]
    assert agent["runner_entrypoint"]["source"].endswith("claudecode_runner.py")
    assert agent["env"]["PATCHEVAL_CLAUDE_TOOL_LIMIT"] == "100"
    assert (
        agent["env"]["PATCHEVAL_CLAUDE_GATEWAY_BASE_URL"]
        == "http://gateway:18000/v1/sessions"
    )
    assert agent["env"]["PATCHEVAL_CLAUDE_MODEL"] == "claude-route"
    assert agent["env"]["PATCHEVAL_CLAUDE_MAX_THINKING_TOKENS"] == "1024"
    task_file = next((output / "datasets").glob("*.jsonl"))
    task = json.loads(task_file.read_text(encoding="utf-8"))
    assert task["agent_framework"] == "claude-code"
    assert task["agent_experiment"] == "exp1"
    assert "problem_statement" in task
    assert "prompt_template" not in task
    eval_spec = discover_rule_eval_spec(agent_name=env_name, env_root=output)
    assert eval_spec is not None
    assert eval_spec.rule_evaluator.endswith(f"/{env_name}/rule_evaluator.py")


def test_anthropic_sse_aggregates_thinking_signature_and_text() -> None:
    stream = "\n\n".join(
        [
            'event: message_start\ndata: {"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude","content":[],"usage":{"input_tokens":7,"output_tokens":0}}}',
            'event: content_block_start\ndata: {"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":"","signature":""}}',
            'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"inspect"}}',
            'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"signed-blob"}}',
            'event: content_block_start\ndata: {"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}',
            'event: content_block_delta\ndata: {"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"done"}}',
            'event: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":4}}',
            'event: message_stop\ndata: {"type":"message_stop"}',
        ]
    )

    response = _anthropic_response_from_sse(stream, "claude")

    assert response["content"][0] == {
        "type": "thinking",
        "thinking": "inspect",
        "signature": "signed-blob",
    }
    assert response["content"][1] == {"type": "text", "text": "done"}
    assert response["stop_reason"] == "end_turn"


def test_fixed_thinking_compatibility_normalizes_anthropic_payload() -> None:
    target = LLMRouteTarget(
        route_model="claude",
        base_url="http://upstream/v1",
        api_key=None,
        anthropic_compatibility="fixed_thinking",
        anthropic_thinking_budget_tokens=1024,
        anthropic_max_tokens=8192,
    )
    original = {
        "model": "claude",
        "max_tokens": 64000,
        "thinking": {"type": "adaptive"},
        "context_management": {"edits": []},
        "output_config": {"effort": "high"},
        "messages": [{"role": "user", "content": "hello"}],
    }

    prepared = InferenceForwarder.prepare_anthropic_payload(target, original)

    assert prepared["thinking"] == {"type": "enabled", "budget_tokens": 1024}
    assert prepared["max_tokens"] == 8192
    assert "context_management" not in prepared
    assert "output_config" not in prepared
    assert original["thinking"] == {"type": "adaptive"}


def test_provider_trace_writes_signature_artifact(tmp_path) -> None:
    writer = ProviderTraceWriter(str(tmp_path), "full")
    metadata = asyncio.run(
        writer.write(
            session_id="session-1",
            request_id="request-1",
            llm_step_index=2,
            model="claude",
            endpoint="messages",
            request_body={"model": "claude", "messages": []},
            response_body={
                "type": "message",
                "content": [
                    {
                        "type": "thinking",
                        "thinking": "private",
                        "signature": "signed-blob",
                    }
                ],
            },
            status_code=200,
            capture_complete=True,
        )
    )

    assert metadata is not None
    assert metadata["signature_count"] == 1
    assert metadata["artifact"]["response"]["content"][0]["signature"] == "signed-blob"
    artifact = json.loads(
        open(metadata["artifact_path"], encoding="utf-8").read()
    )
    assert artifact["signatures"][0]["signature"] == "signed-blob"
    assert artifact["capture"]["complete"] is True


def test_provider_trace_keeps_database_artifact_when_external_write_fails(tmp_path) -> None:
    invalid_root = tmp_path / "not-a-directory"
    invalid_root.write_text("occupied", encoding="utf-8")
    writer = ProviderTraceWriter(str(invalid_root), "full")

    metadata = asyncio.run(
        writer.write(
            session_id="session-1",
            request_id="request-1",
            llm_step_index=1,
            model="claude",
            endpoint="messages",
            request_body={"model": "claude", "messages": []},
            response_body={
                "type": "message",
                "content": [
                    {
                        "type": "thinking",
                        "thinking": "private",
                        "signature": "signed-blob",
                    }
                ],
            },
            status_code=200,
            capture_complete=True,
        )
    )

    assert metadata is not None
    assert metadata["artifact_path"] is None
    assert metadata["capture_complete"] is False
    assert metadata["capture_error"] == "external_artifact_write_failed"
    assert metadata["artifact"]["signatures"][0]["signature"] == "signed-blob"


def test_gateway_streams_native_anthropic_and_records_trace(tmp_path, monkeypatch) -> None:
    stream = "\n\n".join(
        [
            'event: message_start\ndata: {"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude","content":[],"usage":{"input_tokens":3,"output_tokens":0}}}',
            'event: content_block_start\ndata: {"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":"","signature":""}}',
            'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"inspect"}}',
            'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"signed-blob"}}',
            'event: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":2}}',
            'event: message_stop\ndata: {"type":"message_stop"}',
            "",
        ]
    ).encode()

    class FakeContent:
        async def iter_any(self):
            yield stream[:73]
            yield stream[73:]

    class FakeResponse:
        status = 200
        headers = {"content-type": "text/event-stream"}
        content = FakeContent()

        def close(self) -> None:
            return None

    class FakeForwarder:
        def __init__(self, cfg) -> None:
            del cfg

        def build_anthropic_headers(self, target, inbound_headers, **kwargs):
            del target, inbound_headers
            assert kwargs["session_id"] == "session-1"
            return {"Content-Type": "application/json", "x-api-key": "upstream-key"}

        def prepare_anthropic_payload(self, target, payload):
            del target
            return payload

        async def open_anthropic_messages_stream(self, target, payload, headers):
            del target, payload, headers
            return StreamForwardContext(
                response=FakeResponse(),
                status_code=200,
                headers={"content-type": "text/event-stream"},
                upstream_latency_ms=1.0,
                upstream_started_perf=time.perf_counter(),
            )

        async def forward_anthropic_messages(self, target, payload, headers):
            del target, payload, headers
            return ForwardResult(
                body={
                    "id": "msg_2",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude",
                    "content": [
                        {
                            "type": "thinking",
                            "thinking": "inspect",
                            "signature": "nonstream-signature",
                        },
                        {"type": "text", "text": "done"},
                    ],
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 3, "output_tokens": 2},
                },
                status_code=200,
                headers={},
                upstream_latency_ms=1.0,
            )

        async def forward_anthropic_count_tokens(self, target, payload, headers):
            del target, payload, headers
            return ForwardResult(
                body={"input_tokens": 17},
                status_code=200,
                headers={},
                upstream_latency_ms=1.0,
            )

        async def close(self) -> None:
            return None

        def normalize_error(self, exc):
            return 500, {"error": {"message": str(exc)}}

    class FakeStorage:
        def __init__(self) -> None:
            self.records = []

        async def bind_session_environment(self, binding) -> None:
            del binding

        async def record_telemetry_batch(self, batch) -> None:
            self.records.extend(batch)

        async def record_inference_step(self, binding, record) -> None:
            self.records.append((binding, record))

        async def close(self) -> None:
            return None

    monkeypatch.setattr(gateway_app, "InferenceForwarder", FakeForwarder)
    storage = FakeStorage()
    cfg = GatewayConfig(
        storage_type="sqlite",
        storage_config={"db_url": "sqlite://unused.db"},
        request_log_enabled=False,
        provider_trace_capture="full",
        provider_trace_dir=str(tmp_path),
        llm_routes={
            "claude": LLMRouteConfig(
                base_url="https://provider.example/v1",
                api_key="upstream-key",
            )
        },
    )
    app = gateway_app.create_app(cfg, storage=storage)

    with TestClient(app) as client:
        response = client.post(
            "/v1/sessions/session-1/v1/messages",
            json={
                "model": "claude",
                "max_tokens": 128,
                "stream": True,
                "messages": [{"role": "user", "content": "fix it"}],
            },
        )
        count_response = client.post(
            "/v1/sessions/session-1/v1/messages/count_tokens",
            json={
                "model": "claude",
                "messages": [{"role": "user", "content": "fix it"}],
            },
        )
        nonstream_response = client.post(
            "/v1/sessions/session-1/v1/messages",
            json={
                "model": "claude",
                "max_tokens": 128,
                "stream": False,
                "messages": [{"role": "user", "content": "fix it"}],
            },
        )

    assert response.status_code == 200
    assert "signature_delta" in response.text
    assert count_response.json() == {"input_tokens": 17}
    assert nonstream_response.json()["content"][0]["signature"] == "nonstream-signature"
    assert len(storage.records) == 2
    record = storage.records[0][1]
    metadata = GatewayStorage._metadata(record)
    assert metadata["provider_trace"]["signature_count"] == 1
    assert "artifact" not in metadata["provider_trace"]
    stored_response = json.loads(GatewayStorage._provider_response(record))
    assert stored_response["response"]["content"][0]["signature"] == "signed-blob"
    assert stored_response["stream_text"].startswith("event: message_start")
    artifact_path = Path(metadata["provider_trace"]["artifact_path"])
    assert artifact_path.is_file()
    assert GatewayStorage._provider_response(record) == artifact_path.read_text(encoding="utf-8")
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert artifact["signatures"][0]["signature"] == "signed-blob"
