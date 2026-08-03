from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import sys
import time

import yaml
from fastapi.testclient import TestClient

from env.patcheval import claudecode_runner, generate_full_config
from env.patcheval.claude_adapter import conversion
from env.patcheval.claude_adapter.app import create_app as create_adapter_app
from evaluator.rule_evaluator import discover_rule_eval_spec
from core.data_manager.strategy.sqlite_strategy_impl import SqliteStrategy
from gateway import app as gateway_app
from gateway.config import GatewayConfig, LLMRouteConfig
from gateway.inference_forwarder import (
    ForwardResult,
    InferenceForwarder,
    StreamForwardContext,
)
from gateway.llm_router import LLMRouteTarget
from gateway.storage import GatewayStorage


def test_shared_adapter_health(monkeypatch) -> None:
    monkeypatch.setenv(
        "CLAUDE_ADAPTER_GATEWAY_SESSION_BASE_URL",
        "http://127.0.0.1:18000/v1/sessions",
    )
    monkeypatch.setenv("CLAUDE_ADAPTER_ROUTE_MODEL", "route/model")
    with TestClient(create_adapter_app()) as client:
        assert client.get("/readyz").json() == {"status": "ready"}


def test_sqlite_runtime_schema_adds_provider_request_column(tmp_path) -> None:
    database = tmp_path / "legacy.db"
    connection = sqlite3.connect(database)
    connection.execute("CREATE TABLE session_steps (id INTEGER PRIMARY KEY)")
    connection.commit()
    connection.close()

    strategy = SqliteStrategy(
        "job",
        f"sqlite://{database}",
        enable_buffer=False,
    )
    asyncio.run(strategy._ensure_runtime_schema())

    connection = sqlite3.connect(database)
    columns = {
        row[1] for row in connection.execute("PRAGMA table_info(session_steps)")
    }
    connection.close()
    assert "request" in columns


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


def test_anthropic_payload_uses_single_adaptive_max_path() -> None:
    target = LLMRouteTarget(
        route_model="claude",
        base_url="http://upstream/v1",
        api_key=None,
        anthropic_interleaved_thinking=True,
    )
    original = {
        "model": "claude",
        "max_tokens": 64000,
        "thinking": {"type": "adaptive"},
        "context_management": {"edits": []},
        "output_config": {"effort": "high"},
        "messages": [{"role": "user", "content": "hello"}],
    }

    prepared = InferenceForwarder.prepare_anthropic_payload(original)

    assert prepared["thinking"] == {"type": "adaptive"}
    assert prepared["output_config"] == {"effort": "max"}
    assert prepared["display"] == "summarized"
    assert prepared["max_tokens"] == 64000
    assert "context_management" not in prepared
    assert original["context_management"] == {"edits": []}

    headers = object.__new__(InferenceForwarder).build_anthropic_headers(
        target,
        {"anthropic-beta": "claude-code-20250219,unsupported-beta"},
    )
    assert headers["anthropic-beta"] == "interleaved-thinking-2025-05-14"


def test_gateway_streams_native_anthropic_and_records_request_response(monkeypatch) -> None:
    forwarded_payloads = []
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

        def prepare_anthropic_payload(self, payload):
            return payload

        def build_anthropic_headers(self, target, inbound_headers, **kwargs):
            del target, inbound_headers
            assert kwargs["session_id"] == "session-1"
            return {
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01",
                "anthropic-beta": "interleaved-thinking-2025-05-14",
                "x-api-key": "upstream-key",
                "Authorization": "Bearer upstream-key",
            }

        async def open_anthropic_messages_stream(self, target, payload, headers):
            del target, headers
            forwarded_payloads.append(json.loads(json.dumps(payload)))
            return StreamForwardContext(
                response=FakeResponse(),
                status_code=200,
                headers={"content-type": "text/event-stream"},
                upstream_latency_ms=1.0,
                upstream_started_perf=time.perf_counter(),
            )

        async def forward_anthropic_messages(self, target, payload, headers):
            del target, headers
            forwarded_payloads.append(json.loads(json.dumps(payload)))
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
        llm_routes={
            "claude": LLMRouteConfig(
                base_url="https://provider.example/v1",
                api_key="upstream-key",
                anthropic_interleaved_thinking=True,
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
        nonstream_response = client.post(
            "/v1/sessions/session-1/v1/messages",
            json={
                "model": "claude",
                "max_tokens": 128,
                "stream": False,
                "messages": [
                    {"role": "user", "content": "fix it"},
                    {"role": "assistant", "content": []},
                    {"role": "user", "content": "continue"},
                ],
            },
        )

    assert response.status_code == 200
    assert "signature_delta" in response.text
    assert nonstream_response.json()["content"][0]["signature"] == "nonstream-signature"
    assert forwarded_payloads[1]["messages"][1]["content"] == []
    assert len(storage.records) == 2
    record = storage.records[0][1]
    stored_request = json.loads(record.request)
    metadata = GatewayStorage._metadata(record)
    assert stored_request == forwarded_payloads[0]
    assert record.response == stream.decode()
    assert '"signature":"signed-blob"' in record.response
    assert metadata["request_method"] == "POST"
    assert metadata["request_url"] == "https://provider.example/v1/messages"
    assert metadata["request_headers"]["anthropic-version"] == "2023-06-01"
    assert metadata["request_headers"]["Accept"] == "text/event-stream"
    assert metadata["request_headers"]["x-api-key"] == "[REDACTED]"
    assert metadata["request_headers"]["Authorization"] == "[REDACTED]"
