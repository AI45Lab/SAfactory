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


def test_anthropic_headers_preserve_native_headers_and_add_safe_beta() -> None:
    target = LLMRouteTarget(
        route_model="claude",
        base_url="http://upstream/v1",
        api_key="upstream-key",
        anthropic_interleaved_thinking=True,
    )

    headers = object.__new__(InferenceForwarder).build_anthropic_headers(
        target,
        {
            "anthropic-beta": "claude-code-20250219,unsupported-beta",
            "anthropic-version": "2023-06-01",
            "x-api-key": "client-key",
        },
    )
    assert headers["anthropic-beta"] == "interleaved-thinking-2025-05-14"
    assert headers["x-api-key"] == "upstream-key"
    assert "X-Safactory-Session-Id" not in headers


def test_gateway_streams_native_anthropic_and_records_request_response(monkeypatch) -> None:
    forwarded_payloads = []
    forwarded_query_strings = []
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
            return {
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01",
                "anthropic-beta": "interleaved-thinking-2025-05-14",
                "x-api-key": "upstream-key",
                "Authorization": "Bearer upstream-key",
            }

        async def open_anthropic_messages_stream(
            self, target, payload, headers, *, query_string=None
        ):
            del target, headers
            forwarded_payloads.append(json.loads(json.dumps(payload)))
            forwarded_query_strings.append(query_string)
            return StreamForwardContext(
                response=FakeResponse(),
                status_code=200,
                headers={"content-type": "text/event-stream"},
                upstream_latency_ms=1.0,
                upstream_started_perf=time.perf_counter(),
            )

        async def forward_anthropic_messages(
            self, target, payload, headers, *, query_string=None
        ):
            del target, headers
            forwarded_payloads.append(json.loads(json.dumps(payload)))
            forwarded_query_strings.append(query_string)
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
            "/v1/sessions/session-1/v1/messages?beta=true&preserve=value",
            json={
                "model": "claude",
                "max_tokens": 128,
                "stream": True,
                "context_management": {"edits": []},
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
    assert forwarded_payloads[0]["max_tokens"] == 128
    assert forwarded_payloads[0]["context_management"] == {"edits": []}
    assert forwarded_query_strings == ["preserve=value", None]
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
