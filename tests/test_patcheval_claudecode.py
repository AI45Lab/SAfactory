from __future__ import annotations

import json
import os
import sys

import yaml
from fastapi.testclient import TestClient

from env.patcheval import claudecode_runner, generate_full_config
from env.patcheval.claude_adapter import conversion
from env.patcheval.claude_adapter.app import create_app
from evaluator.rule_evaluator import discover_rule_eval_spec


def test_shared_adapter_health(monkeypatch) -> None:
    monkeypatch.setenv(
        "CLAUDE_ADAPTER_GATEWAY_SESSION_BASE_URL",
        "http://127.0.0.1:18000/v1/sessions",
    )
    monkeypatch.setenv("CLAUDE_ADAPTER_ROUTE_MODEL", "route/model")
    with TestClient(create_app()) as client:
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
            "--claude-adapter-base-url",
            "http://gateway:18001/v1/sessions",
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
        agent["env"]["PATCHEVAL_CLAUDE_ADAPTER_BASE_URL"]
        == "http://gateway:18001/v1/sessions"
    )
    task_file = next((output / "datasets").glob("*.jsonl"))
    task = json.loads(task_file.read_text(encoding="utf-8"))
    assert task["agent_framework"] == "claude-code"
    assert task["agent_experiment"] == "exp1"
    assert "problem_statement" in task
    assert "prompt_template" not in task
    eval_spec = discover_rule_eval_spec(agent_name=env_name, env_root=output)
    assert eval_spec is not None
    assert eval_spec.rule_evaluator.endswith(f"/{env_name}/rule_evaluator.py")
