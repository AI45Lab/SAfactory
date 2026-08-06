import json
from types import SimpleNamespace

import pytest

from gateway.anthropic_messages import (
    AnthropicMessageConversionError,
    normalize_anthropic_request,
)
from gateway.storage import _trajectory_messages
from gateway.telemetry import SENSITIVE_KEY_PARTS
from core.data_manager.strategy.cloud_strategy_impl import CloudStrategy
from core.data_manager.strategy import cloud_strategy_impl


def test_normalize_anthropic_request_preserves_reasoning_signature_and_tools():
    request = {
        "system": [{"type": "text", "text": "Follow the task."}],
        "messages": [
            {"role": "user", "content": "Inspect the repository."},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "thinking",
                        "thinking": "I should inspect the files first.",
                        "signature": "claude-signature",
                    },
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "Read",
                        "input": {"path": "README.md"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_1",
                        "content": [{"type": "text", "text": "contents"}],
                    }
                ],
            },
        ],
    }

    assert normalize_anthropic_request(json.dumps(request)) == [
        {"role": "system", "content": "Follow the task."},
        {"role": "user", "content": "Inspect the repository."},
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": "I should inspect the files first.",
            "encrypted_content": "claude-signature",
            "tool_calls": [
                {
                    "id": "toolu_1",
                    "type": "function",
                    "function": {
                        "name": "Read",
                        "arguments": '{"path":"README.md"}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "toolu_1", "content": "contents"},
    ]


def test_normalize_anthropic_request_preserves_interleaved_text_and_image():
    messages = normalize_anthropic_request(
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "before"},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": "abc",
                            },
                        },
                        {"type": "text", "text": "after"},
                    ],
                }
            ]
        }
    )

    assert messages == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "before"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
                {"type": "text", "text": "after"},
            ],
        }
    ]


def test_normalize_anthropic_request_skips_redacted_thinking():
    assert normalize_anthropic_request(
        {
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "redacted_thinking", "data": "opaque"},
                        {"type": "text", "text": "visible answer"},
                    ],
                }
            ]
        }
    ) == [{"role": "assistant", "content": "visible answer"}]


def test_normalize_anthropic_request_preserves_signature_when_thinking_is_empty():
    assert normalize_anthropic_request(
        {
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "thinking",
                            "thinking": "",
                            "signature": "opaque-claude-signature",
                        }
                    ],
                }
            ]
        }
    ) == [
        {
            "role": "assistant",
            "content": "",
            "encrypted_content": "opaque-claude-signature",
        }
    ]


def test_normalize_anthropic_request_accepts_system_messages_in_history():
    assert normalize_anthropic_request(
        {
            "messages": [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": "Cached instruction.",
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                }
            ]
        }
    ) == [{"role": "system", "content": "Cached instruction."}]


def test_normalize_anthropic_request_rejects_unknown_blocks():
    with pytest.raises(AnthropicMessageConversionError, match="unsupported Anthropic content block"):
        normalize_anthropic_request(
            {"messages": [{"role": "user", "content": [{"type": "document"}]}]}
        )


def test_storage_normalizes_native_anthropic_requests_and_falls_back_on_failure():
    normalized = _trajectory_messages(
        SimpleNamespace(
            endpoint="messages",
            request_id="request-1",
            request=json.dumps({"messages": [{"role": "user", "content": "hello"}]}),
            messages=[{"role": "user", "content": "raw"}],
        )
    )
    fallback = _trajectory_messages(
        SimpleNamespace(
            endpoint="messages",
            request_id="request-2",
            request="not-json",
            messages=[{"role": "user", "content": "raw"}],
        )
    )
    non_anthropic = _trajectory_messages(
        SimpleNamespace(
            endpoint="chat/completions",
            request_id="request-3",
            request="not-json",
            messages=[{"role": "user", "content": "already-openai"}],
        )
    )

    assert normalized == [{"role": "user", "content": "hello"}]
    assert fallback == [{"role": "user", "content": "raw"}]
    assert non_anthropic == [{"role": "user", "content": "already-openai"}]


def test_claude_signature_fields_are_not_telemetry_redaction_keys():
    assert "signature" not in SENSITIVE_KEY_PARTS
    assert "encrypted_content" not in SENSITIVE_KEY_PARTS


def test_cloud_adapter_preserves_extended_fields_in_legacy_content_blocks():
    strategy = object.__new__(CloudStrategy)

    messages = strategy._convert_to_chat_messages(
        [
            {
                "role": "assistant",
                "content": "final answer",
                "reasoning_content": "reasoning",
                "encrypted_content": "signature",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "Read", "arguments": '{"path":"README.md"}'},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": "contents",
            },
        ]
    )

    assert [
        (item.type, item.text) for item in messages[0].content
    ] == [
        ("reasoning_content", "reasoning"),
        ("encrypted_content", "signature"),
        ("text", "final answer"),
    ]
    assert messages[0].tool_calls[0].function.name == "Read"
    assert messages[1].tool_call_id == "call_1"
    assert [(item.type, item.text) for item in messages[1].content] == [
        ("text", "contents")
    ]


def test_cloud_json_update_value_preserves_extended_message_fields():
    strategy = object.__new__(CloudStrategy)
    payload = [
        {
            "role": "assistant",
            "content": "answer",
            "reasoning_content": "reasoning",
            "encrypted_content": "signature",
        }
    ]

    serialized = strategy._messages_to_landing_value(payload)
    assert json.loads(serialized) == payload

    cloud_strategy_impl._load_wt_sdk()
    legacy_record = cloud_strategy_impl.LandingRecord.model_construct(
        dataset_type="test",
        id="record",
        created_at=1,
        messages=serialized,
        response='{"role":"assistant","content":"answer"}',
    )
    assert legacy_record.model_dump()["messages"] == serialized
