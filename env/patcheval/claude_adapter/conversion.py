"""Anthropic Messages ↔ OpenAI Chat Completions conversion helpers."""
from __future__ import annotations

import json
import uuid
from typing import Any


def anthropic_to_openai(payload: dict[str, Any], model: str) -> dict[str, Any]:
    messages: list[dict[str, Any]] = []
    system_text = content_text(payload.get("system"))
    if system_text:
        messages.append({"role": "system", "content": system_text})

    for message in payload.get("messages") or []:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "")
        content = message.get("content")
        if isinstance(content, str):
            messages.append({"role": role, "content": content})
            continue
        if not isinstance(content, list):
            continue
        if role == "assistant":
            text_parts: list[str] = []
            tool_calls: list[dict[str, Any]] = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "text":
                    text_parts.append(str(block.get("text") or ""))
                elif block.get("type") == "tool_use":
                    tool_calls.append(
                        {
                            "id": str(block.get("id") or f"tool_{uuid.uuid4().hex}"),
                            "type": "function",
                            "function": {
                                "name": str(block.get("name") or ""),
                                "arguments": json.dumps(block.get("input") or {}, ensure_ascii=False),
                            },
                        }
                    )
            converted: dict[str, Any] = {"role": "assistant", "content": "\n".join(text_parts) or None}
            if tool_calls:
                converted["tool_calls"] = tool_calls
            messages.append(converted)
            continue

        pending_text: list[str] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type == "tool_result":
                if pending_text:
                    messages.append({"role": "user", "content": "\n".join(pending_text)})
                    pending_text = []
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": str(block.get("tool_use_id") or ""),
                        "content": content_text(block.get("content")),
                    }
                )
            elif block_type == "text":
                pending_text.append(str(block.get("text") or ""))
        if pending_text:
            messages.append({"role": "user", "content": "\n".join(pending_text)})

    request: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": int(payload.get("max_tokens") or 16384),
        # The adapter converts one complete OpenAI response into Anthropic SSE.
        "stream": False,
    }
    if payload.get("temperature") is not None:
        request["temperature"] = payload["temperature"]
    if payload.get("stop_sequences"):
        request["stop"] = payload["stop_sequences"]
    tools = []
    for tool in payload.get("tools") or []:
        if not isinstance(tool, dict) or not tool.get("name"):
            continue
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description") or "",
                    "parameters": without_json_schema(tool.get("input_schema") or {}),
                },
            }
        )
    if tools:
        request["tools"] = tools
        request["tool_choice"] = "auto"
    return request


def openai_to_anthropic(
    response: dict[str, Any],
    original_request: dict[str, Any],
) -> dict[str, Any]:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("OpenAI-compatible response has no choices")
    choice = choices[0] if isinstance(choices[0], dict) else {}
    message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
    blocks: list[dict[str, Any]] = []
    text = content_text(message.get("content"))
    if text:
        blocks.append({"type": "text", "text": text})
    for call in message.get("tool_calls") or []:
        if not isinstance(call, dict):
            continue
        function = call.get("function") if isinstance(call.get("function"), dict) else {}
        raw_arguments = function.get("arguments") or "{}"
        try:
            arguments = json.loads(raw_arguments) if isinstance(raw_arguments, str) else raw_arguments
        except json.JSONDecodeError:
            arguments = {"raw": str(raw_arguments)}
        blocks.append(
            {
                "type": "tool_use",
                "id": str(call.get("id") or f"toolu_{uuid.uuid4().hex}"),
                "name": str(function.get("name") or ""),
                "input": arguments if isinstance(arguments, dict) else {"value": arguments},
            }
        )
    if not blocks:
        blocks.append({"type": "text", "text": ""})
    stop_reason = {
        "tool_calls": "tool_use",
        "length": "max_tokens",
        "stop": "end_turn",
    }.get(str(choice.get("finish_reason")), "end_turn")
    usage = response.get("usage") if isinstance(response.get("usage"), dict) else {}
    return {
        "id": str(response.get("id") or f"msg_{uuid.uuid4().hex}"),
        "type": "message",
        "role": "assistant",
        "model": str(original_request.get("model") or response.get("model") or "claude"),
        "content": blocks,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": int(usage.get("prompt_tokens") or 0),
            "output_tokens": int(usage.get("completion_tokens") or 0),
        },
    }


def anthropic_sse(message: dict[str, Any]) -> str:
    initial = dict(message)
    initial["content"] = []
    initial["stop_reason"] = None
    initial["usage"] = {
        "input_tokens": message["usage"]["input_tokens"],
        "output_tokens": 0,
    }
    events: list[tuple[str, dict[str, Any]]] = [
        ("message_start", {"type": "message_start", "message": initial}),
        ("ping", {"type": "ping"}),
    ]
    for index, block in enumerate(message["content"]):
        if block.get("type") == "tool_use":
            start_block = {
                "type": "tool_use",
                "id": block["id"],
                "name": block["name"],
                "input": {},
            }
            delta = {
                "type": "input_json_delta",
                "partial_json": json.dumps(block.get("input") or {}, ensure_ascii=False),
            }
        else:
            start_block = {"type": "text", "text": ""}
            delta = {"type": "text_delta", "text": str(block.get("text") or "")}
        events.extend(
            [
                (
                    "content_block_start",
                    {"type": "content_block_start", "index": index, "content_block": start_block},
                ),
                (
                    "content_block_delta",
                    {"type": "content_block_delta", "index": index, "delta": delta},
                ),
                ("content_block_stop", {"type": "content_block_stop", "index": index}),
            ]
        )
    events.extend(
        [
            (
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {
                        "stop_reason": message["stop_reason"],
                        "stop_sequence": message["stop_sequence"],
                    },
                    "usage": {"output_tokens": message["usage"]["output_tokens"]},
                },
            ),
            ("message_stop", {"type": "message_stop"}),
        ]
    )
    return "".join(
        f"event: {event_name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
        for event_name, data in events
    )


def without_json_schema(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: without_json_schema(item) for key, item in value.items() if key != "$schema"}
    if isinstance(value, list):
        return [without_json_schema(item) for item in value]
    return value


def content_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text") or ""))
        return "\n".join(part for part in parts if part)
    return ""
