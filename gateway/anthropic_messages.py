"""Normalize Anthropic request messages to SAfactory chat-completions JSON."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any


class AnthropicMessageConversionError(ValueError):
    """Raised when an Anthropic request cannot be converted without data loss."""


def normalize_anthropic_request(request: str | Mapping[str, Any]) -> list[dict[str, Any]]:
    """Convert one native Anthropic request's history to Chat Completions messages."""

    document = _request_object(request)
    normalized = _normalize_system(document.get("system"))
    messages = document.get("messages", [])
    if not isinstance(messages, list):
        raise AnthropicMessageConversionError("Anthropic request.messages must be a list")
    for message in messages:
        normalized.extend(_normalize_message(message))
    return normalized


def _request_object(request: str | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(request, Mapping):
        return dict(request)
    if not isinstance(request, str) or not request.strip():
        raise AnthropicMessageConversionError("Anthropic request is missing")
    try:
        parsed = json.loads(request)
    except json.JSONDecodeError as exc:
        raise AnthropicMessageConversionError("Anthropic request is not valid JSON") from exc
    if not isinstance(parsed, dict):
        raise AnthropicMessageConversionError("Anthropic request must be a JSON object")
    return parsed


def _normalize_system(system: Any) -> list[dict[str, Any]]:
    if system in (None, ""):
        return []
    if isinstance(system, str):
        return [{"role": "system", "content": system}]
    if not isinstance(system, list):
        raise AnthropicMessageConversionError("Anthropic system must be a string or text blocks")
    text_parts = []
    for block in system:
        if (
            not isinstance(block, dict)
            or block.get("type") != "text"
            or not isinstance(block.get("text"), str)
        ):
            raise AnthropicMessageConversionError(
                "Anthropic system supports only text blocks"
            )
        text_parts.append(block["text"])
    return [{"role": "system", "content": "\n\n".join(text_parts)}] if text_parts else []


def _normalize_message(message: Any) -> list[dict[str, Any]]:
    if not isinstance(message, dict):
        raise AnthropicMessageConversionError("Anthropic message must be an object")
    role = message.get("role")
    if role not in {"system", "user", "assistant"}:
        raise AnthropicMessageConversionError(f"unsupported Anthropic message role: {role!r}")
    content = message.get("content")
    if isinstance(content, str):
        return [{"role": role, "content": content}]
    if not isinstance(content, list):
        raise AnthropicMessageConversionError(
            "Anthropic message content must be a string or list"
        )
    if role == "system":
        return _normalize_system(content)
    return _normalize_blocks(role, content)


def _normalize_blocks(role: str, blocks: list[Any]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    text_parts: list[str] = []
    content_parts: list[dict[str, Any]] = []
    reasoning_parts: list[str] = []
    signatures: list[str] = []
    tool_calls: list[dict[str, Any]] = []

    def flush_message() -> None:
        if (
            not text_parts
            and not content_parts
            and not reasoning_parts
            and not signatures
            and not tool_calls
        ):
            return
        message: dict[str, Any] = {"role": role}
        if content_parts:
            content_parts.extend({"type": "text", "text": text} for text in text_parts)
            message["content"] = list(content_parts)
        else:
            message["content"] = "".join(text_parts)
        if reasoning_parts:
            message["reasoning_content"] = "\n\n".join(reasoning_parts)
        if signatures:
            if len(signatures) != 1:
                raise AnthropicMessageConversionError(
                    "multiple thinking signatures in one Anthropic message are unsupported"
                )
            message["encrypted_content"] = signatures[0]
        if tool_calls:
            message["tool_calls"] = list(tool_calls)
        output.append(message)
        text_parts.clear()
        content_parts.clear()
        reasoning_parts.clear()
        signatures.clear()
        tool_calls.clear()

    for block in blocks:
        if not isinstance(block, dict):
            raise AnthropicMessageConversionError("Anthropic content block must be an object")
        block_type = block.get("type")
        if block_type == "text":
            text = block.get("text")
            if not isinstance(text, str):
                raise AnthropicMessageConversionError("Anthropic text block must contain text")
            if content_parts:
                content_parts.append({"type": "text", "text": text})
            else:
                text_parts.append(text)
        elif block_type == "image":
            if role == "assistant":
                raise AnthropicMessageConversionError("assistant image blocks are unsupported")
            if text_parts:
                content_parts.extend({"type": "text", "text": text} for text in text_parts)
                text_parts.clear()
            content_parts.append({"type": "image_url", "image_url": {"url": _image_url(block)}})
        elif block_type == "thinking":
            if role != "assistant":
                raise AnthropicMessageConversionError("thinking blocks require assistant role")
            thinking = block.get("thinking", "")
            signature = block.get("signature")
            if not isinstance(thinking, str):
                raise AnthropicMessageConversionError("Anthropic thinking must be a string")
            if signature is not None and not isinstance(signature, str):
                raise AnthropicMessageConversionError("Anthropic signature must be a string")
            if thinking:
                reasoning_parts.append(thinking)
            if signature:
                signatures.append(signature)
        elif block_type == "tool_use":
            if role != "assistant":
                raise AnthropicMessageConversionError("tool_use blocks require assistant role")
            call_id = block.get("id")
            name = block.get("name")
            if not isinstance(call_id, str) or not isinstance(name, str):
                raise AnthropicMessageConversionError(
                    "tool_use blocks require string id and name"
                )
            tool_calls.append(
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(
                            block.get("input", {}),
                            ensure_ascii=False,
                            separators=(",", ":"),
                            sort_keys=True,
                        ),
                    },
                }
            )
        elif block_type == "tool_result":
            if role != "user":
                raise AnthropicMessageConversionError("tool_result blocks require user role")
            call_id = block.get("tool_use_id")
            if not isinstance(call_id, str):
                raise AnthropicMessageConversionError(
                    "tool_result blocks require a string tool_use_id"
                )
            flush_message()
            output.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": _tool_result_content(block.get("content")),
                }
            )
        elif block_type == "redacted_thinking":
            continue
        else:
            raise AnthropicMessageConversionError(
                f"unsupported Anthropic content block: {block_type!r}"
            )
    flush_message()
    return output


def _image_url(block: dict[str, Any]) -> str:
    source = block.get("source")
    if not isinstance(source, dict):
        raise AnthropicMessageConversionError("Anthropic image source must be an object")
    if source.get("type") == "url" and isinstance(source.get("url"), str):
        return source["url"]
    if source.get("type") == "base64":
        media_type = source.get("media_type")
        data = source.get("data")
        if isinstance(media_type, str) and isinstance(data, str):
            return f"data:{media_type};base64,{data}"
    raise AnthropicMessageConversionError("unsupported Anthropic image source")


def _tool_result_content(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        text_parts = [
            block["text"]
            for block in value
            if isinstance(block, dict)
            and block.get("type") == "text"
            and isinstance(block.get("text"), str)
        ]
        if len(text_parts) == len(value):
            return "\n".join(text_parts)
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


__all__ = ["AnthropicMessageConversionError", "normalize_anthropic_request"]
