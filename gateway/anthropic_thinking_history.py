from __future__ import annotations

import copy
import json
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any


_PRIVATE_BLOCK_TYPES = {"thinking", "redacted_thinking"}


@dataclass(frozen=True)
class _SignedAssistantHistory:
    public_fingerprint: str
    private_blocks: tuple[dict[str, Any], ...]


class AnthropicThinkingHistory:
    """Restore signed Anthropic thinking blocks omitted by an API client."""

    def __init__(
        self,
        *,
        max_sessions: int = 4096,
        max_records_per_session: int = 256,
    ) -> None:
        self._max_sessions = max_sessions
        self._max_records_per_session = max_records_per_session
        self._sessions: OrderedDict[str, list[_SignedAssistantHistory]] = OrderedDict()

    def record_response(self, session_id: str, response: dict[str, Any]) -> bool:
        content = response.get("content")
        if not isinstance(content, list):
            return False
        private_blocks = tuple(
            copy.deepcopy(block)
            for block in content
            if isinstance(block, dict) and block.get("type") in _PRIVATE_BLOCK_TYPES
        )
        if not any(
            block.get("type") == "thinking"
            and isinstance(block.get("signature"), str)
            and bool(block["signature"])
            for block in private_blocks
        ):
            return False
        fingerprint = _public_fingerprint(content)
        records = self._sessions.setdefault(session_id, [])
        candidate = _SignedAssistantHistory(fingerprint, private_blocks)
        if candidate not in records:
            records.append(candidate)
            del records[:-self._max_records_per_session]
        self._sessions.move_to_end(session_id)
        while len(self._sessions) > self._max_sessions:
            self._sessions.popitem(last=False)
        return True

    def restore_request(
        self,
        session_id: str,
        payload: dict[str, Any],
    ) -> tuple[dict[str, Any], int]:
        records = self._sessions.get(session_id)
        messages = payload.get("messages")
        if not records or not isinstance(messages, list):
            return payload, 0

        by_fingerprint: dict[str, list[_SignedAssistantHistory]] = {}
        for record in records:
            by_fingerprint.setdefault(record.public_fingerprint, []).append(record)

        restored_payload: dict[str, Any] | None = None
        restored_count = 0
        for message_index, message in enumerate(messages):
            if not isinstance(message, dict) or message.get("role") != "assistant":
                continue
            content = message.get("content")
            if not isinstance(content, list):
                continue
            if any(
                isinstance(block, dict) and block.get("type") in _PRIVATE_BLOCK_TYPES
                for block in content
            ):
                continue
            candidates = by_fingerprint.get(_public_fingerprint(content), [])
            if len(candidates) != 1:
                continue
            if restored_payload is None:
                restored_payload = copy.deepcopy(payload)
            restored_message = restored_payload["messages"][message_index]
            restored_message["content"] = [
                *copy.deepcopy(list(candidates[0].private_blocks)),
                *restored_message["content"],
            ]
            restored_count += 1

        if restored_payload is None:
            return payload, 0
        self._sessions.move_to_end(session_id)
        return restored_payload, restored_count

    def discard(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)


def _public_fingerprint(content: list[Any]) -> str:
    public_content = [
        block
        for block in content
        if not (
            isinstance(block, dict) and block.get("type") in _PRIVATE_BLOCK_TYPES
        )
    ]
    return json.dumps(
        public_content,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
