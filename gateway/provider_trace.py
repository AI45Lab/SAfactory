from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any


class ProviderTraceWriter:
    """Build provider-native evidence for database persistence."""

    def __init__(self, capture_mode: str = "off"):
        self.capture_mode = str(capture_mode or "off").strip().lower()
        if self.capture_mode not in {"off", "metadata", "full"}:
            raise ValueError("provider_trace_capture must be one of: off, metadata, full")

    @property
    def enabled(self) -> bool:
        return self.capture_mode != "off"

    async def write(
        self,
        *,
        session_id: str,
        request_id: str,
        llm_step_index: int | None,
        model: str,
        endpoint: str,
        request_body: dict[str, Any],
        response_body: dict[str, Any] | None = None,
        stream_text: str | None = None,
        status_code: int,
        capture_complete: bool,
        capture_error: str | None = None,
    ) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        try:
            signatures = _signature_index(response_body, stream_text)
            persisted_signatures = (
                signatures
                if self.capture_mode == "full"
                else [
                    {key: value for key, value in item.items() if key != "signature"}
                    for item in signatures
                ]
            )
            artifact: dict[str, Any] = {
                "schema_version": 1,
                "boundary": "provider_raw",
                "session_id": session_id,
                "request_id": request_id,
                "llm_step_index": llm_step_index,
                "model": model,
                "endpoint": endpoint,
                "status_code": status_code,
                "request": request_body if self.capture_mode == "full" else None,
                "response": response_body if self.capture_mode == "full" else None,
                "stream_text": stream_text if self.capture_mode == "full" else None,
                "signatures": persisted_signatures,
                "capture": {
                    "complete": bool(capture_complete),
                    "streamed": stream_text is not None,
                    "error": capture_error,
                    "captured_at": datetime.now(timezone.utc).isoformat(),
                },
            }
            return {"artifact": artifact}
        except Exception:
            return None


def _signature_index(
    response_body: dict[str, Any] | None,
    stream_text: str | None,
) -> list[dict[str, Any]]:
    found: list[tuple[str, str, str]] = []
    _collect_signatures(response_body, "$.response", found)
    if stream_text and not found:
        for event_index, event in enumerate(_iter_sse_data(stream_text)):
            _collect_signatures(event, f"$.stream[{event_index}]", found)

    indexed: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for field_name, path, signature in found:
        key = (field_name, signature)
        if key in seen:
            continue
        seen.add(key)
        encoded = signature.encode("utf-8")
        indexed.append(
            {
                "field": field_name,
                "json_path": path,
                "signature": signature,
                "sha256": hashlib.sha256(encoded).hexdigest(),
                "byte_length": len(encoded),
            }
        )
    return indexed


def _collect_signatures(
    value: Any,
    path: str,
    found: list[tuple[str, str, str]],
) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if key in {"signature", "encrypted_content"} and isinstance(child, str) and child:
                found.append((key, child_path, child))
            _collect_signatures(child, child_path, found)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _collect_signatures(child, f"{path}[{index}]", found)


def _iter_sse_data(stream_text: str):
    for block in stream_text.replace("\r\n", "\n").split("\n\n"):
        data_lines = [
            line[5:].lstrip()
            for line in block.splitlines()
            if line.startswith("data:")
        ]
        if not data_lines:
            continue
        raw = "\n".join(data_lines)
        if raw == "[DONE]":
            continue
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            yield value
