from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_SAFE_SEGMENT_RE = re.compile(r"[^A-Za-z0-9._-]+")


class ProviderTraceWriter:
    """Build provider-native evidence and persist an external audit copy."""

    def __init__(self, root: str | None, capture_mode: str = "off"):
        self.root = Path(root).expanduser().resolve() if root else None
        self.capture_mode = str(capture_mode or "off").strip().lower()
        if self.capture_mode not in {"off", "metadata", "full"}:
            raise ValueError("provider_trace_capture must be one of: off, metadata, full")
        if self.capture_mode != "off" and self.root is None:
            raise ValueError("provider_trace_dir is required when provider trace capture is enabled")

    @property
    def enabled(self) -> bool:
        return self.capture_mode != "off" and self.root is not None

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
            return await asyncio.to_thread(
                self._write_sync,
                session_id=session_id,
                request_id=request_id,
                llm_step_index=llm_step_index,
                model=model,
                endpoint=endpoint,
                request_body=request_body,
                response_body=response_body,
                stream_text=stream_text,
                status_code=status_code,
                capture_complete=capture_complete,
                capture_error=capture_error,
            )
        except Exception as exc:
            return {
                "schema_version": 1,
                "boundary": "provider_raw",
                "capture_mode": self.capture_mode,
                "capture_complete": False,
                "capture_error": type(exc).__name__,
                "signature_count": 0,
                "signature_total_bytes": 0,
            }

    def _write_sync(
        self,
        *,
        session_id: str,
        request_id: str,
        llm_step_index: int | None,
        model: str,
        endpoint: str,
        request_body: dict[str, Any],
        response_body: dict[str, Any] | None,
        stream_text: str | None,
        status_code: int,
        capture_complete: bool,
        capture_error: str | None,
    ) -> dict[str, Any]:
        assert self.root is not None
        signatures = _signature_index(response_body, stream_text)
        persisted_signatures = (
            signatures
            if self.capture_mode == "full"
            else [{key: value for key, value in item.items() if key != "signature"} for item in signatures]
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
        payload = json.dumps(
            artifact,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
        digest = hashlib.sha256(payload).hexdigest()
        target: Path | None = None
        external_write_error: str | None = None
        try:
            self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
            os.chmod(self.root, 0o700)
            session_dir = self.root / _safe_segment(session_id)
            session_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
            os.chmod(session_dir, 0o700)
            target = session_dir / f"{_safe_segment(request_id)}.json"
            fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=session_dir)
            try:
                os.fchmod(fd, 0o600)
                with os.fdopen(fd, "wb") as handle:
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, target)
                os.chmod(target, 0o600)
            except BaseException:
                try:
                    os.close(fd)
                except OSError:
                    pass
                Path(temporary).unlink(missing_ok=True)
                raise
        except Exception as exc:
            target = None
            external_write_error = type(exc).__name__

        return {
            "schema_version": 1,
            "boundary": "provider_raw",
            "capture_mode": self.capture_mode,
            "artifact_path": str(target) if target is not None else None,
            "artifact_sha256": digest,
            "signature_count": len(signatures),
            "signature_total_bytes": sum(item["byte_length"] for item in signatures),
            "capture_complete": bool(capture_complete) and external_write_error is None,
            "capture_error": (
                "external_artifact_write_failed"
                if external_write_error
                else "capture_incomplete"
                if capture_error
                else None
            ),
            "artifact": artifact,
        }


def _safe_segment(value: str) -> str:
    cleaned = _SAFE_SEGMENT_RE.sub("_", str(value)).strip("._")
    return cleaned[:200] or "unknown"


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
