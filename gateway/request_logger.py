from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from typing import Any

from gateway.config import GatewayConfig
from gateway.llm_router import LLMRouteTarget
from gateway.models import GatewayRequestContext, GatewaySessionBinding

log = logging.getLogger("gateway.request_logger")

SENSITIVE_KEY_PARTS = ("authorization", "api_key", "token", "password", "secret")


@dataclass
class CapturedBody:
    text: str
    total_bytes: int
    captured_bytes: int
    truncated: bool


class StreamBodyCapture:
    def __init__(self, limit_bytes: int, *, enabled: bool = True):
        self.limit_bytes = limit_bytes
        self.enabled = enabled
        self.total_bytes = 0
        self.captured_bytes = 0
        self.truncated = False
        self._chunks: list[bytes] = []

    def append(self, chunk: bytes) -> None:
        self.total_bytes += len(chunk)
        if not self.enabled or not chunk:
            return
        if self.limit_bytes == 0:
            self._chunks.append(bytes(chunk))
            self.captured_bytes += len(chunk)
            return
        remaining = self.limit_bytes - self.captured_bytes
        if remaining <= 0:
            self.truncated = True
            return
        captured = bytes(chunk[:remaining])
        self._chunks.append(captured)
        self.captured_bytes += len(captured)
        if len(chunk) > remaining:
            self.truncated = True

    def snapshot(self) -> CapturedBody:
        text = b"".join(self._chunks).decode("utf-8", errors="replace")
        return CapturedBody(
            text=text,
            total_bytes=self.total_bytes,
            captured_bytes=self.captured_bytes,
            truncated=self.truncated,
        )


class GatewayRequestLogger:
    def __init__(self, cfg: GatewayConfig):
        self.cfg = cfg
        self.enabled = bool(cfg.request_log_enabled and cfg.request_log_path)
        self.body_limit_bytes = int(cfg.request_log_body_limit_bytes)
        self._logger = logging.getLogger(f"gateway.request_log.{id(self)}")
        self._logger.setLevel(logging.INFO)
        self._logger.propagate = False
        self._handler: logging.Handler | None = None

    def start(self) -> None:
        if not self.enabled or self._handler is not None:
            return

        path = str(self.cfg.request_log_path)
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        try:
            self._handler = RotatingFileHandler(
                path,
                maxBytes=int(self.cfg.request_log_max_bytes),
                backupCount=int(self.cfg.request_log_backup_count),
                encoding="utf-8",
            )
            self._handler.setFormatter(logging.Formatter("%(message)s"))
            self._logger.addHandler(self._handler)
            log.info("Gateway request log enabled: %s", path)
        except Exception:
            self.enabled = False
            log.exception("Gateway request log could not be opened: %s", path)

    def close(self) -> None:
        if self._handler is None:
            return
        self._logger.removeHandler(self._handler)
        self._handler.close()
        self._handler = None

    def new_stream_capture(self) -> StreamBodyCapture:
        return StreamBodyCapture(self.body_limit_bytes, enabled=self.enabled)

    async def log_request(
        self,
        ctx: GatewayRequestContext,
        binding: GatewaySessionBinding,
        target: LLMRouteTarget,
        request_body: dict[str, Any],
    ) -> None:
        await self._write(
            {
                "event": "gateway_request",
                **self._base_fields(ctx, binding, target),
                "request": self._safe_body(request_body),
            }
        )

    async def log_response(
        self,
        ctx: GatewayRequestContext,
        binding: GatewaySessionBinding,
        target: LLMRouteTarget,
        *,
        status_code: int,
        response_body: Any,
        latency_ms: float,
        upstream_latency_ms: float | None,
    ) -> None:
        await self._write(
            {
                "event": "gateway_response",
                **self._base_fields(ctx, binding, target),
                "status_code": status_code,
                "latency_ms": latency_ms,
                "upstream_latency_ms": upstream_latency_ms,
                "response": self._safe_body(response_body),
            }
        )

    async def log_stream_response(
        self,
        ctx: GatewayRequestContext,
        binding: GatewaySessionBinding,
        target: LLMRouteTarget,
        *,
        status_code: int,
        stream_body: CapturedBody,
        stream_summary: dict[str, Any],
        latency_ms: float,
        upstream_latency_ms: float | None,
        ttft_ms: float | None,
        output_chunk_count: int,
        client_cancelled: bool,
        upstream_cancelled: bool,
        error_text: str | None,
        upstream_open_latency_ms: float | None = None,
        upstream_stream_total_ms: float | None = None,
    ) -> None:
        response: dict[str, Any] = {
            "stream_text": stream_body.text,
            "stream_total_bytes": stream_body.total_bytes,
            "stream_captured_bytes": stream_body.captured_bytes,
            "stream_truncated": stream_body.truncated,
            "stream_summary": stream_summary,
        }
        if error_text:
            response["error_text"] = error_text

        await self._write(
            {
                "event": "gateway_stream_response",
                **self._base_fields(ctx, binding, target),
                "status_code": status_code,
                "latency_ms": latency_ms,
                "upstream_latency_ms": upstream_latency_ms,
                "upstream_open_latency_ms": upstream_open_latency_ms,
                "upstream_stream_total_ms": upstream_stream_total_ms,
                "ttft_ms": ttft_ms,
                "output_chunk_count": output_chunk_count,
                "client_cancelled": client_cancelled,
                "upstream_cancelled": upstream_cancelled,
                "response": self._safe_body(response),
            }
        )

    async def log_stop_request(
        self,
        ctx: GatewayRequestContext,
        binding: GatewaySessionBinding,
        request_body: dict[str, Any],
        reason: str,
    ) -> None:
        await self._write(
            {
                "event": "gateway_stop_response",
                **self._base_fields(ctx, binding, None),
                "reason": reason,
                "max_steps": self.cfg.max_steps,
                "llm_step_count": binding.step_count_for(ctx.requested_model),
                "request": self._safe_body(request_body),
            }
        )

    async def log_error(
        self,
        *,
        endpoint: str,
        path_session_id: str,
        request_body: Any | None,
        error_body: dict[str, Any],
        error_text: str,
        status_code: int,
        latency_ms: float,
        upstream_latency_ms: float | None = None,
        ctx: GatewayRequestContext | None = None,
        binding: GatewaySessionBinding | None = None,
        target: LLMRouteTarget | None = None,
    ) -> None:
        event: dict[str, Any] = {
            "event": "gateway_error",
            "timestamp": _utc_timestamp(),
            "endpoint": endpoint,
            "path_session_id": path_session_id,
            "status_code": status_code,
            "latency_ms": latency_ms,
            "upstream_latency_ms": upstream_latency_ms,
            "error_text": error_text,
            "error": self._safe_body(error_body),
        }
        if request_body:
            event["request"] = self._safe_body(request_body)
        if ctx is not None:
            event.update(self._base_fields(ctx, binding, target))
        await self._write(event)

    async def _write(self, event: dict[str, Any]) -> None:
        if not self.enabled or self._handler is None:
            return
        event.setdefault("timestamp", _utc_timestamp())
        line = json.dumps(event, ensure_ascii=False, default=str)
        try:
            await asyncio.to_thread(self._logger.info, line)
        except Exception:
            log.exception("Gateway request log write failed")

    def _base_fields(
        self,
        ctx: GatewayRequestContext,
        binding: GatewaySessionBinding | None,
        target: LLMRouteTarget | None,
    ) -> dict[str, Any]:
        return {
            "request_id": ctx.request_id,
            "session_id": ctx.session_id,
            "endpoint": ctx.endpoint,
            "requested_model": ctx.requested_model,
            "is_stream": ctx.is_stream,
            "route_model": target.route_model if target else None,
            "upstream_base_url": target.base_url if target else None,
            "upstream_url": _upstream_url(target, ctx.endpoint),
            "session_status": binding.status if binding else None,
            "session_request_count": binding.request_count if binding else None,
            "llm_step_index": ctx.llm_step_index,
            "synthetic_stop": ctx.synthetic_stop,
            "llm_step_count": binding.step_count_for(ctx.requested_model) if binding else None,
            "session_llm_step_count": binding.llm_step_count if binding else None,
            "llm_step_count_by_model": dict(binding.llm_step_count_by_model) if binding else None,
            "truncated": binding.is_model_truncated(ctx.requested_model) if binding else None,
            "session_truncated": binding.truncated if binding else None,
            "truncate_reason": binding.model_truncate_reason(ctx.requested_model) if binding else None,
            "truncated_models": dict(binding.truncated_models) if binding else None,
        }

    def _safe_body(self, body: Any) -> Any:
        safe = _redact(body) if self.cfg.redact_sensitive_fields else body
        return _limit_json_body(safe, self.body_limit_bytes)


def _limit_json_body(body: Any, limit_bytes: int) -> Any:
    if limit_bytes == 0:
        return body
    encoded = json.dumps(body, ensure_ascii=False, default=str).encode("utf-8")
    if len(encoded) <= limit_bytes:
        return body
    preview = encoded[:limit_bytes].decode("utf-8", errors="replace")
    return {
        "truncated": True,
        "total_bytes": len(encoded),
        "captured_bytes": limit_bytes,
        "preview": preview,
    }


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        redacted = {}
        for key, item in value.items():
            if any(part in str(key).lower() for part in SENSITIVE_KEY_PARTS):
                redacted[key] = "[REDACTED]"
            else:
                redacted[key] = _redact(item)
        return redacted
    if isinstance(value, list):
        return [_redact(item) for item in value]
    return value


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _upstream_url(target: LLMRouteTarget | None, endpoint: str) -> str | None:
    if target is None:
        return None
    return f"{target.base_url.rstrip('/')}/{endpoint}"
