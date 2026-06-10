from __future__ import annotations

import asyncio
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any

from gateway.config import GatewayConfig
from gateway.models import GatewayRequestContext, GatewaySessionBinding

log = logging.getLogger("gateway.session_resolver")


class SessionResolutionError(Exception):
    pass


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class SessionResolver:
    def __init__(self, cfg: GatewayConfig):
        self.cfg = cfg
        self._bindings: dict[str, GatewaySessionBinding] = {}
        self._lock = asyncio.Lock()
        self._last_eviction_at = 0.0

    async def resolve(
        self,
        payload: dict[str, Any],
        *,
        endpoint: str,
        path_session_id: str,
    ) -> GatewayRequestContext:
        if not path_session_id:
            raise SessionResolutionError("session_id must be supplied in the URL path")

        requested_model = payload.get("model")
        if not isinstance(requested_model, str) or not requested_model:
            raise SessionResolutionError("request body requires a non-empty model field")

        return GatewayRequestContext(
            request_id=str(uuid.uuid4()),
            session_id=path_session_id,
            requested_model=requested_model,
            endpoint=endpoint,
            is_stream=bool(payload.get("stream", False)),
            created_at=_utcnow(),
        )

    async def get_or_create_binding(self, ctx: GatewayRequestContext) -> GatewaySessionBinding:
        await self._evict_expired()
        now = _utcnow()
        async with self._lock:
            binding = self._bindings.get(ctx.session_id)
            if binding is None:
                binding = GatewaySessionBinding(
                    session_id=ctx.session_id,
                    model=ctx.requested_model,
                    upstream_base_url=None,
                    status="active",
                    last_seen_at=now,
                    first_seen_at=now,
                )
                self._bindings[ctx.session_id] = binding
            else:
                binding.model = ctx.requested_model or binding.model
                binding.last_seen_at = now
        return binding

    async def close_session(self, session_id: str, reason: str = "gateway_close") -> GatewaySessionBinding:
        now = _utcnow()
        async with self._lock:
            binding = self._bindings.get(session_id)
            if binding is None:
                binding = GatewaySessionBinding(
                    session_id=session_id,
                    model="",
                    upstream_base_url=None,
                    status="closed",
                    last_seen_at=now,
                    first_seen_at=now,
                    closed_at=now,
                    close_reason=reason,
                )
                self._bindings[session_id] = binding
            else:
                binding.close(reason, now)
            return binding

    async def get_status(self, session_id: str) -> dict[str, Any] | None:
        async with self._lock:
            binding = self._bindings.get(session_id)
            if binding is None:
                return None
            return {
                "session_id": binding.session_id,
                "status": binding.status,
                "last_model": binding.model,
                "upstream_base_url": binding.upstream_base_url,
                "first_seen_at": binding.first_seen_at.isoformat() if binding.first_seen_at else None,
                "last_seen_at": binding.last_seen_at.isoformat(),
                "closed_at": binding.closed_at.isoformat() if binding.closed_at else None,
                "close_reason": binding.close_reason,
                "active_request_count": binding.active_request_count,
                "active_stream_count": binding.active_stream_count,
                "request_count": binding.request_count,
                "error_count": binding.error_count,
                "max_steps": self.cfg.max_steps,
                "llm_step_count": binding.llm_step_count,
                "llm_step_count_by_model": dict(binding.llm_step_count_by_model),
                "truncated": binding.truncated,
                "truncate_reason": binding.truncate_reason,
                "stop_response_sent": binding.stop_response_sent,
                "truncated_models": dict(binding.truncated_models),
                "stop_response_sent_models": sorted(binding.stop_response_sent_models),
            }

    async def _evict_expired(self) -> None:
        now_mono = time.monotonic()
        if now_mono - self._last_eviction_at < 60:
            return
        self._last_eviction_at = now_mono
        cutoff_ts = _utcnow().timestamp() - self.cfg.session_cache_ttl_s
        async with self._lock:
            expired = [
                session_id
                for session_id, binding in self._bindings.items()
                if binding.active_request_count <= 0
                and binding.active_stream_count <= 0
                and binding.last_seen_at.timestamp() < cutoff_ts
            ]
            for session_id in expired:
                self._bindings.pop(session_id, None)
        if expired:
            log.debug("Evicted %d inactive gateway session bindings", len(expired))
