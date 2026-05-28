from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal

from gateway.config import GatewayConfig
from gateway.llm_router import LLMRouteTarget
from gateway.models import GatewayRequestContext, GatewaySessionBinding


@dataclass(frozen=True)
class AdmissionRejected(Exception):
    reason: str
    status_code: int


@dataclass(frozen=True)
class AdmissionDecision:
    action: Literal["forward", "stop"]
    llm_step_index: int | None = None
    stop_reason: str | None = None


class AdmissionController:
    def __init__(self, cfg: GatewayConfig):
        self.cfg = cfg
        self.draining = False
        self._lock = asyncio.Lock()
        self._inflight_requests = 0
        self._active_streams = 0
        self._per_session_inflight: dict[str, int] = {}
        self._per_route_inflight: dict[str, int] = {}
        self._request_acquired: set[str] = set()
        self._route_acquired: set[tuple[str, str]] = set()
        self.accepted_total = 0
        self.rejected_total = 0

    async def acquire_request(
        self,
        ctx: GatewayRequestContext,
        binding: GatewaySessionBinding,
        target: LLMRouteTarget | None = None,
    ) -> AdmissionDecision:
        async with self._lock:
            if self.draining:
                self.rejected_total += 1
                raise AdmissionRejected("gateway is draining", 503)

            if self.cfg.max_steps >= 0 and binding.llm_step_count >= self.cfg.max_steps:
                binding.mark_truncated("max_steps_reached", datetime.now(timezone.utc))
                self.accepted_total += 1
                return AdmissionDecision(action="stop", stop_reason="max_steps_reached")

            if self._inflight_requests >= self.cfg.max_inflight_requests:
                self.rejected_total += 1
                raise AdmissionRejected("gateway inflight limit reached", 503)

            if ctx.is_stream and self._active_streams >= self.cfg.max_active_streams:
                self.rejected_total += 1
                raise AdmissionRejected("gateway active stream limit reached", 503)

            session_inflight = self._per_session_inflight.get(ctx.session_id, 0)
            if session_inflight >= self.cfg.per_session_max_inflight:
                self.rejected_total += 1
                raise AdmissionRejected("per-session inflight limit reached", 429)

            if target is not None:
                route_inflight = self._per_route_inflight.get(target.route_model, 0)
                if route_inflight >= target.max_concurrency:
                    self.rejected_total += 1
                    raise AdmissionRejected("LLM route concurrency limit reached", 503)
                self._per_route_inflight[target.route_model] = route_inflight + 1
                self._route_acquired.add((ctx.request_id, target.route_model))

            self._inflight_requests += 1
            if ctx.is_stream:
                self._active_streams += 1
                binding.active_stream_count += 1
            binding.active_request_count += 1
            self._per_session_inflight[ctx.session_id] = session_inflight + 1
            self._request_acquired.add(ctx.request_id)
            self.accepted_total += 1
            binding.llm_step_count += 1
            return AdmissionDecision(action="forward", llm_step_index=binding.llm_step_count)

    async def release(
        self,
        ctx: GatewayRequestContext | None,
        binding: GatewaySessionBinding | None,
        target: LLMRouteTarget | None = None,
    ) -> None:
        if ctx is None:
            return
        async with self._lock:
            if ctx.request_id in self._request_acquired:
                self._request_acquired.discard(ctx.request_id)
                self._inflight_requests = max(0, self._inflight_requests - 1)
                if ctx.is_stream:
                    self._active_streams = max(0, self._active_streams - 1)
                    if binding is not None:
                        binding.active_stream_count = max(0, binding.active_stream_count - 1)
                if binding is not None:
                    binding.active_request_count = max(0, binding.active_request_count - 1)

                session_inflight = max(0, self._per_session_inflight.get(ctx.session_id, 0) - 1)
                if session_inflight:
                    self._per_session_inflight[ctx.session_id] = session_inflight
                else:
                    self._per_session_inflight.pop(ctx.session_id, None)

            if target is not None and (ctx.request_id, target.route_model) in self._route_acquired:
                self._route_acquired.discard((ctx.request_id, target.route_model))
                route_inflight = max(0, self._per_route_inflight.get(target.route_model, 0) - 1)
                if route_inflight:
                    self._per_route_inflight[target.route_model] = route_inflight
                else:
                    self._per_route_inflight.pop(target.route_model, None)

    async def snapshot(self) -> dict[str, int | bool]:
        async with self._lock:
            return {
                "draining": self.draining,
                "inflight_requests": self._inflight_requests,
                "active_streams": self._active_streams,
                "accepted_total": self.accepted_total,
                "rejected_total": self.rejected_total,
            }
