from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal

from gateway.config import GatewayConfig
from gateway.llm_router import LLMRouteTarget
from gateway.models import GatewayRequestContext, GatewaySessionBinding


class AdmissionRejected(Exception):
    """Admission rejected by the gateway (draining, queue full, concurrency, etc).

    NB: must NOT be a @dataclass(frozen=True). Python's `raise` statement sets
    `__traceback__` on the raised instance; frozen dataclasses forbid attribute
    assignment, so `raise AdmissionRejected(...)` would itself raise
    `AttributeError: cannot assign to field '__traceback__'` and surface as a
    500 instead of the intended status_code.
    """

    def __init__(self, reason: str, status_code: int) -> None:
        super().__init__(reason)
        self.reason = reason
        self.status_code = status_code


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
        # Per-route semaphores: instead of hard-rejecting when a route is at
        # max_concurrency, requests WAIT for a slot. Hard 503-rejects fail RL
        # episodes on transient concurrency spikes (slow sglang, retry bursts,
        # launcher over-subscription). A bounded wait lets episodes queue and
        # proceed as soon as a slot frees, which is the correct behavior for an
        # RL gateway. Capacity is fixed at first use from target.max_concurrency.
        self._route_semaphores: dict[str, asyncio.Semaphore] = {}
        self.accepted_total = 0
        self.rejected_total = 0

    async def acquire_request(
        self,
        ctx: GatewayRequestContext,
        binding: GatewaySessionBinding,
        target: LLMRouteTarget | None = None,
    ) -> AdmissionDecision:
        # Acquire the route concurrency slot BEFORE taking the admission lock:
        # the semaphore WAITS when the route is at max_concurrency instead of
        # hard-rejecting. Waiting outside the lock means a queued request does
        # not block other admissions (draining checks, other routes, etc.).
        route_sem: asyncio.Semaphore | None = None
        if target is not None and target.max_concurrency > 0:
            route_sem = self._route_semaphore(target.route_model, target.max_concurrency)
            await route_sem.acquire()

        async with self._lock:
            try:
                if self.draining:
                    self.rejected_total += 1
                    raise AdmissionRejected("gateway is draining", 503)

                if self.cfg.max_steps >= 0 and binding.step_count_for(ctx.requested_model) >= self.cfg.max_steps:
                    binding.mark_model_truncated(
                        ctx.requested_model,
                        "max_steps_reached",
                        datetime.now(timezone.utc),
                    )
                    self.accepted_total += 1
                    if route_sem is not None:
                        route_sem.release()
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
                    # The semaphore already guarantees a concurrency slot; just
                    # track the counter for snapshot/reporting.
                    route_inflight = self._per_route_inflight.get(target.route_model, 0)
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
                llm_step_index = binding.increment_step_count(ctx.requested_model)
                return AdmissionDecision(action="forward", llm_step_index=llm_step_index)
            except BaseException:
                # If we got a route slot but a later admission check rejected us,
                # release the slot so a queued request can proceed.
                if route_sem is not None:
                    route_sem.release()
                raise

    def _route_semaphore(self, route_model: str, capacity: int) -> asyncio.Semaphore:
        sem = self._route_semaphores.get(route_model)
        if sem is None:
            sem = asyncio.Semaphore(max(1, capacity))
            self._route_semaphores[route_model] = sem
        return sem

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
                # Release the concurrency slot so a queued request can proceed.
                sem = self._route_semaphores.get(target.route_model)
                if sem is not None:
                    sem.release()

    async def snapshot(self) -> dict[str, int | bool]:
        async with self._lock:
            return {
                "draining": self.draining,
                "inflight_requests": self._inflight_requests,
                "active_streams": self._active_streams,
                "accepted_total": self.accepted_total,
                "rejected_total": self.rejected_total,
            }
