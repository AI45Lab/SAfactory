from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass

from gateway.config import GatewayConfig, LLMRouteConfig
from gateway.models import GatewayRequestContext, GatewaySessionBinding


def _pick_api_key(api_key: str | list[str] | None) -> str | None:
    if isinstance(api_key, list):
        return random.choice(api_key) if api_key else None
    return api_key


class ModelNotFoundError(Exception):
    pass


class LLMRouteUnavailableError(Exception):
    pass


@dataclass(frozen=True)
class LLMRouteTarget:
    route_model: str
    base_url: str
    api_key: str | None
    supports_stream: bool = True
    max_concurrency: int = 256
    anthropic_interleaved_thinking: bool = True


@dataclass
class LLMRouteRuntimeState:
    route_model: str
    healthy: bool = True
    inflight_requests: int = 0
    active_streams: int = 0
    recent_error_rate: float = 0.0
    recent_429_rate: float = 0.0


class LLMRouter:
    def __init__(self, cfg: GatewayConfig):
        self.cfg = cfg
        self._routes: dict[str, LLMRouteConfig] = dict(cfg.llm_routes or {})
        self._states: dict[str, LLMRouteRuntimeState] = {
            model: LLMRouteRuntimeState(route_model=model) for model in self._routes
        }
        self._lock = asyncio.Lock()

    async def select_target(
        self,
        ctx: GatewayRequestContext,
        binding: GatewaySessionBinding,
    ) -> LLMRouteTarget:
        return await self.select_standard_target(
            requested_model=ctx.requested_model,
            is_stream=ctx.is_stream,
        )

    async def select_standard_target(
        self,
        *,
        requested_model: str,
        is_stream: bool,
    ) -> LLMRouteTarget:
        route = self._routes.get(requested_model)
        if route is None:
            raise ModelNotFoundError(f"model {requested_model!r} is not configured")

        async with self._lock:
            state = self._states[requested_model]
            if not state.healthy:
                raise LLMRouteUnavailableError(f"model route {requested_model!r} is unhealthy")

        if is_stream and not route.supports_stream:
            raise LLMRouteUnavailableError(f"model route {requested_model!r} does not support streaming")

        max_concurrency = (
            route.max_concurrency
            if route.max_concurrency is not None
            else self.cfg.per_llm_route_max_concurrency
        )
        return LLMRouteTarget(
            route_model=requested_model,
            base_url=route.base_url,
            api_key=_pick_api_key(route.api_key),
            supports_stream=route.supports_stream,
            max_concurrency=max_concurrency,
            anthropic_interleaved_thinking=route.anthropic_interleaved_thinking,
        )

    async def on_acquire(self, route_model: str, *, is_stream: bool) -> None:
        async with self._lock:
            state = self._states[route_model]
            state.inflight_requests += 1
            if is_stream:
                state.active_streams += 1

    async def on_release(self, route_model: str, *, is_stream: bool) -> None:
        async with self._lock:
            state = self._states[route_model]
            state.inflight_requests = max(0, state.inflight_requests - 1)
            if is_stream:
                state.active_streams = max(0, state.active_streams - 1)

    async def mark_route_result(
        self,
        route_model: str,
        ok: bool,
        latency_ms: float,
        status_code: int = 200,
    ) -> None:
        async with self._lock:
            state = self._states.get(route_model)
            if state is None:
                return
            error_sample = 0.0 if ok else 1.0
            rate_limit_sample = 1.0 if status_code == 429 else 0.0
            state.recent_error_rate = (state.recent_error_rate * 0.9) + (error_sample * 0.1)
            state.recent_429_rate = (state.recent_429_rate * 0.9) + (rate_limit_sample * 0.1)

    def list_models(self) -> list[str]:
        return sorted(self._routes)

    async def snapshot(self) -> dict[str, dict[str, float | int | bool | str]]:
        async with self._lock:
            return {
                model: {
                    "healthy": state.healthy,
                    "inflight_requests": state.inflight_requests,
                    "active_streams": state.active_streams,
                    "recent_error_rate": state.recent_error_rate,
                    "recent_429_rate": state.recent_429_rate,
                }
                for model, state in self._states.items()
            }
