from __future__ import annotations

import asyncio
import random
import uuid
from dataclasses import dataclass, field, replace
from typing import Any

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


class RouteManagementError(Exception):
    def __init__(self, code: str, message: str, status_code: int = 409, **details: Any):
        super().__init__(message)
        self.status_code = status_code
        self.error = {"code": code, "source": "gateway", "message": message, **details}


@dataclass(frozen=True)
class LLMRouteTarget:
    route_model: str
    base_url: str
    api_key: str | None = field(repr=False)
    supports_stream: bool = True
    max_concurrency: int = 256
    anthropic_interleaved_thinking: bool = True
    route_id: str = ""
    lease_id: str = ""


@dataclass
class LLMRouteRuntimeState:
    route_model: str
    healthy: bool = True
    inflight_requests: int = 0
    active_streams: int = 0
    recent_error_rate: float = 0.0
    recent_429_rate: float = 0.0
    route_id: str = field(default_factory=lambda: "rt_" + uuid.uuid4().hex)
    source: str = "config"
    probe_endpoint: str | None = None


class LLMRouter:
    def __init__(self, cfg: GatewayConfig):
        self.cfg = cfg
        self._routes: dict[str, LLMRouteConfig] = dict(cfg.llm_routes or {})
        self._states: dict[str, LLMRouteRuntimeState] = {
            model: LLMRouteRuntimeState(route_model=model) for model in self._routes
        }
        self._lock = asyncio.Lock()
        self._static_models = frozenset(model.strip() for model in self._routes)
        self._pending: dict[str, str] = {}
        self._leases: dict[str, tuple[str, str, bool]] = {}

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
        # Read-only compatibility helper. Inference must use acquire_target().
        async with self._lock:
            return self._select_target(requested_model, is_stream)

    def _select_target(self, requested_model: str, is_stream: bool) -> LLMRouteTarget:
        route = self._routes.get(requested_model)
        if route is None:
            raise ModelNotFoundError(f"model {requested_model!r} is not configured")

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
            route_id=state.route_id,
        )

    async def acquire_target(self, *, requested_model: str, is_stream: bool) -> LLMRouteTarget:
        """Select and reserve in the same critical section used by deletion."""
        async with self._lock:
            target = self._select_target(requested_model, is_stream)
            state = self._states[requested_model]
            if state.inflight_requests >= target.max_concurrency:
                raise LLMRouteUnavailableError("LLM route concurrency limit reached")
            lease_id = uuid.uuid4().hex
            self._leases[lease_id] = (requested_model, state.route_id, is_stream)
            state.inflight_requests += 1
            if is_stream:
                state.active_streams += 1
            return replace(target, lease_id=lease_id)

    async def release_target(self, target: LLMRouteTarget) -> None:
        async with self._lock:
            lease = self._leases.pop(target.lease_id, None)
            if lease is None:
                return
            model, route_id, was_stream = lease
            state = self._states.get(model)
            if state is not None and state.route_id == route_id:
                state.inflight_requests -= 1
                if was_stream:
                    state.active_streams -= 1

    async def mark_route_result(
        self,
        route_model: str | LLMRouteTarget,
        ok: bool,
        latency_ms: float,
        status_code: int = 200,
    ) -> None:
        async with self._lock:
            target = route_model if isinstance(route_model, LLMRouteTarget) else None
            state = self._states.get(target.route_model if target else route_model)
            if state is None:
                return
            if target is not None and target.route_id != state.route_id:
                return
            error_sample = 0.0 if ok else 1.0
            rate_limit_sample = 1.0 if status_code == 429 else 0.0
            state.recent_error_rate = (state.recent_error_rate * 0.9) + (error_sample * 0.1)
            state.recent_429_rate = (state.recent_429_rate * 0.9) + (rate_limit_sample * 0.1)

    async def reserve_registration(self, model: str, operation_id: str) -> None:
        async with self._lock:
            if model in self._static_models:
                raise RouteManagementError("static_model_conflict", "Model name is reserved by YAML configuration")
            if model in self._routes:
                raise RouteManagementError("model_already_exists", "Model is already registered")
            if model in self._pending:
                raise RouteManagementError("registration_in_progress", "Model registration is in progress")
            if len(self._routes) - len(self._static_models) + len(self._pending) >= 256:
                raise RouteManagementError("route_quota_exceeded", "Dynamic route limit reached", 429)
            self._pending[model] = operation_id

    async def publish_registration(
        self, model: str, operation_id: str, route: LLMRouteConfig, probe_endpoint: str,
    ) -> dict[str, Any]:
        async with self._lock:
            if self._pending.get(model) != operation_id:
                raise RouteManagementError("registration_cancelled", "Registration is no longer current")
            self._routes[model] = route
            self._states[model] = LLMRouteRuntimeState(
                route_model=model, source="dynamic", probe_endpoint=probe_endpoint,
            )
            del self._pending[model]
            return self._describe(model)

    async def abort_registration(self, model: str, operation_id: str) -> None:
        async with self._lock:
            if self._pending.get(model) == operation_id:
                del self._pending[model]

    def _describe(self, model: str) -> dict[str, Any]:
        route, state = self._routes[model], self._states[model]
        key_count = len(route.api_key) if isinstance(route.api_key, list) else int(bool(route.api_key))
        return {
            "route_id": state.route_id, "model": model, "status": "active",
            "source": state.source, "base_url": route.base_url,
            "api_key_count": key_count, "probe_endpoint": state.probe_endpoint,
            "inflight_requests": state.inflight_requests, "active_streams": state.active_streams,
        }

    async def describe_routes(self, model: str | None = None) -> list[dict[str, Any]]:
        async with self._lock:
            return [self._describe(name) for name in sorted(self._routes) if model is None or name == model]

    async def has_route_id(self, route_id: str) -> bool:
        async with self._lock:
            return any(state.route_id == route_id for state in self._states.values())

    async def delete_route(self, route_id: str) -> None:
        async with self._lock:
            state = next((state for state in self._states.values() if state.route_id == route_id), None)
            if state is None:
                return
            if state.source == "config":
                raise RouteManagementError("static_model_protected", "YAML routes cannot be deleted")
            if state.inflight_requests or state.active_streams:
                raise RouteManagementError(
                    "model_in_use", "Model has in-flight requests; deletion failed",
                    route_id=route_id, inflight_requests=state.inflight_requests,
                    active_streams=state.active_streams,
                )
            del self._routes[state.route_model]
            del self._states[state.route_model]

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
