"""Process-local dynamic routes: validate, probe every key, then publish once."""
from __future__ import annotations

import asyncio
import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlencode, urlsplit

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

from gateway.config import GatewayConfig, LLMRouteConfig
from gateway.inference_forwarder import InferenceForwarder, RouteProbeError
from gateway.llm_router import LLMRouter, LLMRouteTarget, RouteManagementError

_IDEMPOTENCY_TTL_S = 3600
_MAX_REGISTRATIONS = 1024
_MAX_BODY_BYTES = 64 * 1024
_MAX_API_KEYS = 8


@dataclass
class _Registration:
    fingerprint: str
    operation_id: str
    completed_at: float | None = None
    status_code: int = 0
    result: dict[str, Any] | None = None
    headers: dict[str, str] = field(default_factory=dict)


def _validate(payload: Any) -> tuple[str, LLMRouteConfig, str]:
    def invalid(message: str) -> None:
        raise RouteManagementError("invalid_route_config", message, 422)

    if not isinstance(payload, dict) or set(payload) - {"model", "base_url", "api_key", "probe_endpoint"}:
        invalid("Expected model, base_url, api_key and optional probe_endpoint")
    model = payload.get("model")
    if not isinstance(model, str) or not model.strip() or len(model.strip()) > 150:
        invalid("model must be a non-empty string of at most 150 characters (session storage limit)")
    model = model.strip()
    if any(character.isspace() or ord(character) < 32 or ord(character) == 127 for character in model):
        invalid("model must not contain whitespace or control characters")
    base_url = payload.get("base_url")
    if not isinstance(base_url, str) or not base_url or len(base_url) > 2048:
        invalid("base_url must be an HTTP(S) URL")
    try:
        parsed = urlsplit(base_url)
        if (parsed.scheme not in {"http", "https"} or not parsed.hostname
                or parsed.username is not None or parsed.password is not None
                or parsed.query or parsed.fragment or parsed.port == 0
                or any(character.isspace() or ord(character) < 32 for character in base_url)):
            invalid("base_url must be HTTP(S), without credentials, query or fragment")
    except ValueError:
        invalid("base_url is invalid")
    api_key = payload.get("api_key")
    keys = api_key if isinstance(api_key, list) else ([] if api_key is None else [api_key])
    if isinstance(api_key, list) and not api_key:
        invalid("api_key list must not be empty; use null for unauthenticated upstreams")
    if len(keys) > _MAX_API_KEYS or any(
        not isinstance(key, str) or not key.strip() or len(key) > 4096
        or any(ord(character) < 32 or ord(character) == 127 for character in key)
        for key in keys
    ):
        invalid(f"api_key must be null, a non-empty string, or up to {_MAX_API_KEYS} non-empty strings")
    if len(set(keys)) != len(keys):
        invalid("api_key list must not contain duplicates")
    endpoint = payload.get("probe_endpoint", "chat/completions")
    if not isinstance(endpoint, str) or endpoint not in {"chat/completions", "responses", "messages"}:
        invalid("probe_endpoint must be chat/completions, responses or messages")
    # Normalize the one-key spellings so safe retries have the same identity.
    return model, LLMRouteConfig(base_url=base_url.rstrip("/"), api_key=keys or None), endpoint


class _Registrations:
    def __init__(self, cfg: GatewayConfig, router: LLMRouter, forwarder: InferenceForwarder):
        self.cfg, self.router, self.forwarder = cfg, router, forwarder
        self.records: dict[str, _Registration] = {}
        self.lock = asyncio.Lock()

    async def register(self, key: str, payload: Any) -> tuple[int, dict[str, Any], dict[str, str]]:
        model, route, endpoint = _validate(payload)
        fingerprint = hashlib.sha256(
            json.dumps([model, route.base_url, route.api_key, endpoint], ensure_ascii=False).encode(),
        ).hexdigest()
        async with self.lock:
            cutoff = time.monotonic() - _IDEMPOTENCY_TTL_S
            self.records = {
                name: record for name, record in self.records.items()
                if record.completed_at is None or record.completed_at >= cutoff
            }
            record = self.records.get(key)
            if record is not None:
                if record.fingerprint != fingerprint:
                    raise RouteManagementError("idempotency_conflict", "Idempotency-Key was used with another configuration")
                if record.result is None:
                    raise RouteManagementError("registration_in_progress", "Registration is still running; retry the same key")
                if record.status_code == 201:
                    if not await self.router.has_route_id(record.result["route_id"]):
                        raise RouteManagementError("route_deleted", "This registration was deleted; use a new Idempotency-Key", 410)
                    return 200, record.result, record.headers
                return record.status_code, record.result, record.headers
            if len(self.records) >= _MAX_REGISTRATIONS:
                raise RouteManagementError("registration_quota_exceeded", "Registration history is full; retry later", 429)
            record = _Registration(fingerprint, uuid.uuid4().hex)
            # Lock order is always registration -> router. No network work here.
            await self.router.reserve_registration(model, record.operation_id)
            self.records[key] = record

        key_index = 0
        try:
            async def probe_all() -> None:
                nonlocal key_index
                for key_index, api_key in enumerate(route.api_key or [None]):
                    await self.forwarder.probe(
                        LLMRouteTarget(route_model=model, base_url=route.base_url, api_key=api_key),
                        endpoint, timeout_s=self.cfg.route_probe_timeout_s,
                    )

            await asyncio.wait_for(probe_all(), timeout=self.cfg.route_registration_timeout_s)
            result = await self.router.publish_registration(model, record.operation_id, route, endpoint)
            result["validated_key_count"] = result["api_key_count"]
            record.status_code, record.result = 201, result
        except RouteProbeError as exc:
            # Scrub all supplied keys, including keys echoed by an unusual upstream.
            def safe(value: str | None) -> str | None:
                if value is None:
                    return None
                for api_key in sorted(route.api_key or [], key=len, reverse=True):
                    value = value.replace(api_key, "[REDACTED]")
                return value[:2048]

            record.status_code = exc.status_code
            record.result = {"error": {
                "code": "upstream_probe_failed", "source": "upstream",
                "message": safe(exc.message), "upstream_status_code": exc.upstream_status_code,
                "upstream_error_code": safe(exc.upstream_error_code), "key_index": key_index,
            }}
            if exc.retry_after and safe(exc.retry_after) == exc.retry_after:
                record.headers["Retry-After"] = exc.retry_after
        except asyncio.TimeoutError:
            record.status_code = 504
            record.result = {"error": {
                "code": "registration_timeout", "source": "gateway",
                "message": "Registration probe deadline exceeded", "key_index": key_index,
                "upstream_status_code": None,
            }}
        except asyncio.CancelledError:
            record.status_code = 503
            record.result = {"error": {
                "code": "registration_cancelled", "source": "gateway",
                "message": "Registration was cancelled before publication; use a new Idempotency-Key",
            }}
            raise
        except RouteManagementError as exc:
            record.status_code, record.result = exc.status_code, {"error": exc.error}
        except Exception:
            # Never expose a registration payload or exception containing credentials.
            record.status_code = 500
            record.result = {"error": {
                "code": "registration_failed", "source": "gateway", "message": "Registration failed",
            }}
        finally:
            record.completed_at = time.monotonic()
            await asyncio.shield(self.router.abort_registration(model, record.operation_id))
        return record.status_code, record.result, record.headers


def install_model_route_api(app: FastAPI, cfg: GatewayConfig) -> None:
    @app.exception_handler(RouteManagementError)
    async def management_error(_request: Request, exc: RouteManagementError) -> JSONResponse:
        headers = {"Retry-After": "1"} if exc.error["code"] in {"registration_in_progress", "model_in_use"} else None
        return JSONResponse({"error": exc.error}, status_code=exc.status_code, headers=headers)

    @app.post("/v1/model-routes")
    async def register_route(request: Request) -> Response:
        if not app.state.gateway_ready or app.state.gateway_draining:
            raise RouteManagementError("gateway_unavailable", "Gateway is not accepting registrations", 503)
        key = request.headers.get("idempotency-key", "")
        if not key.strip() or len(key) > 200:
            raise RouteManagementError("invalid_idempotency_key", "A non-empty Idempotency-Key of at most 200 characters is required", 422)
        body = bytearray()
        async for chunk in request.stream():
            body.extend(chunk)
            if len(body) > _MAX_BODY_BYTES:
                raise RouteManagementError("request_too_large", "Registration body exceeds 64 KiB", 413)
        try:
            payload = json.loads(body)
        except (ValueError, UnicodeError):
            raise RouteManagementError("invalid_route_config", "Body must be valid JSON", 422) from None
        router = app.state.gateway_router
        registrations = getattr(app.state, "gateway_registrations", None)
        if registrations is None or registrations.router is not router:
            registrations = _Registrations(cfg, router, app.state.gateway_forwarder)
            app.state.gateway_registrations = registrations
        status, result, cached_headers = await registrations.register(key, payload)
        headers = dict(cached_headers)
        if status == 201:
            headers["Location"] = "/v1/model-routes?" + urlencode({"model": result["model"]})
        return JSONResponse(result, status_code=status, headers=headers)

    @app.get("/v1/model-routes")
    async def list_routes(model: str | None = None) -> dict[str, Any]:
        return {"data": await app.state.gateway_router.describe_routes(model)}

    @app.delete("/v1/model-routes/{route_id}")
    async def delete_route(route_id: str) -> Response:
        await app.state.gateway_router.delete_route(route_id)
        return Response(status_code=204)

    # Only model names are public, just as with the existing metrics endpoint.
    session_root = cfg.base_session_path.rstrip("/")
    api_root = (session_root[:-len("/sessions")] or "/v1") if session_root.endswith("/sessions") else "/v1"

    @app.get(f"{api_root}/models")
    async def list_models() -> dict[str, Any]:
        return {"object": "list", "data": [
            {"id": model, "object": "model"} for model in app.state.gateway_router.list_models()
        ]}
