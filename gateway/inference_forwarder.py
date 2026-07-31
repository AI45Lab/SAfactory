from __future__ import annotations

import asyncio
import copy
import ipaddress
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import aiohttp

from gateway.admission_control import AdmissionRejected
from gateway.config import GatewayConfig
from gateway.llm_router import LLMRouteTarget, LLMRouteUnavailableError, ModelNotFoundError
from gateway.session_resolver import SessionResolutionError

log = logging.getLogger("gateway.inference_forwarder")
_INTERLEAVED_THINKING_BETA = "interleaved-thinking-2025-05-14"


class SessionClosedError(Exception):
    pass


@dataclass(frozen=True)
class ForwardResult:
    body: dict[str, Any]
    status_code: int
    headers: dict[str, str]
    upstream_latency_ms: float


@dataclass
class StreamForwardContext:
    response: aiohttp.ClientResponse
    status_code: int
    headers: dict[str, str]
    upstream_latency_ms: float
    upstream_started_perf: float

    @property
    def media_type(self) -> str:
        return self.response.headers.get("content-type", "text/event-stream")


class UpstreamHTTPError(Exception):
    def __init__(
        self,
        status_code: int,
        body: Any,
        upstream_request_id: str | None = None,
        upstream_latency_ms: float | None = None,
    ):
        super().__init__(f"upstream returned HTTP {status_code}")
        self.status_code = status_code
        self.body = body
        self.upstream_request_id = upstream_request_id
        self.upstream_latency_ms = upstream_latency_ms


class InferenceForwarder:
    def __init__(self, cfg: GatewayConfig):
        self.cfg = cfg
        self._http_proxy = cfg.upstream_http_proxy.strip() if cfg.upstream_http_proxy else None
        raw_no_proxy = cfg.upstream_no_proxy or ()
        if isinstance(raw_no_proxy, str):
            raw_no_proxy = raw_no_proxy.split(",")
        env_no_proxy = os.environ.get("NO_PROXY") or os.environ.get("no_proxy") or ""
        self._no_proxy = [
            str(item).strip().lower()
            for item in [*raw_no_proxy, *env_no_proxy.split(",")]
            if str(item).strip()
        ]

        timeout = aiohttp.ClientTimeout(
            total=cfg.upstream_request_timeout_s,
            connect=cfg.upstream_connect_timeout_s,
        )
        connector = aiohttp.TCPConnector(
            limit=cfg.upstream_max_connections,
            limit_per_host=0,
            enable_cleanup_closed=True,
        )
        self._client = aiohttp.ClientSession(timeout=timeout, connector=connector)
        log.info(
            "Gateway forwarder initialized: request_timeout_s=%.2f connect_timeout_s=%.2f "
            "max_connections=%d keepalive_connections=%d proxy_configured=%s no_proxy_count=%d",
            cfg.upstream_request_timeout_s,
            cfg.upstream_connect_timeout_s,
            cfg.upstream_max_connections,
            cfg.upstream_keepalive_connections,
            bool(self._http_proxy),
            len(self._no_proxy),
        )

    async def forward_chat(
        self,
        target: LLMRouteTarget,
        payload: dict[str, Any],
        headers: dict[str, str],
    ) -> ForwardResult:
        return await self._forward_json(target, "chat/completions", payload, headers)

    async def forward_responses(
        self,
        target: LLMRouteTarget,
        payload: dict[str, Any],
        headers: dict[str, str],
    ) -> ForwardResult:
        return await self._forward_json(target, "responses", payload, headers)

    async def forward_anthropic_messages(
        self,
        target: LLMRouteTarget,
        payload: dict[str, Any],
        headers: dict[str, str],
    ) -> ForwardResult:
        return await self._forward_json(target, "messages", payload, headers)

    async def forward_anthropic_count_tokens(
        self,
        target: LLMRouteTarget,
        payload: dict[str, Any],
        headers: dict[str, str],
    ) -> ForwardResult:
        return await self._forward_json(target, "messages/count_tokens", payload, headers)

    async def open_chat_stream(
        self,
        target: LLMRouteTarget,
        payload: dict[str, Any],
        headers: dict[str, str],
    ) -> StreamForwardContext:
        return await self._open_stream(target, "chat/completions", payload, headers)

    async def open_responses_stream(
        self,
        target: LLMRouteTarget,
        payload: dict[str, Any],
        headers: dict[str, str],
    ) -> StreamForwardContext:
        return await self._open_stream(target, "responses", payload, headers)

    async def open_anthropic_messages_stream(
        self,
        target: LLMRouteTarget,
        payload: dict[str, Any],
        headers: dict[str, str],
    ) -> StreamForwardContext:
        return await self._open_stream(target, "messages", payload, headers)

    async def _forward_json(
        self,
        target: LLMRouteTarget,
        endpoint: str,
        payload: dict[str, Any],
        headers: dict[str, str],
    ) -> ForwardResult:
        started = time.perf_counter()
        url = self._url(target, endpoint)
        log.debug(
            "Gateway forwarder json POST begin: route_model=%s endpoint=%s url=%s proxy=%s",
            target.route_model,
            endpoint,
            url,
            bool(self._proxy_for(url)),
        )
        try:
            async with self._client.post(
                url,
                json=payload,
                headers=headers,
                proxy=self._proxy_for(url),
            ) as response:
                log.debug(
                    "Gateway forwarder json response headers received: route_model=%s status=%d",
                    target.route_model,
                    response.status,
                )
                body_bytes = await response.read()
                latency_ms = (time.perf_counter() - started) * 1000
                upstream_request_id = response.headers.get("x-request-id") or response.headers.get("x-openai-request-id")
                body = self._parse_bytes_body(body_bytes)
                status_code = response.status
                response_headers = dict(response.headers)
                log.debug(
                    "Gateway forwarder json response body read: route_model=%s status=%d bytes=%d latency_ms=%.2f upstream_request_id=%s",
                    target.route_model,
                    status_code,
                    len(body_bytes),
                    latency_ms,
                    upstream_request_id,
                )
        except Exception as exc:
            latency_ms = (time.perf_counter() - started) * 1000
            setattr(exc, "upstream_latency_ms", latency_ms)
            log.warning(
                "Gateway forwarder json request failed: route_model=%s endpoint=%s latency_ms=%.2f error=%s",
                target.route_model,
                endpoint,
                latency_ms,
                exc,
            )
            raise

        if status_code >= 400:
            log.warning(
                "Gateway forwarder json upstream error: route_model=%s status=%d latency_ms=%.2f upstream_request_id=%s",
                target.route_model,
                status_code,
                latency_ms,
                upstream_request_id,
            )
            raise UpstreamHTTPError(
                status_code,
                body,
                upstream_request_id,
                upstream_latency_ms=latency_ms,
            )

        return ForwardResult(
            body=body if isinstance(body, dict) else {"data": body},
            status_code=status_code,
            headers=response_headers,
            upstream_latency_ms=latency_ms,
        )

    async def _open_stream(
        self,
        target: LLMRouteTarget,
        endpoint: str,
        payload: dict[str, Any],
        headers: dict[str, str],
    ) -> StreamForwardContext:
        started = time.perf_counter()
        url = self._url(target, endpoint)
        log.debug(
            "Gateway forwarder stream POST begin: route_model=%s endpoint=%s url=%s proxy=%s",
            target.route_model,
            endpoint,
            url,
            bool(self._proxy_for(url)),
        )
        stream_headers = {
            **headers,
            "Accept": "text/event-stream",
            "Cache-Control": "no-cache",
            "Accept-Encoding": "identity",
        }
        try:
            response = await self._client.post(
                url,
                json=payload,
                headers=stream_headers,
                proxy=self._proxy_for(url),
            )
        except Exception as exc:
            latency_ms = (time.perf_counter() - started) * 1000
            setattr(exc, "upstream_latency_ms", latency_ms)
            log.warning(
                "Gateway forwarder stream open failed: route_model=%s endpoint=%s latency_ms=%.2f error=%s",
                target.route_model,
                endpoint,
                latency_ms,
                exc,
            )
            raise
        latency_ms = (time.perf_counter() - started) * 1000
        upstream_request_id = response.headers.get("x-request-id") or response.headers.get("x-openai-request-id")
        log.debug(
            "Gateway forwarder stream response opened: route_model=%s status=%d latency_ms=%.2f upstream_request_id=%s",
            target.route_model,
            response.status,
            latency_ms,
            upstream_request_id,
        )

        if response.status >= 400:
            body_bytes = await response.read()
            latency_ms = (time.perf_counter() - started) * 1000
            response.close()
            body = self._parse_bytes_body(body_bytes)
            log.warning(
                "Gateway forwarder stream upstream error: route_model=%s status=%d bytes=%d latency_ms=%.2f upstream_request_id=%s",
                target.route_model,
                response.status,
                len(body_bytes),
                latency_ms,
                upstream_request_id,
            )
            raise UpstreamHTTPError(
                response.status,
                body,
                upstream_request_id,
                upstream_latency_ms=latency_ms,
            )

        return StreamForwardContext(
            response=response,
            status_code=response.status,
            headers=dict(response.headers),
            upstream_latency_ms=latency_ms,
            upstream_started_perf=started,
        )

    def build_upstream_headers(
        self,
        target: LLMRouteTarget,
        session_id: str | None = None,
    ) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
        }
        if target.api_key:
            headers["Authorization"] = f"Bearer {target.api_key}"
        # Forward the Safactory session id out-of-band so session-aware upstreams
        # (e.g. the RL llm_proxy) can bind trajectories to the same session.
        # Plain OpenAI-compatible upstreams ignore the unknown header.
        if session_id:
            headers["X-Safactory-Session-Id"] = session_id
        return headers

    @staticmethod
    def prepare_anthropic_payload(
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        prepared = copy.deepcopy(payload)
        prepared.pop("context_management", None)
        prepared["thinking"] = {"type": "adaptive"}
        output_config = prepared.get("output_config")
        if not isinstance(output_config, dict):
            output_config = {}
        prepared["output_config"] = {**output_config, "effort": "max"}
        prepared["display"] = "summarized"
        prepared["max_tokens"] = 64000
        return prepared

    def build_anthropic_headers(
        self,
        target: LLMRouteTarget,
        inbound_headers: Any,
        *,
        session_id: str | None = None,
        request_id: str | None = None,
        llm_step_index: int | None = None,
    ) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        for name in ("anthropic-version", "user-agent"):
            value = inbound_headers.get(name) if inbound_headers is not None else None
            if value:
                headers[name] = str(value)
        if target.anthropic_interleaved_thinking:
            beta_values = [
                value.strip()
                for value in headers.get("anthropic-beta", "").split(",")
                if value.strip()
            ]
            if _INTERLEAVED_THINKING_BETA not in beta_values:
                beta_values.append(_INTERLEAVED_THINKING_BETA)
            headers["anthropic-beta"] = ",".join(beta_values)
        headers.setdefault("anthropic-version", "2023-06-01")
        if target.api_key:
            headers["x-api-key"] = target.api_key
            headers["Authorization"] = f"Bearer {target.api_key}"
        if session_id:
            headers["X-Safactory-Session-Id"] = session_id
        if request_id:
            headers["X-Safactory-Request-Id"] = request_id
        if llm_step_index is not None:
            headers["X-Safactory-Step-Index"] = str(llm_step_index)
        return headers

    def normalize_error(self, exc: Exception) -> tuple[int, dict[str, Any]]:
        if isinstance(exc, AdmissionRejected):
            return exc.status_code, self._error("admission_rejected", exc.reason)
        if isinstance(exc, SessionResolutionError):
            return 400, self._error("session_resolution_error", str(exc))
        if isinstance(exc, SessionClosedError):
            return 409, self._error("session_closed", str(exc))
        if isinstance(exc, ModelNotFoundError):
            return 404, self._error("model_not_found", str(exc))
        if isinstance(exc, LLMRouteUnavailableError):
            return 503, self._error("llm_route_unavailable", str(exc))
        if isinstance(exc, (asyncio.TimeoutError, aiohttp.ServerTimeoutError)):
            return 504, self._error("upstream_timeout", str(exc) or "upstream timeout")
        if isinstance(exc, aiohttp.ClientError):
            return 503, self._error("upstream_request_error", str(exc))
        if isinstance(exc, UpstreamHTTPError):
            if isinstance(exc.body, dict) and "error" in exc.body:
                return exc.status_code, exc.body
            return exc.status_code, self._error("upstream_http_error", str(exc), payload=exc.body)
        if isinstance(exc, ValueError):
            return 400, self._error("bad_request", str(exc))
        return 500, self._error("gateway_internal_error", str(exc) or exc.__class__.__name__)

    async def close(self) -> None:
        log.info("Gateway forwarder close begin")
        await self._client.close()
        log.info("Gateway forwarder close complete")

    def _proxy_for(self, url: str) -> str | None:
        parsed = urlparse(url)
        proxy = self._http_proxy
        if proxy is None:
            if parsed.scheme == "https":
                proxy = (
                    os.environ.get("HTTPS_PROXY")
                    or os.environ.get("https_proxy")
                    or os.environ.get("HTTP_PROXY")
                    or os.environ.get("http_proxy")
                    or os.environ.get("ALL_PROXY")
                    or os.environ.get("all_proxy")
                )
            else:
                proxy = (
                    os.environ.get("HTTP_PROXY")
                    or os.environ.get("http_proxy")
                    or os.environ.get("ALL_PROXY")
                    or os.environ.get("all_proxy")
                )
        if not proxy:
            return None

        host = (parsed.hostname or "").lower()
        if not host:
            return proxy

        target = f"{host}:{parsed.port}" if parsed.port else host
        for rule in self._no_proxy:
            if rule == "*":
                return None
            if rule == host or rule == target:
                return None
            if rule.startswith(".") and (host == rule[1:] or host.endswith(rule)):
                return None
            try:
                if "/" in rule and ipaddress.ip_address(host) in ipaddress.ip_network(rule, strict=False):
                    return None
            except ValueError:
                pass
        return proxy

    @staticmethod
    def _url(target: LLMRouteTarget, endpoint: str) -> str:
        return f"{target.base_url.rstrip('/')}/{endpoint}"

    @staticmethod
    def _parse_bytes_body(body: bytes) -> Any:
        text = body.decode("utf-8", errors="replace")
        try:
            return json.loads(text)
        except ValueError:
            return {"text": text}

    @staticmethod
    def _error(error_type: str, message: str, payload: Any | None = None) -> dict[str, Any]:
        body: dict[str, Any] = {
            "error": {
                "type": error_type,
                "message": message,
            }
        }
        if payload is not None:
            body["error"]["upstream"] = payload
        return body
