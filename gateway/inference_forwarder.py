from __future__ import annotations

import asyncio
import ipaddress
import json
import logging
import math
import os
import time
from dataclasses import dataclass
from datetime import timezone
from email.utils import format_datetime, parsedate_to_datetime
from typing import Any
from urllib.parse import quote, quote_plus, urlparse

import aiohttp

from gateway.admission_control import AdmissionRejected
from gateway.config import GatewayConfig
from gateway.llm_router import LLMRouteTarget, LLMRouteUnavailableError, ModelNotFoundError
from gateway.session_resolver import SessionResolutionError

log = logging.getLogger("gateway.inference_forwarder")
_ANTHROPIC_CLIENT_HEADERS = {
    "accept",
    "anthropic-dangerous-direct-browser-access",
    "anthropic-version",
    "user-agent",
    "x-app",
}
_INTERLEAVED_THINKING_BETA = "interleaved-thinking-2025-05-14"
_PROBE_MAX_RESPONSE_BYTES = 1024 * 1024


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


class RouteProbeError(Exception):
    def __init__(
        self,
        status_code: int,
        message: str,
        upstream_status_code: int | None = None,
        upstream_error_code: str | None = None,
        retry_after: str | None = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.upstream_status_code = upstream_status_code
        self.upstream_error_code = upstream_error_code
        self.retry_after = retry_after


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
        self._client = aiohttp.ClientSession(
            timeout=timeout, connector=connector, cookie_jar=aiohttp.DummyCookieJar(),
        )
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

    async def probe(
        self,
        target: LLMRouteTarget,
        endpoint: str,
        *,
        timeout_s: float,
    ) -> None:
        """Verify one credential with a real inference without publishing the route."""
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise RouteProbeError(400, "probe timeout must be positive and finite")
        payload = _probe_payload(target.route_model, endpoint)
        headers = (
            self.build_anthropic_headers(target, None)
            if endpoint == "messages"
            else self.build_upstream_headers(target)
        )
        url = self._url(target, endpoint)
        status_code: int | None = None
        retry_after: str | None = None
        try:
            async with self._client.post(
                url,
                json=payload,
                headers=headers,
                proxy=self._proxy_for(url),
                timeout=aiohttp.ClientTimeout(total=timeout_s),
                allow_redirects=False,
            ) as response:
                status_code = response.status
                if status_code in (429, 503):
                    retry_after = _probe_retry_after(
                        response.headers.get("Retry-After"), target.api_key
                    )
                chunks = bytearray()
                async for chunk in response.content.iter_chunked(64 * 1024):
                    chunks.extend(chunk)
                    if len(chunks) > _PROBE_MAX_RESPONSE_BYTES:
                        raise RouteProbeError(
                            status_code if 400 <= status_code <= 599 else 502,
                            "upstream probe response exceeds the size limit",
                            upstream_status_code=status_code,
                            retry_after=retry_after,
                        )
                raw_body = bytes(chunks).decode("utf-8", errors="replace")
        except asyncio.CancelledError:
            raise
        except asyncio.TimeoutError:
            raise RouteProbeError(
                504, "upstream route probe timed out", upstream_status_code=status_code,
                retry_after=retry_after,
            ) from None
        except (aiohttp.ClientError, OSError) as exc:
            raise RouteProbeError(
                503,
                _probe_safe_text(f"upstream route probe failed: {exc}", target.api_key),
                upstream_status_code=status_code,
                retry_after=retry_after,
            ) from None

        try:
            body = json.loads(raw_body)
        except (ValueError, TypeError, RecursionError):
            body = None

        has_error = isinstance(body, dict) and (
            ("error" in body and body["error"] is not None)
            or body.get("type") == "error"
            or body.get("status") in ("failed", "cancelled")
            or (
                body.get("status") == "incomplete"
                and not (endpoint == "responses" and _probe_output_budget_exhausted(body))
            )
        )
        if not 200 <= status_code < 300 or has_error:
            message, error_code = _probe_error_details(body, raw_body, status_code)
            raise RouteProbeError(
                status_code if 400 <= status_code <= 599 else 502,
                _probe_safe_text(message, target.api_key),
                upstream_status_code=status_code,
                upstream_error_code=(
                    _probe_safe_text(error_code, target.api_key)
                    if error_code is not None
                    else None
                ),
                retry_after=retry_after,
            )
        if not _valid_probe_response(body, endpoint):
            raise RouteProbeError(
                502,
                f"upstream returned an invalid {endpoint} probe response",
                upstream_status_code=status_code,
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
        *,
        query_string: str | None = None,
    ) -> ForwardResult:
        return await self._forward_json(
            target,
            "messages",
            payload,
            headers,
            query_string=query_string,
        )

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
        *,
        query_string: str | None = None,
    ) -> StreamForwardContext:
        return await self._open_stream(
            target,
            "messages",
            payload,
            headers,
            query_string=query_string,
        )

    async def _forward_json(
        self,
        target: LLMRouteTarget,
        endpoint: str,
        payload: dict[str, Any],
        headers: dict[str, str],
        *,
        query_string: str | None = None,
    ) -> ForwardResult:
        started = time.perf_counter()
        url = self._url(target, endpoint, query_string=query_string)
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
        *,
        query_string: str | None = None,
    ) -> StreamForwardContext:
        started = time.perf_counter()
        url = self._url(target, endpoint, query_string=query_string)
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
        inbound_header_names: set[str] = set()
        if inbound_headers is not None:
            for name, value in inbound_headers.items():
                normalized_name = str(name).lower()
                inbound_header_names.add(normalized_name)
                if (
                    normalized_name in _ANTHROPIC_CLIENT_HEADERS
                    or normalized_name.startswith("x-stainless-")
                ):
                    headers[normalized_name] = str(value)
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
            if "x-api-key" in inbound_header_names:
                headers["x-api-key"] = target.api_key
            if "authorization" in inbound_header_names:
                headers["Authorization"] = f"Bearer {target.api_key}"
            if not {"x-api-key", "authorization"} & inbound_header_names:
                headers["x-api-key"] = target.api_key
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
    def _url(
        target: LLMRouteTarget,
        endpoint: str,
        *,
        query_string: str | None = None,
    ) -> str:
        url = f"{target.base_url.rstrip('/')}/{endpoint}"
        if query_string:
            url = f"{url}?{query_string.lstrip('?')}"
        return url

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


def _probe_payload(model: str, endpoint: str) -> dict[str, Any]:
    payload: dict[str, Any] = {"model": model, "stream": False}
    if endpoint == "responses":
        payload.update(input="Reply with OK.", max_output_tokens=16)
    elif endpoint in {"chat/completions", "messages"}:
        payload.update(
            messages=[{"role": "user", "content": "Reply with OK."}],
            max_tokens=16,
        )
    else:
        raise RouteProbeError(400, "unsupported route probe endpoint")
    return payload


def _probe_safe_text(value: str, api_key: str | None) -> str:
    if api_key:
        variants = {
            api_key,
            quote(api_key, safe=""),
            quote_plus(api_key),
            json.dumps(api_key, ensure_ascii=False)[1:-1],
            json.dumps(api_key, ensure_ascii=True)[1:-1],
        }
        for secret in sorted(variants, key=len, reverse=True):
            value = value.replace(secret, "[REDACTED]")
    return value[:2048]


def _probe_retry_after(value: str | None, api_key: str | None) -> str | None:
    if not value or len(value) > 128 or "\r" in value or "\n" in value:
        return None
    value = value.strip()
    if not value.isascii() or _probe_safe_text(value, api_key) != value:
        return None
    if value.isdigit():
        return value
    try:
        parsed = parsedate_to_datetime(value)
        if parsed.tzinfo is None:
            return None
        return format_datetime(parsed.astimezone(timezone.utc), usegmt=True)
    except (ValueError, TypeError, OverflowError):
        return None


def _probe_error_details(
    body: Any, raw_body: str, status_code: int
) -> tuple[str, str | None]:
    fallback = f"upstream route probe returned HTTP {status_code}"
    if not isinstance(body, dict):
        return raw_body.strip() or fallback, None
    error = body.get("error")
    source = error if isinstance(error, dict) else body
    message = source.get("message") or source.get("detail")
    if not message and isinstance(error, str):
        message = error
    if not message and body.get("status") in ("failed", "cancelled", "incomplete"):
        message = f"upstream route probe response status is {body['status']}"
    code = source.get("code")
    if code is None:
        code = source.get("type")
    return str(message or fallback), str(code) if code is not None else None


def _has_probe_text(content: Any, *, block_type: str) -> bool:
    if isinstance(content, str):
        return bool(content.strip())
    return isinstance(content, list) and any(
        isinstance(block, dict)
        and block.get("type") == block_type
        and isinstance(block.get("text"), str)
        and bool(block["text"].strip())
        for block in content
    )


def _probe_output_budget_exhausted(body: dict[str, Any]) -> bool:
    details = body.get("incomplete_details")
    return (
        body.get("status") == "incomplete"
        and isinstance(details, dict)
        and details.get("reason") == "max_output_tokens"
    )


def _valid_probe_reasoning(item: dict[str, Any]) -> bool:
    summary = item.get("summary")
    return (
        item.get("type") == "reasoning"
        and isinstance(item.get("id"), str)
        and bool(item["id"])
        and isinstance(summary, list)
        and all(
            isinstance(block, dict)
            and block.get("type") == "summary_text"
            and isinstance(block.get("text"), str)
            for block in summary
        )
    )


def _valid_probe_response(body: Any, endpoint: str) -> bool:
    if not isinstance(body, dict):
        return False
    if endpoint == "chat/completions":
        choices = body.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            return False
        choice = choices[0]
        message = choice.get("message")
        return (
            isinstance(message, dict)
            and message.get("role") == "assistant"
            and choice.get("error") is None
            and message.get("error") is None
            and isinstance(choice.get("finish_reason"), str)
            and choice["finish_reason"] not in {"", "error"}
            and (
                _has_probe_text(message.get("content"), block_type="text")
                or (
                    choice["finish_reason"] == "length"
                    and "content" in message
                    and message["content"] in (None, "", [])
                )
            )
        )
    if endpoint == "responses":
        output = body.get("output")
        budget_exhausted = _probe_output_budget_exhausted(body)
        if (
            body.get("status") != "completed" and not budget_exhausted
        ) or not isinstance(output, list):
            return False
        if any(
            not isinstance(item, dict)
            or item.get("error") is not None
            or item.get("status") not in (
                (None, "completed", "incomplete") if budget_exhausted else (None, "completed")
            )
            for item in output
        ):
            return False
        return any(
            (
                item.get("type") == "message"
                and item.get("role") == "assistant"
                and isinstance(item.get("content"), list)
                and _has_probe_text(item.get("content"), block_type="output_text")
            )
            or (budget_exhausted and _valid_probe_reasoning(item))
            for item in output
        )
    return (
        body.get("type") == "message"
        and body.get("role") == "assistant"
        and isinstance(body.get("content"), list)
        and isinstance(body.get("stop_reason"), str)
        and body["stop_reason"] not in {"", "error"}
        and _has_probe_text(body.get("content"), block_type="text")
    )
