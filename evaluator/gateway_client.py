from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx

log = logging.getLogger("evaluator.gateway_client")


class GatewayClient:
    def __init__(
        self,
        *,
        gateway_base_url: str,
        timeout_s: float = 10.0,
        close_timeout_s: float = 120.0,
        close_retries: int = 1,
        retry_backoff_s: float = 10.0,
    ) -> None:
        self.gateway_base_url = gateway_base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.close_timeout_s = close_timeout_s
        self.close_retries = max(0, int(close_retries))
        self.retry_backoff_s = max(0.0, float(retry_backoff_s))
        self._client = httpx.AsyncClient(timeout=self.timeout_s)

    async def close_session(
        self,
        session_id: str,
        reason: str,
        completion_mode: str = "complete",
    ) -> dict[str, Any]:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.close_timeout_s
        attempt = 0
        last_error: Exception | None = None
        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                log.warning(
                    "Gateway close_session polling timed out: session_id=%s timeout_s=%.1f error=%s",
                    session_id,
                    self.close_timeout_s,
                    last_error,
                )
                return {
                    "session_id": session_id,
                    "status": "closed",
                    "drained": False,
                    "telemetry_status": "timeout",
                    "client_timed_out": True,
                    "completion_mode": completion_mode,
                }
            try:
                request_timeout = min(self.timeout_s, remaining)
                response = await self._client.post(
                    f"{self.gateway_base_url}/{session_id}/close",
                    json={"reason": reason, "completion_mode": completion_mode},
                    timeout=httpx.Timeout(request_timeout),
                )
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, dict):
                    raise ValueError("Gateway close response must be a JSON object")
                if payload.get("status") != "closing":
                    return payload
                delay = self._retry_delay(response, attempt)
                attempt += 1
                await asyncio.sleep(min(delay, max(0.0, deadline - loop.time())))
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code < 500:
                    raise
                last_error = exc
                delay = self._retry_delay(exc.response, attempt)
                attempt += 1
                await self._sleep_before_retry(session_id, attempt, delay, exc, deadline)
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = exc
                delay = self._retry_delay(None, attempt)
                attempt += 1
                await self._sleep_before_retry(session_id, attempt, delay, exc, deadline)

    async def get_latest_success_step(self, session_id: str, model: str) -> int | None:
        response = await self._client.get(
            f"{self.gateway_base_url}/{session_id}/latest-success-step",
            params={"model": model},
        )
        response.raise_for_status()
        payload = response.json()
        step_id = payload.get("step_id") if isinstance(payload, dict) else None
        return int(step_id) if step_id is not None else None

    async def get_session_status(self, session_id: str) -> dict[str, Any] | None:
        response = await self._client.get(f"{self.gateway_base_url}/{session_id}")
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return response.json()

    async def clear_session_cache(self, session_ids: list[str]) -> dict[str, Any]:
        if not session_ids:
            return {"session_ids": [], "removed": {}}
        response = await self._client.post(
            f"{self.gateway_base_url}/cache/cleanup",
            json={"session_ids": session_ids},
        )
        response.raise_for_status()
        return response.json()

    async def aclose(self) -> None:
        await self._client.aclose()

    async def wait_telemetry_flush(
        self,
        session_id: str,
        *,
        timeout_s: float = 10.0,
        poll_interval_s: float = 0.2,
    ) -> None:
        deadline = asyncio.get_running_loop().time() + timeout_s
        while True:
            try:
                status = await self.get_session_status(session_id)
            except httpx.HTTPError as exc:
                if asyncio.get_running_loop().time() >= deadline:
                    log.warning(
                        "Gateway telemetry flush status check failed after timeout: session_id=%s error=%s",
                        session_id,
                        exc,
                    )
                    return
                await asyncio.sleep(poll_interval_s)
                continue
            if status is None or status.get("status") == "closed":
                return
            if asyncio.get_running_loop().time() >= deadline:
                return
            await asyncio.sleep(poll_interval_s)

    async def _sleep_before_retry(
        self,
        session_id: str,
        attempt: int,
        delay: float,
        exc: Exception,
        deadline: float,
    ) -> None:
        log.warning(
            "Gateway close_session failed; retrying: session_id=%s attempt=%d delay_s=%.2f error=%s",
            session_id,
            attempt,
            delay,
            exc,
        )
        remaining = deadline - asyncio.get_running_loop().time()
        if delay > 0 and remaining > 0:
            await asyncio.sleep(min(delay, remaining))

    def _retry_delay(self, response: httpx.Response | None, attempt: int) -> float:
        retry_after = 0.0
        if response is not None:
            try:
                retry_after = float(response.headers.get("Retry-After") or 0.0)
            except (TypeError, ValueError):
                retry_after = 0.0
        base = max(10.0, self.retry_backoff_s, retry_after)
        return min(40.0, base * (2 ** max(0, attempt)))
