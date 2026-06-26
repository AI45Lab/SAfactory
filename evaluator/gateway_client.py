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
        close_retries: int = 2,
        retry_backoff_s: float = 1.0,
    ) -> None:
        self.gateway_base_url = gateway_base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.close_timeout_s = close_timeout_s
        self.close_retries = max(0, int(close_retries))
        self.retry_backoff_s = max(0.0, float(retry_backoff_s))

    async def close_session(self, session_id: str, reason: str) -> None:
        timeout = httpx.Timeout(self.close_timeout_s, connect=min(self.timeout_s, self.close_timeout_s))
        attempts = self.close_retries + 1
        for attempt in range(1, attempts + 1):
            try:
                async with httpx.AsyncClient(timeout=timeout) as client:
                    response = await client.post(
                        f"{self.gateway_base_url}/{session_id}/close",
                        json={"reason": reason},
                    )
                    response.raise_for_status()
                    return
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code < 500 or attempt >= attempts:
                    raise
                await self._sleep_before_retry(session_id, attempt, attempts, exc)
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                if attempt >= attempts:
                    raise
                await self._sleep_before_retry(session_id, attempt, attempts, exc)

    async def get_session_status(self, session_id: str) -> dict[str, Any] | None:
        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            response = await client.get(f"{self.gateway_base_url}/{session_id}")
            if response.status_code == 404:
                return None
            response.raise_for_status()
            return response.json()

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
        attempts: int,
        exc: Exception,
    ) -> None:
        delay = self.retry_backoff_s * (2 ** (attempt - 1))
        log.warning(
            "Gateway close_session failed; retrying: session_id=%s attempt=%d/%d delay_s=%.2f error=%s",
            session_id,
            attempt,
            attempts,
            delay,
            exc,
        )
        if delay > 0:
            await asyncio.sleep(delay)
