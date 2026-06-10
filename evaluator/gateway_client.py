from __future__ import annotations

import asyncio
from typing import Any

import httpx


class GatewayClient:
    def __init__(self, *, gateway_base_url: str, timeout_s: float = 10.0) -> None:
        self.gateway_base_url = gateway_base_url.rstrip("/")
        self.timeout_s = timeout_s

    async def close_session(self, session_id: str, reason: str) -> None:
        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            response = await client.post(
                f"{self.gateway_base_url}/{session_id}/close",
                json={"reason": reason},
            )
            response.raise_for_status()

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
            status = await self.get_session_status(session_id)
            if status is None or status.get("status") == "closed":
                return
            if asyncio.get_running_loop().time() >= deadline:
                return
            await asyncio.sleep(poll_interval_s)
