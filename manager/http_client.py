from __future__ import annotations

from typing import Any, Optional

import httpx


class HttpServiceClient:
    """
    Small wrapper around httpx.AsyncClient with readiness probe.
    """

    def __init__(self, timeout_s: float = 10.0, trust_env: bool = True) -> None:
        self._timeout_s = float(timeout_s)
        self._trust_env = bool(trust_env)
        self._client: Optional[httpx.AsyncClient] = None

    async def start(self) -> None:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._timeout_s, trust_env=self._trust_env)

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError("HTTP client not initialized. Call await start() first.")
        return self._client

    async def get(self, url: str, *, timeout_s: Optional[float] = None) -> httpx.Response:
        timeout = self._timeout_s if timeout_s is None else float(timeout_s)
        return await self.client.get(url, timeout=timeout)

    async def post(self, url: str, *, json: Any = None, timeout_s: Optional[float] = None) -> httpx.Response:
        timeout = self._timeout_s if timeout_s is None else float(timeout_s)
        return await self.client.post(url, json=json, timeout=timeout)

    async def check_envs_ready(self, host: str, port: int, *, timeout_s: float = 5.0) -> bool:
        """
        Check GET /envs returns 200.
        """
        url = f"http://{host}:{int(port)}/envs"
        try:
            resp = await self.get(url, timeout_s=float(timeout_s))
            return resp.status_code == 200
        except Exception:
            return False
