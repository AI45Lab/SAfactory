from __future__ import annotations

from typing import Any, Optional
import httpx


class HttpServiceClient:
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

    async def request(
        self,
        method: str,
        url: str,
        *,
        json: Any = None,
        timeout_s: Optional[float] = None,
    ) -> httpx.Response:
        timeout = self._timeout_s if timeout_s is None else float(timeout_s)
        return await self.client.request(method=method, url=url, json=json, timeout=timeout)

    async def get(self, url: str, *, timeout_s: Optional[float] = None) -> httpx.Response:
        return await self.request("GET", url, timeout_s=timeout_s)

    async def post(self, url: str, *, json: Any = None, timeout_s: Optional[float] = None) -> httpx.Response:
        return await self.request("POST", url, json=json, timeout_s=timeout_s)

    async def delete(self, url: str, *, timeout_s: Optional[float] = None) -> httpx.Response:
        return await self.request("DELETE", url, timeout_s=timeout_s)

    async def check_envs_ready(self, host: str, port: int, *, timeout_s: float = 5.0) -> bool:
        url = f"http://{host}:{int(port)}/envs"
        try:
            resp = await self.get(url, timeout_s=float(timeout_s))
            return resp.status_code == 200
        except Exception:
            return False
