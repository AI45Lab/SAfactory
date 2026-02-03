from __future__ import annotations

import os
import ipaddress
from typing import Any, Optional, List
from urllib.parse import urlparse
import httpx


class HttpServiceClient:
    def __init__(self, timeout_s: float = 10.0, trust_env: bool = True) -> None:
        self._timeout_s = float(timeout_s)
        self._trust_env = bool(trust_env)


        self._client: Optional[httpx.AsyncClient] = None
        self._direct_client: Optional[httpx.AsyncClient] = None
        self._no_proxy_cidrs: List[ipaddress.IPv4Network | ipaddress.IPv6Network] = []

    def _parse_no_proxy_cidrs(self) -> None:
        """parse the CIDR in the env"""
        no_proxy = os.environ.get("NO_PROXY") or os.environ.get("no_proxy")
        if not no_proxy:
            return

        for item in no_proxy.split(","):
            item = item.strip()
            if "/" in item:
                try:
                    self._no_proxy_cidrs.append(ipaddress.ip_network(item, strict=False))
                except ValueError:
                    continue

    async def start(self) -> None:
        if self._client is None:
            self._parse_no_proxy_cidrs()


            self._client = httpx.AsyncClient(
                timeout=self._timeout_s,
                trust_env=self._trust_env,
                limits=httpx.Limits(max_connections=None, max_keepalive_connections=None),
            )


            self._direct_client = httpx.AsyncClient(
                timeout=self._timeout_s,
                trust_env=False,
                proxy=None,
                limits=httpx.Limits(max_connections=None, max_keepalive_connections=None),
            )

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
        if self._direct_client is not None:
            await self._direct_client.aclose()
            self._direct_client = None

    @property
    def client(self) -> httpx.AsyncClient:

        if self._client is None:
            raise RuntimeError("HTTP client not initialized. Call await start() first.")
        return self._client

    def _should_bypass_proxy(self, url: str) -> bool:
        """内部判断逻辑：当前 URL 是否命中 CIDR"""
        if not self._no_proxy_cidrs:
            return False
        try:
            hostname = urlparse(url).hostname
            if not hostname:
                return False
            target_ip = ipaddress.ip_address(hostname)
            return any(target_ip in network for network in self._no_proxy_cidrs)
        except ValueError:

            return False


    async def request(
            self,
            method: str,
            url: str,
            *,
            json: Any = None,
            timeout_s: Optional[float] = None,
    ) -> httpx.Response:
        if self._client is None or self._direct_client is None:
            raise RuntimeError("HTTP client not initialized. Call await start() first.")

        if self._should_bypass_proxy(url):
            use_client = self._direct_client
        else:
            use_client = self._client

        timeout = self._timeout_s if timeout_s is None else float(timeout_s)

        return await use_client.request(
            method=method,
            url=url,
            json=json,
            timeout=timeout
        )

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