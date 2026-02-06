import os
import ipaddress
import logging
from typing import Optional, List, Union
from urllib.parse import urlparse
import aiohttp

log = logging.getLogger("core.http_client")


class HttpClient:
    """
    A generic asynchronous HTTP client wrapper based on aiohttp.

    Features:
    - Manages aiohttp ClientSession lifecycle.
    - Supports connection pooling configurations.
    - Implements NO_PROXY logic based on CIDR ranges.
    """

    def __init__(
            self,
            timeout_s: float = 10.0,
            trust_env: bool = True,
            max_connections: int = 100,
            max_keepalive_connections: int = 50,
    ) -> None:
        self._timeout_s = float(timeout_s)
        self._trust_env = bool(trust_env)
        self._max_connections = int(max_connections)
        self._max_keepalive = int(max_keepalive_connections)

        self._session: Optional[aiohttp.ClientSession] = None
        self._direct_session: Optional[aiohttp.ClientSession] = None

        self._no_proxy_cidrs: List[Union[ipaddress.IPv4Network, ipaddress.IPv6Network]] = []
        self._parse_no_proxy_cidrs()

    def _parse_no_proxy_cidrs(self) -> None:
        """Parses CIDR ranges from NO_PROXY environment variable."""
        no_proxy = os.environ.get("NO_PROXY") or os.environ.get("no_proxy")
        if not no_proxy:
            return

        for item in no_proxy.split(","):
            item = item.strip()
            # Basic CIDR detection
            if "/" in item:
                try:
                    self._no_proxy_cidrs.append(ipaddress.ip_network(item, strict=False))
                except ValueError:
                    log.warning("Invalid CIDR format in NO_PROXY: %s", item)
                    continue

    async def start(self) -> None:
        """Initializes the client sessions."""
        if self._session is not None:
            return

        timeout = aiohttp.ClientTimeout(total=self._timeout_s)

        # 1. Default Session (respects system proxy settings)
        self._session = aiohttp.ClientSession(
            timeout=timeout,
            connector=aiohttp.TCPConnector(
                limit=self._max_connections,
                limit_per_host=0,
                enable_cleanup_closed=True
            ),
            trust_env=self._trust_env
        )

        # 2. Direct Session (bypasses proxy, used for internal CIDR matches)
        self._direct_session = aiohttp.ClientSession(
            timeout=timeout,
            connector=aiohttp.TCPConnector(
                limit=self._max_connections,
                enable_cleanup_closed=True
            ),
            trust_env=False
        )

    async def close(self) -> None:
        """Closes all active sessions."""
        if self._session:
            await self._session.close()
            self._session = None
        if self._direct_session:
            await self._direct_session.close()
            self._direct_session = None

    def _get_session(self, url: str) -> aiohttp.ClientSession:
        """Selects the appropriate session based on the URL and NO_PROXY rules."""
        if not self._session:
            raise RuntimeError("HttpClient not started. Call await start() first.")

        if not self._no_proxy_cidrs:
            return self._session

        try:
            hostname = urlparse(url).hostname
            if hostname:
                # If hostname is an IP, check if it falls within NO_PROXY CIDRs
                target_ip = ipaddress.ip_address(hostname)
                if any(target_ip in net for net in self._no_proxy_cidrs):
                    return self._direct_session
        except ValueError:
            # Hostname is not an IP address
            pass
        except Exception:
            pass

        return self._session

    def request(self, method: str, url: str, **kwargs) -> aiohttp.ClientResponse:
        """
        Performs an HTTP request using the appropriate session.

        Returns:
            aiohttp.ClientResponse: The response object (must be awaited).
        """
        session = self._get_session(url)
        return session.request(method, url, **kwargs)

    def get(self, url: str, **kwargs) -> aiohttp.ClientResponse:
        return self.request("GET", url, **kwargs)

    def post(self, url: str, **kwargs) -> aiohttp.ClientResponse:
        return self.request("POST", url, **kwargs)

    def delete(self, url: str, **kwargs) -> aiohttp.ClientResponse:
        return self.request("DELETE", url, **kwargs)

    def put(self, url: str, **kwargs) -> aiohttp.ClientResponse:
        return self.request("PUT", url, **kwargs)