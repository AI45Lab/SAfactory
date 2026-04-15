import asyncio
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
            connect_timeout_s: Optional[float] = None,
            sock_read_timeout_s: Optional[float] = None,
            trust_env: bool = True,
            max_connections: int = 100,
            max_keepalive_connections: int = 50,
            ttl_dns_cache_s: Optional[int] = None,
    ) -> None:
        self._timeout_s = float(timeout_s)
        self._connect_timeout_s = None if connect_timeout_s is None else float(connect_timeout_s)
        self._sock_read_timeout_s = None if sock_read_timeout_s is None else float(sock_read_timeout_s)
        self._trust_env = bool(trust_env)
        self._max_connections = int(max_connections)
        self._max_keepalive = int(max_keepalive_connections)
        self._ttl_dns_cache_s = None if ttl_dns_cache_s is None else int(ttl_dns_cache_s)

        self._session: Optional[aiohttp.ClientSession] = None
        self._direct_session: Optional[aiohttp.ClientSession] = None
        self._lifecycle_lock = asyncio.Lock()

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
        if self._sessions_ready():
            return

        async with self._lifecycle_lock:
            if self._sessions_ready():
                return

            timeout = aiohttp.ClientTimeout(
                total=self._timeout_s,
                connect=self._connect_timeout_s,
                sock_read=self._sock_read_timeout_s,
            )
            connector_kwargs = {
                "limit": self._max_connections,
                "limit_per_host": 0,
                "enable_cleanup_closed": True,
            }
            if self._ttl_dns_cache_s is not None:
                connector_kwargs["ttl_dns_cache"] = self._ttl_dns_cache_s

            session: Optional[aiohttp.ClientSession] = None
            direct_session: Optional[aiohttp.ClientSession] = None
            try:
                session = aiohttp.ClientSession(
                    timeout=timeout,
                    connector=aiohttp.TCPConnector(**connector_kwargs),
                    trust_env=self._trust_env,
                )
                direct_session = aiohttp.ClientSession(
                    timeout=timeout,
                    connector=aiohttp.TCPConnector(**connector_kwargs),
                    trust_env=False,
                )
            except Exception:
                if direct_session is not None:
                    await direct_session.close()
                if session is not None:
                    await session.close()
                raise

            self._session = session
            self._direct_session = direct_session
            log.debug(
                "http client started: max_connections=%d connect_timeout=%s sock_read_timeout=%s trust_env=%s "
                "ttl_dns_cache_s=%s no_proxy_cidrs=%d",
                self._max_connections,
                self._connect_timeout_s,
                self._sock_read_timeout_s,
                self._trust_env,
                self._ttl_dns_cache_s,
                len(self._no_proxy_cidrs),
            )

    async def close(self) -> None:
        """Closes all active sessions."""
        if self._session is None and self._direct_session is None:
            return

        async with self._lifecycle_lock:
            session = self._session
            direct_session = self._direct_session
            self._session = None
            self._direct_session = None

        if session is not None:
            await session.close()
        if direct_session is not None:
            await direct_session.close()

        log.debug("http client closed")

    def _sessions_ready(self) -> bool:
        return bool(
            self._session is not None
            and not self._session.closed
            and self._direct_session is not None
            and not self._direct_session.closed
        )

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
