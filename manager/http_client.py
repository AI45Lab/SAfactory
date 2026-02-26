from __future__ import annotations
import logging
from core.http.http_client import HttpClient

# Define logger identity
log = logging.getLogger("manager.http_client")


class HttpServiceClient(HttpClient):
    """
    Manager-specific HTTP client extending the core HttpClient.
    """

    async def check_envs_ready(self, host: str, port: int, *, timeout_s: float = 5.0) -> bool:
        """Checks if the environment service on the target host is ready."""
        url = f"http://{host}:{int(port)}/envs"
        try:
            # Explicitly passing timeout for this specific check
            async with await self.get(url, timeout=timeout_s) as resp:
                return resp.status == 200
        except Exception:
            return False