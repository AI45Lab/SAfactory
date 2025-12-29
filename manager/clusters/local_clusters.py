from __future__ import annotations

import asyncio
from typing import Dict

from ..binding_plan import BindingPlan
from ..http_client import HttpServiceClient
from ..types import ClusterRegistry, EnvClusterBinding, RayClusterInfo
from .base import ClusterBackend


class LocalHTTPBackend(ClusterBackend):
    """
    Local mode:
      - no RayJob creation
      - all routes go to local HTTP service (host:port)
    """

    def __init__(
        self,
        *,
        host: str,
        http: HttpServiceClient,
        http_port: int,
        poll_interval_s: float = 1.0,
        poll_timeout_s: float = 60.0,
    ) -> None:
        self._host = (host or "").strip() or "127.0.0.1"
        self._http = http
        self._http_port = int(http_port)
        self._poll_interval_s = float(poll_interval_s)
        self._poll_timeout_s = float(poll_timeout_s)

    async def start(self, plan: BindingPlan) -> ClusterRegistry:
        if not plan.env_to_image:
            return ClusterRegistry(clusters_by_image={}, env_bindings={})

        await self._wait_for_local_http_ready()

        clusters_by_image: Dict[str, RayClusterInfo] = {}
        for img in (plan.images_needed or set(plan.env_to_image.values())):
            if not img:
                continue
            clusters_by_image[img] = RayClusterInfo(
                image=img,
                project="local",
                job_name="local",
                head_ip=self._host,
            )

        env_bindings: Dict[str, EnvClusterBinding] = {}
        for env_name, image in plan.env_to_image.items():
            env_bindings[env_name] = EnvClusterBinding(
                env_name=env_name,
                image=image,
                project="local",
                job_name="local",
                head_ip=self._host,
            )

        return ClusterRegistry(clusters_by_image=clusters_by_image, env_bindings=env_bindings)

    async def close(self) -> None:
        # No-op for local mode
        return

    async def _wait_for_local_http_ready(self) -> None:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._poll_timeout_s
        attempt = 0

        while True:
            ok = await self._http.check_envs_ready(self._host, self._http_port)
            if ok:
                print(f"[manager] local HTTP service ready at {self._host}:{self._http_port}")
                return

            attempt += 1
            if loop.time() >= deadline:
                raise RuntimeError(
                    f"Timeout waiting for local HTTP service at {self._host}:{self._http_port}"
                )

            print(
                f"[manager] local HTTP not ready (attempt {attempt}), "
                f"retry in {self._poll_interval_s}s: {self._host}:{self._http_port}"
            )
            await asyncio.sleep(self._poll_interval_s)
