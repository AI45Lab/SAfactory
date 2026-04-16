from __future__ import annotations

import asyncio
import logging
from typing import Dict

from ..binding_plan import BindingPlan
from ..http_client import HttpServiceClient
from ..types import ClusterRegistry, EnvClusterBinding, RayClusterInfo
from .base import ClusterBackend

log = logging.getLogger("manager.local_clusters")


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
        require_http_ready: bool = True,
        poll_interval_s: float = 1.0,
        poll_timeout_s: float = 60.0,
    ) -> None:
        self._host = (host or "").strip() or "127.0.0.1"
        self._http = http
        self._http_port = int(http_port)
        self._require_http_ready = bool(require_http_ready)
        self._poll_interval_s = float(poll_interval_s)
        self._poll_timeout_s = float(poll_timeout_s)

    async def start(self, plan: 'BindingPlan') -> 'ClusterRegistry':
        """
        Initializes the ClusterRegistry based on the provided BindingPlan.
        Supports multiple cluster instances per environment based on job counts.
        """
        # 1. Early exit if no environment mapping is provided
        if not plan.env_to_image:
            return ClusterRegistry(clusters_by_id={}, env_bindings={})

        # 2. Wait for the local environment to be ready
        if self._require_http_ready:
            await self._wait_for_local_http_ready()

        # 3. Initialize counts and storage
        env_job_counts = dict(getattr(plan, "env_job_counts", None) or {})
        clusters_by_id: Dict[str, 'RayClusterInfo'] = {}
        env_bindings: Dict[str, 'EnvClusterBinding'] = {}

        # 4. Iterate through environments to set up clusters and bindings
        for env_name, image in plan.env_to_image.items():
            image = (image or "").strip()
            if not image:
                continue

            # Determine the number of cluster slots required for this env (default 1)
            n = max(1, int(env_job_counts.get(env_name, 1) or 1))

            # Create 'n' cluster instances for this specific environment
            for idx in range(1, n + 1):
                # CID uses the format "env_name#index"
                cid = f"{env_name}#{idx}"
                job_name = f"local-{env_name}-{idx}"

                clusters_by_id[cid] = RayClusterInfo(
                    image=image,
                    project="local",
                    job_name=job_name,
                    head_ip=self._host,
                )

            # Define the default binding for this environment (mapping to the first slot)
            env_bindings[env_name] = EnvClusterBinding(
                env_name=env_name,
                image=image,
                project="local",
                job_name=f"local-{env_name}-1",
                head_ip=self._host,
            )

        # 5. Return the populated registry
        return ClusterRegistry(
            clusters_by_id=clusters_by_id,
            env_bindings=env_bindings
        )


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
                log.info("local HTTP service ready at %s:%d", self._host, self._http_port)
                return

            attempt += 1
            if loop.time() >= deadline:
                raise RuntimeError(
                    f"Timeout waiting for local HTTP service at {self._host}:{self._http_port}"
                )

            log.info(
                "local HTTP not ready (attempt %d), retry in %.1fs: %s:%d",
                attempt,
                self._poll_interval_s,
                self._host,
                self._http_port,
            )
            await asyncio.sleep(self._poll_interval_s)
