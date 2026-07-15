from __future__ import annotations

from typing import Any, Dict, Protocol

from clusters.docker_clusters import DockerContainerBackend, DockerContainerRecord
from clusters.rjob_cluster import RJobClusterBackend
from clusters.sandbox_cluster import SandboxClusterBackend

from .binding_plan import BindingPlan
from .types import PoolEntry


class RuntimeLeaseAllocator(Protocol):
    runtime: str
    startup_concurrency: int

    async def start(self, plan: BindingPlan) -> None:
        ...

    async def allocate(
        self,
        *,
        row_id: Any,
        env_name: str,
        env_id: str,
        image: str,
        env_params: Dict[str, Any],
        group_id: str,
    ) -> PoolEntry:
        ...

    async def release(self, entry: PoolEntry, *, succeeded: bool) -> None:
        ...

    async def remove(self, entry: PoolEntry) -> None:
        ...

    async def close(self) -> None:
        ...


class DockerLeaseAllocator:
    runtime = "docker"

    def __init__(self, *, cluster_cfg: Dict[str, Any]) -> None:
        self._cluster_cfg = dict(cluster_cfg or {})
        docker_cfg = dict(self._cluster_cfg.get("docker", {}) or {})
        self.startup_concurrency = int(docker_cfg.get("startup_concurrency", 8) or 8)
        self._backend = DockerContainerBackend(cluster_cfg=self._cluster_cfg)

    @property
    def docker_bin(self) -> str:
        return self._backend.docker_bin

    async def start(self, plan: BindingPlan) -> None:
        await self._backend.start(plan)

    async def allocate(
        self,
        *,
        row_id: Any,
        env_name: str,
        env_id: str,
        image: str,
        env_params: Dict[str, Any],
        group_id: str,
    ) -> PoolEntry:
        container = await self._backend.acquire(env_name=env_name, image=image)
        return self._build_pool_entry(
            row_id=row_id,
            env_id=env_id,
            env_params=env_params,
            group_id=group_id,
            container=container,
        )

    async def release(self, entry: PoolEntry, *, succeeded: bool) -> None:
        if entry.container_id:
            await self._backend.release(entry.container_id, succeeded=succeeded)

    async def remove(self, entry: PoolEntry) -> None:
        if entry.container_id:
            await self._backend.remove(entry.container_id)

    async def close(self) -> None:
        await self._backend.close()

    def _build_pool_entry(
        self,
        *,
        row_id: Any,
        env_id: str,
        env_params: Dict[str, Any],
        group_id: str,
        container: DockerContainerRecord,
    ) -> PoolEntry:
        return PoolEntry(
            env_name=container.env_name,
            env_id=str(env_id or ""),
            row_id=row_id,
            image=container.image,
            job_name=container.container_name,
            env_params=dict(env_params or {}),
            group_id=str(group_id or ""),
            status="ready",
            runtime=self.runtime,
            runtime_config={},
            resource_id=container.container_id,
            resource_name=container.container_name,
            container_id=container.container_id,
            container_name=container.container_name,
            docker_bin=self._backend.docker_bin,
            workdir=container.workdir,
            run_command=container.run_command,
            result_mode=container.result_mode,
            cleanup_command=container.cleanup_command,
            healthcheck_command=container.healthcheck_command,
            reuse_container=container.reuse_container,
        )


class RJobLeaseAllocator(RJobClusterBackend):
    """RJob lease allocation is implemented by the cluster backend."""


class SandboxLeaseAllocator(SandboxClusterBackend):
    """Sandbox lease allocation is implemented by the cluster backend."""
