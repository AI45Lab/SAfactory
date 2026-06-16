from __future__ import annotations

import logging
from typing import Any, Dict, List, Protocol

from clusters.docker_clusters import DockerContainerBackend, DockerContainerRecord

from .binding_plan import BindingPlan
from .types import PoolEntry

log = logging.getLogger("manager.runtime_allocator")

_DEFAULT_RUNNER_CONTAINER_PATH = "/tmp/safactory-openclaw-runner.mjs"
_DEFAULT_RUN_COMMAND = f"node {_DEFAULT_RUNNER_CONTAINER_PATH}"


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


class RJobLeaseAllocator:
    runtime = "rjob"

    def __init__(self, *, cluster_cfg: Dict[str, Any]) -> None:
        self._cluster_cfg = dict(cluster_cfg or {})
        self._rjob_cfg = dict(self._cluster_cfg.get("rjob", {}) or {})
        self._env_types = dict(self._cluster_cfg.get("env_types", {}) or {})
        self.startup_concurrency = max(1, int(self._rjob_cfg.get("submit_concurrency", 0) or 1))

    async def start(self, plan: BindingPlan) -> None:
        if not plan.env_to_image:
            log.warning("empty RJob binding plan")
            return
        for env_name in sorted(plan.env_to_image):
            docker_cfg = self._docker_cfg_for_env(env_name)
            rjob_cfg = self._rjob_cfg_for_env(env_name)
            self._validate_mounts(env_name, docker_cfg, rjob_cfg)
        log.info("RJob allocator ready for %d agent image(s)", len(plan.env_to_image))

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
        docker_cfg = self._docker_cfg_for_env(env_name)
        rjob_cfg = self._rjob_cfg_for_env(env_name)
        self._validate_mounts(env_name, docker_cfg, rjob_cfg)

        run_command = str(
            rjob_cfg.get("run_command") or docker_cfg.get("run_command") or _DEFAULT_RUN_COMMAND
        ).strip()
        if not run_command:
            raise RuntimeError(f"RJob lease for {env_name}/{env_id} is missing run_command")

        workdir = str(rjob_cfg.get("workdir") or docker_cfg.get("workdir") or "").strip()
        result_mode = str(
            rjob_cfg.get("result_mode") or docker_cfg.get("result_mode") or docker_cfg.get("run_result_mode") or "json"
        ).strip().lower()
        env = _merge_dicts(docker_cfg.get("env"), rjob_cfg.get("env"))
        runtime_config = dict(rjob_cfg)
        runtime_config["env"] = env
        runtime_config["docker_volumes"] = list(docker_cfg.get("volumes", []) or [])

        pending_name = f"rjob-pending-{str(env_id).replace('-', '')[:12]}"
        return PoolEntry(
            env_name=str(env_name),
            env_id=str(env_id),
            row_id=row_id,
            image=str(image),
            job_name=pending_name,
            env_params=dict(env_params or {}),
            group_id=str(group_id or ""),
            status="ready",
            runtime=self.runtime,
            runtime_config=runtime_config,
            resource_id="",
            resource_name=pending_name,
            workdir=workdir,
            run_command=run_command,
            result_mode=result_mode,
            reuse_container=False,
        )

    async def release(self, entry: PoolEntry, *, succeeded: bool) -> None:
        return

    async def remove(self, entry: PoolEntry) -> None:
        return

    async def close(self) -> None:
        return

    def _docker_cfg_for_env(self, env_name: str) -> Dict[str, Any]:
        env_cfg = dict(self._env_types.get(env_name, {}) or {})
        docker_cfg = dict(self._cluster_cfg.get("docker", {}) or {})
        return _merge_dicts(docker_cfg, env_cfg.get("docker"))

    def _rjob_cfg_for_env(self, env_name: str) -> Dict[str, Any]:
        env_cfg = dict(self._env_types.get(env_name, {}) or {})
        return _merge_dicts(self._rjob_cfg, env_cfg.get("rjob"))

    @staticmethod
    def _validate_mounts(env_name: str, docker_cfg: Dict[str, Any], rjob_cfg: Dict[str, Any]) -> None:
        docker_volumes = docker_cfg.get("volumes", docker_cfg.get("mounts", [])) or []
        rjob_mounts = rjob_cfg.get("mount_config") or rjob_cfg.get("mount") or []
        if docker_volumes and not rjob_mounts:
            raise RuntimeError(
                f"RJob mode cannot map local Docker mounts for agent {env_name!r}. "
                "Configure rjob.mount_config or rjob.mount with cluster-accessible storage."
            )


def _merge_dicts(*values: Any) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    for value in values:
        if isinstance(value, dict):
            merged.update(value)
    return merged
