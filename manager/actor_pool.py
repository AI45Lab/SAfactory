from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional

from clusters.docker_clusters import DockerContainerBackend, DockerContainerRecord

from .types import AgentKey, PoolEntry

log = logging.getLogger("manager.docker_agent_pool")


class DockerAgentPool:
    """
    Assigns DB rows to OpenClaw Docker containers.

    A lease is ready once a container has been allocated. The actual OpenClaw
    episode runs later through docker exec.
    """

    def __init__(
        self,
        *,
        repo,
        docker: DockerContainerBackend,
        pool_size: int,
        startup_concurrency: int,
    ) -> None:
        self._repo = repo
        self._docker = docker
        self._pool_size = max(0, int(pool_size))
        self._fill_sem = asyncio.Semaphore(max(1, int(startup_concurrency or 1)))
        self._lock = asyncio.Lock()
        self._pool: Dict[AgentKey, PoolEntry] = {}
        self._image_by_env: Dict[str, str] = {}

    def _repo_fetch_batch_size(self, requested: int = 1) -> int:
        return max(1, self._pool_size, int(requested))

    async def reset(self) -> None:
        async with self._lock:
            entries = list(self._pool.values())
            self._pool.clear()
            self._repo.reset_cursor()
        await asyncio.gather(
            *[self._docker.remove(entry.container_id) for entry in entries if entry.container_id],
            return_exceptions=True,
        )

    async def list_instances(self) -> List[PoolEntry]:
        async with self._lock:
            return list(self._pool.values())

    async def prewarm(self, rows: Optional[List[Dict[str, Any]]] = None) -> None:
        if self._pool_size <= 0:
            log.info("pool_size <= 0, skip Docker prewarm")
            return

        if rows is None:
            rows = await self._repo.reserve_rows(
                self._pool_size,
                fetch_batch_size=self._repo_fetch_batch_size(self._pool_size),
            )
        if not rows:
            log.info("no active rows, skip Docker prewarm")
            return

        self._image_by_env = self._repo.get_env_image_map()
        log.info("Docker prewarm start: target_pool_size=%d initial_rows=%d", self._pool_size, len(rows))
        tasks = [asyncio.create_task(self._fill_slot(row)) for row in rows]
        await asyncio.gather(*tasks, return_exceptions=True)
        await self.ensure_capacity()

    async def ensure_capacity(self) -> None:
        while True:
            async with self._lock:
                deficit = max(0, self._pool_size - len(self._pool))
            if deficit <= 0:
                return

            rows = await self._repo.reserve_rows(deficit, fetch_batch_size=self._repo_fetch_batch_size(deficit))
            if not rows:
                log.info("ensure_capacity: no DB rows currently available to fill Docker pool")
                return
            tasks = [asyncio.create_task(self._fill_slot(row)) for row in rows]
            await asyncio.gather(*tasks, return_exceptions=True)

    async def close_and_refill(
        self,
        *,
        env: str,
        env_id: str,
        succeeded: bool,
    ) -> Optional[PoolEntry]:
        key = (str(env), str(env_id))
        async with self._lock:
            old_entry = self._pool.pop(key, None)

        if old_entry is not None:
            await self._docker.release(old_entry.container_id, succeeded=succeeded)

        next_row = await self._repo.reserve_one(fetch_batch_size=self._repo_fetch_batch_size())
        if not next_row:
            log.info("close_and_refill: no DB row currently available to refill Docker pool")
            return None
        return await self._fill_slot(next_row)

    async def _fill_slot(self, row: Dict[str, Any]) -> Optional[PoolEntry]:
        async with self._fill_sem:
            return await self._fill_slot_once(row)

    async def _fill_slot_once(self, row: Dict[str, Any]) -> Optional[PoolEntry]:
        env_name = str(row.get("env_name", "")).strip()
        env_id = str(row.get("env_id", "")).strip()
        image = self._resolve_image(row, env_name)
        if not env_name or not env_id or not image:
            log.error("Invalid DB row for Docker lease: %s", row)
            return None

        try:
            container = await self._docker.acquire(env_name=env_name, image=image)
        except Exception:
            log.error("failed to allocate Docker container for %s/%s", env_name, env_id, exc_info=True)
            return None

        entry = self._build_pool_entry(row=row, container=container)
        async with self._lock:
            self._pool[(env_name, env_id)] = entry
        log.info(
            "allocated Docker lease: agent=%s/%s container=%s reuse=%s",
            env_name,
            env_id,
            container.container_name,
            container.reuse_container,
        )
        return entry

    def _resolve_image(self, row: Dict[str, Any], env_name: str) -> str:
        return str(row.get("image") or row.get("env_image") or self._image_by_env.get(env_name) or "").strip()

    def _build_pool_entry(self, *, row: Dict[str, Any], container: DockerContainerRecord) -> PoolEntry:
        return PoolEntry(
            env_name=str(row.get("env_name") or container.env_name),
            env_id=str(row.get("env_id") or ""),
            row_id=row.get("id"),
            image=container.image,
            job_name=container.container_name,
            env_params=self._normalize_env_params(row.get("env_params")),
            group_id=str(row.get("group_id") or ""),
            status="ready",
            container_id=container.container_id,
            container_name=container.container_name,
            docker_bin=self._docker.docker_bin,
            workdir=container.workdir,
            run_command=container.run_command,
            cleanup_command=container.cleanup_command,
            healthcheck_command=container.healthcheck_command,
            reuse_container=container.reuse_container,
        )

    @staticmethod
    def _normalize_env_params(value: Any) -> Dict[str, Any]:
        if isinstance(value, dict):
            return dict(value)
        if isinstance(value, str) and value.strip():
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                return {"raw": value}
            return dict(parsed) if isinstance(parsed, dict) else {"value": parsed}
        return {}
