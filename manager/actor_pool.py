from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional

from .runtime_allocator import RuntimeLeaseAllocator
from .types import AgentKey, PoolEntry

log = logging.getLogger("manager.runtime_agent_pool")


class RuntimeAgentPool:
    """
    Assigns DB rows to runtime leases.

    Docker mode allocates a real container before the lease becomes ready.
    RJob mode only reserves the row and attaches RJob runtime config; the actual
    remote job is submitted later by the episode runner.
    """

    def __init__(
        self,
        *,
        repo,
        allocator: RuntimeLeaseAllocator,
        pool_size: int,
        startup_concurrency: int,
        row_wait_timeout_s: float = 60.0,
        row_fetch_timeout_s: float = 30.0,
    ) -> None:
        self._repo = repo
        self._allocator = allocator
        self._pool_size = max(0, int(pool_size))
        self._fill_sem = asyncio.Semaphore(max(1, int(startup_concurrency or 1)))
        self._row_wait_timeout_s = max(1.0, float(row_wait_timeout_s or 60.0))
        self._row_fetch_timeout_s = max(1.0, float(row_fetch_timeout_s or 30.0))
        self._lock = asyncio.Lock()
        self._pool: Dict[AgentKey, PoolEntry] = {}
        self._image_by_env: Dict[str, str] = {}
        self._background_tasks: set[asyncio.Task] = set()

    def _repo_fetch_batch_size(self, requested: int = 1) -> int:
        return max(1, self._pool_size, int(requested))

    async def reset(self) -> None:
        async with self._lock:
            entries = list(self._pool.values())
            self._pool.clear()
            self._repo.reset_cursor()
        await asyncio.gather(
            *[self._allocator.remove(entry) for entry in entries],
            return_exceptions=True,
        )
        await self.drain_background_tasks(timeout_s=30.0)

    async def list_instances(self) -> List[PoolEntry]:
        async with self._lock:
            return list(self._pool.values())

    async def prewarm(self, rows: Optional[List[Dict[str, Any]]] = None) -> None:
        if self._pool_size <= 0:
            log.debug("pool_size <= 0, skip %s prewarm", self._allocator.runtime)
            return

        if rows is None:
            rows = await self._repo.reserve_rows(
                self._pool_size,
                fetch_batch_size=self._repo_fetch_batch_size(self._pool_size),
                fetch_timeout_s=self._row_fetch_timeout_s,
            )
        if not rows:
            log.debug("no active rows, skip %s prewarm", self._allocator.runtime)
            return

        self._image_by_env = self._repo.get_env_image_map()
        log.debug(
            "%s prewarm start: target_pool_size=%d initial_rows=%d",
            self._allocator.runtime,
            self._pool_size,
            len(rows),
        )
        tasks = [asyncio.create_task(self._fill_slot(row)) for row in rows]
        await asyncio.gather(*tasks, return_exceptions=True)
        await self.ensure_capacity()

    async def ensure_capacity(self, *, wait_for_rows: bool = False) -> None:
        while True:
            async with self._lock:
                deficit = max(0, self._pool_size - len(self._pool))
            if deficit <= 0:
                return

            rows = await self._repo.reserve_rows(
                deficit,
                fetch_batch_size=self._repo_fetch_batch_size(deficit),
                wait_for_rows=wait_for_rows,
                max_wait_s=self._row_wait_timeout_s if wait_for_rows else None,
                fetch_timeout_s=self._row_fetch_timeout_s,
            )
            if not rows:
                log.debug(
                    "ensure_capacity: no DB rows available to fill %s pool",
                    self._allocator.runtime,
                )
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
            self._schedule_release_entry(old_entry, succeeded=succeeded)

        next_row = await self._repo.reserve_one(
            fetch_batch_size=self._repo_fetch_batch_size(),
            wait_for_rows=True,
            max_wait_s=self._row_wait_timeout_s,
            fetch_timeout_s=self._row_fetch_timeout_s,
        )
        if not next_row:
            log.debug("close_and_refill: DB rows exhausted for %s pool", self._allocator.runtime)
            return None
        return await self._fill_slot_for_refill(next_row)

    async def _release_entry(self, entry: PoolEntry, *, succeeded: bool) -> None:
        task = asyncio.create_task(
            self._allocator.release(entry, succeeded=succeeded),
            name=f"{self._allocator.runtime}-release-{entry.env_name}-{entry.env_id}",
        )
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            self._track_background_task(
                task,
                action=(
                    f"release {self._allocator.runtime} lease "
                    f"{entry.env_name}/{entry.env_id} resource={entry.resource_name or entry.container_id}"
                ),
            )
            raise

    def _schedule_release_entry(self, entry: PoolEntry, *, succeeded: bool) -> None:
        task = asyncio.create_task(
            self._allocator.release(entry, succeeded=succeeded),
            name=f"{self._allocator.runtime}-release-{entry.env_name}-{entry.env_id}",
        )
        self._track_background_task(
            task,
            action=(
                f"release {self._allocator.runtime} lease "
                f"{entry.env_name}/{entry.env_id} resource={entry.resource_name or entry.container_id}"
            ),
        )

    async def _fill_slot_for_refill(self, row: Dict[str, Any]) -> Optional[PoolEntry]:
        task = asyncio.create_task(
            self._fill_slot(row),
            name=f"{self._allocator.runtime}-refill-{row.get('env_name')}-{row.get('env_id')}",
        )
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            self._track_background_task(
                task,
                action=(
                    f"refill {self._allocator.runtime} lease "
                    f"{row.get('env_name')}/{row.get('env_id')}"
                ),
            )
            raise

    def _track_background_task(self, task: asyncio.Task, *, action: str) -> None:
        self._background_tasks.add(task)

        def _done(completed: asyncio.Task) -> None:
            self._background_tasks.discard(completed)
            try:
                completed.result()
            except asyncio.CancelledError:
                log.warning("background %s was cancelled", action)
            except Exception:
                log.warning("background %s failed", action, exc_info=True)

        task.add_done_callback(_done)

    async def drain_background_tasks(self, *, timeout_s: float) -> None:
        if not self._background_tasks:
            return
        pending = set(self._background_tasks)
        log.debug("draining %d background %s task(s)", len(pending), self._allocator.runtime)
        done, still_pending = await asyncio.wait(pending, timeout=max(0.0, float(timeout_s or 0.0)))
        for task in done:
            try:
                task.result()
            except asyncio.CancelledError:
                pass
            except Exception:
                pass
        if still_pending:
            log.warning(
                "leaving %d background %s task(s) unfinished after %.1fs",
                len(still_pending),
                self._allocator.runtime,
                float(timeout_s or 0.0),
            )

    async def _fill_slot(self, row: Dict[str, Any]) -> Optional[PoolEntry]:
        async with self._fill_sem:
            return await self._fill_slot_once(row)

    async def _fill_slot_once(self, row: Dict[str, Any]) -> Optional[PoolEntry]:
        env_name = str(row.get("env_name", "")).strip()
        env_id = str(row.get("env_id", "")).strip()
        image = self._resolve_image(row, env_name)
        if not env_name or not env_id or not image:
            log.error("Invalid DB row for %s lease: %s", self._allocator.runtime, row)
            return None

        try:
            entry = await self._allocator.allocate(
                row_id=row.get("id"),
                env_name=env_name,
                env_id=env_id,
                image=image,
                env_params=self._normalize_env_params(row.get("env_params")),
                group_id=str(row.get("group_id") or ""),
            )
        except Exception:
            log.error(
                "failed to allocate %s lease for %s/%s",
                self._allocator.runtime,
                env_name,
                env_id,
                exc_info=True,
            )
            return None

        async with self._lock:
            self._pool[(env_name, env_id)] = entry
        log.debug(
            "allocated %s lease: env=%s agent_id=%s resource=%s reuse=%s",
            entry.runtime,
            env_name,
            env_id,
            entry.resource_name or entry.container_name or entry.job_name,
            entry.reuse_container,
        )
        return entry

    def _resolve_image(self, row: Dict[str, Any], env_name: str) -> str:
        return str(row.get("image") or row.get("env_image") or self._image_by_env.get(env_name) or "").strip()

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


class _LegacyDockerAllocator:
    runtime = "docker"

    def __init__(self, docker: Any, startup_concurrency: int) -> None:
        self._docker = docker
        self.startup_concurrency = max(1, int(startup_concurrency or 1))

    async def start(self, _plan) -> None:
        return

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
        container = await self._docker.acquire(env_name=env_name, image=image)
        return PoolEntry(
            env_name=str(env_name),
            env_id=str(env_id),
            row_id=row_id,
            image=container.image,
            job_name=container.container_name,
            env_params=dict(env_params or {}),
            group_id=str(group_id or ""),
            status="ready",
            runtime="docker",
            resource_id=container.container_id,
            resource_name=container.container_name,
            container_id=container.container_id,
            container_name=container.container_name,
            docker_bin=self._docker.docker_bin,
            workdir=container.workdir,
            run_command=container.run_command,
            result_mode=container.result_mode,
            cleanup_command=container.cleanup_command,
            healthcheck_command=container.healthcheck_command,
            reuse_container=container.reuse_container,
        )

    async def release(self, entry: PoolEntry, *, succeeded: bool) -> None:
        if entry.container_id:
            await self._docker.release(entry.container_id, succeeded=succeeded)

    async def remove(self, entry: PoolEntry) -> None:
        if entry.container_id:
            await self._docker.remove(entry.container_id)

    async def close(self) -> None:
        return


class DockerAgentPool(RuntimeAgentPool):
    """Backward-compatible DockerAgentPool wrapper."""

    def __init__(
        self,
        *,
        repo,
        docker: Any,
        pool_size: int,
        startup_concurrency: int,
    ) -> None:
        super().__init__(
            repo=repo,
            allocator=_LegacyDockerAllocator(docker, startup_concurrency),
            pool_size=pool_size,
            startup_concurrency=startup_concurrency,
        )
