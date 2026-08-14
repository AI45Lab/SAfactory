from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Dict, List, Optional

from .actor_pool import RuntimeAgentPool
from .binding_plan import build_binding_plan
from .repository import AgentDataRepository
from .runtime_allocator import (
    DockerLeaseAllocator,
    RJobLeaseAllocator,
    RuntimeLeaseAllocator,
    SandboxLeaseAllocator,
)
from .types import PoolEntry

log = logging.getLogger("manager")


class AgentPoolManager:
    """
    Safactory runtime scheduler facade.

    It exposes ready runtime leases and refills from the DB after each episode.
    Docker and Sandbox modes warm runtime instances; RJob mode reserves rows
    and submits the remote job later in the episode runner.
    """

    def __init__(
        self,
        cfg: dict,
        data_manager: Any,
        *,
        job_id: str = "",
        db_processing_done_checker: Optional[Callable[[], bool]] = None,
    ) -> None:
        self.cfg = cfg or {}
        self._job_id = str(job_id or "").strip()
        self._repo = AgentDataRepository(
            data_manager,
            job_id=self._job_id,
            db_processing_done_checker=db_processing_done_checker,
        )
        self._pool_size = int(self.cfg.get("pool_size", 0) or 0)
        self._mode = str(self.cfg.get("mode", "docker") or "docker").strip().lower()
        if self._mode not in {"docker", "rjob", "sandbox"}:
            raise ValueError(f"Unsupported runtime workflow mode: {self._mode!r}")
        self._row_wait_timeout_s = float(self.cfg.get("row_wait_timeout_s", 60.0) or 60.0)
        self._row_fetch_timeout_s = float(self.cfg.get("row_fetch_timeout_s", 30.0) or 30.0)

        cluster_cfg: Dict = dict(self.cfg.get("cluster", {}) or {})
        self._allocator: RuntimeLeaseAllocator = self._build_allocator(cluster_cfg)
        startup_concurrency = int(getattr(self._allocator, "startup_concurrency", 8) or 8)

        self._pool = RuntimeAgentPool(
            repo=self._repo,
            allocator=self._allocator,
            pool_size=self._pool_size,
            startup_concurrency=startup_concurrency,
            row_wait_timeout_s=self._row_wait_timeout_s,
            row_fetch_timeout_s=self._row_fetch_timeout_s,
        )
        self._state_lock = asyncio.Lock()
        self._initialized = False
        self._closed = False

    async def start(self) -> None:
        async with self._state_lock:
            if self._initialized:
                return
            self._closed = False

            plan = await build_binding_plan(self._repo)
            if not plan.env_to_image:
                log.warning("No agent/image mapping found in DB; nothing to start.")
                self._initialized = True
                return

            prewarm_rows = (
                await self._repo.prime(self._pool_size, fetch_timeout_s=self._row_fetch_timeout_s)
                if self._pool_size > 0
                else []
            )
            await self._allocator.start(plan)
            await self._pool.prewarm(rows=prewarm_rows)

            self._initialized = True
            log.info(
                "Safactory %s scheduler started: pool_size=%d job_id=%s",
                self._mode,
                self._pool_size,
                self._job_id or "<all>",
            )

    async def close_all(self) -> None:
        async with self._state_lock:
            if self._closed:
                log.debug("AgentPoolManager.close_all(): already closed")
                return
            self._initialized = False
            self._closed = True
            await self._pool.reset()

        try:
            await self._allocator.close()
        except Exception:
            log.warning("%s allocator close failed (ignored)", self._mode, exc_info=True)
        self._repo.close()

    async def list_pool_instances(self) -> List[PoolEntry]:
        return await self._pool.list_instances()

    async def ensure_capacity(self, *, wait_for_rows: bool = False) -> None:
        if not self._initialized:
            raise RuntimeError("AgentPoolManager not started. Call await start() first.")
        await self._pool.ensure_capacity(wait_for_rows=wait_for_rows)

    async def is_data_exhausted(self) -> bool:
        return await self._repo.is_exhausted()

    async def close_and_refill(
        self,
        agent: str,
        id_: str,
        *,
        succeeded: bool,
    ) -> Optional[PoolEntry]:
        if not self._initialized:
            raise RuntimeError("AgentPoolManager not started. Call await start() first.")
        return await self._pool.close_and_refill(env=str(agent), env_id=str(id_), succeeded=bool(succeeded))

    def _build_allocator(self, cluster_cfg: Dict) -> RuntimeLeaseAllocator:
        if self._mode == "docker":
            return DockerLeaseAllocator(cluster_cfg=cluster_cfg)
        if self._mode == "rjob":
            return RJobLeaseAllocator(cluster_cfg=cluster_cfg)
        if self._mode == "sandbox":
            return SandboxLeaseAllocator(cluster_cfg=cluster_cfg)
        raise ValueError(f"Unsupported runtime workflow mode: {self._mode!r}")
