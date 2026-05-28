from __future__ import annotations

import asyncio
import logging
from collections import deque
from typing import Deque, Optional, Set, Tuple

from .manager import AgentPoolManager
from .types import PoolEntry, SimulationAgentLease, SimulationStartResult

log = logging.getLogger("manager.simulation_lease_pool")


class _LeaseRuntimeState:
    def __init__(self) -> None:
        self._ready: Deque[SimulationAgentLease] = deque()
        self._known: Set[Tuple[str, str]] = set()
        self._leased = 0
        self._refills_in_flight = 0
        self._initial_load_done = False
        self._cond = asyncio.Condition()

    async def add_ready_lease(self, key: Tuple[str, str], lease: SimulationAgentLease) -> bool:
        async with self._cond:
            if key in self._known:
                return False
            self._known.add(key)
            self._ready.append(lease)
            self._cond.notify(1)
            return True

    async def mark_initial_load_done(self) -> None:
        async with self._cond:
            self._initial_load_done = True
            self._cond.notify_all()

    async def acquire(self) -> Optional[SimulationAgentLease]:
        async with self._cond:
            while True:
                if self._ready:
                    self._leased += 1
                    return self._ready.popleft()
                if self._is_exhausted_locked():
                    return None
                await self._cond.wait()

    async def begin_refill(self, _old_key: Tuple[str, str]) -> None:
        async with self._cond:
            if self._leased > 0:
                self._leased -= 1
            self._refills_in_flight += 1
            self._cond.notify_all()

    async def finish_refill(
        self,
        old_key: Tuple[str, str],
        new_key: Optional[Tuple[str, str]] = None,
        lease: Optional[SimulationAgentLease] = None,
    ) -> bool:
        async with self._cond:
            self._known.discard(old_key)
            if self._refills_in_flight > 0:
                self._refills_in_flight -= 1

            added = False
            if new_key is not None and lease is not None and new_key not in self._known:
                self._known.add(new_key)
                self._ready.append(lease)
                added = True

            self._cond.notify_all()
            return added

    async def fail_refill(self, old_key: Tuple[str, str]) -> None:
        async with self._cond:
            self._known.discard(old_key)
            if self._refills_in_flight > 0:
                self._refills_in_flight -= 1
            self._cond.notify_all()

    def _is_exhausted_locked(self) -> bool:
        return (
            self._initial_load_done
            and not self._ready
            and self._leased == 0
            and self._refills_in_flight == 0
        )


class SimulationLeasePool:
    """
    Manager-side lease view over initialized agent instances.

    This class owns only lease queueing and close/refill coordination. It does
    not run the episode and it does not know anything about LLM routing.
    """

    def __init__(self, mgr: AgentPoolManager, *, pool_size: int):
        self.mgr = mgr
        self.pool_size = max(1, int(pool_size or 1))
        self._runtime = _LeaseRuntimeState()
        self._lifecycle_lock = asyncio.Lock()
        self._closing = False
        self._closed = False

    async def start(self) -> None:
        log.info("starting AgentPoolManager for simulation lease pool")
        await self.mgr.start()

        entries = await self.mgr.list_pool_instances()
        log.info("manager reports %d warmed agent instance(s)", len(entries))
        for entry in entries:
            lease = self._entry_to_agent_lease(entry)
            if lease is None:
                continue
            added = await self._runtime.add_ready_lease((lease.agent_name, lease.agent_id), lease)
            if added:
                log.debug("enqueued agent lease: %s/%s", lease.agent_name, lease.agent_id)

        await self._runtime.mark_initial_load_done()

    async def acquire(self) -> Optional[SimulationAgentLease]:
        lease = await self._runtime.acquire()
        if lease is None:
            log.debug("acquire(): exhausted")
        else:
            log.debug("acquire(): got %s/%s", lease.agent_name, lease.agent_id)
        return lease

    async def done(
        self,
        lease: SimulationAgentLease,
        result: Optional[SimulationStartResult] = None,
        reusable: Optional[bool] = None,
    ) -> None:
        old_key = (str(lease.agent_name), str(lease.agent_id))
        async with self._lifecycle_lock:
            if self._closing or self._closed:
                log.info("done(): pool is closing; skip refill for agent=%s id=%s", lease.agent_name, lease.agent_id)
                await self._finish_without_replacement(old_key)
                return

            await self._runtime.begin_refill(old_key)
            ok = await self._close_and_refill(lease, old_key, result, reusable)
            if not ok:
                await self._runtime.fail_refill(old_key)
                raise RuntimeError(f"close_and_refill failed for {lease.agent_name}/{lease.agent_id}")

    async def aclose(self) -> None:
        async with self._lifecycle_lock:
            if self._closed:
                log.info("SimulationLeasePool.aclose(): already closed")
                return
            self._closing = True

        async with self._lifecycle_lock:
            if self._closed:
                return
            log.info("SimulationLeasePool.aclose(): closing AgentPoolManager")
            await self.mgr.close_all()
            self._closed = True

    async def _finish_without_replacement(self, old_key: Tuple[str, str]) -> None:
        await self._runtime.begin_refill(old_key)
        await self._runtime.fail_refill(old_key)

    async def _close_and_refill(
        self,
        lease: SimulationAgentLease,
        old_key: Tuple[str, str],
        result: Optional[SimulationStartResult],
        reusable: Optional[bool],
    ) -> bool:
        try:
            succeeded = bool(reusable) if reusable is not None else result is not None and result.status == "succeeded"
            replacement = await self.mgr.close_and_refill(lease.agent_name, lease.agent_id, succeeded=succeeded)
        except asyncio.CancelledError:
            await self._runtime.fail_refill(old_key)
            raise
        except Exception:
            log.exception("close_and_refill failed for %s/%s", lease.agent_name, lease.agent_id)
            return False

        if replacement is None:
            await self._runtime.finish_refill(old_key)
            return True

        new_lease = self._entry_to_agent_lease(replacement)
        if new_lease is None:
            log.error("replacement agent instance could not be converted for %s/%s", replacement.env_name, replacement.env_id)
            return False

        new_key = (new_lease.agent_name, new_lease.agent_id)
        added = await self._runtime.finish_refill(old_key, new_key, new_lease)
        if added:
            log.debug("registered replacement agent lease: %s/%s", new_key[0], new_key[1])
        return True

    def _entry_to_agent_lease(self, entry: PoolEntry) -> Optional[SimulationAgentLease]:
        agent_name = str(entry.env_name)
        agent_id = str(entry.env_id)

        return SimulationAgentLease(
            agent_name=agent_name,
            agent_id=agent_id,
            group_id=str(entry.group_id or ""),
            image=str(entry.image or ""),
            row_id=entry.row_id,
            env_params=dict(entry.env_params or {}),
            container_id=str(getattr(entry, "container_id", "") or ""),
            container_name=str(getattr(entry, "container_name", "") or ""),
            docker_bin=str(getattr(entry, "docker_bin", "docker") or "docker"),
            workdir=str(getattr(entry, "workdir", "") or ""),
            run_command=str(getattr(entry, "run_command", "") or ""),
            cleanup_command=str(getattr(entry, "cleanup_command", "") or ""),
            healthcheck_command=str(getattr(entry, "healthcheck_command", "") or ""),
            reuse_container=bool(getattr(entry, "reuse_container", False)),
        )
