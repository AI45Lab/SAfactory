from __future__ import annotations

import asyncio
import logging
import aiohttp
from typing import Any, Dict, List, Optional, Tuple

from .http_client import HttpServiceClient
from .types import ActorKey, ActorRoute, ClusterRegistry, EnvClusterBinding, PoolEntry

log = logging.getLogger("manager.actor_pool")


class ActorPool:
    """
    Manages prewarm + close/refill via HTTP service.
    Works for both remote clusters and local mode.
    """

    def __init__(
            self,
            *,
            repo,
            http: HttpServiceClient,
            pool_size: int,
            http_port: int,
            http_concurrency: int,
            startup_concurrency: Optional[int] = None,
            base_image: str,
            default_seed: int = 123,
            env_limits: Optional[Dict[str, int]] = None,
    ) -> None:
        self._repo = repo
        self._http = http

        self._pool_size = int(pool_size)
        self._http_port = int(http_port)
        self._http_concurrency = int(http_concurrency)
        if startup_concurrency is None:
            startup_concurrency = min(self._http_concurrency, 16)
        self._startup_concurrency = max(1, int(startup_concurrency))
        self._base_image = (base_image or "").strip()
        self._default_seed = int(default_seed)
        self._env_limits: Dict[str, int] = {}
        for k, v in (env_limits or {}).items():
            try:
                self._env_limits[str(k)] = int(v)
            except Exception:
                continue

        self._lock = asyncio.Lock()
        # Limit how many cold-start/reset attempts can run concurrently.
        # This is intentionally separate from the broader HTTP concurrency.
        self._fill_sem = asyncio.Semaphore(self._startup_concurrency)
        self._pool: Dict[ActorKey, PoolEntry] = {}
        self._actor_routes: Dict[ActorKey, ActorRoute] = {}
        self._job_load: Dict[Tuple[str, str], int] = {}



    async def reset(self) -> None:
        async with self._lock:
            self._pool.clear()
            self._actor_routes.clear()
            self._repo.reset_cursor()
            self._job_load.clear()

    async def list_actors(self) -> List[dict]:
        async with self._lock:
            return [{"env_name": e.env_name, "env_id": e.env_id, "group_id": e.group_id} for e in self._pool.values()]

    def get_actor_route(self, env: str, env_id: str, fallback: Optional[EnvClusterBinding]) -> Optional[ActorRoute]:
        key = (str(env), str(env_id))
        route = self._actor_routes.get(key)
        if route:
            return route
        if fallback and fallback.head_ip:
            return (fallback.head_ip, self._http_port)
        return None

    async def prewarm(self, registry: ClusterRegistry, rows: Optional[List[Dict[str, Any]]] = None) -> None:
        if self._pool_size <= 0:
            log.info("pool_size <= 0, skip prewarm")
            return

        # Reserve rows under lock, but do not do network under lock.
        if rows is None:
            async with self._lock:
                rows = self._repo.fetch_active_rows(self._pool_size)

        if not rows:
            log.info("no active rows, skip prewarm")
            return

        log.info(
            "prewarm start: target_pool_size=%d initial_rows=%d startup_concurrency=%d http_concurrency=%d",
            self._pool_size,
            len(rows),
            self._startup_concurrency,
            self._http_concurrency,
        )

        tasks = [asyncio.create_task(self._robust_fill_slot(registry, self._fill_sem, initial_row=row)) for row in rows]
        await asyncio.gather(*tasks, return_exceptions=True)

        await self.ensure_capacity(registry)

    async def ensure_capacity(self, registry: ClusterRegistry) -> None:
        """
        Continuously fill the pool until it reaches pool_size or DB is empty.
        """
        while True:
            async with self._lock:
                deficit = max(0, self._pool_size - len(self._pool))
                if deficit <= 0:
                    return
                # Fetch just enough rows to fill deficit
                rows = self._repo.fetch_active_rows(deficit)

            if not rows:
                log.info("[manager] ensure_capacity: no more DB rows to fill pool")
                return

            tasks = [asyncio.create_task(self._robust_fill_slot(registry, self._fill_sem, initial_row=row)) for row in rows]
            await asyncio.gather(*tasks, return_exceptions=True)

    async def close_and_refill(self, *, env: str, env_id: str, registry):
        """
        Close the specified actor and immediately try to create a new one to replace it.
        """
        env = str(env)
        env_id = str(env_id)
        key = (env, env_id)

        # 1. Cleanup old actor state
        async with self._lock:
            route = self._actor_routes.get(key)
            binding = registry.env_bindings.get(env)

            old_entry = self._pool.pop(key, None)
            self._actor_routes.pop(key, None)
            if old_entry is not None:
                old_job = str(getattr(old_entry, "job_name", "") or "").strip()
                if old_job:
                    lk = (env, old_job)
                    cur = int(self._job_load.get(lk, 0) or 0)
                    if cur <= 1:
                        self._job_load.pop(lk, None)
                    else:
                        self._job_load[lk] = cur - 1

            # Fetch one new row specifically for this refill
            next_row = self._repo.fetch_one_active_row()

        # 2. Issue HTTP Delete (Close)
        host = ""
        if route:
            host = route[0]
        elif binding and binding.head_ip:
            host = binding.head_ip

        if host:
            delete_url = f"http://{host}:{self._http_port}/{env}/{env_id}"
            legacy_post_url = f"http://{host}:{self._http_port}/{env}/{env_id}/close"

            try:
                # Try DELETE first
                async with await self._http.delete(delete_url) as resp:
                    status = resp.status
                    if status == 404:
                        # Fallback to POST
                        async with await self._http.post(legacy_post_url) as resp2:
                            if resp2.status >= 400:
                                log.error("close failed (fallback): %s status=%s", legacy_post_url, resp2.status)
                    elif status >= 400:
                        log.error("close failed: %s status=%s", delete_url, status)

            except Exception:
                log.error(
                    "close error: env='%s', id='%s'",
                    env,
                    env_id,
                    exc_info=True,
                )
        else:
            log.warning("close skipped: no route/binding for env='%s' id='%s'", env, env_id)

        # 3. Refill the slot
        # If we got a row, start the robust loop. If no row, we just exit (pool shrinks).
        if not next_row:
            log.info("close_and_refill: no more DB rows to refill pool")
            return

        # No semaphore needed for single replacement, or create a dummy one
        await self._robust_fill_slot(registry, sem=self._fill_sem, initial_row=next_row)

    # ------------------------------------------------------------------ #

    async def _robust_fill_slot(
            self,
            registry: ClusterRegistry,
            sem: Optional[asyncio.Semaphore],
            initial_row: Optional[Dict[str, Any]]
    ) -> None:
        """
        Attempts to fill ONE pool slot.
        If initial_row fails (after retries), it fetches the next row from DB.
        It repeats until success or DB runs out.
        """
        current_row = initial_row

        while True:
            # 1. Ensure we have a row
            if current_row is None:
                async with self._lock:
                    current_row = self._repo.fetch_one_active_row()

                if current_row is None:
                    log.info("robust_fill_slot: DB exhausted, stopping slot fill.")
                    return

            # 2. Try to create actor for this row
            env_key = f"{current_row.get('env_name')}/{current_row.get('env_id')}"
            try:
                if sem:
                    async with sem:
                        created = await self._attempt_create_actor(current_row, registry)
                else:
                    created = await self._attempt_create_actor(current_row, registry)

                if created:
                    log.info("Successfully created actor for %s", env_key)
                    return

                log.error("Failed to create actor for %s after retries. Skipping row.", env_key)
                current_row = None
                continue

            except Exception:
                # Unexpected failures should not kill the whole fill loop.
                log.error(
                    "Unexpected error while creating actor for %s. Skipping row.",
                    env_key,
                    exc_info=True,
                )
                current_row = None

    async def _attempt_create_actor(
            self,
            row: Dict[str, Any],
            registry: ClusterRegistry,
    ) -> bool:
        """
        Tries to create an actor for a SPECIFIC row.
        Retries 'reset' 2 times (total 3 attempts) as requested.
        Returns whether the actor was created successfully.
        """
        env_name = str(row.get("env_name", "")).strip()
        env_id = str(row.get("env_id", "")).strip()
        if not env_name or not env_id:
            log.error("Invalid DB row (missing env_name/env_id): %s", row)
            return False

        image = (row.get("image") or "").strip()
        if not image:
            binding = registry.env_bindings.get(env_name)
            if binding and binding.image:
                image = binding.image
            else:
                image = self._base_image

        if not image:
            log.error("Cannot resolve image for env='%s' id='%s'", env_name, env_id)
            return False

        # Reservation Logic
        async with self._lock:
            try:
                cluster = self._choose_cluster_and_reserve_locked(env_name=env_name, image=image, registry=registry)
                reserved_key = (env_name, str(cluster.job_name))
            except Exception:
                log.error(
                    "Reservation failed for %s/%s",
                    env_name,
                    env_id,
                    exc_info=True,
                )
                return False

        url = f"http://{cluster.head_ip}:{self._http_port}/{env_name}/{env_id}/reset"
        delete_url = f"http://{cluster.head_ip}:{self._http_port}/{env_name}/{env_id}"
        payload = {
            "env_param": row.get("env_params"),
            "seed": row.get("seed", self._default_seed),
        }

        async def cleanup_remote_actor(stage: str) -> None:
            try:
                async with await self._http.delete(delete_url) as cleanup_resp:
                    log.info(
                        "Cleanup response for %s/%s during %s: status=%s",
                        env_name,
                        env_id,
                        stage,
                        cleanup_resp.status,
                    )
            except Exception:
                log.warning(
                    "Cleanup failed for %s/%s during %s (ignoring)",
                    env_name,
                    env_id,
                    stage,
                    exc_info=True,
                )

        # --- RETRY LOGIC (Requirement: retry twice if reset encounters any error) ---
        max_reset_attempts = 3  # 1 initial + 2 retries
        cleanup_attempted_for_failure = False

        for attempt in range(1, max_reset_attempts + 1):
            cleanup_attempted_for_failure = False
            try:
                # Use aiohttp post
                async with await self._http.post(url, json=payload) as resp:
                    if resp.status in (500, 502, 503, 504):
                        # Treat server errors as retryable
                        raise RuntimeError(f"Server error status={resp.status}")

                    resp.raise_for_status()

                    # Success - Register in Pool
                    key: ActorKey = (env_name, env_id)
                    async with self._lock:
                        self._pool[key] = PoolEntry(
                            env_name=env_name,
                            env_id=env_id,
                            row_id=row.get("id"),
                            image=image,
                            job_name=cluster.job_name,
                            head_ip=cluster.head_ip,
                            group_id=str(row.get("group_id") or ""),
                            status="ready",
                        )
                        self._actor_routes[key] = (cluster.head_ip, self._http_port)
                    return True

            except (asyncio.TimeoutError, aiohttp.ClientError, RuntimeError):
                log.warning(
                    "Reset attempt %d/%d failed for %s/%s",
                    attempt,
                    max_reset_attempts,
                    env_name,
                    env_id,
                    exc_info=True,
                )

                await cleanup_remote_actor(f"attempt_{attempt}")
                cleanup_attempted_for_failure = True

                if attempt < max_reset_attempts:
                    await asyncio.sleep(1.0)  # wait a bit before retry
            except Exception:
                # Non-recoverable errors (e.g. serialization)
                log.error(
                    "Non-recoverable reset failure for %s/%s on attempt %d/%d",
                    env_name,
                    env_id,
                    attempt,
                    max_reset_attempts,
                    exc_info=True,
                )
                break

        if not cleanup_attempted_for_failure:
            log.warning("Reset failed for %s/%s. Sending CLEANUP (DELETE) to %s...", env_name, env_id, cluster.head_ip)
            await cleanup_remote_actor("final_failure")
        else:
            log.warning("Reset failed %d times for %s/%s. Cleanup was attempted after each failed attempt.",
                        max_reset_attempts, env_name, env_id)

        # Cleanup reservation
        async with self._lock:
            cur = int(self._job_load.get(reserved_key, 0) or 0)
            if cur <= 1:
                self._job_load.pop(reserved_key, None)
            else:
                self._job_load[reserved_key] = cur - 1

        return False

    def _choose_cluster_and_reserve_locked(self, *, env_name: str, image: str, registry: 'ClusterRegistry'):
        """
        Pick the best cluster for this env and reserve 1 slot.
        MUST be called under self._lock.
        """
        prefix = f"{env_name}#"
        candidates = []

        # 1. Look for clusters belonging to this env.
        clusters = registry.clusters_by_id or {}
        for cid, info in clusters.items():
            if str(cid).startswith(prefix):
                if info is not None and getattr(info, "head_ip", None):
                    candidates.append(info)

        # 2. Error handling if no candidates are found
        if not candidates:
            raise RuntimeError(f"No cluster/head_ip available for env='{env_name}', image='{image}'")

        # 3. Load balancing and environment limits logic
        lim = int(self._env_limits.get(env_name, 0) or 0)

        def load_of(info) -> int:
            """Helper to get the current job load for a specific cluster."""
            job_name = str(getattr(info, "job_name", "") or "")
            jk = (env_name, job_name)
            return int(self._job_load.get(jk, 0) or 0)

        if lim > 0:
            # Filter candidates that are below the defined limit
            below = [c for c in candidates if load_of(c) < lim]
            pool = below or candidates
        else:
            pool = candidates

        # 4. Select the cluster with the minimum load
        # Tie-break using job_name for deterministic selection
        chosen = min(pool, key=lambda c: (load_of(c), str(getattr(c, "job_name", "") or "")))

        # 5. Increment the load counter and reserve the slot
        chosen_job = str(getattr(chosen, "job_name", "") or "")
        lk = (env_name, chosen_job)
        self._job_load[lk] = int(self._job_load.get(lk, 0) or 0) + 1

        return chosen
