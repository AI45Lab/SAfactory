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
            base_image: str,
            default_seed: int = 123,
            env_limits: Optional[Dict[str, int]] = None,
    ) -> None:
        self._repo = repo
        self._http = http

        self._pool_size = int(pool_size)
        self._http_port = int(http_port)
        self._http_concurrency = int(http_concurrency)
        self._base_image = (base_image or "").strip()
        self._default_seed = int(default_seed)
        self._env_limits: Dict[str, int] = {}
        for k, v in (env_limits or {}).items():
            try:
                self._env_limits[str(k)] = int(v)
            except Exception:
                continue

        self._lock = asyncio.Lock()
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
            print("[manager] pool_size <= 0, skip prewarm")
            return

        # Reserve rows under lock, but do not do network under lock.
        if rows is None:
            async with self._lock:
                rows = self._repo.fetch_active_rows(self._pool_size)

        if not rows:
            print("[manager] no active rows, skip prewarm")
            return

        sem = asyncio.Semaphore(self._http_concurrency)
        # Change: Use _robust_fill_slot instead of _create_actor_for_row
        tasks = [asyncio.create_task(self._robust_fill_slot(registry, sem, initial_row=row)) for row in rows]
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

            sem = asyncio.Semaphore(self._http_concurrency)
            tasks = [asyncio.create_task(self._robust_fill_slot(registry, sem, initial_row=row)) for row in rows]
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

            except Exception as e:
                log.error("close error: env='%s', id='%s', err=%s", env, env_id, e)
        else:
            log.warning("close skipped: no route/binding for env='%s' id='%s'", env, env_id)

        # 3. Refill the slot
        # If we got a row, start the robust loop. If no row, we just exit (pool shrinks).
        if not next_row:
            log.info("close_and_refill: no more DB rows to refill pool")
            return

        # No semaphore needed for single replacement, or create a dummy one
        await self._robust_fill_slot(registry, sem=None, initial_row=next_row)

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
                        await self._attempt_create_actor(current_row, registry)
                else:
                    await self._attempt_create_actor(current_row, registry)

                # Success!
                log.info("Successfully created actor for %s", env_key)
                return

            except Exception as e:
                # 3. Failure logic
                log.error("Failed to create actor for %s after retries. Skipping row. Error: %s", env_key, e)
                # Important: Set current_row to None so next loop fetches a NEW one
                current_row = None
                # Loop continues...

    async def _attempt_create_actor(
            self,
            row: Dict[str, Any],
            registry: ClusterRegistry,
    ) -> None:
        """
        Tries to create an actor for a SPECIFIC row.
        Retries 'reset' 2 times (total 3 attempts) as requested.
        Raises exception if all attempts fail.
        """
        env_name = str(row.get("env_name", "")).strip()
        env_id = str(row.get("env_id", "")).strip()
        if not env_name or not env_id:
            raise ValueError(f"Invalid DB row (missing env_name/env_id): {row}")

        image = (row.get("image") or "").strip()
        if not image:
            binding = registry.env_bindings.get(env_name)
            if binding and binding.image:
                image = binding.image
            else:
                image = self._base_image

        if not image:
            raise RuntimeError(f"Cannot resolve image for env='{env_name}' id='{env_id}'")

        # Reservation Logic
        async with self._lock:
            try:
                cluster = self._choose_cluster_and_reserve_locked(env_name=env_name, image=image, registry=registry)
                reserved_key = (env_name, str(cluster.job_name))
            except Exception as e:
                raise RuntimeError(f"Reservation failed: {e}")

        url = f"http://{cluster.head_ip}:{self._http_port}/{env_name}/{env_id}/reset"
        payload = {
            "env_param": row.get("env_params"),
            "seed": row.get("seed", self._default_seed),
        }

        # --- RETRY LOGIC (Requirement: retry twice if reset encounters any error) ---
        max_reset_attempts = 3  # 1 initial + 2 retries
        last_error = None

        for attempt in range(1, max_reset_attempts + 1):
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
                    return  # Exit function on success

            except (asyncio.TimeoutError, aiohttp.ClientError, RuntimeError) as e:
                last_error = e
                log.warning("Reset attempt %d/%d failed for %s/%s: %s", attempt, max_reset_attempts, env_name, env_id,
                            e)

                if attempt < max_reset_attempts:
                    await asyncio.sleep(1.0)  # wait a bit before retry
            except Exception as e:
                # Non-recoverable errors (e.g. serialization)
                last_error = e
                break
        log.warning("Reset failed 3 times. Sending CLEANUP (DELETE) for %s/%s to %s...", env_name, env_id, cluster.head_ip)
        delete_url = f"http://{cluster.head_ip}:{self._http_port}/{env_name}/{env_id}"
        try:
            # Send a best-effort DELETE request to kill any zombie actor
            async with await self._http.delete(delete_url) as cleanup_resp:
                log.info("Cleanup response for %s/%s: status=%s", env_name, env_id, cleanup_resp.status)
        except Exception as cleanup_err:
            log.warning("Cleanup failed for %s/%s: %s (ignoring)", env_name, env_id, cleanup_err)
            pass
        # Cleanup reservation
        async with self._lock:
            cur = int(self._job_load.get(reserved_key, 0) or 0)
            if cur <= 1:
                self._job_load.pop(reserved_key, None)
            else:
                self._job_load[reserved_key] = cur - 1

        raise RuntimeError(
            f"Failed to reset {env_name}/{env_id} after {max_reset_attempts} attempts. Last error: {last_error}")

    def _choose_cluster_and_reserve_locked(self, *, env_name: str, image: str, registry: 'ClusterRegistry'):
        """
        Pick the best cluster for this env and reserve 1 slot.
        MUST be called under self._lock.
        """
        prefix = f"{env_name}#"
        candidates = []

        # 1. Look for clusters matching env_name or starting with "env_name#"
        clusters = registry.clusters_by_image or {}
        for cid, info in clusters.items():
            if cid == env_name or str(cid).startswith(prefix):
                if info is not None and getattr(info, "head_ip", None):
                    candidates.append(info)

        # 2. Backward-compatible fallback: try matching directly by image key
        if not candidates and image:
            info = clusters.get(image)
            if info is not None and getattr(info, "head_ip", None):
                candidates = [info]

        # 3. Second fallback: iterate through all clusters to find a matching image attribute
        if not candidates and image:
            for info in clusters.values():
                if info is None:
                    continue
                if getattr(info, "image", None) == image and getattr(info, "head_ip", None):
                    candidates.append(info)

        # 4. Error handling if no candidates are found
        if not candidates:
            raise RuntimeError(f"No cluster/head_ip available for env='{env_name}', image='{image}'")

        # 5. Load balancing and environment limits logic
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

        # 6. Select the cluster with the minimum load
        # Tie-break using job_name for deterministic selection
        chosen = min(pool, key=lambda c: (load_of(c), str(getattr(c, "job_name", "") or "")))

        # 7. Increment the load counter and reserve the slot
        chosen_job = str(getattr(chosen, "job_name", "") or "")
        lk = (env_name, chosen_job)
        self._job_load[lk] = int(self._job_load.get(lk, 0) or 0) + 1

        return chosen
