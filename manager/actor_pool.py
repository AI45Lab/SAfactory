from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional, Tuple

import httpx

from .http_client import HttpServiceClient
from .types import ActorKey, ActorRoute, ClusterRegistry, EnvClusterBinding, PoolEntry


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
        # Per-env limit: how many actors a single RayJob should host.
        self._env_limits: Dict[str, int] = {}
        for k, v in (env_limits or {}).items():
            try:
                self._env_limits[str(k)] = int(v)
            except Exception:
                continue

        self._lock = asyncio.Lock()
        self._pool: Dict[ActorKey, PoolEntry] = {}
        self._actor_routes: Dict[ActorKey, ActorRoute] = {}
        # Counts of actors assigned to each (env_name, job_name).
        # NOTE: includes in-flight reservations for stable concurrent scheduling.
        self._job_load: Dict[Tuple[str, str], int] = {}

    async def reset(self) -> None:
        async with self._lock:
            self._pool.clear()
            self._actor_routes.clear()
            self._repo.reset_cursor()
            self._job_load.clear()

    async def list_actors(self) -> List[dict]:
        async with self._lock:
            return [{"env_name": e.env_name, "env_id": e.env_id} for e in self._pool.values()]

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
        tasks = [asyncio.create_task(self._create_actor_for_row(row, registry, sem)) for row in rows]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Log failures but continue
        failed = sum(1 for r in results if isinstance(r, Exception))
        if failed:
            print(f"[manager] prewarm finished with {failed} actor create error(s)")

        await self.ensure_capacity(registry)

    async def ensure_capacity(self, registry: ClusterRegistry) -> None:
        while True:
            async with self._lock:
                deficit = max(0, self._pool_size - len(self._pool))
                if deficit <= 0:
                    return
                rows = self._repo.fetch_active_rows(deficit)

            if not rows:
                print("[manager] no more DB rows to fill pool")
                return

            sem = asyncio.Semaphore(self._http_concurrency)
            tasks = [asyncio.create_task(self._create_actor_for_row(row, registry, sem)) for row in rows]
            await asyncio.gather(*tasks, return_exceptions=True)

    async def close_and_refill(self, *, env: str, env_id: str, registry):
        env = str(env)
        env_id = str(env_id)
        key = (env, env_id)

        # Resolve route without holding lock during HTTP
        async with self._lock:
            route = self._actor_routes.get(key)
            binding = registry.env_bindings.get(env)

            # remove immediately from local pool (even if remote close fails)
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

            next_row = self._repo.fetch_one_active_row()

        host = ""
        if route:
            host = route[0]
        elif binding and binding.head_ip:
            host = binding.head_ip

        if host:
            delete_url = f"http://{host}:{self._http_port}/{env}/{env_id}"

            legacy_post_url = f"http://{host}:{self._http_port}/{env}/{env_id}/close"

            try:
                resp = await self._http.delete(delete_url)
                if resp.status_code == 404:
                    # fallback to legacy
                    resp2 = await self._http.post(legacy_post_url)
                    if resp2.status_code >= 400:
                        print(
                            f"[manager] close failed: {legacy_post_url} "
                            f"status={resp2.status_code} body={resp2.text}"
                        )
                elif resp.status_code >= 400:
                    print(
                        f"[manager] close failed: {delete_url} "
                        f"status={resp.status_code} body={resp.text}"
                    )

            except Exception as e:
                print(f"[manager] close error: env='{env}', id='{env_id}', err={e}")
        else:
            print(f"[manager] close skipped: no route/binding for env='{env}' id='{env_id}'")

        if not next_row:
            print("[manager] no more DB rows to refill pool")
            return

        await self._create_actor_for_row(row=next_row, registry=registry, sem=None)

    # ------------------------------------------------------------------ #

    async def _create_actor_for_row(
        self,
        row: Dict[str, Any],
        registry: ClusterRegistry,
        sem: Optional[asyncio.Semaphore],
    ) -> None:
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
            raise RuntimeError(f"Cannot resolve image for env='{env_name}' id='{env_id}' (no row.image, no base_image)")

        # Choose the best cluster (RayJob) for this env based on current load.
        # Reserve a slot immediately (under lock) so that concurrent scheduling is stable.
        async with self._lock:
            cluster = self._choose_cluster_and_reserve_locked(env_name=env_name, image=image, registry=registry)
            reserved_key = (env_name, str(cluster.job_name))

        url = f"http://{cluster.head_ip}:{self._http_port}/{env_name}/{env_id}/reset"
        payload = {
            "env_param": row.get("env_params"),
            "seed": row.get("seed", self._default_seed),
        }

        async def _do_post() -> None:
            max_attempts = 3
            for attempt in range(1, max_attempts + 1):
                try:
                    resp = await self._http.post(url, json=payload)
                    # Retry on transient status
                    if resp.status_code in (429, 500, 502, 503, 504):
                        raise RuntimeError(f"Transient status={resp.status_code} body={resp.text[:200]}")
                    resp.raise_for_status()

                    key: ActorKey = (env_name, env_id)
                    async with self._lock:
                        self._pool[key] = PoolEntry(
                            env_name=env_name,
                            env_id=env_id,
                            row_id=row.get("id"),
                            image=image,
                            job_name=cluster.job_name,
                            head_ip=cluster.head_ip,
                            status="ready",
                        )
                        self._actor_routes[key] = (cluster.head_ip, self._http_port)

                    return

                except (httpx.TimeoutException, asyncio.TimeoutError) as e:
                    if attempt == max_attempts:
                        raise RuntimeError(f"Timeout creating actor: {url}, err={e}") from e
                    await asyncio.sleep(min(2.0, 0.5 * attempt))

                except Exception as e:
                    if attempt == max_attempts:
                        raise
                    await asyncio.sleep(min(2.0, 0.5 * attempt))

        if sem is None:
            try:
                await _do_post()
            except Exception:
            # Release reservation on failure
                async with self._lock:
                    cur = int(self._job_load.get(reserved_key, 0) or 0)
                    if cur <= 1:
                        self._job_load.pop(reserved_key, None)
                    else:
                        self._job_load[reserved_key] = cur - 1
        else:
            async with sem:
                await _do_post()


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
