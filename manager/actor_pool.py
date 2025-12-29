from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

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
    ) -> None:
        self._repo = repo
        self._http = http

        self._pool_size = int(pool_size)
        self._http_port = int(http_port)
        self._http_concurrency = int(http_concurrency)
        self._base_image = (base_image or "").strip()
        self._default_seed = int(default_seed)

        self._lock = asyncio.Lock()
        self._pool: Dict[ActorKey, PoolEntry] = {}
        self._actor_routes: Dict[ActorKey, ActorRoute] = {}

    async def reset(self) -> None:
        async with self._lock:
            self._pool.clear()
            self._actor_routes.clear()
            self._repo.reset_cursor()

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

    async def prewarm(self, registry: ClusterRegistry) -> None:
        if self._pool_size <= 0:
            print("[manager] pool_size <= 0, skip prewarm")
            return

        # Reserve rows under lock, but do not do network under lock.
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

    async def close_and_refill(self, *, env: str, env_id: str, registry: ClusterRegistry) -> None:
        env = str(env)
        env_id = str(env_id)
        key: ActorKey = (env, env_id)

        # Resolve route without holding lock during HTTP
        async with self._lock:
            route = self._actor_routes.get(key)
            binding = registry.env_bindings.get(env)

            # remove immediately from local pool (even if remote close fails)
            self._pool.pop(key, None)
            self._actor_routes.pop(key, None)

            next_row = self._repo.fetch_one_active_row()

        host = ""
        if route:
            host = route[0]
        elif binding and binding.head_ip:
            host = binding.head_ip

        if host:
            close_url = f"http://{host}:{self._http_port}/{env}/{env_id}/close"
            try:
                resp = await self._http.post(close_url)
                if resp.status_code >= 400:
                    print(f"[manager] close failed: {close_url} status={resp.status_code} body={resp.text[:200]}")
            except Exception as e:
                print(f"[manager] close error: {close_url} err={e}")
        else:
            print(f"[manager] close skipped: no route/binding for env='{env}' id='{env_id}'")

        if not next_row:
            print("[manager] no more DB rows to refill pool")
            return

        await self._create_actor_for_row(next_row, registry, sem=None)

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

        cluster = registry.clusters_by_image.get(image)
        if cluster is None:
            # fallback: try env binding's image if row image was unexpected
            binding = registry.env_bindings.get(env_name)
            if binding:
                cluster = registry.clusters_by_image.get(binding.image)
                image = binding.image

        if cluster is None or not cluster.head_ip:
            raise RuntimeError(f"No cluster/head_ip for env='{env_name}', image='{image}'")

        url = f"http://{cluster.head_ip}:{self._http_port}/{env_name}/{env_id}/reset"
        payload = {
            "env_param": row.get("env_param"),
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
            await _do_post()
        else:
            async with sem:
                await _do_post()
