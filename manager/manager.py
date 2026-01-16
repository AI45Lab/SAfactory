from __future__ import annotations

import asyncio
import sqlite3
from typing import Any, Dict, List, Optional

from .actor_pool import ActorPool
from .binding_plan import build_binding_plan
from .http_client import HttpServiceClient
from .types import ActorRoute, ClusterRegistry, EnvClusterBinding
from .repository import EnvDataRepository
from .clusters.base import ClusterBackend


def _detect_mode(cfg: Dict[str, Any]) -> str:
    """
    Priority:
      1) root `mode` if set (new config)
      2) cluster.mode if set (backward compatible)
      3) if rayjob required keys exist -> remote
      4) otherwise -> local
    """

    mode = str(cfg.get("mode", "")).strip().lower()
    if mode in ("local", "localhost"):
        return "local"
    if mode in ("remote", "rayjob", "cluster"):
        return "remote"

    cluster_cfg = dict(cfg.get("cluster", {}) or {})
    mode = str(cluster_cfg.get("mode", "")).strip().lower()
    if mode in ("local", "localhost"):
        return "local"
    if mode in ("remote", "rayjob", "cluster"):
        return "remote"

    rayjob_cfg = dict(cfg.get("rayjob", {}) or {})
    required = ["domain", "tenant", "access_key", "secret_key"]
    if all(str(rayjob_cfg.get(k, "")).strip() for k in required):
        return "remote"
    return "local"


class EnvPoolManager:
    def __init__(self, cfg: dict, conn: sqlite3.Connection) -> None:
        self.cfg = cfg or {}
        self._repo = EnvDataRepository(conn)

        self._pool_size: int = int(self.cfg.get("pool_size", 0) or 0)

        cluster_cfg: Dict[str, Any] = dict(self.cfg.get("cluster", {}) or {})
        rayjob_cfg: Dict[str, Any] = dict(self.cfg.get("rayjob", {}) or {})

        self._base_image: str = str(cluster_cfg.get("base_image") or self.cfg.get("base_image") or "").strip()

        http_cfg = dict(cluster_cfg.get("http", {}) or {})
        self._http_port: int = int(http_cfg.get("port", self.cfg.get("server", {}).get("port", 36663)))
        self._http_timeout_s: float = float(http_cfg.get("timeout_s", 10.0))
        self._http_concurrency: int = int(http_cfg.get("concurrency", 64))

        self._default_seed: int = int(self.cfg.get("seed", 123))

        self._http = HttpServiceClient(timeout_s=self._http_timeout_s, trust_env=True)

        self._mode: str = _detect_mode(self.cfg)

        # IMPORTANT: backend is created lazily with imports inside _build_backend()
        self._backend: ClusterBackend = self._build_backend(cluster_cfg=cluster_cfg, rayjob_cfg=rayjob_cfg)

        self._pool = ActorPool(
            repo=self._repo,
            http=self._http,
            pool_size=self._pool_size,
            http_port=self._http_port,
            http_concurrency=self._http_concurrency,
            base_image=self._base_image,
            default_seed=self._default_seed,
        )

        self._registry: ClusterRegistry = ClusterRegistry(clusters_by_image={}, env_bindings={})
        self._state_lock = asyncio.Lock()
        self._initialized: bool = False

    async def start(self) -> None:
        async with self._state_lock:
            if self._initialized:
                return

            await self._http.start()

            plan = build_binding_plan(self._repo, base_image=self._base_image)
            if not plan.env_to_image:
                print("[manager] No env/image mapping found in DB; nothing to start.")
                self._registry = ClusterRegistry(clusters_by_image={}, env_bindings={})
                self._initialized = True
                return

            self._registry = await self._backend.start(plan)
            await self._pool.prewarm(self._registry)

            self._initialized = True
            print(f"[manager] started in mode='{self._mode}', pool_size={self._pool_size}")

    async def close_all(self) -> None:
        async with self._state_lock:
            self._initialized = False
            await self._pool.reset()
            self._registry = ClusterRegistry(clusters_by_image={}, env_bindings={})

        try:
            await self._http.close()
        except Exception as e:
            print(f"[manager] http client close failed (ignored): {e}")

        try:
            await self._backend.close()
        except Exception as e:
            print(f"[manager] backend close failed (ignored): {e}")

    @property
    def env_cluster_map(self) -> Dict[str, EnvClusterBinding]:
        return self._registry.env_bindings

    def get_cluster_for_env(self, env_name: str) -> Optional[EnvClusterBinding]:
        return self._registry.env_bindings.get(env_name)

    def get(self, env: str, id_: str) -> Optional[EnvClusterBinding]:
        return self.get_cluster_for_env(env)

    async def list_status(self, parallelism: int = 128) -> List[dict]:
        return [
            {
                "env": b.env_name,
                "image": b.image,
                "project": b.project,
                "job_name": b.job_name,
                "head_ip": b.head_ip,
            }
            for b in self._registry.env_bindings.values()
        ]

    async def list_pool_actors(self) -> List[dict]:
        return await self._pool.list_actors()

    def get_actor_route(self, env: str, id_: str) -> Optional[ActorRoute]:
        binding = self._registry.env_bindings.get(env)
        return self._pool.get_actor_route(env, str(id_), fallback=binding)

    async def close_and_remove(self, env: str, id_: str) -> None:
        await self.close_and_refill(env, id_)

    async def close_and_refill(self, env: str, id_: str) -> None:
        if not self._initialized:
            raise RuntimeError("EnvPoolManager not started. Call await start() first.")
        await self._pool.close_and_refill(env=str(env), env_id=str(id_), registry=self._registry)

    def _build_backend(self, *, cluster_cfg: Dict[str, Any], rayjob_cfg: Dict[str, Any]) -> ClusterBackend:
        """
        Lazy-import backend modules so local mode doesn't require rayjob_sdk installed.
        """
        if self._mode == "local":
            from .clusters.local_clusters import LocalHTTPBackend

            local_cfg = dict(cluster_cfg.get("local", {}) or {})
            host = str(local_cfg.get("host", "127.0.0.1")).strip() or "127.0.0.1"
            if host == "0.0.0.0":
                host = "127.0.0.1"

            return LocalHTTPBackend(
                host=host,
                http=self._http,
                http_port=self._http_port,
                poll_interval_s=float(local_cfg.get("poll_interval_s", 1.0)),
                poll_timeout_s=float(local_cfg.get("poll_timeout_s", 60.0)),
            )

        from .clusters.ray_clusters import RemoteRayJobBackend
        return RemoteRayJobBackend(
            rayjob_cfg=rayjob_cfg,
            cluster_cfg=cluster_cfg,
            http=self._http,
            http_port=self._http_port,
        )
