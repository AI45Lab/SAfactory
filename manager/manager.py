from __future__ import annotations

import asyncio
import sqlite3
import logging
from typing import Any, Callable, Dict, List, Optional

from .actor_pool import ActorPool
from .binding_plan import build_binding_plan
from .http_client import HttpServiceClient
from .types import ActorRoute, ClusterRegistry, EnvClusterBinding, PoolEntry
from .repository import EnvDataRepository
from .clusters.base import ClusterBackend

log = logging.getLogger("manager")


def _detect_mode(cfg: Dict[str, Any]) -> str:
    """
    Read root `mode` and normalize to `local` or `remote`.

    Recognized aliases:
      local:  `local`, `localhost`
      remote: `remote`, `rayjob`, `cluster`

    Returns `remote` if mode is unset or unrecognized
    (backward-compatible default).
    """
    top_mode = str(cfg.get("mode", "")).strip().lower()
    if top_mode in ("local", "localhost"):
        return "local"
    if top_mode in ("remote", "rayjob", "cluster"):
        return "remote"
    if not top_mode:
        log.warning("'mode' not set in config, defaulting to 'remote'")
    else:
        log.warning("unrecognized mode=%r, defaulting to 'remote'", top_mode)
    return "remote"


class EnvPoolManager:
    def __init__(
        self,
        cfg: dict,
        conn: Optional[sqlite3.Connection],
        *,
        job_id: str = "",
        db_processing_done_checker: Optional[Callable[[], bool]] = None,
    ) -> None:
        self.cfg = cfg or {}
        self._job_id = str(job_id or "").strip()
        self._repo = EnvDataRepository(
            conn,
            job_id=self._job_id,
            db_processing_done_checker=db_processing_done_checker,
        )

        self._pool_size: int = int(self.cfg.get("pool_size", 0) or 0)

        cluster_cfg: Dict[str, Any] = dict(self.cfg.get("cluster", {}) or {})
        rayjob_cfg: Dict[str, Any] = dict(self.cfg.get("rayjob", {}) or {})

        env_types_cfg: Dict[str, Any] = dict(cluster_cfg.get("env_types", {}) or {})
        self._env_limits: Dict[str, int] = {}
        for env_name, env_cfg in env_types_cfg.items():
            if not isinstance(env_cfg, dict):
                continue
            lim = env_cfg.get("limit")
            if lim is None:
                continue
            try:
                self._env_limits[str(env_name)] = int(lim)
            except Exception:
                continue

        self._base_image: str = str(cluster_cfg.get("base_image") or self.cfg.get("base_image") or "").strip()
        transport_raw = cluster_cfg.get("transport", self.cfg.get("transport", "http"))
        self._transport_mode: str = str(transport_raw or "http").strip().lower() or "http"

        http_cfg = dict(cluster_cfg.get("http", {}) or {})
        self._http_port: int = int(http_cfg.get("port", self.cfg.get("server", {}).get("port", 36663)))
        self._http_timeout_s: float = float(http_cfg.get("timeout_s", 10.0))
        self._http_concurrency: int = int(http_cfg.get("concurrency", 64))
        startup_default = min(self._http_concurrency, 16)
        self._startup_concurrency: int = int(http_cfg.get("startup_concurrency", startup_default) or startup_default)

        self._default_seed: int = int(self.cfg.get("seed", 123))

        self._http = HttpServiceClient(timeout_s=self._http_timeout_s, trust_env=True)

        self._mode: str = _detect_mode(self.cfg)
        if self._transport_mode not in {"http", "inproc"}:
            raise ValueError(f"Unsupported transport mode: {self._transport_mode}")
        if self._transport_mode == "inproc" and self._mode != "local":
            raise ValueError("transport=inproc is currently supported only in local mode")

        # IMPORTANT: backend is created lazily with imports inside _build_backend()
        self._backend: ClusterBackend = self._build_backend(cluster_cfg=cluster_cfg, rayjob_cfg=rayjob_cfg)

        self._pool = ActorPool(
            repo=self._repo,
            http=self._http,
            pool_size=self._pool_size,
            http_port=self._http_port,
            http_concurrency=self._http_concurrency,
            startup_concurrency=self._startup_concurrency,
            base_image=self._base_image,
            default_seed=self._default_seed,
            env_limits=self._env_limits,
            transport_mode=self._transport_mode,
        )

        self._registry: ClusterRegistry = ClusterRegistry(clusters_by_id={}, env_bindings={})
        self._state_lock = asyncio.Lock()
        self._initialized: bool = False
        self._closed: bool = False

    async def start(self) -> None:
        async with self._state_lock:
            if self._initialized:
                return
            self._closed = False

            if self._transport_mode == "http":
                await self._http.start()

            plan_base_image = self._base_image
            if self._transport_mode == "inproc" and not plan_base_image:
                plan_base_image = "__inproc_local__"
            tmp_plan = build_binding_plan(self._repo, base_image=plan_base_image)
            if not tmp_plan.env_to_image:
                log.warning("No env/image mapping found in DB; nothing to start.")
                self._registry = ClusterRegistry(clusters_by_id={}, env_bindings={})
                self._initialized = True
                return

            # Prime the initial warm-pool batch up front so prewarm does not have
            # to assemble its first usable rows through repeated DB probes.
            prewarm_rows: Optional[List[dict]] = None
            batch_counts: Dict[str, int] = {}
            if self._pool_size > 0:
                prewarm_rows = await self._repo.prime(self._pool_size)
                for row in prewarm_rows:
                    env_name = str(row.get("env_name", "")).strip()
                    if not env_name:
                        continue
                    batch_counts[env_name] = batch_counts.get(env_name, 0) + 1

            # compute per-env RayJob counts using env.limit
            env_job_counts: Dict[str, int] = {}
            for env_name in tmp_plan.env_to_image.keys():
                cnt = int(batch_counts.get(env_name, 0) or 0)
                lim = int(self._env_limits.get(env_name, 0) or 0)
                if lim > 0:
                    jobs = (max(cnt, 1) + lim - 1) // lim
                else:
                    jobs = 1
                env_job_counts[env_name] = max(1, int(jobs))
            final_plan = type(tmp_plan)(
                env_to_image=tmp_plan.env_to_image,
                image_to_env=tmp_plan.image_to_env,
                images_needed=tmp_plan.images_needed,
                env_job_counts=env_job_counts,
            )

            self._registry = await self._backend.start(final_plan)
            await self._pool.prewarm(self._registry, rows=prewarm_rows)

            self._initialized = True
            log.info(
                "started in mode='%s', pool_size=%d, startup_concurrency=%d, http_concurrency=%d, job_id=%s",
                self._mode,
                self._pool_size,
                self._startup_concurrency,
                self._http_concurrency,
                self._job_id or "<all>",
            )

    async def close_all(self) -> None:
        async with self._state_lock:
            if self._closed:
                log.info("EnvPoolManager.close_all(): already closed")
                return
            self._initialized = False
            self._closed = True
            await self._pool.reset()
            await self._pool.shutdown()
            self._registry = ClusterRegistry(clusters_by_id={}, env_bindings={})

        try:
            await self._http.close()
        except Exception as e:
            log.warning("http client close failed (ignored): %s", e)

        try:
            await self._backend.close()
        except Exception as e:
            log.warning("backend close failed (ignored): %s", e)

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

    async def list_pool_actors(self) -> List[PoolEntry]:
        return await self._pool.list_actors()

    def get_actor_route(self, env: str, id_: str) -> Optional[ActorRoute]:
        binding = self._registry.env_bindings.get(env)
        return self._pool.get_actor_route(env, str(id_), fallback=binding)

    async def close_and_remove(self, env: str, id_: str) -> None:
        await self.close_and_refill(env, id_)

    async def close_and_refill(self, env: str, id_: str) -> Optional[PoolEntry]:
        if not self._initialized:
            raise RuntimeError("EnvPoolManager not started. Call await start() first.")
        return await self._pool.close_and_refill(env=str(env), env_id=str(id_), registry=self._registry)

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
                require_http_ready=(self._transport_mode == "http"),
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
