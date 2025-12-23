import asyncio
import sqlite3
import traceback
import httpx
import time
import requests
import asyncio

from dataclasses import dataclass
from typing import Dict, Optional, Any, List, Tuple

from db_loader import (
    get_active_data,
    get_env_image_map,
    get_all_image
)
from cluster import RayJobManager


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class RayClusterInfo:
    """
    Descriptor of one Ray cluster (backed by a RayJob).
    job_name is the *actual* RayJob name returned by the create API.
    """
    image: str
    project: str
    job_name: str
    head_ip: str  # filled after RayJob becomes ready


@dataclass(slots=True)
class EnvClusterBinding:
    """
    Binding between an env_name and a Ray cluster.
    Different envs may share the same cluster if they use the same image.
    """
    env_name: str
    image: str
    project: str
    job_name: str
    head_ip: str


@dataclass(slots=True)
class PoolEntry:
    """
    Lightweight record of a *remote* actor in the pool.

    We do NOT keep Ray actor handles locally. The identity is solely:
        - env_name
        - env_id

    The actor itself lives inside the target Ray cluster, created lazily on
    the cluster-side HTTP service (first POST /{env}/{id}/reset).
    """
    env_name: str
    env_id: str
    row_id: Optional[int]  # DB primary key if available (for tracing)
    image: str  # quick reverse lookup
    job_name: str  # which RayJob it belongs to
    head_ip: str  # cluster head IP for routing
    status: str = "ready"  # reserved for future (ready/closing/closed)


# ---------------------------------------------------------------------------
# EnvPoolManager: manages Ray clusters + remote actor pool (prewarm & refill)
# ---------------------------------------------------------------------------


class EnvPoolManager:
    """
    Cluster-aware pool manager.

    Startup:
      1) scan DB to build env_name -> image mapping;
      2) create ONE RayJob (Ray cluster) per distinct image (record *actual* job_name);
      3) bind env_name -> cluster (head_ip initially empty);
      4) poll RayJob until every cluster has head_ip;
      5) prewarm remote actor pool: read first N rows from DB and, for each row,
         call cluster HTTP  POST /{env}/{id}/reset  to lazily create the actor,
         then record (env, id) in the local pool map.

    Runtime:
      - close_and_refill(env, id): close a remote actor (POST /{env}/{id}/close),
        remove it from the pool, then pull the next DB row and create a new
        remote actor (POST /{env}/{id}/reset) to keep the pool at target size.
    """

    def __init__(self, cfg: dict, conn: sqlite3.Connection) -> None:
        self.cfg = cfg or {}
        self.conn = conn

        # ---- pool config ----
        self.pool_size: int = int(self.cfg.get("pool_size", 0) or 0)

        # ---- cluster & RayJob config ----
        cluster_cfg = dict(self.cfg.get("cluster", {}) or {})
        rayjob_cfg = dict(self.cfg.get("rayjob", {}) or {})

        # base image for envs without explicit image in DB
        self._base_image: str = str(cluster_cfg.get("base_image") or "").strip()
        if not self._base_image:
            # legacy fallback
            self._base_image = str(self.cfg.get("base_image", "")).strip()

        if not rayjob_cfg:
            raise RuntimeError(
                "rayjob configuration is required (domain, tenant, access_key, "
                "secret_key, project, etc.)"
            )

        # RayJob project and manager
        self._rayjob_project: str = str(rayjob_cfg.get("project", "default"))
        self._rayjob_manager = RayJobManager(
            domain=rayjob_cfg["domain"],
            tenant=rayjob_cfg["tenant"],
            access_key=rayjob_cfg["access_key"],
            secret_key=rayjob_cfg["secret_key"],
            token=rayjob_cfg.get("token"),
            verify=bool(rayjob_cfg.get("verify", False)),
        )

        # Entrypoint & metadata for the Ray cluster pods
        self._cluster_entrypoint = cluster_cfg.get("entrypoint", "{}")
        self._cluster_quotagroup: str = str(cluster_cfg.get("quotagroup", "")).strip()
        self._cluster_description: str = str(
            cluster_cfg.get("description", "RL env Ray cluster")
        ).strip()

        # HTTP configuration for talking to per-cluster services
        http_cfg = dict(cluster_cfg.get("http", {}) or {})
        self._cluster_http_port: int = int(
            http_cfg.get(
                "port",
                self.cfg.get("server", {}).get("port", 36663),
            )
        )
        self._http_timeout_s: float = float(http_cfg.get("timeout_s", 10.0))
        self._http_concurrency: int = int(http_cfg.get("concurrency", 64))

        # Head IP polling config
        self._head_ip_poll_interval_s: float = float(
            cluster_cfg.get("head_ip_poll_interval_s", 5.0)
        )
        self._head_ip_poll_timeout_s: float = float(
            cluster_cfg.get("head_ip_poll_timeout_s", 600.0)
        )

        # ---- in-memory state ----
        # image -> RayClusterInfo
        self._clusters_by_image: Dict[str, RayClusterInfo] = {}
        # env_name -> EnvClusterBinding
        self._env_bindings: Dict[str, EnvClusterBinding] = {}
        # (env_name, env_id) -> PoolEntry (represents a *remote* actor)
        self._pool: Dict[Tuple[str, str], PoolEntry] = {}
        # per-actor routing index (same keys as _pool; separated for clarity)
        self._actor_routes: Dict[Tuple[str, str], Tuple[str, str]] = {}  # (head_ip, port)

        # TODO: db cursor should be locked as there may be not more than one requests at the meantime
        # DB cursor offset for sequential row reservation
        self._db_offset: int = 0

        # HTTP client (created on start())
        self._http_client: Optional[httpx.AsyncClient] = None

        # concurrency primitives
        self._state_lock = asyncio.Lock()  # guards start/close_all
        self._pool_lock = asyncio.Lock()  # guards pool & DB offset
        self._initialized: bool = False

    # ------------------------------------------------------------------ #
    # Lifecycle                                                         #
    # ------------------------------------------------------------------ #

    async def start(self) -> None:
        """
        Initialize Ray clusters and pre-warm remote actor pool.
        """
        async with self._state_lock:
            if self._initialized:
                return

            if self._http_client is None:
                self._http_client = httpx.AsyncClient(timeout=self._http_timeout_s, trust_env=True)

            # 1) clusters + env bindings (head_ip initially empty)
            await self._init_clusters_and_bindings()

            # 2) poll until head_ip is ready for each cluster
            await self._wait_for_head_ips()

            await self._wait_for_head_http_services()

            # 3) pre-warm remote actor pool
            await self._prewarm_pool()

            self._initialized = True

    async def close_all(self) -> None:
        """
        Shutdown hook: stop + delete all RayJobs created/owned by this manager.
        """
        async with self._state_lock:
            jobs: List[Tuple[str, str]] = []
            for info in self._clusters_by_image.values():
                if info and info.job_name:
                    jobs.append((info.project, info.job_name))

            jobs = list(dict.fromkeys(jobs))

            client, self._http_client = self._http_client, None

            self._initialized = False

            async with self._pool_lock:
                self._pool.clear()
                self._actor_routes.clear()
                self._db_offset = 0

            self._env_bindings.clear()
            self._clusters_by_image.clear()

        if client is not None:
            try:
                await client.aclose()
            except Exception as e:
                print(f"[manager] http client close failed (ignored): {e}")

        if not jobs:
            return

        for project, job_name in jobs:
            try:
                await asyncio.to_thread(self._rayjob_manager.stop, project, job_name)
            except Exception as e:
                print(
                    "[manager] stop rayjob failed (ignored): "
                    f"project='{project}', job_name='{job_name}', err={e}"
                )

        for project, job_name in jobs:
            try:
                await asyncio.to_thread(self._rayjob_manager.delete, project, job_name)
            except Exception as e:
                print(
                    "[manager] delete rayjob failed (ignored): "
                    f"project='{project}', job_name='{job_name}', err={e}"
                )



    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #

    @property
    def env_cluster_map(self) -> Dict[str, EnvClusterBinding]:
        """
        Read-only view of env -> cluster binding (for routing by env_name).
        """
        return self._env_bindings

    def get_cluster_for_env(self, env_name: str) -> Optional[EnvClusterBinding]:
        return self._env_bindings.get(env_name)

    # Backward-compatible shim for old interface
    def get(self, env: str, id_: str) -> Optional[EnvClusterBinding]:
        return self.get_cluster_for_env(env)

    async def list_status(self, parallelism: int = 128) -> List[dict]:
        """
        List bindings between envs and clusters.
        """
        return [
            {
                "env": b.env_name,
                "image": b.image,
                "project": b.project,
                "job_name": b.job_name,
                "head_ip": b.head_ip,
            }
            for b in self._env_bindings.values()
        ]

    async def list_pool_actors(self) -> List[dict]:
        """
        List all remote env actors currently tracked in the local pool.
        Each entry contains the env_name and env_id of an actor.
        """
        async with self._pool_lock:
            return [
                {
                    "env_name": entry.env_name,
                    "env_id": entry.env_id,
                }
                for entry in self._pool.values()
            ]

    def get_actor_route(self, env: str, id_: str) -> Optional[Tuple[str, str]]:
        """
        Return (head_ip, port) for a specific remote actor if tracked;
        fallback to env-level binding if the exact (env,id) was not yet warmed.
        """
        key = (env, str(id_))
        if key in self._actor_routes:
            return self._actor_routes[key]
        binding = self._env_bindings.get(env)
        if not binding or not binding.head_ip:
            return None
        return (binding.head_ip, self._cluster_http_port)

    async def close_and_remove(self, env: str, id_: str) -> None:
        """
        Legacy API compatibility; alias to close_and_refill().
        """
        await self.close_and_refill(env, id_)

    async def close_and_refill(self, env: str, id_: str) -> None:

        if self._http_client is None:
            raise RuntimeError("HTTP client is not initialized")

        async with self._pool_lock:
            key = (env, str(id_))
            route = self._actor_routes.get(key)

            if route is not None:
                head_ip, port = route
                close_url = f"http://{head_ip}:{port}/{env}/{str(id_)}/close"
            else:
                binding = self._env_bindings.get(env)
                if not binding or not binding.head_ip:
                    raise RuntimeError(
                        f"Cluster binding for env='{env}' is not ready; head_ip missing"
                    )
                close_url = self._build_actor_close_url(binding, env, id_)

            try:
                resp = await self._http_client.post(close_url)
                if resp.status_code >= 400:
                    print(
                        f"[manager] remote close failed: env='{env}', id={id_}, "
                        f"status={resp.status_code}, body={resp.text}"
                    )
            except Exception as e:
                print(f"[manager] remote close error: env='{env}', id={id_}, err={e}")

            self._pool.pop(key, None)
            self._actor_routes.pop(key, None)

            next_row = self._reserve_one_row()
            if not next_row:
                print("[manager] no more DB rows to refill the pool")
                return
            await self._create_remote_actor_for_row(next_row)

    # ------------------------------------------------------------------ #
    # Internals: cluster bootstrap & env binding                         #
    # ------------------------------------------------------------------ #

    async def _init_clusters_and_bindings(self) -> None:
        """
        Scan DB, create Ray clusters per image and build env -> cluster map.
        """
        env_image_map = get_env_image_map(self.conn)
        if not env_image_map:
            print("[manager] No env/image mapping found in DB; nothing to start.")
            return

        if not self._base_image:
            raise RuntimeError(
                "cluster.base_image must be configured in config.yaml "
                "or each env must have an explicit image in the DB."
            )

        final_env_image: Dict[str, str] = {}

        image_to_env = get_all_image(self.conn)
        images_needed = set(image_to_env.keys())

        for env_name, image in env_image_map.items():
            env_name = str(env_name)
            effective_image = (image or "").strip() or self._base_image
            final_env_image[env_name] = effective_image
            images_needed.add(effective_image)
            if effective_image not in image_to_env:
                image_to_env[effective_image] = env_name

        if not final_env_image:
            print("[manager] No envs found when building bindings; nothing to do.")
            return

        # Create one RayJob per image (head_ip left empty for now)
        tasks = [
            self._ensure_cluster_for_image(img, "envmanager", image_to_env[img])
            for img in images_needed
        ]
        if tasks:
            await asyncio.gather(*tasks)

        # Bind each env_name to the corresponding cluster
        self._env_bindings.clear()
        for env_name, image in final_env_image.items():
            cluster = self._clusters_by_image.get(image)
            if cluster is None:
                print(
                    f"[manager] no Ray cluster found for env='{env_name}', "
                    f"image='{image}', skip binding."
                )
                continue

            self._env_bindings[env_name] = EnvClusterBinding(
                env_name=env_name,
                image=image,
                project=cluster.project,
                job_name=cluster.job_name,
                head_ip=cluster.head_ip,
            )

    async def _ensure_cluster_for_image(self, image: str, job_name_hint: Optional[str], env_name: str) -> None:
        """
        Ensure that there is a Ray cluster for a given image.

        We only create the RayJob and capture its *actual* job_name here.
        head_ip is resolved later by _wait_for_head_ips().
        """
        if image in self._clusters_by_image:
            return

        def _create_job_sync() -> str:
            result = self._rayjob_manager.create(
                project=self._rayjob_project,
                name=job_name_hint,
                image=image,
                entrypoint=str(self._cluster_entrypoint.get(env_name, "python /app/app.py")),
                quotagroup=self._cluster_quotagroup,
                description=self._cluster_description or f"Env cluster for image={image}",
            )
            actual_name = result
            print(
                f"[manager] RayJob {actual_name} created for image='{image}' "
            )
            return actual_name

        job_name = await asyncio.to_thread(_create_job_sync)

        # Record cluster with empty head_ip for now
        self._clusters_by_image[image] = RayClusterInfo(
            image=image, project=self._rayjob_project, job_name=job_name, head_ip=""
        )

    async def _wait_for_head_ips(self) -> None:
        """
        Poll RayJob API until every known cluster has a non-empty head_ip.
        """
        if not self._clusters_by_image:
            return

        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._head_ip_poll_timeout_s
        attempt = 0

        def _missing() -> List[str]:
            return [img for img, info in self._clusters_by_image.items() if not info.head_ip]

        while True:
            missing_images = _missing()
            if not missing_images:
                print("[manager] all Ray clusters have head_ip")
                return

            attempt += 1
            print(f"[manager] head_ip poll attempt {attempt}, missing={missing_images}")

            tasks = [
                asyncio.to_thread(
                    self._rayjob_manager.get_head_ip,
                    self._rayjob_project,
                    self._clusters_by_image[img].job_name,
                )
                for img in missing_images
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for img, ip_or_exc in zip(missing_images, results):
                if isinstance(ip_or_exc, Exception):
                    print(
                        f"[manager] get_head_ip failed for image='{img}', "
                        f"job='{self._clusters_by_image[img].job_name}': {ip_or_exc}"
                    )
                    continue
                ip = (ip_or_exc or "").strip()
                if not ip:
                    continue

                # Update cluster info
                info = self._clusters_by_image[img]
                info.head_ip = ip
                # Update all env bindings pointing to this image
                for b in self._env_bindings.values():
                    if b.image == img:
                        b.head_ip = ip
                print(f"[manager] head_ip resolved for image='{img}': {ip}")

            if not _missing():
                print("[manager] head_ip polling finished successfully")
                return

            if loop.time() >= deadline:
                raise RuntimeError(
                    f"Timeout waiting for head IPs for images: {_missing()}"
                )

            await asyncio.sleep(self._head_ip_poll_interval_s)

    async def _wait_for_head_http_services(self) -> None:
        """
        Poll Ray head HTTP services on all known cluster heads until they are ready.

        This uses check_head_http_ready(head_ip, port) to probe /envs on the
        per-cluster service for every head_ip we have recorded.
        """
        if not self._clusters_by_image:
            return

        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._head_ip_poll_timeout_s
        attempt = 0

        while True:
            clusters_with_ip = [
                (image, info)
                for image, info in self._clusters_by_image.items()
                if info.head_ip
            ]
            if not clusters_with_ip:
                print(
                    "[manager] no head_ip available for HTTP readiness check; "
                    "skipping head HTTP wait."
                )
                return

            tasks = [
                check_head_http_ready(info.head_ip, self._cluster_http_port)
                for image, info in clusters_with_ip
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            not_ready: List[Tuple[str, RayClusterInfo]] = []
            for (image, info), res in zip(clusters_with_ip, results):
                if isinstance(res, Exception):
                    print(
                        f"[manager] check_head_http_ready failed for image='{image}', "
                        f"head_ip='{info.head_ip}': {res}"
                    )
                    not_ready.append((image, info))
                elif not res:
                    not_ready.append((image, info))

            if not not_ready:
                print("[manager] all Ray head HTTP services are ready")
                return

            if loop.time() >= deadline:
                remaining = ", ".join(
                    f"image='{image}', head_ip='{info.head_ip}'"
                    for image, info in not_ready
                )
                raise RuntimeError(
                    "Timeout waiting for head HTTP services to be ready: "
                    + remaining,
                )

            attempt += 1
            print(
                f"[manager] head HTTP readiness poll attempt {attempt}, ",
                f"not ready images={[image for image, _ in not_ready]}",
            )

            await asyncio.sleep(self._head_ip_poll_interval_s)

    # ------------------------------------------------------------------ #
    # Remote actor pool (prewarm & refill)                               #
    # ------------------------------------------------------------------ #

    async def _prewarm_pool(self) -> None:
        """
        Pre-warm the *remote* actor pool based on the first N DB rows.

        For each row we call:
            POST http://{head_ip}:{port}/{env}/{id}/reset
        The cluster-side service lazily creates the actor on first reset().
        """
        if self.pool_size <= 0:
            print("[manager] pool_size <= 0; skip pool prewarm.")
            return

        rows = get_active_data(self.conn, self.pool_size, self._db_offset)
        self._db_offset += len(rows)

        if not rows:
            print("[manager] No active rows found in DB; skip pool prewarm.")
            return
        env_name = str(rows[0].get("env_name"))
        binding = self._env_bindings.get(env_name)
        ok = await check_head_http_ready(binding.head_ip, 36663)
        if not ok:
            print(f"[manager] ERROR: head HTTP service not ready at {binding.head_ip}:36663")

        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=self._http_timeout_s, trust_env=True)

        sem = asyncio.Semaphore(self._http_concurrency)
        tasks: List[asyncio.Task[Any]] = []

        async with self._pool_lock:
            for row in rows:
                tasks.append(asyncio.create_task(self._create_remote_actor_for_row(row, sem)))

        if tasks:
            await asyncio.gather(*tasks)

    async def _create_remote_actor_for_row(
            self,
            row: Dict[str, Any],
            sem: Optional[asyncio.Semaphore] = None,
    ) -> None:
        """
            POST http://{head_ip}:{port}/{env}/{id}/reset
        """
        if self._http_client is None:
            raise RuntimeError("HTTP client is not initialized")

        env_name = str(row.get("env_name"))
        env_id = str(row.get("env_id"))

        image = (row.get("image") or "").strip()
        if not image:
            env_binding = self._env_bindings.get(env_name)
            if env_binding and env_binding.image:
                image = env_binding.image
            else:
                image = self._base_image

        cluster = self._clusters_by_image.get(image)
        if cluster is None:
            print(
                f"[manager] skip creating actor for row id={row.get('id')}: "
                f"no Ray cluster for image='{image}' (env='{env_name}')"
            )
            return

        if not cluster.head_ip:
            print(
                f"[manager] skip creating actor for row id={row.get('id')}: "
                f"cluster for image='{image}' has empty head_ip"
            )
            return

        binding = EnvClusterBinding(
            env_name=env_name,
            image=image,
            project=cluster.project,
            job_name=cluster.job_name,
            head_ip=cluster.head_ip,
        )

        url = self._build_actor_reset_url(binding, env_name, env_id)
        payload = {
            "env_param": row["env_param"],
            "seed": 123,
        }
        print(url)
        print(payload)

        async def _do_post():
            max_attempts = 2
            attempt = 1
            while attempt <= max_attempts:
                try:
                    resp = await self._http_client.post(url, json=payload)
                    resp.raise_for_status()

                    key = (env_name, env_id)
                    self._pool[key] = PoolEntry(
                        env_name=env_name,
                        env_id=env_id,
                        row_id=row.get("id"),
                        image=image,
                        job_name=cluster.job_name,
                        head_ip=cluster.head_ip,
                        status="ready",
                    )
                    self._actor_routes[key] = (cluster.head_ip, self._cluster_http_port)

                    if attempt > 1:
                        print(
                            "[manager] remote actor ready after retry: "
                            f"env='{env_name}', id={env_id}, image='{image}', head_ip='{cluster.head_ip}'"
                        )
                    else:
                        print(
                            "[manager] remote actor ready (via reset) and added to pool: "
                            f"env='{env_name}', id={env_id}, image='{image}', head_ip='{cluster.head_ip}'"
                        )
                    return

                except (httpx.TimeoutException, asyncio.TimeoutError) as e:
                    print(
                        f"[manager] timeout creating remote actor (attempt {attempt}/{max_attempts}): "
                        f"env='{env_name}', id={env_id}, image='{image}', url='{url}', err={e}"
                    )
                    attempt += 1
                    if attempt > max_attempts:
                        print(
                            f"[manager] giving up creating remote actor after timeout retries: "
                            f"env='{env_name}', id={env_id}, image='{image}'"
                        )
                        return
                    continue

                except Exception as e:
                    print(
                        f"[manager] failed to create remote actor via reset: "
                        f"env='{env_name}', id={env_id}, image='{image}', err={e}"
                    )
                    return

        if sem is None:
            await _do_post()
        else:
            async with sem:
                await _do_post()

    async def _ensure_capacity(self) -> None:
        """
        Ensure the pool has at least `pool_size` remote actors.
        """
        while True:
            async with self._pool_lock:
                deficit = max(0, self.pool_size - len(self._pool))
            if deficit <= 0:
                return

            rows = get_active_data(self.conn, deficit, self._db_offset)
            if not rows:
                print("[manager] no more DB rows to fill the pool.")
                return

            self._db_offset += len(rows)

            sem = asyncio.Semaphore(self._http_concurrency)
            tasks: List[asyncio.Task[Any]] = []
            async with self._pool_lock:
                for row in rows:
                    tasks.append(asyncio.create_task(self._create_remote_actor_for_row(row, sem)))
            if tasks:
                await asyncio.gather(*tasks)

    def _reserve_one_row(self) -> Optional[Dict[str, Any]]:
        """
        Reserve the next DB row (sequentially by offset) for pool refill.
        """
        rows = get_active_data(self.conn, limit=1, offset=self._db_offset)
        if not rows:
            return None
        self._db_offset += 1
        return rows[0]

    # ------------------------------------------------------------------ #
    # HTTP helpers                                                       #
    # ------------------------------------------------------------------ #

    def _build_actor_reset_url(self, binding: EnvClusterBinding, env: str, env_id: str) -> str:
        host = binding.head_ip
        return f"http://{host}:{self._cluster_http_port}/{env}/{str(env_id)}/reset"

    def _build_actor_close_url(self, binding: EnvClusterBinding, env: str, id_: str) -> str:
        host = binding.head_ip
        return f"http://{host}:{self._cluster_http_port}/{env}/{str(id_)}/close"

    def _build_actor_step_url(self, binding: EnvClusterBinding, env: str, id_: str) -> str:
        host = binding.head_ip
        return f"http://{host}:{self._cluster_http_port}/{env}/{str(id_)}/step"


async def check_head_http_ready(head_ip: str, port: int = 36663, timeout: float = 5.0) -> bool:
    """
    Check whether the Ray head HTTP service is alive by GET /envs.

    Returns:
        True  -> service alive
        False -> service unreachable or non-200 status
    """
    url = f"http://{head_ip}:{port}/envs"

    def _do_request():
        try:
            resp = requests.get(url, timeout=timeout)
            return resp.status_code == 200
        except Exception:
            return False

    return await asyncio.to_thread(_do_request)