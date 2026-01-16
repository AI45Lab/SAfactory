from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional, Tuple

from ..binding_plan import BindingPlan
from ..http_client import HttpServiceClient
from ..types import ClusterRegistry, EnvClusterBinding, RayClusterInfo

from .base import ClusterBackend
from .ryajob import RayJobManager


# If you don't provide a default entrypoint, we will try per-env entrypoints only.
DEFAULT_ENTRYPOINT = "python env/app.py"


def build_rayjob_config(cluster_cfg: Dict[str, Any], env_name: str) -> Tuple[Optional[Any], List[Any]]:
    """
    Build rayjob_sdk.HeadConfig and a list of rayjob_sdk.Volume objects.

    Returns:
        Tuple(HeadConfig, List[Volume])
    """
    env_types = dict(cluster_cfg.get("env_types", {}) or {})
    env_cfg = dict(env_types.get(str(env_name), {}) or {})

    # Expected shape:
    #   cluster.env_types.<env>.resources.head: {cpu, gpu, memory}
    head_res = dict(((env_cfg.get("resources") or {}).get("head") or {}) or {})
    raw_volumes = env_cfg.get("volumes")

    if not head_res and not raw_volumes:
        return None, []

    resources: Dict[str, str] = {}
    if "cpu" in head_res and head_res.get("cpu") is not None:
        resources["cpu"] = str(head_res.get("cpu"))
    if "memory" in head_res and head_res.get("memory") is not None:
        resources["memory"] = str(head_res.get("memory"))

    gpu = head_res.get("gpu")
    if gpu is None:
        gpu = head_res.get("nvidia.com/gpu")
    if gpu is not None:
        resources["nvidia.com/gpu"] = str(gpu)

    from rayjob_sdk import HeadConfig, Volume

    sdk_volumes = []
    if raw_volumes and isinstance(raw_volumes, list):
        for vol_data in raw_volumes:
            if isinstance(vol_data, dict):
                sdk_volumes.append(Volume(**vol_data))

    kwargs: Dict[str, Any] = {}
    if resources:
        kwargs["resources"] = resources

    if raw_volumes:
        kwargs["volumes"] = raw_volumes

    head_config = None
    if kwargs:
        try:
            head_config = HeadConfig(**kwargs)
        except TypeError:
            kwargs.pop("volumes", None)
            if kwargs:
                head_config = HeadConfig(**kwargs)

    return head_config, sdk_volumes



def _normalize_entrypoints(raw: Any) -> Dict[str, str]:
    if raw is None:
        return {}

    if isinstance(raw, str):
        ep = raw.strip()
        return {"*": ep} if ep else {}

    if isinstance(raw, dict):
        out: Dict[str, str] = {}
        for k, v in raw.items():
            env = str(k).strip()
            ep = str(v).strip()
            if env and ep:
                out[env] = ep
        return out

    if isinstance(raw, list):
        out: Dict[str, str] = {}
        for item in raw:
            if not isinstance(item, dict):
                continue
            env = str(item.get("env", "")).strip()
            ep = str(item.get("entrypoint", "")).strip()
            if env and ep:
                out[env] = ep
        return out

    raise TypeError(f"Unsupported entrypoint config type: {type(raw)!r}")


class RemoteRayJobBackend(ClusterBackend):
    """
    Remote backend that manages Ray clusters via RayJobManager (rayjob_sdk).
    """

    def __init__(
        self,
        *,
        rayjob_cfg: Dict[str, Any],
        cluster_cfg: Dict[str, Any],
        http: HttpServiceClient,
        http_port: int,
    ) -> None:
        self._http = http
        self._http_port = int(http_port)

        required = ["domain", "tenant", "access_key", "secret_key"]
        missing = [k for k in required if not str(rayjob_cfg.get(k, "")).strip()]
        if missing:
            raise RuntimeError(f"Remote mode requires rayjob config keys: {missing}")

        self._rayjob_project: str = str(rayjob_cfg.get("project", "default")).strip() or "default"

        self._rayjob_manager = RayJobManager(
            domain=str(rayjob_cfg["domain"]),
            tenant=str(rayjob_cfg["tenant"]),
            access_key=str(rayjob_cfg["access_key"]),
            secret_key=str(rayjob_cfg["secret_key"]),
            token=rayjob_cfg.get("token"),
            verify=bool(rayjob_cfg.get("verify", False)),
        )

        self._cluster_cfg: Dict[str, Any] = dict(cluster_cfg or {})
        self._env_types: Dict[str, Any] = dict(self._cluster_cfg.get("env_types", {}) or {})

        self._quotagroup: str = str(self._cluster_cfg.get("quotagroup", "")).strip()
        self._description: str = str(
             rayjob_cfg.get("description", self._cluster_cfg.get("description", "RL env Ray cluster"))
        ).strip()
        self._entrypoints: Dict[str, str] = _normalize_entrypoints(self._cluster_cfg.get("entrypoint"))

        self._default_entrypoint: str = (
            str(self._cluster_cfg.get("default_entrypoint", DEFAULT_ENTRYPOINT)).strip() or DEFAULT_ENTRYPOINT
        )

        self._poll_interval_s: float = float(cluster_cfg.get("head_ip_poll_interval_s", 5.0))
        self._poll_timeout_s: float = float(cluster_cfg.get("head_ip_poll_timeout_s", 600.0))

        self._clusters_by_image: Dict[str, RayClusterInfo] = {}

    async def start(self, plan: BindingPlan) -> ClusterRegistry:
        if not plan.images_needed:
            return ClusterRegistry(clusters_by_image={}, env_bindings={})

        tasks = [self._ensure_cluster_for_image(image=img, plan=plan) for img in plan.images_needed]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        errors = [r for r in results if isinstance(r, Exception)]
        if errors:
            for e in errors[:3]:
                print(f"[manager] ERROR: rayjob create failed: {e}")
            raise RuntimeError(f"RayJob cluster creation failed for {len(errors)} image(s).")

        await self._wait_for_head_ips()
        await self._wait_for_head_http_services()

        env_bindings: Dict[str, EnvClusterBinding] = {}
        for env_name, image in plan.env_to_image.items():
            info = self._clusters_by_image.get(image)
            if not info:
                continue
            env_bindings[env_name] = EnvClusterBinding(
                env_name=env_name,
                image=image,
                project=info.project,
                job_name=info.job_name,
                head_ip=info.head_ip,
            )

        return ClusterRegistry(clusters_by_image=dict(self._clusters_by_image), env_bindings=env_bindings)

    async def close(self) -> None:
        jobs: List[Tuple[str, str]] = []
        for info in self._clusters_by_image.values():
            if info.job_name:
                jobs.append((info.project, info.job_name))

        jobs = list(dict.fromkeys(jobs))
        self._clusters_by_image.clear()

        # stop then delete; each best-effort
        for project, job_name in jobs:
            try:
                await asyncio.to_thread(self._rayjob_manager.stop, project, job_name)
            except Exception as e:
                print(f"[manager] stop rayjob failed (ignored): project={project}, job={job_name}, err={e}")

        for project, job_name in jobs:
            try:
                await asyncio.to_thread(self._rayjob_manager.delete, project, job_name)
            except Exception as e:
                print(f"[manager] delete rayjob failed (ignored): project={project}, job={job_name}, err={e}")

    # ------------------------------------------------------------------ #

    async def _ensure_cluster_for_image(self, *, image: str, plan: BindingPlan) -> None:
        if not image:
            return
        if image in self._clusters_by_image:
            return

        env_name = plan.image_to_env.get(image, "")
        env_cfg = dict(self._env_types.get(env_name, {}) or {})
        entrypoint = str(
            env_cfg.get("entrypoint")
            or self._entrypoints.get(env_name)
            or self._entrypoints.get("*")
            or self._default_entrypoint
        ).strip()

        if not entrypoint:
            raise RuntimeError(
                f"No entrypoint configured for env='{env_name}'. "
                f"Provide cluster.env_types['{env_name}'].entrypoint (preferred) "
                "or cluster_cfg['entrypoint']/default_entrypoint (legacy)."
            )

        quotagroup = str(env_cfg.get("quotagroup") or self._quotagroup).strip()
        head_config,volumes = build_rayjob_config(self._cluster_cfg, env_name)

        def _create_job_sync() -> str:
            return str(
                self._rayjob_manager.create(
                    project=self._rayjob_project,
                    image=image,
                    entrypoint=str(entrypoint),
                    quotagroup=str(quotagroup),
                    head_config=head_config,
                    volumes=volumes,
                    description=self._description or f"Env cluster for image={image}",
                )
            )

        max_attempts = 3
        last_err: Optional[Exception] = None
        for attempt in range(1, max_attempts + 1):
            try:
                job_name = await asyncio.to_thread(_create_job_sync)
                self._clusters_by_image[image] = RayClusterInfo(
                    image=image,
                    project=self._rayjob_project,
                    job_name=job_name,
                    head_ip="",
                )
                print(f"[manager] RayJob created for image='{image}', job_name='{job_name}'")
                return
            except Exception as e:
                last_err = e
                sleep_s = min(5.0, 0.5 * (2 ** (attempt - 1)))
                print(
                    f"[manager] RayJob create failed (attempt {attempt}/{max_attempts}) "
                    f"for image='{image}': {e}. Retry in {sleep_s}s"
                )
                if attempt < max_attempts:
                    await asyncio.sleep(sleep_s)

        raise RuntimeError(f"RayJob create failed for image='{image}': {last_err}")

    async def _wait_for_head_ips(self) -> None:
        if not self._clusters_by_image:
            return

        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._poll_timeout_s
        attempt = 0

        def missing_images() -> List[str]:
            return [img for img, info in self._clusters_by_image.items() if not info.head_ip]

        while True:
            missing = missing_images()
            if not missing:
                print("[manager] all clusters have head_ip")
                return

            attempt += 1
            print(f"[manager] head_ip poll attempt {attempt}, missing={missing}")

            tasks = [
                asyncio.to_thread(
                    self._rayjob_manager.get_head_ip,
                    self._rayjob_project,
                    self._clusters_by_image[img].job_name,
                )
                for img in missing
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for img, res in zip(missing, results):
                if isinstance(res, Exception):
                    print(f"[manager] get_head_ip failed for image='{img}': {res}")
                    continue
                ip = (res or "").strip()
                if ip:
                    self._clusters_by_image[img].head_ip = ip
                    print(f"[manager] head_ip resolved: image='{img}' -> {ip}")

            if not missing_images():
                print("[manager] head_ip polling finished")
                return

            if loop.time() >= deadline:
                raise RuntimeError(f"Timeout waiting for head IPs: {missing_images()}")

            await asyncio.sleep(self._poll_interval_s)

    async def _wait_for_head_http_services(self) -> None:
        if not self._clusters_by_image:
            return

        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._poll_timeout_s
        attempt = 0

        while True:
            clusters = [(img, info) for img, info in self._clusters_by_image.items() if info.head_ip]
            if not clusters:
                raise RuntimeError("No head_ip resolved; cannot check HTTP readiness")

            tasks = [self._http.check_envs_ready(info.head_ip, self._http_port) for _, info in clusters]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            not_ready: List[Tuple[str, RayClusterInfo]] = []
            for (img, info), res in zip(clusters, results):
                if isinstance(res, Exception) or not res:
                    not_ready.append((img, info))

            if not not_ready:
                print("[manager] all head HTTP services are ready")
                return

            attempt += 1
            if loop.time() >= deadline:
                remaining = ", ".join(f"{img}@{info.head_ip}" for img, info in not_ready)
                raise RuntimeError(f"Timeout waiting for head HTTP services: {remaining}")

            print(
                f"[manager] head HTTP not ready (attempt {attempt}), retry in {self._poll_interval_s}s. "
                f"not_ready={[img for img, _ in not_ready]}"
            )
            await asyncio.sleep(self._poll_interval_s)
