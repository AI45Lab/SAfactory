from __future__ import annotations

import asyncio
import base64
import logging
import shlex
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict

import httpx

from manager.binding_plan import BindingPlan
from manager.types import PoolEntry

from .base import ClusterBackend

log = logging.getLogger("clusters.sandbox_cluster")


@dataclass(frozen=True, slots=True)
class SandboxCommandOutput:
    exit_code: int
    stdout: str
    stderr: str

    @classmethod
    def from_sdk(cls, result: Any) -> "SandboxCommandOutput":
        logs = getattr(result, "logs", None)

        def stream_text(name: str) -> str:
            values = getattr(logs, name, []) if logs is not None else []
            if isinstance(values, str):
                return values
            return "".join(str(getattr(item, "text", item)) for item in (values or []))

        exit_code = getattr(result, "exit_code", getattr(result, "code", 0))
        return cls(int(exit_code or 0), stream_text("stdout"), stream_text("stderr"))


async def run_sandbox_command(sandbox: Any, command: str, *, timeout_s: float) -> SandboxCommandOutput:
    result = await asyncio.wait_for(sandbox.commands.run(command), timeout=max(1.0, float(timeout_s)))
    return SandboxCommandOutput.from_sdk(result)


class SandboxClusterBackend(ClusterBackend):
    """OpenSandbox/Brainbox runtime allocator."""

    runtime = "sandbox"

    def __init__(self, *, cluster_cfg: Dict[str, Any] | None = None) -> None:
        self._cluster_cfg = dict(cluster_cfg or {})
        self._sandbox_cfg = dict(self._cluster_cfg.get("sandbox", {}) or {})
        self._env_types = dict(self._cluster_cfg.get("env_types", {}) or {})
        self.startup_concurrency = max(1, int(self._sandbox_cfg.get("startup_concurrency", 8) or 8))
        self._instances: Dict[str, Any] = {}

    async def start(self, plan: BindingPlan) -> None:
        if not plan.env_to_image:
            log.warning("empty Sandbox binding plan")
            return
        if not self._sandbox_cfg.get("api_key"):
            env_name = str(self._sandbox_cfg.get("api_key_env") or "OPEN_SANDBOX_API_KEY")
            raise ValueError(f"Sandbox API key is required; set {env_name} or sandbox.api_key")
        for env_name, image in sorted(plan.env_to_image.items()):
            await self._validate_environment(env_name, image, self._config_for_env(env_name))
        log.debug("Sandbox backend ready for %d agent image(s)", len(plan.env_to_image))

    async def allocate(
        self,
        *,
        row_id: Any,
        env_name: str,
        env_id: str,
        image: str,
        env_params: Dict[str, Any],
        group_id: str,
    ) -> PoolEntry:
        cfg = self._config_for_env(env_name)
        environment_id = str(cfg.get("environment_id") or "").strip()
        run_command = str(cfg.get("run_command") or "").strip()
        if not environment_id:
            raise ValueError(f"Sandbox environment_id is required for agent {env_name!r}")
        if not run_command:
            raise ValueError(f"Sandbox run_command is required for agent {env_name!r}")

        sandbox = await self._create_sandbox(image=image, environment_id=environment_id, cfg=cfg)
        try:
            endpoint = await sandbox.get_endpoint(int(cfg.get("command_port", 44772) or 44772))
            await self._prepare_instance(sandbox, cfg)
        except Exception:
            try:
                await self._terminate(sandbox)
            except Exception:
                log.warning("failed to clean up Sandbox after allocation error", exc_info=True)
            raise

        sandbox_id = str(getattr(sandbox, "id", None) or getattr(sandbox, "sandbox_id", None) or "")
        resource_id = sandbox_id or f"sandbox-{env_id}"
        endpoint_url = str(getattr(endpoint, "endpoint", "") or "")
        if endpoint_url and not endpoint_url.startswith(("http://", "https://")):
            endpoint_url = f"{str(cfg.get('protocol') or 'https')}://{endpoint_url}"
        self._instances[resource_id] = sandbox
        return PoolEntry(
            env_name=str(env_name),
            env_id=str(env_id),
            row_id=row_id,
            image=str(image),
            job_name=resource_id,
            env_params=dict(env_params or {}),
            group_id=str(group_id or ""),
            status="ready",
            runtime=self.runtime,
            runtime_config={
                "command_timeout_s": float(cfg.get("command_timeout_s", 720.0) or 720.0),
                "endpoint": endpoint_url,
                "endpoint_headers": dict(getattr(endpoint, "headers", {}) or {}),
                "env": dict(cfg.get("env", {}) or {}),
                "gateway_base_url": str(cfg.get("gateway_base_url") or ""),
            },
            runtime_handle=sandbox,
            resource_id=resource_id,
            resource_name=resource_id,
            workdir=str(cfg.get("workdir") or ""),
            run_command=run_command,
            result_mode=str(cfg.get("result_mode") or "json").strip().lower(),
            reuse_container=False,
        )

    async def release(self, entry: PoolEntry, *, succeeded: bool) -> None:
        del succeeded
        if bool(self._config_for_env(entry.env_name).get("cleanup_on_finish", True)):
            await self.remove(entry)
            return
        sandbox = self._instances.pop(entry.resource_id, None) or entry.runtime_handle
        if sandbox is not None:
            await sandbox.close()
        log.info("preserving Sandbox instance after run: %s", entry.resource_id)

    async def remove(self, entry: PoolEntry) -> None:
        sandbox = self._instances.pop(entry.resource_id, None) or entry.runtime_handle
        if sandbox is None:
            return
        try:
            await self._terminate(sandbox)
        except Exception:
            log.warning("failed to delete Sandbox instance %s", entry.resource_id, exc_info=True)

    async def close(self) -> None:
        instances = list(self._instances.items())
        self._instances.clear()
        results = await asyncio.gather(
            *(self._terminate(sandbox) for _, sandbox in instances),
            return_exceptions=True,
        )
        for (sandbox_id, _), result in zip(instances, results):
            if isinstance(result, Exception):
                log.warning("failed to delete Sandbox instance %s during shutdown: %s", sandbox_id, result)

    def _config_for_env(self, env_name: str) -> Dict[str, Any]:
        env_cfg = dict(self._env_types.get(str(env_name), {}) or {})
        sandbox_cfg = dict(env_cfg.get("sandbox", {}) or {})
        merged = {**self._sandbox_cfg, **sandbox_cfg}
        merged["env"] = {
            **dict(self._sandbox_cfg.get("env", {}) or {}),
            **dict(sandbox_cfg.get("env", {}) or {}),
        }
        merged["resource"] = {
            **dict(self._sandbox_cfg.get("resource", {}) or {}),
            **dict(sandbox_cfg.get("resource", {}) or {}),
        }
        merged["extensions"] = {
            **dict(self._sandbox_cfg.get("extensions", {}) or {}),
            **dict(sandbox_cfg.get("extensions", {}) or {}),
        }
        merged["embedded_files"] = [
            *list(self._sandbox_cfg.get("embedded_files", []) or []),
            *list(sandbox_cfg.get("embedded_files", []) or []),
        ]
        merged["required_mount_paths"] = [
            *list(self._sandbox_cfg.get("required_mount_paths", []) or []),
            *list(sandbox_cfg.get("required_mount_paths", []) or []),
        ]
        merged["environment_id"] = str(
            sandbox_cfg.get("environment_id") or self._sandbox_cfg.get("environment_id") or ""
        ).strip()
        return merged

    async def _validate_environment(self, env_name: str, image: str, cfg: Dict[str, Any]) -> None:
        environment_id = str(cfg.get("environment_id") or "").strip()
        if not environment_id:
            raise ValueError(f"Sandbox environment_id is required for agent {env_name!r}")
        params = {"project": cfg["project"]} if cfg.get("project") else None
        async with httpx.AsyncClient(timeout=float(cfg.get("request_timeout_s", 60.0))) as client:
            response = await client.get(
                f"{str(cfg['domain']).rstrip('/')}/v1/sandbox-environments/{environment_id}",
                params=params,
                headers={"OPEN-SANDBOX-API-KEY": str(cfg["api_key"])},
            )
            response.raise_for_status()
            environment = response.json()

        environment_image = str((environment.get("image") or {}).get("uri") or "")
        if environment_image and image and environment_image != image:
            raise ValueError(
                f"Sandbox environment {environment_id!r} image {environment_image!r} "
                f"does not match agent {env_name!r} image {image!r}"
            )
        ports = {
            int(port["containerPort"])
            for port in environment.get("ports", [])
            if port.get("containerPort") is not None
        }
        command_port = int(cfg.get("command_port", 44772) or 44772)
        if command_port not in ports:
            raise ValueError(f"Sandbox environment {environment_id!r} does not expose command port {command_port}")
        mounts = {str(volume.get("mountPath") or "") for volume in environment.get("volumes", [])}
        missing = set(cfg.get("required_mount_paths", []) or []) - mounts
        if missing:
            raise ValueError(f"Sandbox environment {environment_id!r} is missing volume mounts: {sorted(missing)}")

    async def _create_sandbox(self, *, image: str, environment_id: str, cfg: Dict[str, Any]) -> Any:
        try:
            from opensandbox import Sandbox
            from opensandbox.config import ConnectionConfig
        except ImportError as exc:
            raise RuntimeError("Sandbox mode requires opensandbox==0.1.9") from exc

        connection = ConnectionConfig(
            domain=str(cfg["domain"]),
            api_key=str(cfg["api_key"]),
            protocol=str(cfg.get("protocol") or "https"),
            use_server_proxy=bool(cfg.get("use_server_proxy", True)),
            request_timeout=timedelta(seconds=float(cfg.get("request_timeout_s", 60.0))),
        )
        kwargs: Dict[str, Any] = {
            "connection_config": connection,
            "timeout": timedelta(minutes=int(cfg.get("lifecycle_minutes", 120) or 120)),
            "ready_timeout": timedelta(seconds=float(cfg.get("create_timeout_s", 600.0) or 600.0)),
            "extensions": {**dict(cfg.get("extensions", {}) or {}), "environmentId": environment_id},
            "skip_health_check": bool(cfg.get("skip_health_check", False)),
        }
        if cfg.get("resource"):
            kwargs["resource"] = dict(cfg["resource"])
        return await asyncio.wait_for(
            Sandbox.create(image, **kwargs),
            timeout=float(cfg.get("create_timeout_s", 600.0) or 600.0),
        )

    async def _prepare_instance(self, sandbox: Any, cfg: Dict[str, Any]) -> None:
        commands = []
        for item in cfg.get("embedded_files", []) or []:
            source = Path(str(item["source"]))
            if not source.is_file():
                raise FileNotFoundError(f"Sandbox embedded file not found: {source}")
            target = str(item["target"])
            encoded = base64.b64encode(source.read_bytes()).decode("ascii")
            commands.append(
                f"mkdir -p {shlex.quote(str(Path(target).parent))} && "
                f"printf %s {shlex.quote(encoded)} | base64 -d > {shlex.quote(target)}"
            )
        if not commands:
            return
        output = await run_sandbox_command(
            sandbox,
            " && ".join(commands),
            timeout_s=float(cfg.get("command_timeout_s", 720.0) or 720.0),
        )
        if output.exit_code != 0:
            raise RuntimeError(f"Sandbox bootstrap failed: {output.stderr or output.stdout}")

    @staticmethod
    async def _terminate(sandbox: Any) -> None:
        try:
            await sandbox.kill()
        finally:
            await sandbox.close()
