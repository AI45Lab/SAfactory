from __future__ import annotations

import asyncio
import base64
import inspect
import logging
import shlex
import time
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, Mapping

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
    async def invoke() -> Any:
        result = await asyncio.to_thread(sandbox.commands.run, command)
        if inspect.isawaitable(result):
            return await result
        return result

    result = await asyncio.wait_for(invoke(), timeout=max(1.0, float(timeout_s)))
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
        self._connection: Any | None = None
        self._service: Any | None = None
        self._manager: Any | None = None

    async def start(self, plan: BindingPlan) -> None:
        if not plan.env_to_image:
            log.warning("empty Sandbox binding plan")
            return
        if not self._sandbox_cfg.get("api_key"):
            env_name = str(self._sandbox_cfg.get("api_key_env") or "OPEN_SANDBOX_API_KEY")
            raise ValueError(f"Sandbox API key is required; set {env_name} or sandbox.api_key")
        if not str(self._sandbox_cfg.get("project") or "").strip():
            raise ValueError("Sandbox project is required; set sandbox.project")
        await self._ensure_sdk_clients()
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
        run_command = str(cfg.get("run_command") or "").strip()
        if not run_command:
            raise ValueError(f"Sandbox run_command is required for agent {env_name!r}")
        lifecycle_s = float(cfg.get("lifecycle_minutes", 120) or 120) * 60.0
        command_timeout_s = float(cfg.get("command_timeout_s", 720.0) or 720.0)
        if lifecycle_s < command_timeout_s + 600.0:
            raise ValueError(
                f"Sandbox lifecycle for agent {env_name!r} must cover command timeout plus 600 seconds"
            )

        sandbox_id, sandbox = await self._create_sandbox(
            image=image,
            cfg=cfg,
        )
        try:
            endpoint = await asyncio.to_thread(
                sandbox.get_endpoint,
                int(cfg.get("command_port", 44772) or 44772),
            )
            await self._prepare_instance(sandbox, cfg)
        except Exception:
            try:
                await self._terminate(sandbox_id, sandbox)
            except Exception:
                log.warning("failed to clean up Sandbox after allocation error", exc_info=True)
            raise

        resource_id = sandbox_id or f"sandbox-{env_id}"
        if isinstance(endpoint, str):
            endpoint_url = endpoint
        elif isinstance(endpoint, Mapping):
            endpoint_url = str(endpoint.get("endpoint") or endpoint.get("url") or "")
        else:
            endpoint_url = str(
                getattr(endpoint, "endpoint", "")
                or getattr(endpoint, "url", "")
                or ""
            )
        if endpoint_url and not endpoint_url.startswith(("http://", "https://")):
            endpoint_url = f"{str(cfg.get('protocol') or 'https')}://{endpoint_url}"
        endpoint_headers = (
            dict(endpoint.get("headers", {}) or {})
            if isinstance(endpoint, Mapping)
            else dict(getattr(endpoint, "headers", {}) or {})
        )
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
                "command_timeout_s": command_timeout_s,
                "endpoint": endpoint_url,
                "endpoint_headers": endpoint_headers,
                "env": dict(cfg.get("env", {}) or {}),
                "gateway_base_url": str(cfg.get("gateway_base_url") or ""),
                "result_artifact_root": str(cfg.get("result_artifact_root") or ""),
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
            await asyncio.to_thread(sandbox.close)
        log.info("preserving Sandbox instance after run: %s", entry.resource_id)

    async def remove(self, entry: PoolEntry) -> None:
        sandbox = self._instances.pop(entry.resource_id, None) or entry.runtime_handle
        if sandbox is None:
            return
        try:
            await self._terminate(entry.resource_id, sandbox)
        except Exception:
            log.warning("failed to delete Sandbox instance %s", entry.resource_id, exc_info=True)

    async def close(self) -> None:
        instances = list(self._instances.items())
        self._instances.clear()
        results = await asyncio.gather(
            *(self._terminate(sandbox_id, sandbox) for sandbox_id, sandbox in instances),
            return_exceptions=True,
        )
        for (sandbox_id, _), result in zip(instances, results):
            if isinstance(result, Exception):
                log.warning("failed to delete Sandbox instance %s during shutdown: %s", sandbox_id, result)
        manager = self._manager
        connection = self._connection
        self._manager = None
        self._service = None
        self._connection = None
        try:
            if manager is not None:
                await asyncio.to_thread(manager.close)
        finally:
            if connection is not None:
                close_transport = getattr(connection, "close_transport_if_owned", None)
                if callable(close_transport):
                    await asyncio.to_thread(close_transport)

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
        merged["volumes"] = list(
            sandbox_cfg.get("volumes", self._sandbox_cfg.get("volumes", [])) or []
        )
        merged["metadata"] = {
            **dict(self._sandbox_cfg.get("metadata", {}) or {}),
            **dict(sandbox_cfg.get("metadata", {}) or {}),
        }
        return merged

    async def _create_sandbox(
        self,
        *,
        image: str,
        cfg: Dict[str, Any],
    ) -> tuple[str, Any]:
        await self._ensure_sdk_clients()
        service = self._service
        manager = self._manager
        if service is None or manager is None:
            raise RuntimeError("Sandbox SDK clients are not initialized")

        extensions = dict(cfg.get("extensions", {}) or {})
        if "environmentId" in extensions or "environment_id" in extensions:
            raise ValueError(
                "Sandbox mode creates instances directly via the SDK and does not support "
                "extensions.environmentId; configure sandbox.project, image, and volumes instead"
            )
        extensions["project"] = str(extensions.get("project") or cfg.get("project") or "").strip()
        if not extensions["project"]:
            raise ValueError("Sandbox project is required; set sandbox.project")
        entrypoint = cfg.get("idle_entrypoint") or ["tail", "-f", "/dev/null"]
        if isinstance(entrypoint, str):
            entrypoint = shlex.split(entrypoint)
        metadata = {
            "source": "safactory",
            "runtime": "sandbox",
            **{str(key): str(value) for key, value in dict(cfg.get("metadata", {}) or {}).items()},
        }
        network_policy = cfg.get("network_policy") or None
        try:
            from opensandbox.models.sandboxes import SandboxImageSpec
        except ImportError as exc:
            raise RuntimeError("Sandbox mode requires opensandbox==0.1.9") from exc
        if isinstance(network_policy, dict):
            try:
                from opensandbox.models.sandboxes import NetworkPolicy, NetworkRule

                network_policy = NetworkPolicy(
                    defaultAction=str(network_policy.get("defaultAction", "allow")),
                    egress=[NetworkRule(**dict(rule)) for rule in network_policy.get("egress", [])],
                )
            except (ImportError, TypeError, ValueError) as exc:
                raise ValueError("invalid sandbox.network_policy") from exc
        kwargs: Dict[str, Any] = {
            "spec": SandboxImageSpec(image),
            "entrypoint": list(entrypoint),
            "env": {str(key): str(value) for key, value in dict(cfg.get("env", {}) or {}).items()},
            "metadata": metadata,
            "timeout": timedelta(minutes=int(cfg.get("lifecycle_minutes", 120) or 120)),
            "resource": dict(cfg.get("resource", {}) or {}),
            "network_policy": network_policy,
            "extensions": extensions,
            # Passing None preserves the project's implicit AFS mount.
            "volumes": list(cfg.get("volumes", []) or []) or None,
        }
        try:
            created = await asyncio.wait_for(
                asyncio.to_thread(service.create_sandbox, **kwargs),
                timeout=float(cfg.get("create_timeout_s", 600.0) or 600.0),
            )
            sandbox_id = str(getattr(created, "id", "") or "").strip()
            if not sandbox_id:
                raise RuntimeError("Sandbox create response did not contain an id")
            await self._wait_running(
                sandbox_id,
                timeout_s=float(cfg.get("create_timeout_s", 600.0) or 600.0),
                poll_interval_s=float(cfg.get("poll_interval_s", 2.0) or 2.0),
            )
            sandbox = await self._connect_sandbox(sandbox_id, cfg)
            return sandbox_id, sandbox
        except Exception:
            sandbox_id = locals().get("sandbox_id")
            if sandbox_id:
                try:
                    await asyncio.to_thread(manager.kill_sandbox, sandbox_id)
                except Exception:
                    log.warning("failed to clean up Sandbox %s after creation error", sandbox_id, exc_info=True)
            raise

    async def _ensure_sdk_clients(self) -> tuple[Any, Any, Any]:
        if self._connection is not None and self._service is not None and self._manager is not None:
            return self._connection, self._service, self._manager
        try:
            from opensandbox.config.connection_sync import ConnectionConfigSync
            from opensandbox.sync.adapters.factory import AdapterFactorySync
            from opensandbox.sync.manager import SandboxManagerSync
        except ImportError as exc:
            raise RuntimeError("Sandbox mode requires opensandbox==0.1.9") from exc

        cfg = self._sandbox_cfg
        connection = ConnectionConfigSync(
            domain=str(cfg.get("domain") or ""),
            protocol=str(cfg.get("protocol") or "https"),
            headers={"OPEN-SANDBOX-API-KEY": str(cfg.get("api_key") or "")},
            request_timeout=timedelta(seconds=float(cfg.get("request_timeout_s", 60.0) or 60.0)),
            use_server_proxy=bool(cfg.get("use_server_proxy", True)),
        ).with_transport_if_missing()
        service = AdapterFactorySync(connection).create_sandbox_service()
        manager = SandboxManagerSync.create(connection)
        self._connection = connection
        self._service = service
        self._manager = manager
        return connection, service, manager

    async def _wait_running(
        self,
        sandbox_id: str,
        *,
        timeout_s: float,
        poll_interval_s: float,
    ) -> Dict[str, Any]:
        manager = self._manager
        if manager is None:
            raise RuntimeError("Sandbox manager is not initialized")
        deadline = time.monotonic() + max(1.0, float(timeout_s))
        last_info: Any = None
        while time.monotonic() < deadline:
            last_info = await asyncio.to_thread(manager.get_sandbox_info, sandbox_id)
            status = getattr(last_info, "status", None)
            state = str(getattr(status, "state", "") or "").rsplit(".", 1)[-1].lower()
            if state == "running":
                return {"id": sandbox_id, "state": state, "reason": getattr(status, "reason", "")}
            if state in {"failed", "stopped", "terminated"}:
                raise RuntimeError(
                    f"Sandbox {sandbox_id} entered terminal state: "
                    f"state={state} reason={getattr(status, 'reason', '')!r} "
                    f"message={getattr(status, 'message', '')!r}"
                )
            await asyncio.sleep(max(0.1, float(poll_interval_s)))
        status = getattr(last_info, "status", None)
        raise TimeoutError(
            f"Sandbox {sandbox_id} did not become running within {float(timeout_s):.1f}s; "
            f"state={getattr(status, 'state', None)!r} reason={getattr(status, 'reason', None)!r}"
        )

    async def _connect_sandbox(self, sandbox_id: str, cfg: Mapping[str, Any]) -> Any:
        try:
            from opensandbox.sync.sandbox import SandboxSync
        except ImportError as exc:
            raise RuntimeError("Sandbox mode requires opensandbox==0.1.9") from exc
        if self._connection is None:
            raise RuntimeError("Sandbox connection is not initialized")
        return await asyncio.to_thread(
            SandboxSync.connect,
            sandbox_id,
            connection_config=self._connection,
            connect_timeout=timedelta(seconds=float(cfg.get("request_timeout_s", 60.0) or 60.0)),
            skip_health_check=bool(cfg.get("skip_health_check", False)),
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

    async def _terminate(self, sandbox_id: str, sandbox: Any | None) -> None:
        if sandbox is not None:
            try:
                await asyncio.to_thread(sandbox.close)
            except Exception:
                log.debug("failed to close Sandbox client %s", sandbox_id, exc_info=True)
        if self._manager is not None and sandbox_id:
            try:
                await asyncio.to_thread(self._manager.kill_sandbox, sandbox_id)
            except Exception as exc:
                # Deletion is idempotent from Safactory's perspective; preserve
                # the first cleanup error for callers that need to report it.
                if "not found" not in str(exc).lower():
                    raise
