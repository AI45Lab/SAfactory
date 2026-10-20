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
from urllib.parse import urlsplit

import httpx

from manager.binding_plan import BindingPlan
from manager.simulation_config import _normalize_sandbox_config
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


@dataclass(slots=True)
class SandboxRuntimeHandle:
    """SDK services sharing a transport owned by the cluster backend."""

    id: str
    commands: Any
    files: Any
    _service: Any
    _connection: Any

    def get_endpoint(self, port: int) -> Any:
        return self._service.get_sandbox_endpoint(
            self.id, port, self._connection.use_server_proxy,
        )

    def close(self) -> None:
        # Releasing one episode must not close the transport used by other
        # instances. The backend closes it after all instances are released.
        pass


async def run_sandbox_command(sandbox: Any, command: str, *, timeout_s: float) -> SandboxCommandOutput:
    async def invoke() -> Any:
        result = await asyncio.to_thread(sandbox.commands.run, command)
        if inspect.isawaitable(result):
            return await result
        return result

    result = await asyncio.wait_for(invoke(), timeout=max(1.0, float(timeout_s)))
    return SandboxCommandOutput.from_sdk(result)


class SandboxClusterBackend(ClusterBackend):
    """DeepEnv allocator using the lifecycle API and OpenSandbox execd SDK."""

    runtime = "sandbox"

    def __init__(self, *, cluster_cfg: Dict[str, Any] | None = None) -> None:
        self._cluster_cfg = dict(cluster_cfg or {})
        self._sandbox_cfg = _normalize_sandbox_config(self._cluster_cfg.get("sandbox", {}) or {})
        self._env_types = dict(self._cluster_cfg.get("env_types", {}) or {})
        self.startup_concurrency = max(1, int(self._sandbox_cfg.get("startup_concurrency", 8) or 8))
        self._instances: Dict[str, Any] = {}
        self._connection: Any | None = None
        self._service: Any | None = None
        self._manager: Any | None = None
        self._clients: Dict[tuple[Any, ...], tuple[Any, Any, Any]] = {}
        self._instance_managers: Dict[str, Any] = {}

    async def start(self, plan: BindingPlan) -> None:
        if not plan.env_to_image:
            log.warning("empty Sandbox binding plan")
            return
        for env_name in plan.env_to_image:
            cfg = self._config_for_env(env_name)
            if not cfg.get("api_key"):
                key_env = str(cfg.get("api_key_env") or "OPEN_SANDBOX_API_KEY")
                raise ValueError(f"Sandbox API key is required; set {key_env} or sandbox.api_key")
            if not str(cfg.get("project") or cfg.get("extensions", {}).get("project") or "").strip():
                project_env = str(cfg.get("project_env") or "OPEN_SANDBOX_PROJECT")
                raise ValueError(f"Sandbox project is required; set {project_env} or sandbox.project")
            await self._ensure_sdk_clients(cfg)
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
            endpoint_url = f"{str(cfg.get('protocol') or 'http')}://{endpoint_url}"
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
        self._instance_managers.pop(entry.resource_id, None)
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
        clients = list(self._clients.values())
        self._clients.clear()
        self._instance_managers.clear()
        self._manager = None
        self._service = None
        self._connection = None
        for connection, _, manager in clients:
            try:
                await asyncio.to_thread(manager.close)
            except Exception:
                log.warning("failed to close Sandbox manager", exc_info=True)
            finally:
                close_transport = getattr(connection, "close_transport_if_owned", None)
                if callable(close_transport):
                    try:
                        await asyncio.to_thread(close_transport)
                    except Exception:
                        log.warning("failed to close Sandbox transport", exc_info=True)

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
        # Agent-specific environment variables override values already resolved
        # from the global configuration.
        for key in ("api_key", "project"):
            if f"{key}_env" in sandbox_cfg and key not in sandbox_cfg:
                merged[key] = ""
        return _normalize_sandbox_config(merged)

    async def _create_sandbox(
        self,
        *,
        image: str,
        cfg: Dict[str, Any],
    ) -> tuple[str, Any]:
        extensions = dict(cfg.get("extensions", {}) or {})
        if "environmentId" in extensions or "environment_id" in extensions:
            raise ValueError(
                "Sandbox mode creates instances directly via the API and does not support "
                "extensions.environmentId; configure sandbox.project, image, and volumes instead"
            )
        extensions = {str(key): str(value) for key, value in extensions.items()}
        extensions["project"] = str(extensions.get("project") or cfg.get("project") or "").strip()
        if not extensions["project"]:
            raise ValueError("Sandbox project is required; set OPEN_SANDBOX_PROJECT or sandbox.project")
        connection, _, manager = await self._ensure_sdk_clients(cfg)
        entrypoint = cfg.get("idle_entrypoint") or ["tail", "-f", "/dev/null"]
        if isinstance(entrypoint, str):
            entrypoint = shlex.split(entrypoint)
        metadata = {
            "source": "safactory",
            "runtime": "sandbox",
            **{str(key): str(value) for key, value in dict(cfg.get("metadata", {}) or {}).items()},
        }
        payload: Dict[str, Any] = {
            "image": {"uri": image},
            "entrypoint": list(entrypoint),
            "env": {str(key): str(value) for key, value in dict(cfg.get("env", {}) or {}).items()},
            "metadata": metadata,
            "timeout": int(cfg.get("lifecycle_minutes", 120) or 120) * 60,
            "resourceRequests": dict(cfg.get("resource", {}) or {}),
            "ports": [{
                "containerPort": int(cfg.get("command_port", 44772) or 44772),
                "name": "execd",
                "protocol": "tcp",
                "purpose": "system service",
            }],
            "extensions": extensions,
        }
        if cfg.get("network_policy"):
            payload["networkPolicy"] = dict(cfg["network_policy"])
        # Omit volumes to preserve the project's implicit AFS mount. Direct JSON
        # also supports DeepEnv volumes without an upstream SDK storage backend.
        if cfg.get("volumes"):
            payload["volumes"] = list(cfg["volumes"])
        try:
            created = await asyncio.wait_for(
                asyncio.to_thread(self._post_sandbox, cfg, payload),
                timeout=float(cfg.get("create_timeout_s", 600.0) or 600.0),
            )
            sandbox_id = str(created.get("id") or "").strip()
            if not sandbox_id:
                raise RuntimeError("Sandbox create response did not contain an id")
            self._instance_managers[sandbox_id] = manager
            await self._wait_running(
                sandbox_id,
                manager=manager,
                timeout_s=float(cfg.get("create_timeout_s", 600.0) or 600.0),
                poll_interval_s=float(cfg.get("poll_interval_s", 2.0) or 2.0),
            )
            sandbox = await self._connect_sandbox(sandbox_id, cfg, connection=connection)
            return sandbox_id, sandbox
        except Exception:
            sandbox_id = locals().get("sandbox_id")
            if sandbox_id:
                self._instance_managers.pop(sandbox_id, None)
                try:
                    await asyncio.to_thread(manager.kill_sandbox, sandbox_id)
                except Exception:
                    log.warning("failed to clean up Sandbox %s after creation error", sandbox_id, exc_info=True)
            raise

    @staticmethod
    def _post_sandbox(cfg: Mapping[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
        # opensandbox==0.1.9 cannot include ports in create_sandbox. The proxy
        # requires the execd port to be declared independently of the image.
        with httpx.Client(timeout=float(cfg.get("request_timeout_s", 60.0) or 60.0)) as client:
            response = client.post(
                f"{cfg['protocol']}://{cfg['domain']}/v1/sandboxes",
                headers={"OPEN-SANDBOX-API-KEY": str(cfg.get("api_key") or "")},
                json=payload,
            )
            response.raise_for_status()
            created = response.json()
        if not isinstance(created, dict):
            raise RuntimeError("Sandbox create response must be an object")
        return created

    async def _ensure_sdk_clients(self, cfg: Mapping[str, Any] | None = None) -> tuple[Any, Any, Any]:
        cfg = cfg if cfg is not None else self._sandbox_cfg
        key = tuple(cfg.get(name) for name in (
            "domain", "protocol", "api_key", "request_timeout_s", "use_server_proxy",
        ))
        if key in self._clients:
            return self._clients[key]
        try:
            from opensandbox.config.connection_sync import ConnectionConfigSync
            from opensandbox.sync.adapters.factory import AdapterFactorySync
            from opensandbox.sync.manager import SandboxManagerSync
        except ImportError as exc:
            raise RuntimeError("Sandbox mode requires opensandbox==0.1.9") from exc

        connection = ConnectionConfigSync(
            domain=str(cfg.get("domain") or ""),
            protocol=str(cfg.get("protocol") or "http"),
            api_key=str(cfg.get("api_key") or ""),
            headers={"OPEN-SANDBOX-API-KEY": str(cfg.get("api_key") or "")},
            request_timeout=timedelta(seconds=float(cfg.get("request_timeout_s", 60.0) or 60.0)),
            use_server_proxy=bool(cfg.get("use_server_proxy", True)),
        ).with_transport_if_missing()
        service = AdapterFactorySync(connection).create_sandbox_service()
        manager = SandboxManagerSync.create(connection)
        self._connection = connection
        self._service = service
        self._manager = manager
        clients = (connection, service, manager)
        self._clients[key] = clients
        return clients

    async def _wait_running(
        self,
        sandbox_id: str,
        *,
        timeout_s: float,
        poll_interval_s: float,
        manager: Any | None = None,
    ) -> Dict[str, Any]:
        manager = manager if manager is not None else self._manager
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

    async def _connect_sandbox(
        self, sandbox_id: str, cfg: Mapping[str, Any], *, connection: Any | None = None,
    ) -> Any:
        try:
            from opensandbox.sync.adapters.factory import AdapterFactorySync
        except ImportError as exc:
            raise RuntimeError("Sandbox mode requires opensandbox==0.1.9") from exc
        connection = connection if connection is not None else self._connection
        if connection is None:
            raise RuntimeError("Sandbox connection is not initialized")
        factory = AdapterFactorySync(connection)
        service = factory.create_sandbox_service()
        # SandboxSync.connect hardcodes 44772 and also resolves an unused
        # egress endpoint on 18080. Connect the required execd services directly
        # so DeepEnv only needs the configured command port declared.
        endpoint = await asyncio.to_thread(
            service.get_sandbox_endpoint,
            sandbox_id,
            int(cfg.get("command_port", 44772) or 44772),
            connection.use_server_proxy,
        )
        # DeepEnv can return an absolute proxy URL. SDK service adapters prepend
        # connection.protocol themselves, so supply a scheme-free endpoint.
        if "://" in endpoint.endpoint:
            parsed = urlsplit(endpoint.endpoint)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                raise ValueError("Sandbox execd endpoint must use http or https")
            endpoint = endpoint.model_copy(update={
                "endpoint": parsed.netloc + parsed.path,
            })
            connection = connection.model_copy(update={"protocol": parsed.scheme})
            factory = AdapterFactorySync(connection)
        sandbox = SandboxRuntimeHandle(
            id=sandbox_id,
            commands=factory.create_command_service(endpoint),
            files=factory.create_filesystem_service(endpoint),
            _service=service,
            _connection=connection,
        )
        if bool(cfg.get("skip_health_check", False)):
            return sandbox
        health = factory.create_health_service(endpoint)
        timeout_s = float(cfg.get("request_timeout_s", 60.0) or 60.0)
        deadline = time.monotonic() + timeout_s
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            try:
                if await asyncio.wait_for(
                    asyncio.to_thread(health.ping, sandbox_id),
                    timeout=max(0.001, deadline - time.monotonic()),
                ):
                    return sandbox
            except Exception as exc:
                last_error = exc
            await asyncio.sleep(min(0.2, max(0.0, deadline - time.monotonic())))
        raise TimeoutError(f"Sandbox {sandbox_id} execd was not ready within {timeout_s:.1f}s") from last_error

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
        manager = self._instance_managers.pop(sandbox_id, None) or self._manager
        if manager is not None and sandbox_id:
            try:
                await asyncio.to_thread(manager.kill_sandbox, sandbox_id)
            except Exception as exc:
                # Deletion is idempotent from Safactory's perspective; preserve
                # the first cleanup error for callers that need to report it.
                if "not found" not in str(exc).lower():
                    raise
