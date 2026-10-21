from __future__ import annotations

import asyncio
import base64
import logging
import shlex
import time
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from threading import Event
from typing import Any, Dict, Mapping
from urllib.parse import urlsplit

import httpx

from manager.binding_plan import BindingPlan
from manager.episode_common import safe_path_part
from manager.simulation_config import _normalize_sandbox_config
from manager.types import PoolEntry

from .base import ClusterBackend, RuntimeConfigurationError

log = logging.getLogger("clusters.sandbox_cluster")


def sandbox_results_setup_command(result_dir: str) -> str:
    """Create a shared results directory and verify it using this instance's UID."""
    path = Path(result_dir)
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("Sandbox results directory must be an absolute path without '..'")
    target = shlex.quote(str(path))
    ancestors = " ".join(shlex.quote(str(item)) for item in (*reversed(path.parents), path))
    probe = shlex.quote(str(path / ".safactory-write-XXXXXX"))
    return (
        "( "
        f"for directory in {ancestors}; do "
        '[ ! -L "$directory" ] || exit 1; done && '
        # Restrict umask to directory creation; the runner may contain credentials.
        f"(umask 000; mkdir -p -- {target}) && "
        # A different instance may own an already shared directory.
        f"(chmod -- a+rwX {target} 2>/dev/null || true) && "
        f"probe=$(mktemp {probe}) && "
        'printf safactory > "$probe" && rm -- "$probe"'
        " ) || { "
        "printf '%s\\n' SAFACTORY_RESULTS_NOT_WRITABLE >&2; "
        "id >&2; "
        "printf '%s\\n' /proc/self/uid_map /proc/self/gid_map >&2; "
        "cat /proc/self/uid_map /proc/self/gid_map >&2 2>/dev/null || true; "
        f"stat -c '%a %u:%g %n' -- {target} >&2 2>/dev/null || ls -ld -- {target} >&2; "
        f"findmnt -T {target} >&2 2>/dev/null || true; "
        "exit 73; }"
    )


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

        exit_code = getattr(result, "exit_code", getattr(result, "code", None))
        if exit_code is None:
            raise RuntimeError(f"Sandbox command ended without a completion status: {getattr(result, 'error', None)}")
        return cls(int(exit_code), stream_text("stdout"), stream_text("stderr"))


@dataclass(slots=True)
class SandboxRuntimeHandle:
    """One instance's async execd services, separate from lifecycle clients."""

    id: str
    commands: Any
    files: Any
    _service: Any
    _connection: Any
    cancel_timeout_s: float = 10.0
    _active_commands: set[asyncio.Task] = field(default_factory=set, repr=False)
    _closed: bool = False

    def get_endpoint(self, port: int) -> Any:
        return self._service.get_sandbox_endpoint(
            self.id, port, self._connection.use_server_proxy,
        )

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        tasks = self._active_commands - {asyncio.current_task()}
        for task in tasks:
            task.cancel()
        try:
            await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=self.cancel_timeout_s + 1.0,
            )
        finally:
            await self._connection.close_transport_if_owned()


async def run_sandbox_command(sandbox: Any, command: str, *, timeout_s: float) -> SandboxCommandOutput:
    from opensandbox.models.execd import ExecutionHandlers, RunCommandOpts

    execution_id = ""

    async def on_init(event: Any) -> None:
        nonlocal execution_id
        execution_id = str(event.id or "")

    active = getattr(sandbox, "_active_commands", None)
    if getattr(sandbox, "_closed", False) or active:
        raise RuntimeError("Sandbox instance is closed or already running a command")

    async def invoke() -> SandboxCommandOutput:
        try:
            result = await sandbox.commands.run(
                command,
                opts=RunCommandOpts(timeout=timedelta(seconds=float(timeout_s))),
                handlers=ExecutionHandlers(on_init=on_init),
            )
            return SandboxCommandOutput.from_sdk(result)
        except (Exception, asyncio.CancelledError):
            # Losing the SSE stream does not stop the remote process. Finish
            # interruption before another command can start, even if cancelled.
            if execution_id:
                interruption = asyncio.create_task(asyncio.wait_for(
                    sandbox.commands.interrupt(execution_id),
                    timeout=float(getattr(sandbox, "cancel_timeout_s", 10.0)),
                ))
                while not interruption.done():
                    try:
                        await asyncio.shield(interruption)
                    except asyncio.CancelledError:
                        continue
                    except Exception:
                        break
                try:
                    interruption.result()
                except (Exception, asyncio.CancelledError):
                    log.warning("failed to interrupt Sandbox command: sandbox=%s execution=%s", sandbox.id, execution_id, exc_info=True)
            raise

    task = asyncio.create_task(invoke())
    if active is not None:
        active.add(task)
    try:
        return await asyncio.wait_for(task, timeout=max(0.001, float(timeout_s)))
    finally:
        if active is not None:
            active.discard(task)


class SandboxClusterBackend(ClusterBackend):
    """DeepEnv allocator using the lifecycle API and OpenSandbox execd SDK."""

    runtime = "sandbox"

    def __init__(self, *, cluster_cfg: Dict[str, Any] | None = None) -> None:
        self._cluster_cfg = dict(cluster_cfg or {})
        self._sandbox_cfg = dict(self._cluster_cfg.get("sandbox", {}) or {})
        self._env_types = dict(self._cluster_cfg.get("env_types", {}) or {})
        self._resolved_configs: Dict[str, Dict[str, Any]] = {}
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
        command_timeout_s = float(cfg.get("command_timeout_s", 720.0) or 720.0)

        log.info("Creating Sandbox: agent=%s id=%s", env_name, env_id)
        sandbox_id, sandbox = await self._create_sandbox(
            image=image,
            cfg=cfg,
        )
        try:
            endpoint = await asyncio.to_thread(
                sandbox.get_endpoint,
                int(cfg.get("command_port", 44772) or 44772),
            )
            run_command, runner_path = await self._prepare_instance(sandbox, cfg)
        except (Exception, asyncio.CancelledError):
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
                "runner_path": runner_path,
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
            await sandbox.close()
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
        instances = {
            **dict.fromkeys(self._instance_managers),
            **self._instances,
        }
        self._instances.clear()
        results = await asyncio.gather(
            *(self._terminate(sandbox_id, sandbox) for sandbox_id, sandbox in instances.items()),
            return_exceptions=True,
        )
        for sandbox_id, result in zip(instances, results):
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
        env_name = str(env_name)
        if env_name in self._resolved_configs:
            return self._resolved_configs[env_name]
        env_cfg = dict(self._env_types.get(env_name, {}) or {})
        # Resolve credentials and validate cross-field constraints after overrides.
        # A backend's configuration is fixed for its lifetime, including release.
        cfg = _normalize_sandbox_config(
            self._sandbox_cfg, agent_overrides=env_cfg.get("sandbox", {}) or {},
        )
        self._resolved_configs[env_name] = cfg
        return cfg

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
        # Normalization keeps input mounts and adds a writable results volume.
        payload["volumes"] = list(cfg["volumes"])
        deadline = time.monotonic() + float(cfg.get("create_timeout_s", 600.0))
        sandbox_id = ""
        cancelled_create = Event()
        create_task = asyncio.create_task(asyncio.to_thread(
            self._post_sandbox, {**cfg, "_cancelled_create": cancelled_create}, payload,
        ))
        try:
            created = await asyncio.wait_for(
                asyncio.shield(create_task),
                timeout=max(0.001, deadline - time.monotonic()),
            )
            sandbox_id = str(created.get("id") or "").strip()
            if not sandbox_id:
                raise RuntimeError("Sandbox create response did not contain an id")
            self._instance_managers[sandbox_id] = manager
            log.info("Sandbox created: id=%s; waiting for running state", sandbox_id)
            await self._wait_running(
                sandbox_id,
                manager=manager,
                timeout_s=max(0.001, deadline - time.monotonic()),
                poll_interval_s=float(cfg.get("poll_interval_s", 2.0) or 2.0),
            )
            log.info("Sandbox running: id=%s; connecting execd", sandbox_id)
            sandbox = await asyncio.wait_for(
                self._connect_sandbox(sandbox_id, cfg, connection=connection),
                timeout=max(0.001, deadline - time.monotonic()),
            )
            return sandbox_id, sandbox
        except (Exception, asyncio.CancelledError):
            cancelled_create.set()
            if not sandbox_id and not create_task.cancelled():
                # A cancelled to_thread cannot stop an in-flight create POST.
                # Recover its response before returning so its instance is not orphaned.
                try:
                    created = await asyncio.wait_for(
                        asyncio.shield(create_task),
                        timeout=float(cfg.get("request_timeout_s", 60.0)) + 1.0,
                    )
                    sandbox_id = str(created.get("id") or "")
                except Exception:
                    log.warning("Sandbox create failed before an instance id could be recovered", exc_info=True)
                    create_task.add_done_callback(lambda task: task.exception() if not task.cancelled() else None)
            if sandbox_id:
                self._instance_managers.setdefault(sandbox_id, manager)
                try:
                    await self._terminate(sandbox_id, None)
                except Exception:
                    log.warning("failed to clean up Sandbox %s after creation error", sandbox_id, exc_info=True)
            raise

    @staticmethod
    def _post_sandbox(cfg: Mapping[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
        # opensandbox==0.1.9 cannot include ports in create_sandbox. The proxy
        # requires the execd port to be declared independently of the image.
        headers = {"OPEN-SANDBOX-API-KEY": str(cfg.get("api_key") or "")}
        cancelled = cfg.get("_cancelled_create")
        with httpx.Client(timeout=float(cfg.get("request_timeout_s", 60.0) or 60.0)) as client:
            if cancelled is not None and cancelled.is_set():
                raise RuntimeError("Sandbox creation cancelled before POST")
            response = client.post(
                f"{cfg['protocol']}://{cfg['domain']}/v1/sandboxes",
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            created = response.json()
            # The thread can outlive the caller's recovery deadline. It retains
            # responsibility for deleting a late instance using its own client.
            if cancelled is not None and cancelled.is_set() and isinstance(created, dict) and created.get("id"):
                deleted = client.delete(
                    f"{cfg['protocol']}://{cfg['domain']}/v1/sandboxes/{created['id']}",
                    headers=headers,
                )
                if deleted.status_code != 404:
                    deleted.raise_for_status()
        if not isinstance(created, dict):
            raise RuntimeError("Sandbox create response must be an object")
        return created

    async def _ensure_sdk_clients(self, cfg: Mapping[str, Any] | None = None) -> tuple[Any, Any, Any]:
        cfg = cfg if cfg is not None else self._config_for_env("")
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
        deadline = time.monotonic() + max(0.001, float(timeout_s))
        last_info: Any = None
        while time.monotonic() < deadline:
            last_info = await asyncio.wait_for(
                asyncio.to_thread(manager.get_sandbox_info, sandbox_id),
                timeout=max(0.001, deadline - time.monotonic()),
            )
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
            await asyncio.sleep(min(max(0.1, float(poll_interval_s)), max(0.0, deadline - time.monotonic())))
        status = getattr(last_info, "status", None)
        raise TimeoutError(
            f"Sandbox {sandbox_id} did not become running within {float(timeout_s):.1f}s; "
            f"state={getattr(status, 'state', None)!r} reason={getattr(status, 'reason', None)!r}"
        )

    async def _connect_sandbox(
        self, sandbox_id: str, cfg: Mapping[str, Any], *, connection: Any | None = None,
    ) -> Any:
        try:
            from opensandbox.adapters.factory import AdapterFactory
            from opensandbox.config import ConnectionConfig
            from opensandbox.sync.adapters.factory import AdapterFactorySync
        except ImportError as exc:
            raise RuntimeError("Sandbox mode requires opensandbox==0.1.9") from exc
        connection = connection if connection is not None else self._connection
        if connection is None:
            raise RuntimeError("Sandbox connection is not initialized")
        service = AdapterFactorySync(connection).create_sandbox_service()
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
        protocol = connection.protocol
        if "://" in endpoint.endpoint:
            parsed = urlsplit(endpoint.endpoint)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                raise ValueError("Sandbox execd endpoint must use http or https")
            endpoint = endpoint.model_copy(update={
                "endpoint": parsed.netloc + parsed.path,
            })
            protocol = parsed.scheme
        # Each instance owns its async transport. Cancelling or closing one
        # command must neither occupy lifecycle threads nor close another instance.
        runtime_connection = ConnectionConfig(
            domain=connection.domain,
            protocol=protocol,
            api_key=connection.api_key,
            headers=dict(connection.headers),
            request_timeout=connection.request_timeout,
            use_server_proxy=connection.use_server_proxy,
        ).with_transport_if_missing()
        factory = AdapterFactory(runtime_connection)
        sandbox = SandboxRuntimeHandle(
            id=sandbox_id,
            commands=factory.create_command_service(endpoint),
            files=factory.create_filesystem_service(endpoint),
            _service=service,
            _connection=runtime_connection,
            cancel_timeout_s=float(cfg.get("cancel_timeout_s", 10.0)),
        )
        if bool(cfg.get("skip_health_check", False)):
            return sandbox
        health = factory.create_health_service(endpoint)
        timeout_s = float(cfg.get("request_timeout_s", 60.0) or 60.0)
        deadline = time.monotonic() + timeout_s
        last_error: Exception | None = None
        try:
            while time.monotonic() < deadline:
                try:
                    if await asyncio.wait_for(
                        health.ping(sandbox_id),
                        timeout=max(0.001, deadline - time.monotonic()),
                    ):
                        return sandbox
                except Exception as exc:
                    last_error = exc
                await asyncio.sleep(min(0.2, max(0.0, deadline - time.monotonic())))
            raise TimeoutError(f"Sandbox {sandbox_id} execd was not ready within {timeout_s:.1f}s") from last_error
        except (Exception, asyncio.CancelledError):
            await sandbox.close()
            raise

    async def _prepare_instance(self, sandbox: Any, cfg: Dict[str, Any]) -> tuple[str, str]:
        result_root = str(cfg["result_artifact_root"])
        job_id = str(self._cluster_cfg.get("job_id") or "")
        result_dir = str(Path(result_root) / safe_path_part(job_id)) if job_id else result_root
        commands = [sandbox_results_setup_command(result_dir)]
        runner = dict(cfg.get("runner_entrypoint") or {})
        runner_target = str(runner.get("target") or "")
        runner_path = (
            f"/work/safactory/instances/{sandbox.id}/{Path(runner_target).name}"
            if runner_target else ""
        )
        run_command = str(cfg.get("run_command") or "")
        if runner_path:
            if runner_target not in run_command:
                raise ValueError("Sandbox run_command must reference runner_entrypoint.target")
            run_command = run_command.replace(runner_target, runner_path)
        for item in cfg.get("embedded_files", []) or []:
            source = Path(str(item["source"]))
            if not source.is_file():
                raise FileNotFoundError(f"Sandbox embedded file not found: {source}")
            target = runner_path if str(item["target"]) == runner_target else str(item["target"])
            encoded = base64.b64encode(source.read_bytes()).decode("ascii")
            commands.append(
                f"mkdir -p {shlex.quote(str(Path(target).parent))} && "
                f"printf %s {shlex.quote(encoded)} | base64 -d > {shlex.quote(target)}"
            )
        output = await run_sandbox_command(
            sandbox,
            " && ".join(commands),
            timeout_s=float(cfg.get("bootstrap_timeout_s", 120.0)),
        )
        if output.exit_code == 73:
            raise RuntimeConfigurationError(
                f"Sandbox {sandbox.id} results directory is not writable: {result_dir}; "
                f"readOnly=false does not grant AFS filesystem permissions: {output.stderr or output.stdout}"
            )
        if output.exit_code != 0:
            raise RuntimeError(
                f"Sandbox {sandbox.id} bootstrap failed; verify readOnly=false and AFS permissions "
                f"for {result_root}: {output.stderr or output.stdout}"
            )
        log.info("Sandbox ready: id=%s runner=%s results=%s", sandbox.id, runner_path, result_dir)
        return run_command, runner_path

    async def _terminate(self, sandbox_id: str, sandbox: Any | None) -> None:
        if sandbox is not None:
            try:
                await sandbox.close()
            except Exception:
                log.debug("failed to close Sandbox client %s", sandbox_id, exc_info=True)
        manager = self._instance_managers.get(sandbox_id) or self._manager
        if manager is not None and sandbox_id:
            try:
                await asyncio.to_thread(manager.kill_sandbox, sandbox_id)
            except Exception as exc:
                # Deletion is idempotent from Safactory's perspective; preserve
                # the first cleanup error for callers that need to report it.
                if "not found" not in str(exc).lower():
                    raise
            self._instance_managers.pop(sandbox_id, None)
