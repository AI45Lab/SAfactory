from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import shlex
import signal
import tempfile
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, Dict
from urllib.parse import urlsplit

import httpx

from core.result_artifact import write_json_atomic as _atomic_json
from manager.types import PoolEntry

from .base import ClusterBackend

if TYPE_CHECKING:
    from manager.binding_plan import BindingPlan

log = logging.getLogger("clusters.sandbox_cluster")


class ProjectAfsVolume(dict[str, object]):
    """DeepEnv's backend-less AFS volume, compatible with SDK 0.1.9's converter."""

    def __init__(self, *, sub_path: str | None, mount_path: str, read_only: bool) -> None:
        super().__init__(name="project-default-afs", mountPath=mount_path, readOnly=read_only)
        if sub_path is not None:
            self["subPath"] = sub_path
        self.name = "project-default-afs"
        self.host = self.pvc = self.ossfs = None
        self.sub_path, self.mount_path, self.read_only = sub_path, mount_path, read_only


def _safe_identity(value: Any) -> str:
    value = str(value or "")
    sanitized = "".join(char if char.isalnum() or char in "-_." else "_" for char in value.strip()).strip("._")
    if not value or sanitized != value:
        raise ValueError("Sandbox job/session identity must be a nonempty path component")
    return value


def _storage_volumes(storage: dict) -> list[ProjectAfsVolume] | None:
    if not storage:
        return None
    volumes = []
    paths: set[str] = set()
    for kind in ("inputs", "outputs"):
        for item in storage.get(kind, []) or []:
            subpath, mount = str(item["afs_subpath"]), str(item["mount_path"])
            if (not subpath or subpath.startswith("/") or "\\" in subpath
                    or any(part in {"", ".", ".."} for part in subpath.split("/"))):
                raise ValueError(f"Invalid AFS subpath: {subpath!r}")
            if not mount.startswith("/") or ".." in PurePosixPath(mount).parts or "\\" in mount:
                raise ValueError(f"Invalid Sandbox mount path: {mount!r}")
            if mount in paths:
                raise ValueError(f"Duplicate Sandbox mount path: {mount}")
            paths.add(mount)
            expected = kind == "inputs"
            if item.get("read_only", expected) is not expected:
                raise ValueError(f"Sandbox {kind} read_only must be {expected}")
            volumes.append(ProjectAfsVolume(sub_path=subpath, mount_path=mount, read_only=expected))
    if "/app/results" not in {v.mount_path for v in volumes if not v.read_only}:
        raise ValueError("Sandbox outputs must include writable /app/results")
    return volumes


async def _transfer_process(arguments: list[str], *, env: dict[str, str], stdin: bytes | None = None) -> bytes:
    process = await asyncio.create_subprocess_exec(
        *arguments, env=env, stdin=asyncio.subprocess.PIPE if stdin is not None else asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
    )
    try:
        stdout, stderr = await process.communicate(stdin)
    except BaseException:
        if process.returncode is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                await asyncio.wait_for(process.wait(), 10)
            except asyncio.TimeoutError:
                os.killpg(process.pid, signal.SIGKILL)
                await process.wait()
        raise
    if process.returncode:
        detail = stderr.decode("utf-8", errors="replace")[-2000:]
        for name in ("SK", "SECRET_KEY", "RCLONE_CONFIG_AFS_PASS"):
            secret = env.get(name, "")
            if secret:
                detail = detail.replace(secret, "<redacted>")
        raise RuntimeError(f"AFS transfer exited with {process.returncode}: {detail}")
    return stdout


async def prepare_storage(
    *, inputs: list[dict], outputs: list[dict], project: str,
    webdav_url: str, api_key: str,
) -> list[ProjectAfsVolume]:
    """Publish immutable inputs and prepare output directories; credentials stay on H."""
    volumes = _storage_volumes({"inputs": inputs, "outputs": outputs}) or []
    if ":" not in api_key or not project:
        raise ValueError("AFS staging requires project and OPEN_SANDBOX_API_KEY")
    access, secret = api_key.split(":", 1)
    child_env = os.environ.copy()
    child_env.update(AK=access, SK=secret)
    script = Path(__file__).resolve().parents[1] / "sandbox/migrate_gpfs_data_to_afs.sh"
    for item in inputs:
        await _transfer_process([
            "bash", str(script), "--source", str(item["host_path"]),
            "--dest-subpath", str(item["afs_subpath"]), "--project", project,
            "--webdav-url", webdav_url, "--publish-once",
        ], env=child_env)
    obscured = (await _transfer_process(["rclone", "obscure", "-"], env=child_env, stdin=secret.encode())).decode().strip()
    child_env.update(
        RCLONE_CONFIG_AFS_TYPE="webdav", RCLONE_CONFIG_AFS_URL=webdav_url,
        RCLONE_CONFIG_AFS_VENDOR="other", RCLONE_CONFIG_AFS_USER=f"ak_{access}@{project}",
        RCLONE_CONFIG_AFS_PASS=obscured,
    )
    host = urlsplit(webdav_url).hostname or ""
    child_env["NO_PROXY"] = ",".join(filter(None, [host, "127.0.0.1", "localhost", child_env.get("NO_PROXY", "")]))
    child_env["no_proxy"] = child_env["NO_PROXY"]
    for item in outputs:
        await _transfer_process(["rclone", "mkdir", f"afs:{item['afs_subpath']}"], env=child_env)
    return volumes


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

        exit_code = getattr(result, "exit_code", None)
        error = getattr(result, "error", None)
        if isinstance(exit_code, bool) or not isinstance(exit_code, int) or (error is not None and exit_code == 0):
            raise RuntimeError(f"Sandbox command has no trustworthy exit status: {error!s}")
        stderr = stream_text("stderr")
        if error is not None:
            stderr += f"\n{error}"
        return cls(exit_code, stream_text("stdout"), stderr)


async def _interrupt_and_confirm(sandbox: Any, execution_id: str) -> None:
    await sandbox.commands.interrupt(execution_id)
    while True:
        status = await sandbox.commands.get_command_status(execution_id)
        if status.running is False:
            sandbox.execution_stopped = True
            return
        await asyncio.sleep(0.2)


async def run_sandbox_command(
    sandbox: Any, command: str, *, timeout_s: float,
    on_init: Callable[[str], Awaitable[None]] | None = None,
) -> SandboxCommandOutput:
    from opensandbox.models.execd import ExecutionHandlers, RunCommandOpts

    command_id: str | None = None
    sandbox.execution_stopped = False

    async def handle_init(event: Any) -> None:
        nonlocal command_id
        command_id = str(getattr(event, "id", "") or "") or None
        sandbox.command_id = command_id
        if command_id and on_init:
            await on_init(command_id)

    try:
        result = await asyncio.wait_for(
            sandbox.commands.run(
                command, opts=RunCommandOpts(timeout=timedelta(seconds=max(0.001, timeout_s))),
                handlers=ExecutionHandlers(on_init=handle_init),
            ),
            timeout=max(0.001, timeout_s) + 5,
        )
        command_id = command_id or getattr(result, "id", None)
        if getattr(result, "exit_code", None) is None and command_id:
            status = await sandbox.commands.get_command_status(command_id)
            if status.running is not False or status.exit_code is None:
                raise RuntimeError("Sandbox command returned before a confirmed terminal state")
            result.exit_code = status.exit_code
        output = SandboxCommandOutput.from_sdk(result)
        sandbox.execution_stopped = True
        return output
    except BaseException as original:
        if command_id:
            cleanup = asyncio.create_task(_interrupt_and_confirm(sandbox, command_id))
            try:
                await asyncio.shield(asyncio.wait_for(cleanup, timeout=15))
            except BaseException:
                cleanup.cancel()
                await asyncio.gather(cleanup, return_exceptions=True)
        if not sandbox.execution_stopped:
            sandbox.retain = True
            error = RuntimeError("Sandbox writer state is unknown; retained for recovery, automatic rerun is forbidden")
            error.execution_stopped = False
            raise error from original
        raise


def _load_opensandbox() -> tuple[Any, Any]:
    """Load the optional SDK only when Sandbox mode is used."""
    try:
        from opensandbox.adapters.factory import AdapterFactory
        from opensandbox.config import ConnectionConfig
    except ImportError as exc:
        raise RuntimeError("Sandbox mode requires opensandbox==0.1.9") from exc
    return AdapterFactory, ConnectionConfig


def _normalize_command_endpoint(connection: Any, endpoint: Any) -> tuple[Any, Any]:
    """Return SDK command-adapter inputs without changing the public endpoint."""
    raw_endpoint = str(endpoint.endpoint).strip().rstrip("/")
    if not raw_endpoint:
        raise ValueError("Sandbox command endpoint is empty")

    command_connection = connection
    if "://" in raw_endpoint:
        parsed = urlsplit(raw_endpoint)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError(f"Unsupported Sandbox command endpoint: {endpoint.endpoint!r}")
        if parsed.query or parsed.fragment:
            raise ValueError("Sandbox command endpoint must not contain a query or fragment")
        raw_endpoint = f"{parsed.netloc}{parsed.path}".rstrip("/")
        command_connection = connection.model_copy(update={"protocol": parsed.scheme})

    return command_connection, endpoint.model_copy(update={"endpoint": raw_endpoint})


class _SandboxRuntimeHandle:
    """Minimal runtime handle required by the Sandbox allocator and runner."""

    def __init__(
        self,
        *,
        sandbox_id: str,
        sandbox_service: Any,
        commands: Any,
        files: Any,
        command_endpoint: Any,
        command_port: int,
        connection: Any,
    ) -> None:
        self.id = sandbox_id
        self.commands = commands
        self.files = files
        self.retain = False
        self.execution_stopped = True
        self.command_id: str | None = None
        self._renewal_task: asyncio.Task | None = None
        self._sandbox_service = sandbox_service
        self._command_endpoint = command_endpoint
        self._command_port = command_port
        self._connection = connection
        self._closed = False

    async def renew(self, ttl_seconds: float) -> None:
        await self._sandbox_service.renew_sandbox_expiration(
            self.id, datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds),
        )

    def start_renewal(self, ttl_seconds: float) -> None:
        async def heartbeat() -> None:
            try:
                while True:
                    await asyncio.sleep(max(0.1, min(300, ttl_seconds / 3)))
                    await self.renew(ttl_seconds)
            except asyncio.CancelledError:
                raise
            except Exception:
                self.retain = True
                log.exception("Sandbox TTL renewal failed: %s", self.id)
        self._renewal_task = asyncio.create_task(heartbeat())

    async def get_endpoint(self, port: int) -> Any:
        if port == self._command_port:
            return self._command_endpoint
        return await self._sandbox_service.get_sandbox_endpoint(
            self.id,
            port,
            self._connection.use_server_proxy,
        )

    async def kill(self) -> None:
        await self._sandbox_service.kill_sandbox(self.id)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._renewal_task is not None:
            self._renewal_task.cancel()
            await asyncio.gather(self._renewal_task, return_exceptions=True)
        await self._connection.close_transport_if_owned()


class SandboxClusterBackend(ClusterBackend):
    """OpenSandbox/DeepEnv runtime allocator."""

    runtime = "sandbox"

    def __init__(self, *, cluster_cfg: Dict[str, Any] | None = None) -> None:
        self._cluster_cfg = dict(cluster_cfg or {})
        self._sandbox_cfg = dict(self._cluster_cfg.get("sandbox", {}) or {})
        self._env_types = dict(self._cluster_cfg.get("env_types", {}) or {})
        self.startup_concurrency = max(1, int(self._sandbox_cfg.get("startup_concurrency", 8) or 8))
        self._instances: Dict[str, Any] = {}
        self._volumes: dict[str, list[ProjectAfsVolume] | None] = {}

    async def start(self, plan: BindingPlan) -> None:
        if not plan.env_to_image:
            log.warning("empty Sandbox binding plan")
            return
        if not self._sandbox_cfg.get("api_key"):
            env_name = str(self._sandbox_cfg.get("api_key_env") or "OPEN_SANDBOX_API_KEY")
            raise ValueError(f"Sandbox API key is required; set {env_name} or sandbox.api_key")
        published: set[tuple[str, str]] = set()
        for env_name, image in sorted(plan.env_to_image.items()):
            cfg = self._config_for_env(env_name)
            await self._validate_environment(env_name, image, cfg)
            storage = dict(cfg.get("storage") or {})
            if storage:
                inputs = []
                for item in storage.get("inputs", []):
                    key = (str(item["host_path"]), str(item["afs_subpath"]))
                    if key not in published:
                        inputs.append(item)
                        published.add(key)
                await prepare_storage(
                    inputs=inputs, outputs=list(storage.get("outputs", [])),
                    project=str(cfg.get("project") or ""),
                    webdav_url=str((cfg.get("afs") or {}).get("webdav_url") or "http://10.140.95.76:28081"),
                    api_key=str(cfg["api_key"]),
                )
            self._volumes[env_name] = _storage_volumes(storage)
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

        job_id = _safe_identity(cfg.get("job_id"))
        session_id = _safe_identity(env_id)
        recovery = dict(cfg.get("recovery_by_session", {}).get(session_id) or {})
        runtime_path = self._runtime_path(cfg, session_id)
        if recovery:
            reference = dict(recovery["runtime_ref"])
            self._validate_reference(reference, cfg, session_id, image=image)
            if recovery["kind"] == "local":
                return self._pool_entry(
                    row_id=row_id, env_name=env_name, env_id=env_id, image=image,
                    env_params=env_params, group_id=group_id, cfg=cfg, sandbox=None,
                    reference=reference, runtime_path=runtime_path, recovery=recovery,
                )
            sandbox = await self._recover_handle(reference, cfg, env_name)
            sandbox.retain = True
        else:
            reference = {
                "job_id": job_id, "session_id": session_id, "attempt_id": uuid.uuid4().hex,
                "sandbox_id": None, "image": image, "environment_id": environment_id,
                "gateway_instance_id": cfg.get("gateway_instance_id"),
                "config_digest": self._config_digest(cfg, image),
                "remote_result_path": f"/app/results/{job_id}/{session_id}/safactory_result.json",
            }
            _atomic_json(runtime_path, reference)
            sandbox = await self._create_sandbox(
                image=image, environment_id=environment_id, cfg=cfg,
                volumes=self._volumes.get(env_name, _storage_volumes(dict(cfg.get("storage") or {}))),
                metadata={key: str(reference[key]) for key in ("job_id", "session_id", "attempt_id")}
                | {"env_name": env_name},
            )
            reference["sandbox_id"] = sandbox.id
            try:
                _atomic_json(runtime_path, reference)
            except BaseException:
                sandbox.retain = True
                await sandbox.close()
                raise
        try:
            if not recovery:
                await self._prepare_instance(sandbox, cfg)
        except BaseException:
            try:
                await self._terminate(sandbox)
            except Exception:
                log.warning("failed to clean up Sandbox after allocation error", exc_info=True)
            raise
        return self._pool_entry(
            row_id=row_id, env_name=env_name, env_id=env_id, image=image,
            env_params=env_params, group_id=group_id, cfg=cfg, sandbox=sandbox,
            reference=reference, runtime_path=runtime_path, recovery=recovery,
        )

    def _pool_entry(
        self, *, row_id: Any, env_name: str, env_id: str, image: str,
        env_params: dict, group_id: str, cfg: dict, sandbox: Any,
        reference: dict, runtime_path: Path, recovery: dict,
    ) -> PoolEntry:
        sandbox_id = str(getattr(sandbox, "id", None) or getattr(sandbox, "sandbox_id", None) or "")
        resource_id = sandbox_id or f"sandbox-{env_id}"
        endpoint = getattr(sandbox, "_command_endpoint", None)
        endpoint_url = str(getattr(endpoint, "endpoint", "") or "")
        if endpoint_url and not endpoint_url.startswith(("http://", "https://")):
            endpoint_url = f"{str(cfg.get('protocol') or 'http')}://{endpoint_url}"
        if sandbox is not None:
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
                "gateway_instance_id": reference.get("gateway_instance_id"),
                "attempt_id": reference["attempt_id"],
                "results": dict(cfg.get("results") or {}),
                "runtime_ref_path": str(runtime_path),
                "runtime_ref": reference,
                "recovery": recovery,
            },
            runtime_handle=sandbox,
            resource_id=resource_id,
            resource_name=resource_id,
            workdir=str(cfg.get("workdir") or ""),
            run_command=str(cfg.get("run_command") or ""),
            result_mode=str(cfg.get("result_mode") or "json").strip().lower(),
            reuse_container=False,
        )

    async def release(self, entry: PoolEntry, *, succeeded: bool) -> None:
        del succeeded
        sandbox = self._instances.get(entry.resource_id) or entry.runtime_handle
        if sandbox is None:
            return
        if not sandbox.retain and bool(self._config_for_env(entry.env_name).get("cleanup_on_finish", True)):
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
            *(sandbox.close() if sandbox.retain or not getattr(sandbox, "cleanup_on_finish", True) else self._terminate(sandbox) for _, sandbox in instances),
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

    @staticmethod
    def _runtime_path(cfg: dict, session_id: str) -> Path:
        root = Path(str((cfg.get("results") or {}).get("local_root") or "results")).expanduser().resolve()
        return root / _safe_identity(cfg.get("job_id")) / _safe_identity(session_id) / "runtime.json"

    @staticmethod
    def _config_digest(cfg: dict, image: str) -> str:
        value = {key: cfg.get(key) for key in (
            "environment_id", "workdir", "run_command", "env", "storage", "resource", "extensions",
        )}
        value["image"] = image
        value["files"] = [
            {"target": item["target"], "permissions": item.get("permissions", "0644"),
             "sha256": hashlib.sha256(Path(item["source"]).read_bytes()).hexdigest()}
            for item in cfg.get("embedded_files", [])
        ]
        return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

    def _validate_reference(self, reference: dict, cfg: dict, session_id: str, *, image: str) -> None:
        expected = {
            "job_id": _safe_identity(cfg.get("job_id")), "session_id": session_id,
            "image": image, "environment_id": str(cfg.get("environment_id") or ""),
            "config_digest": self._config_digest(cfg, image),
            "remote_result_path": f"/app/results/{_safe_identity(cfg.get('job_id'))}/{session_id}/safactory_result.json",
        }
        if any(reference.get(key) != value for key, value in expected.items()):
            raise RuntimeError(f"Sandbox runtime reference does not match current session/config: {session_id}")
        _safe_identity(reference.get("attempt_id"))

    @staticmethod
    def _connection(cfg: dict) -> tuple[Any, Any]:
        AdapterFactory, ConnectionConfig = _load_opensandbox()
        connection = ConnectionConfig(
            domain=str(cfg["domain"]), api_key=str(cfg["api_key"]),
            protocol=str(cfg.get("protocol") or "http"),
            headers={"OPEN-SANDBOX-API-KEY": str(cfg["api_key"])},
            use_server_proxy=bool(cfg.get("use_server_proxy", True)),
            request_timeout=timedelta(seconds=float(cfg.get("request_timeout_s", 60.0))),
        ).with_transport_if_missing()
        return connection, AdapterFactory(connection).create_sandbox_service()

    @staticmethod
    async def _find_instance(service: Any, metadata: dict[str, str]) -> str | None:
        from opensandbox.models.sandboxes import SandboxFilter

        found: set[str] = set()
        page = 1
        while True:
            result = await service.list_sandboxes(SandboxFilter(metadata=metadata, page=page, page_size=100))
            items = getattr(result, "sandbox_infos", None)
            if items is None:
                items = getattr(result, "items", None)
            if items is None:
                raise RuntimeError("Sandbox list response did not expose instances")
            found.update(str(item.id) for item in items if (
                metadata.get("purpose") or (getattr(item, "metadata", None) or {}).get("purpose") != "result-recovery"
            ) and all(
                (getattr(item, "metadata", None) or {}).get(key) == value for key, value in metadata.items()
            ))
            if len(items) < 100:
                break
            page += 1
        if len(found) > 1:
            raise RuntimeError("Multiple Sandboxes match one attempt; refusing automatic recovery")
        return next(iter(found), None)

    async def _recover_handle(self, reference: dict, cfg: dict, env_name: str) -> Any:
        connection, service = self._connection(cfg)
        sandbox_id = reference.get("sandbox_id")
        try:
            if not sandbox_id:
                sandbox_id = await self._find_instance(service, {key: str(reference[key]) for key in (
                    "job_id", "session_id", "attempt_id",
                )})
                if not sandbox_id:
                    raise RuntimeError("Created Sandbox identity is unknown; recovery cannot safely retry creation")
            info = await service.get_sandbox_info(str(sandbox_id))
            state = str(info.status.state).rsplit(".", 1)[-1].lower()
            if state not in {"terminated", "stopped", "failed"}:
                handle = await self._connect_instance(
                    sandbox_id=str(sandbox_id), cfg=cfg, connection=connection, sandbox_service=service,
                )
                handle.execution_stopped = reference.get("command_finished") is True
                return handle
        except Exception as exc:
            if not self._is_missing(exc):
                await connection.close_transport_if_owned()
                raise
        await connection.close_transport_if_owned()
        # The old instance is definitively gone. A new instance only mounts AFS;
        # it never stages runner files, changes the attempt, or invokes business code.
        handle = await self._create_sandbox(
            image=str(reference["image"]), environment_id=str(reference["environment_id"]), cfg=cfg,
            volumes=self._volumes.get(env_name, _storage_volumes(dict(cfg.get("storage") or {}))),
            metadata={"job_id": str(reference["job_id"]), "session_id": str(reference["session_id"]),
                      "attempt_id": str(reference["attempt_id"]), "purpose": "result-recovery"},
        )
        handle.retain = True
        handle.execution_stopped = True
        handle.recovery_reader = True
        return handle

    @staticmethod
    def _is_missing(exc: BaseException) -> bool:
        response = getattr(exc, "response", None)
        return getattr(exc, "status_code", getattr(response, "status_code", None)) == 404

    async def prepare_resume(self, environment_rows: list[dict]) -> tuple[list[dict], dict[str, dict]]:
        """Resolve old attempts before callers remove any unfinished trajectory."""
        from core.result_artifact import read_result_file

        rerun_rows: list[dict] = []
        recoveries: dict[str, dict] = {}
        for row in environment_rows:
            env_name = str(row.get("env_name") or row.get("agent_name") or "")
            session_id = _safe_identity(row.get("env_id") or row.get("session_id"))
            cfg = self._config_for_env(env_name)
            image = str(row.get("env_image") or row.get("image") or "")
            path = self._runtime_path(cfg, session_id)
            if not path.is_file():
                connection, service = self._connection(cfg)
                try:
                    old_id = await self._find_instance(service, {
                        "job_id": _safe_identity(cfg.get("job_id")), "session_id": session_id,
                    })
                finally:
                    await connection.close_transport_if_owned()
                if old_id or (path.parent / "safactory_result.json").exists():
                    raise RuntimeError(f"Missing runtime.json for existing Sandbox session {session_id}")
                rerun_rows.append(row)
                continue
            reference = json.loads(path.read_text(encoding="utf-8"))
            self._validate_reference(reference, cfg, session_id, image=image or str(reference.get("image") or ""))
            identity = {key: str(reference[key]) for key in ("job_id", "session_id", "attempt_id")}
            local = path.parent / "safactory_result.json"
            local_valid = False
            if local.is_file():
                try:
                    read_result_file(local, **identity)
                    local_valid = True
                except (ValueError, OSError, TypeError):
                    pass
            if local_valid and reference.get("command_finished") is True:
                recoveries[session_id] = {"kind": "local", "runtime_ref": reference}
                continue
            handle = await self._recover_handle(reference, cfg, env_name)
            handle.retain = True
            try:
                if getattr(handle, "recovery_reader", False) and reference.get("command_finished") is not True:
                    reference.update(command_finished=True, execution_error="Original Sandbox is gone; runner exit status could not be confirmed")
                    _atomic_json(path, reference)
                if not getattr(handle, "execution_stopped", False) or (
                    handle.id == reference.get("sandbox_id") and reference.get("command_finished") is not True
                ):
                    command_id = reference.get("command_id")
                    if not command_id:
                        raise RuntimeError(f"Missing command ID for unfinished Sandbox session {session_id}")
                    status = await handle.commands.get_command_status(str(command_id))
                    interrupted = status.running is not False
                    if interrupted:
                        await asyncio.wait_for(_interrupt_and_confirm(handle, str(command_id)), 20)
                        status = await handle.commands.get_command_status(str(command_id))
                    handle.execution_stopped = True
                    reference.update(command_finished=True, exit_code=status.exit_code)
                    if interrupted or status.exit_code is None or getattr(status, "error", None):
                        reference["execution_error"] = reference.get("execution_error") or (
                            "Previous runner interrupted during recovery" if interrupted else
                            str(getattr(status, "error", None) or "Previous runner exit status is unknown")
                        )
                    _atomic_json(path, reference)
                if local_valid:
                    recoveries[session_id] = {"kind": "local", "runtime_ref": reference}
                    continue
                try:
                    body = await handle.files.read_bytes(str(reference["remote_result_path"]))
                except Exception as exc:
                    if not self._is_missing(exc):
                        raise
                    # No final artifact, and the original producer is confirmed
                    # stopped (or absent). Only now is rerunning permissible.
                    handle.retain = False
                    rerun_rows.append(row)
                    continue
                fd, candidate = tempfile.mkstemp(prefix=".resume-result.", dir=path.parent)
                try:
                    with os.fdopen(fd, "wb") as stream:
                        stream.write(body)
                    read_result_file(candidate, **identity)
                finally:
                    os.unlink(candidate)
                recoveries[session_id] = {"kind": "remote", "runtime_ref": reference}
            finally:
                if handle.retain and not getattr(handle, "recovery_reader", False):
                    await handle.close()
                else:
                    await self._terminate(handle)
        return rerun_rows, recoveries

    async def _validate_environment(self, env_name: str, image: str, cfg: Dict[str, Any]) -> None:
        environment_id = str(cfg.get("environment_id") or "").strip()
        if not environment_id:
            raise ValueError(f"Sandbox environment_id is required for agent {env_name!r}")
        params = {"project": cfg["project"]} if cfg.get("project") else None
        api_domain = str(cfg["domain"]).rstrip("/")
        if not api_domain.startswith(("http://", "https://")):
            api_domain = f"{str(cfg.get('protocol') or 'http')}://{api_domain}"
        async with httpx.AsyncClient(timeout=float(cfg.get("request_timeout_s", 60.0))) as client:
            response = await client.get(
                f"{api_domain}/v1/sandbox-environments/{environment_id}",
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
        overrides = _storage_volumes(dict(cfg.get("storage") or {}))
        mounts = {str(volume.get("mountPath") or "") for volume in (
            overrides if overrides is not None else environment.get("volumes", [])
        )}
        missing = set(cfg.get("required_mount_paths", []) or []) - mounts
        if missing:
            raise ValueError(f"Sandbox environment {environment_id!r} is missing volume mounts: {sorted(missing)}")

    async def _create_sandbox(
        self, *, image: str, environment_id: str, cfg: Dict[str, Any],
        volumes: list[ProjectAfsVolume] | None = None,
        metadata: dict[str, str] | None = None,
    ) -> Any:
        del image  # Environment owns image and entrypoint.
        connection, sandbox_service = self._connection(cfg)
        sandbox_id: str | None = None
        extensions = {
            str(key): str(value) for key, value in dict(cfg.get("extensions", {}) or {}).items()
            if key not in {"project", "environmentId", "environmentName"}
        }
        extensions["environmentId"] = environment_id
        timeout_s = float(cfg.get("create_timeout_s", 600.0) or 600.0)
        try:
            try:
                created = await asyncio.wait_for(
                    sandbox_service.create_sandbox(
                        spec=None, entrypoint=None, env=dict(cfg.get("env") or {}),
                        metadata=dict(metadata or {}),
                        timeout=timedelta(minutes=int(cfg.get("lifecycle_minutes", 120) or 120)),
                        resource=dict(cfg.get("resource", {}) or {}), network_policy=None,
                        extensions=extensions, volumes=volumes,
                    ), timeout=timeout_s,
                )
                sandbox_id = str(created.id)
            except Exception:
                if not metadata or not metadata.get("attempt_id"):
                    raise
                # A timed-out response may hide a successful creation. Enumerate
                # the original identity; never submit a second create request.
                sandbox_id = await asyncio.wait_for(
                    self._find_instance(sandbox_service, metadata), timeout=timeout_s,
                )
                if not sandbox_id:
                    raise RuntimeError("Sandbox create outcome is unknown; retain runtime.json and do not retry blindly")
            handle = await asyncio.wait_for(
                self._connect_instance(
                    sandbox_id=sandbox_id, cfg=cfg, connection=connection,
                    sandbox_service=sandbox_service,
                ), timeout=timeout_s,
            )
            return handle
        except BaseException:
            if sandbox_id is not None:
                # No business command has started during creation/bootstrap.
                try:
                    await asyncio.shield(sandbox_service.kill_sandbox(sandbox_id))
                except Exception:
                    log.warning("failed to clean up Sandbox %s after creation error", sandbox_id, exc_info=True)
            await asyncio.shield(connection.close_transport_if_owned())
            raise

    async def _connect_instance(
        self, *, sandbox_id: str, cfg: dict, connection: Any, sandbox_service: Any,
    ) -> _SandboxRuntimeHandle:
        AdapterFactory, _ = _load_opensandbox()
        timeout_s = float(cfg.get("create_timeout_s", 600.0) or 600.0)
        deadline = asyncio.get_running_loop().time() + timeout_s
        while True:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                raise TimeoutError(f"Sandbox {sandbox_id} did not become ready")
            info = await asyncio.wait_for(sandbox_service.get_sandbox_info(sandbox_id), remaining)
            state = str(info.status.state).rsplit(".", 1)[-1].lower()
            if state == "running":
                break
            if state in {"failed", "stopping", "stopped", "terminated"}:
                raise RuntimeError(f"Sandbox {sandbox_id} entered terminal state {state!r}")
            await asyncio.sleep(min(0.2, remaining))
        command_port = int(cfg.get("command_port", 44772) or 44772)
        endpoint = await sandbox_service.get_sandbox_endpoint(sandbox_id, command_port, connection.use_server_proxy)
        command_connection, adapter_endpoint = _normalize_command_endpoint(connection, endpoint)
        factory = AdapterFactory(command_connection)
        commands = factory.create_command_service(adapter_endpoint)
        files = factory.create_filesystem_service(adapter_endpoint)
        if not bool(cfg.get("skip_health_check", False)):
            health = factory.create_health_service(adapter_endpoint)
            while not await health.ping(sandbox_id):
                if asyncio.get_running_loop().time() >= deadline:
                    raise TimeoutError(f"Sandbox {sandbox_id} execd is not ready")
                await asyncio.sleep(0.2)
        handle = _SandboxRuntimeHandle(
            sandbox_id=sandbox_id, sandbox_service=sandbox_service,
            commands=commands, files=files, command_endpoint=endpoint,
            command_port=command_port, connection=connection,
        )
        handle.cleanup_on_finish = bool(cfg.get("cleanup_on_finish", True))
        handle.start_renewal(float(cfg.get("lifecycle_minutes", 120) or 120) * 60)
        return handle

    async def _prepare_instance(self, sandbox: Any, cfg: Dict[str, Any]) -> None:
        from opensandbox.models.filesystem import SetPermissionEntry, WriteEntry

        entries = []
        payloads: dict[str, bytes] = {}
        for item in cfg.get("embedded_files", []) or []:
            source = Path(str(item["source"]))
            target = str(item["target"])
            if not source.is_file():
                raise FileNotFoundError(f"Sandbox embedded file not found: {source}")
            if target in payloads:
                raise ValueError(f"Duplicate Sandbox uploaded file target: {target}")
            if not target.startswith("/") or ".." in PurePosixPath(target).parts:
                raise ValueError(f"Invalid Sandbox uploaded file path: {target}")
            mode_text = str(item.get("permissions", "0644")).lstrip("0") or "0"
            if any(char not in "01234567" for char in mode_text):
                raise ValueError(f"Invalid file permissions: {mode_text}")
            payloads[target] = source.read_bytes()
            entries.append(WriteEntry(path=target, data=payloads[target], mode=int(mode_text)))
        if entries:
            parents = sorted({str(PurePosixPath(entry.path).parent) for entry in entries})
            await sandbox.files.create_directories([WriteEntry(path=parent, mode=755) for parent in parents])
            await sandbox.files.write_files(entries)
            await sandbox.files.set_permissions([
                SetPermissionEntry(path=entry.path, mode=entry.mode) for entry in entries
            ])
            for target, payload in payloads.items():
                copied = await sandbox.files.read_bytes(target)
                if hashlib.sha256(copied).digest() != hashlib.sha256(payload).digest():
                    raise RuntimeError(f"Sandbox uploaded file checksum mismatch: {target}")
        storage = dict(cfg.get("storage") or {})
        for item in storage.get("inputs", []) or []:
            target = shlex.quote(str(item["mount_path"]))
            result = await run_sandbox_command(
                sandbox, f"test -d {target} && ls -A -- {target} >/dev/null",
                timeout_s=min(30, float(cfg.get("command_timeout_s", 720))),
            )
            if result.exit_code:
                raise RuntimeError(f"Sandbox input is not readable: {item['mount_path']}")
        for item in storage.get("outputs", []) or []:
            probe_dir = str(PurePosixPath(item["mount_path"]) / (".safactory-probe-" + uuid.uuid4().hex))
            probe = probe_dir + "/write-check"
            try:
                await sandbox.files.create_directories([WriteEntry(path=probe_dir, mode=755)])
                await sandbox.files.write_file(probe, b"safactory-write-check", mode=600)
                if await sandbox.files.read_bytes(probe) != b"safactory-write-check":
                    raise RuntimeError("Sandbox output read-back mismatch")
            finally:
                try:
                    await sandbox.files.delete_files([probe])
                    await sandbox.files.delete_directories([probe_dir])
                except Exception:
                    log.warning("could not remove Sandbox output probe: %s", probe_dir, exc_info=True)

    @staticmethod
    async def _terminate(sandbox: Any) -> None:
        try:
            await sandbox.kill()
        finally:
            await sandbox.close()
