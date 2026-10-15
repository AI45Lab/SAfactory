from __future__ import annotations

import asyncio
import base64
import hashlib
import logging
import os
import re
import shlex
import tempfile
from pathlib import Path
from typing import Any

from clusters.sandbox_cluster import run_sandbox_command
from core.result_artifact import read_result_file, update_json_atomic

from .episode_common import (
    normalize_result, request_env, request_payload, result_artifact_path,
    safe_path_part, tail,
)
from .types import SimulationAgentLease, SimulationStartRequest, SimulationStartResult

log = logging.getLogger("manager.sandbox_episode_runner")


class SandboxEpisodeRunner:
    """Run an episode and collect its authoritative file through SDK files APIs."""

    def __init__(self, *, timeout_s: float) -> None:
        self.timeout_s = float(timeout_s)

    async def start(
        self, lease: SimulationAgentLease, request: SimulationStartRequest,
    ) -> SimulationStartResult:
        cfg = dict(lease.runtime_config or {})
        attempt = self._attempt(lease, request)
        recovery = dict(cfg.get("recovery") or {})
        if recovery:
            reference = dict(recovery.get("runtime_ref") or {})
            self._check_reference(reference, request, attempt)
            if reference.get("command_finished") is not True:
                raise RuntimeError("Sandbox recovery has not confirmed the previous writer stopped")
            kind = recovery.get("kind")
            if kind == "local":
                local_path = self._local_path(lease, request)
                body = await asyncio.to_thread(
                    read_result_file, local_path, job_id=request.job_id,
                    session_id=request.session_id, attempt_id=attempt,
                )
                result = self._result(body, local_path, lease, source="local_recovery")
            elif kind == "remote":
                if lease.runtime_handle is None:
                    raise RuntimeError("Remote Sandbox recovery requires a connected file service")
                lease.runtime_handle.retain = True
                result = await self._collect_result(lease, request)
            else:
                raise ValueError(f"Unsupported Sandbox recovery kind: {kind!r}")
            self._apply_execution_error(
                result, str(reference.get("execution_error") or ""), reference.get("exit_code"),
            )
            return result

        sandbox = lease.runtime_handle
        if sandbox is None:
            raise RuntimeError(f"Sandbox lease missing runtime handle: {lease.agent_name}/{lease.agent_id}")
        if not lease.run_command:
            raise RuntimeError(f"Sandbox lease missing run_command: {lease.agent_name}/{lease.agent_id}")
        sandbox.retain = True
        await self._update_reference(lease, request, {"command_finished": False, "execution_error": ""})
        _, payload = request_payload(request)
        env = {
            **dict(cfg.get("env") or {}),
            **request_env(request, payload, gateway_base_url=request.gateway_base_url, containerize_local_gateway=False),
            "SAFACTORY_RUNTIME": "sandbox", "SAFACTORY_ATTEMPT_ID": attempt,
        }
        encoded_payload = base64.b64encode(payload.encode("utf-8")).decode("ascii")
        command = f"printf %s {shlex.quote(encoded_payload)} | base64 -d | env "
        command += " ".join(shlex.quote(f"{key}={value}") for key, value in env.items())
        shell_command = lease.run_command
        if lease.workdir:
            shell_command = f"cd {shlex.quote(lease.workdir)} && {shell_command}"
        command += f" sh -lc {shlex.quote(shell_command)}"
        timeout_s = min(self.timeout_s, float(cfg.get("command_timeout_s") or self.timeout_s))

        async def on_init(command_id: str) -> None:
            await self._update_reference(lease, request, {"command_id": command_id, "command_finished": False})

        try:
            output = await run_sandbox_command(sandbox, command, timeout_s=timeout_s, on_init=on_init)
        except asyncio.CancelledError:
            stopped = getattr(sandbox, "execution_stopped", False) is True
            await asyncio.shield(self._update_reference(
                lease, request, {"command_finished": stopped, "execution_error": "runner cancelled"},
            ))
            if stopped:
                try:
                    await asyncio.shield(asyncio.wait_for(self._collect_result(lease, request), timeout=self._download_budget(cfg)))
                except Exception:
                    log.warning("Sandbox cancelled result collection failed: resource=%s", lease.resource_id, exc_info=True)
            raise
        except Exception as exc:
            stopped = getattr(exc, "execution_stopped", getattr(sandbox, "execution_stopped", False)) is True
            error = f"{type(exc).__name__}: {exc}"
            await self._update_reference(lease, request, {"command_finished": stopped, "execution_error": error})
            if not stopped:
                raise RuntimeError("Sandbox writer state is unknown; retaining instance without scoring") from exc
            result = await self._collect_result(lease, request)
            self._apply_execution_error(result, error, None)
            return result

        error = str(getattr(output, "error", None) or "")
        exit_code = output.exit_code
        if exit_code is None:
            error = error or "Sandbox command returned no trustworthy exit code"
        elif exit_code != 0:
            error = error or f"Sandbox runner exited with code {exit_code}"
        await self._update_reference(lease, request, {
            "command_finished": True, "exit_code": exit_code, "execution_error": error,
        })
        log.debug("Sandbox runner diagnostics: resource=%s stdout=%s stderr=%s", lease.resource_id, tail(output.stdout), tail(output.stderr))
        result = await self._collect_result(lease, request)
        result.metrics.update({"exit_code": exit_code, "stdout_tail": tail(output.stdout), "stderr_tail": tail(output.stderr)})
        self._apply_execution_error(result, error, exit_code)
        return result

    async def _collect_result(
        self, lease: SimulationAgentLease, request: SimulationStartRequest,
    ) -> SimulationStartResult:
        sandbox = lease.runtime_handle
        if sandbox is None or not hasattr(sandbox, "files"):
            raise RuntimeError("Sandbox result collection requires the SDK files service")
        if getattr(sandbox, "execution_stopped", True) is not True:
            raise RuntimeError("Sandbox writer has not stopped; refusing authoritative result collection")
        cfg = dict(lease.runtime_config or {})
        results = dict(cfg.get("results") or {})
        timeout = max(0.1, float(results.get("download_timeout_s", 60.0)))
        retries = max(0, int(results.get("download_retries", 3)))
        remote_path = result_artifact_path(request)
        destination = self._local_path(lease, request)
        attempt = self._attempt(lease, request)
        for retry in range(retries + 1):
            temporary: Path | None = None
            try:
                info = await asyncio.wait_for(sandbox.files.get_file_info([remote_path]), timeout=timeout)
                entry = info.get(remote_path)
                size = getattr(entry, "size", None)
                if isinstance(entry, dict):
                    size = entry.get("size")
                if isinstance(size, bool) or not isinstance(size, int) or size < 0:
                    raise ValueError("Sandbox SDK returned no valid result file size")
                content = await asyncio.wait_for(sandbox.files.read_bytes(remote_path), timeout=timeout)
                if not isinstance(content, bytes) or len(content) != size:
                    raise ValueError("Sandbox result download size does not match remote metadata")
                checksum = await run_sandbox_command(
                    sandbox, f"sha256sum -- {shlex.quote(remote_path)}", timeout_s=timeout,
                )
                digest = checksum.stdout.split(maxsplit=1)[0] if checksum.stdout.strip() else ""
                if checksum.exit_code != 0 or getattr(checksum, "error", None) or not re.fullmatch(r"[0-9a-fA-F]{64}", digest):
                    raise ValueError("Sandbox remote result SHA-256 could not be verified")
                if hashlib.sha256(content).hexdigest() != digest.lower():
                    raise ValueError("Sandbox result download SHA-256 mismatch")
                destination.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=".safactory-result-", suffix=".tmp", delete=False) as stream:
                    temporary = Path(stream.name)
                    stream.write(content)
                    stream.flush()
                    os.fsync(stream.fileno())
                body = await asyncio.to_thread(
                    read_result_file, temporary, job_id=request.job_id,
                    session_id=request.session_id, attempt_id=attempt,
                )
                os.replace(temporary, destination)
                temporary = None
                self._fsync_directory(destination.parent)
                result = self._result(body, destination, lease, source="sdk_file")
                result.metrics["result_sha256"] = digest.lower()
                return result
            except asyncio.CancelledError:
                raise
            except Exception:
                if retry >= retries:
                    raise
                log.warning("Sandbox result download retry: resource=%s attempt=%d", lease.resource_id, retry + 1)
                await asyncio.sleep(min(1.0, 0.1 * (2 ** retry)))
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
        raise AssertionError("unreachable")

    @staticmethod
    def _attempt(lease: SimulationAgentLease, request: SimulationStartRequest) -> str:
        configured = str((lease.runtime_config or {}).get("attempt_id") or "")
        requested = str((request.metadata or {}).get("attempt_id") or "")
        if configured and requested and configured != requested:
            raise ValueError("Sandbox lease and request attempt_id do not match")
        attempt = configured or requested
        if not attempt:
            raise ValueError("Sandbox execution requires a trusted attempt_id")
        return attempt

    @staticmethod
    def _local_path(lease: SimulationAgentLease, request: SimulationStartRequest) -> Path:
        results = dict((lease.runtime_config or {}).get("results") or {})
        root = Path(str(results.get("local_root") or "results")).expanduser().resolve()
        return root / safe_path_part(request.job_id) / safe_path_part(request.session_id) / "safactory_result.json"

    @staticmethod
    def _result(body: dict[str, Any], path: Path, lease: SimulationAgentLease, *, source: str) -> SimulationStartResult:
        result = normalize_result(body, session_id=body["session_id"])
        result.total_reward = None
        result.result_artifact_path = str(path)
        result.metrics.update(runtime="sandbox", sandbox_id=lease.resource_id, result_source=source)
        return result

    @staticmethod
    def _apply_execution_error(result: SimulationStartResult, error: str, exit_code: Any) -> None:
        if exit_code is not None and exit_code != 0:
            error = error or f"Sandbox runner exited with code {exit_code}"
        if error:
            result.status = "failed"
            result.terminated = True
            result.truncated = False
            result.error_text = error
            result.metrics["execution_error"] = error
            result.total_reward = None

    @staticmethod
    def _check_reference(reference: dict, request: SimulationStartRequest, attempt: str) -> None:
        for name, expected in (("job_id", request.job_id), ("session_id", request.session_id), ("attempt_id", attempt)):
            if reference.get(name) != expected:
                raise ValueError(f"Sandbox runtime reference {name} mismatch")

    async def _update_reference(self, lease: SimulationAgentLease, request: SimulationStartRequest, updates: dict) -> None:
        raw_path = (lease.runtime_config or {}).get("runtime_ref_path")
        if not raw_path:
            raise ValueError("Sandbox execution requires runtime_ref_path before starting")
        path = Path(str(raw_path))
        attempt = self._attempt(lease, request)

        await asyncio.to_thread(
            update_json_atomic, path, updates,
            identity={"job_id": request.job_id, "session_id": request.session_id, "attempt_id": attempt},
        )

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    @staticmethod
    def _download_budget(cfg: dict) -> float:
        results = dict(cfg.get("results") or {})
        return max(0.1, float(results.get("download_timeout_s", 60.0))) * 3 * (1 + max(0, int(results.get("download_retries", 3)))) + 5

    async def close(self) -> None:
        return
