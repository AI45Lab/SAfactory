from __future__ import annotations

import asyncio
import base64
import json
import logging
import shlex
from dataclasses import asdict, replace
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .sandbox_input_preparer import SandboxInputPreparer

from clusters.sandbox_cluster import run_sandbox_command, sandbox_results_setup_command

from .episode_common import (
    RESULT_FILENAME,
    RUNNER_DIAGNOSTIC_PREFIXES,
    normalize_result,
    parse_result_output,
    request_env,
    request_payload,
    result_artifact_path,
    safe_path_part,
    tail,
)
from .types import SimulationAgentLease, SimulationStartRequest, SimulationStartResult

log = logging.getLogger("manager.sandbox_episode_runner")


class SandboxEpisodeRunner:
    """Runs one episode in an allocated OpenSandbox instance."""

    def __init__(self, *, timeout_s: float, input_preparer: SandboxInputPreparer | None = None) -> None:
        self.timeout_s = float(timeout_s)
        self.input_preparer = input_preparer

    async def start(
        self,
        lease: SimulationAgentLease,
        request: SimulationStartRequest,
    ) -> SimulationStartResult:
        if "bundle_source" not in request.env_params:
            return await self._start_runner(lease, request)

        from .sandbox_input_preparer import InputPreparationError

        timeout_s = min(self.timeout_s, float(lease.runtime_config.get("command_timeout_s") or self.timeout_s))
        deadline = asyncio.get_running_loop().time() + timeout_s
        artifact = result_artifact_path(request).replace("/app/results/", "/sandbox-data/results/", 1)
        try:
            if self.input_preparer is None:
                raise InputPreparationError("config", "input_storage_not_configured")
            if lease.runtime_handle is None or not lease.run_command:
                raise InputPreparationError("config", "invalid_sandbox_lease")
            if lease.result_mode == "exit_code":
                raise InputPreparationError("config", "json_result_required")
            setup = await run_sandbox_command(
                lease.runtime_handle,
                f"{sandbox_results_setup_command(str(PurePosixPath(artifact).parent))} && rm -f -- {shlex.quote(artifact)}",
                timeout_s=min(30.0, timeout_s),
            )
            if setup.exit_code:
                raise InputPreparationError("results", "results_not_writable")
            receipt = await self.input_preparer.prepare(lease=lease, request=request, deadline=deadline)
            if asyncio.get_running_loop().time() >= deadline:
                raise TimeoutError
            await self._write_input_json(
                lease, str(PurePosixPath(artifact).parent / "infra" / "input-preparation.json"), receipt,
                timeout_s=min(30.0, deadline - asyncio.get_running_loop().time()),
            )
            if asyncio.get_running_loop().time() >= deadline:
                raise TimeoutError
        except asyncio.CancelledError:
            # Preparation owns remote interruption and private-plan cleanup.
            record = asyncio.create_task(self._record_input_failure(
                lease, request, artifact, "Sandbox input preparation cancelled", timed_out=False,
            ))
            while not record.done():
                try:
                    await asyncio.shield(record)
                except asyncio.CancelledError:
                    continue
                except Exception:
                    break
            if not record.cancelled():
                record.exception()
            raise
        except Exception as error:
            timed_out = isinstance(error, TimeoutError)
            message = str(error) if isinstance(error, InputPreparationError) else "Sandbox input preparation failed"
            return await self._record_input_failure(lease, request, artifact, message, timed_out=timed_out)

        remaining = max(0.001, deadline - asyncio.get_running_loop().time())
        prepared_lease = replace(lease, runtime_config={**lease.runtime_config, "command_timeout_s": remaining})
        # The existing runner path retains trajectory, cancellation and artifact
        # validation semantics. Only its remaining command budget changes.
        return await self._start_runner(prepared_lease, request)

    async def _record_input_failure(
        self, lease: SimulationAgentLease, request: SimulationStartRequest,
        artifact: str, message: str, *, timed_out: bool,
    ) -> SimulationStartResult:
        result = SimulationStartResult(
            # No Harbor trial has run: this must remain an infrastructure
            # failure, never the worker's completed/truncated zero-reward path.
            session_id=request.session_id, status="failed",
            total_reward=None, step_count=0, terminated=True, truncated=False,
            error_text=message,
            metrics={"phase": "input_prepare", "timeout_layer": "sandbox_input" if timed_out else "",
                     "sandbox_id": lease.resource_id},
        )
        # Retain a local diagnostic even when AFS itself is unavailable.
        local = Path.cwd() / "results" / safe_path_part(request.job_id) / safe_path_part(request.session_id) / RESULT_FILENAME
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_text(json.dumps(asdict(result), ensure_ascii=False), encoding="utf-8")
        try:
            await self._write_input_json(lease, artifact, asdict(result))
        except Exception:
            log.warning("Could not persist Sandbox input failure to AFS: sandbox=%s", lease.resource_id)
        return result

    @staticmethod
    async def _write_input_json(
        lease: SimulationAgentLease, path: str, body: dict[str, Any], *, timeout_s: float = 30.0,
    ) -> None:
        # Only public receipts / sanitized failures reach this function.
        encoded = base64.b64encode(json.dumps(body, ensure_ascii=False).encode()).decode("ascii")
        parent = str(PurePosixPath(path).parent)
        output = await run_sandbox_command(
            lease.runtime_handle,
            f"{sandbox_results_setup_command(parent)} && "
            f"[ ! -L {shlex.quote(path)} ] && [ ! -L {shlex.quote(path + '.tmp')} ] && "
            f"printf %s {shlex.quote(encoded)} | base64 -d > {shlex.quote(path + '.tmp')} && "
            f"mv -f -- {shlex.quote(path + '.tmp')} {shlex.quote(path)} && chmod a+rw -- {shlex.quote(path)}",
            timeout_s=timeout_s,
        )
        if output.exit_code:
            raise RuntimeError("Unable to persist Sandbox input metadata")

    async def _start_runner(
        self,
        lease: SimulationAgentLease,
        request: SimulationStartRequest,
    ) -> SimulationStartResult:
        sandbox = lease.runtime_handle
        if sandbox is None:
            raise RuntimeError(f"Sandbox lease missing runtime handle: {lease.agent_name}/{lease.agent_id}")
        if not lease.run_command:
            raise RuntimeError(f"Sandbox lease missing run_command: {lease.agent_name}/{lease.agent_id}")
        artifact_required = lease.agent_name == "harbor" or (
            str(lease.env_params.get("reward_source") or "").strip().lower() == "runner"
        )
        if artifact_required and lease.result_mode == "exit_code":
            raise ValueError("Harbor and runner rewards require a JSON result artifact")

        _, payload = request_payload(request)
        cfg = dict(lease.runtime_config or {})
        gateway_base_url = str(cfg.get("gateway_base_url") or request.gateway_base_url).rstrip("/")
        env = {
            **dict(cfg.get("env", {}) or {}),
            **request_env(
                request,
                payload,
                gateway_base_url=gateway_base_url,
                containerize_local_gateway=False,
            ),
        }
        artifact_path = result_artifact_path(request).replace(
            "/app/results/", "/sandbox-data/results/", 1,
        )
        env["SAFACTORY_RESULT_PATH"] = artifact_path
        encoded_payload = base64.b64encode(payload.encode("utf-8")).decode("ascii")
        command = f"printf %s {shlex.quote(encoded_payload)} | base64 -d | env "
        command += " ".join(shlex.quote(f"{key}={value}") for key, value in env.items())
        shell_command = lease.run_command
        if lease.workdir:
            shell_command = f"cd {shlex.quote(lease.workdir)} && {shell_command}"
        command += f" sh -lc {shlex.quote(shell_command)}"
        if artifact_required:
            # Retries share the session's result path. Require this attempt to
            # produce its own result instead of accepting an earlier reward.
            command = f"rm -f -- {shlex.quote(artifact_path)} && {command}"

        session_dir = PurePosixPath(artifact_path).parent
        target = shlex.quote(str(session_dir))
        # Check ancestors again at cleanup: a runner may have replaced a path.
        directories = " && ".join(
            f"[ ! -L {shlex.quote(str(path))} ]"
            for path in (*reversed(session_dir.parents), session_dir)
        )
        cleanup_command = (
            f"if [ -e {target} ] || [ -L {target} ]; then "
            f"{directories} && [ -d {target} ] && "
            f"find -P {target} -type d -exec chmod -- a+rwX {{}} \\; "
            "-o -type f -exec chmod -- a+rwX {} +; fi"
        )
        # The trap covers ordinary runner exits; the management-side finally
        # also runs after SDK errors, interrupts and external cancellation.
        exit_trap = (
            'status=$?; trap - EXIT; '
            f"{cleanup_command}; exit \"$status\""
        )
        command = (
            f"{sandbox_results_setup_command(str(session_dir.parent))} && "
            f"{sandbox_results_setup_command(str(session_dir))} && "
            f"{{ trap {shlex.quote(exit_trap)} EXIT; {command}; }}"
        )

        timeout_s = min(self.timeout_s, float(cfg.get("command_timeout_s") or self.timeout_s))
        execution = asyncio.create_task(run_sandbox_command(sandbox, command, timeout_s=timeout_s))
        cancellation = None
        try:
            while True:
                try:
                    output = await asyncio.shield(execution)
                    break
                except asyncio.CancelledError as exc:
                    if cancellation is None:
                        cancellation = exc
                        execution.cancel()
                    # Repeated cancellation must not interrupt the SDK's remote
                    # interrupt request or let cleanup overlap the main command.
                    if execution.done():
                        raise cancellation
                except asyncio.TimeoutError:
                    if cancellation is not None:
                        raise cancellation
                    return SimulationStartResult(
                        session_id=request.session_id,
                        status="truncated",
                        total_reward=None,
                        step_count=0,
                        terminated=True,
                        truncated=True,
                        error_text=f"sandbox command timed out after {timeout_s:.1f}s",
                        metrics={
                            "timeout_layer": "sandbox_command",
                            "outer_timeout_s": timeout_s,
                            "inner_timeout_s": float(request.agent_start_timeout_s),
                            "sandbox_id": lease.resource_id,
                        },
                    )
                except Exception:
                    if cancellation is not None:
                        raise cancellation
                    raise
        finally:
            cleanup = asyncio.create_task(run_sandbox_command(sandbox, cleanup_command, timeout_s=30.0))
            while True:
                try:
                    permissions = await asyncio.shield(cleanup)
                    if permissions.exit_code != 0 or permissions.stderr:
                        log.warning(
                            "Sandbox results permission cleanup failed: resource=%s path=%s exit_code=%s stderr=%s",
                            lease.resource_id, session_dir, permissions.exit_code, tail(permissions.stderr),
                        )
                    break
                except asyncio.CancelledError as exc:
                    cancellation = cancellation or exc
                    if cleanup.done():
                        log.warning("Sandbox results permission cleanup cancelled: resource=%s path=%s", lease.resource_id, session_dir)
                        break
                except Exception:
                    log.warning(
                        "Sandbox results permission cleanup failed: resource=%s path=%s",
                        lease.resource_id, session_dir, exc_info=True,
                    )
                    break
            if cancellation is not None:
                raise cancellation

        for raw_line in output.stderr.splitlines():
            line = raw_line.strip()
            if any(line.startswith(prefix) for prefix in RUNNER_DIAGNOSTIC_PREFIXES):
                log.debug("Sandbox runner diagnostic: resource=%s payload=%s", lease.resource_id, line)

        if output.exit_code != 0:
            raise RuntimeError(
                f"Sandbox runner failed: resource={lease.resource_id} exit_code={output.exit_code} "
                f"stdout={tail(output.stdout)} stderr={tail(output.stderr)}"
            )
        if lease.result_mode == "exit_code":
            return SimulationStartResult(
                session_id=request.session_id,
                status="succeeded",
                total_reward=None,
                step_count=0,
                terminated=True,
                truncated=False,
                metrics={
                    "result_mode": "exit_code",
                    "sandbox_id": lease.resource_id,
                    "stdout_tail": tail(output.stdout),
                    "stderr_tail": tail(output.stderr),
                },
            )

        if artifact_required:
            # Native rewards come from the persisted runner result, even when
            # stdout contains another parseable result.
            body = await self._read_result_artifact(sandbox, artifact_path, timeout_s, request)
            source = "artifact"
            if body.get("session_id") != request.session_id:
                raise RuntimeError("Sandbox result artifact session_id does not match the current session")
            if not body.get("status"):
                raise RuntimeError("Sandbox result artifact is missing its rollout status")
            if isinstance(body.get("total_reward"), bool):
                raise RuntimeError("Sandbox result artifact total_reward must be numeric, not boolean")
        else:
            try:
                body = parse_result_output(output.stdout)
                source = "stdout"
            except Exception:
                body = await self._read_result_artifact(sandbox, artifact_path, timeout_s, request)
                source = "artifact"

        result = normalize_result(body, session_id=request.session_id)
        result.metrics = dict(result.metrics or {})
        result.metrics.update(
            {
                "runtime": "sandbox",
                "sandbox_id": lease.resource_id,
                "result_source": source,
                "result_artifact_path": artifact_path,
            }
        )
        return result

    @staticmethod
    async def _read_result_artifact(
        sandbox: Any, artifact_path: str, timeout_s: float,
        request: SimulationStartRequest,
    ) -> dict[str, Any]:
        artifact = await run_sandbox_command(
            sandbox, f"cat -- {shlex.quote(artifact_path)}", timeout_s=min(30.0, timeout_s),
        )
        if artifact.exit_code != 0:
            raise RuntimeError(
                f"Sandbox runner result artifact could not be read: {artifact_path}; "
                f"{tail(artifact.stderr or artifact.stdout)}"
            )
        try:
            body = json.loads(artifact.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Sandbox runner result artifact contains invalid JSON: {artifact_path}") from exc
        if not isinstance(body, dict):
            raise RuntimeError("Sandbox runner result artifact must contain a JSON object")
        local_path = (
            Path.cwd() / "results"
            / safe_path_part(request.job_id)
            / safe_path_part(request.session_id)
            / RESULT_FILENAME
        )
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_text(artifact.stdout, encoding="utf-8")
        return body

    async def close(self) -> None:
        return
