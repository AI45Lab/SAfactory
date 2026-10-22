from __future__ import annotations

import asyncio
import base64
import json
import logging
import shlex
from pathlib import PurePosixPath
from typing import Any

from clusters.sandbox_cluster import run_sandbox_command, sandbox_results_setup_command

from .episode_common import (
    RUNNER_DIAGNOSTIC_PREFIXES,
    normalize_result,
    parse_result_output,
    request_env,
    request_payload,
    result_artifact_path,
    tail,
)
from .types import SimulationAgentLease, SimulationStartRequest, SimulationStartResult

log = logging.getLogger("manager.sandbox_episode_runner")


class SandboxEpisodeRunner:
    """Runs one episode in an allocated OpenSandbox instance."""

    def __init__(self, *, timeout_s: float) -> None:
        self.timeout_s = float(timeout_s)

    async def start(
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
            body = await self._read_result_artifact(sandbox, artifact_path, timeout_s)
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
                body = await self._read_result_artifact(sandbox, artifact_path, timeout_s)
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
        return body

    async def close(self) -> None:
        return
