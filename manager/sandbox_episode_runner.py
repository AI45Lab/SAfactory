from __future__ import annotations

import asyncio
import base64
import logging
import shlex

from clusters.sandbox_cluster import run_sandbox_command

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
        encoded_payload = base64.b64encode(payload.encode("utf-8")).decode("ascii")
        command = f"printf %s {shlex.quote(encoded_payload)} | base64 -d | env "
        command += " ".join(shlex.quote(f"{key}={value}") for key, value in env.items())
        shell_command = lease.run_command
        if lease.workdir:
            shell_command = f"cd {shlex.quote(lease.workdir)} && {shell_command}"
        command += f" sh -lc {shlex.quote(shell_command)}"

        timeout_s = min(self.timeout_s, float(cfg.get("command_timeout_s") or self.timeout_s))
        try:
            output = await run_sandbox_command(sandbox, command, timeout_s=timeout_s)
        except asyncio.TimeoutError:
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

        try:
            body = parse_result_output(output.stdout)
            source = "stdout"
        except Exception as stdout_error:
            path = result_artifact_path(request)
            artifact = await run_sandbox_command(
                sandbox,
                f"cat -- {shlex.quote(path)}",
                timeout_s=min(30.0, timeout_s),
            )
            if artifact.exit_code != 0:
                raise RuntimeError(
                    "Sandbox runner produced no parseable SimulationStartResult: "
                    f"stdout_error={stdout_error}; artifact_error={tail(artifact.stderr or artifact.stdout)}"
                ) from stdout_error
            body = parse_result_output(artifact.stdout)
            source = "artifact"

        result = normalize_result(body, session_id=request.session_id)
        result.metrics = dict(result.metrics or {})
        result.metrics.update({"runtime": "sandbox", "sandbox_id": lease.resource_id, "result_source": source})
        return result

    async def close(self) -> None:
        return
