from __future__ import annotations

import asyncio
import logging
import shlex
import subprocess
from typing import List

from .episode_common import (
    RUNNER_DIAGNOSTIC_PREFIX,
    json_for_log,
    normalize_result,
    parse_result_output,
    request_env,
    request_payload,
    tail,
)
from .types import SimulationAgentLease, SimulationStartRequest, SimulationStartResult

log = logging.getLogger("manager.docker_episode_runner")


class DockerEpisodeRunner:
    """Runs one OpenClaw episode inside an allocated Docker container."""

    def __init__(self, *, timeout_s: float) -> None:
        self.timeout_s = float(timeout_s)

    async def start(
        self,
        lease: SimulationAgentLease,
        request: SimulationStartRequest,
    ) -> SimulationStartResult:
        if not lease.container_id:
            raise RuntimeError(f"Docker lease missing container_id: {lease.agent_name}/{lease.agent_id}")
        if not lease.run_command:
            raise RuntimeError(f"Docker lease missing run_command: {lease.agent_name}/{lease.agent_id}")

        request_params, payload = request_payload(request)
        cmd = self._docker_exec_cmd(lease, request, payload)
        log.info(
            "OpenClaw docker exec command: agent=%s/%s container=%s command=%s",
            lease.agent_name,
            lease.agent_id,
            lease.container_name or lease.container_id,
            self._cmd_for_log(cmd),
        )
        log.info(
            "OpenClaw agent start request params: agent=%s/%s params=%s",
            lease.agent_name,
            lease.agent_id,
            json_for_log(request_params),
        )
        result = await asyncio.to_thread(self._run, cmd, payload)
        self._log_runner_diagnostics(result.stderr, lease)
        result_mode = str(getattr(lease, "result_mode", "json") or "json").strip().lower()
        if result_mode == "exit_code":
            if result.returncode != 0:
                raise RuntimeError(
                    "OpenClaw command failed: "
                    f"container={lease.container_name or lease.container_id} "
                    f"returncode={result.returncode} "
                    f"stdout={tail(result.stdout)} stderr={tail(result.stderr)}"
                )
            return SimulationStartResult(
                session_id=str(request.session_id),
                status="succeeded",
                total_reward=0.0,
                step_count=0,
                terminated=True,
                truncated=False,
                error_text=None,
                metrics={
                    "result_mode": result_mode,
                    "stdout_tail": tail(result.stdout),
                    "stderr_tail": tail(result.stderr),
                },
            )
        if result.returncode != 0:
            raise RuntimeError(
                "OpenClaw run failed: "
                f"container={lease.container_name or lease.container_id} "
                f"returncode={result.returncode} "
                f"stdout={tail(result.stdout)} stderr={tail(result.stderr)}"
            )
        body = parse_result_output(result.stdout)
        return normalize_result(body, session_id=request.session_id)

    async def close(self) -> None:
        return

    def _docker_exec_cmd(
        self,
        lease: SimulationAgentLease,
        request: SimulationStartRequest,
        payload: str,
    ) -> List[str]:
        cmd = [
            lease.docker_bin or "docker",
            "exec",
            "-i",
        ]
        if lease.workdir:
            cmd.extend(["-w", lease.workdir])
        for key, value in request_env(request, payload, containerize_local_gateway=True).items():
            cmd.extend(["-e", f"{key}={value}"])
        cmd.extend([lease.container_id, "sh", "-lc", lease.run_command])
        return cmd

    @classmethod
    def _log_runner_diagnostics(cls, stderr: str, lease: SimulationAgentLease) -> None:
        for raw_line in (stderr or "").splitlines():
            line = raw_line.strip()
            if not line.startswith(RUNNER_DIAGNOSTIC_PREFIX):
                continue
            payload = line[len(RUNNER_DIAGNOSTIC_PREFIX) :].strip()
            log.info(
                "OpenClaw agent create params: agent=%s/%s container=%s params=%s",
                lease.agent_name,
                lease.agent_id,
                lease.container_name or lease.container_id,
                payload,
            )

    def _run(self, cmd: List[str], payload: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            cmd,
            input=payload,
            capture_output=True,
            text=True,
            timeout=self.timeout_s,
            check=False,
        )

    @staticmethod
    def _cmd_for_log(cmd: List[str]) -> str:
        return shlex.join([str(part) for part in cmd])
