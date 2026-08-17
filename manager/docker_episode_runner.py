from __future__ import annotations

import asyncio
import logging
import shlex
import subprocess
from typing import List

from .episode_common import (
    RUNNER_DIAGNOSTIC_PREFIXES,
    json_for_log,
    normalize_result,
    parse_result_artifact,
    parse_result_output,
    request_env,
    request_payload,
    result_artifact_path,
    tail,
)
from .types import SimulationAgentLease, SimulationStartRequest, SimulationStartResult

log = logging.getLogger("manager.docker_episode_runner")


class DockerEpisodeRunner:
    """Runs one episode through a runner entrypoint inside an allocated Docker container."""

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
        log.debug(
            "runner entrypoint docker exec command: env=%s agent_id=%s container=%s command=%s",
            lease.agent_name,
            lease.agent_id,
            lease.container_name or lease.container_id,
            self._cmd_for_log(cmd),
        )
        log.debug(
            "runner entrypoint request params: env=%s agent_id=%s params=%s",
            lease.agent_name,
            lease.agent_id,
            json_for_log(request_params),
        )
        log.debug(
            "runner entrypoint result artifact: env=%s agent_id=%s path=%s",
            lease.agent_name,
            lease.agent_id,
            result_artifact_path(request),
        )
        try:
            result = await asyncio.to_thread(self._run, cmd, payload)
        except subprocess.TimeoutExpired as exc:
            log.warning(
                "runner entrypoint docker exec timed out: env=%s agent_id=%s container=%s timeout_s=%.2f",
                lease.agent_name,
                lease.agent_id,
                lease.container_name or lease.container_id,
                self.timeout_s,
            )
            return self._timeout_result(lease, request, exc)
        self._log_runner_diagnostics(result.stderr, lease)
        result_mode = str(getattr(lease, "result_mode", "json") or "json").strip().lower()
        if result_mode == "exit_code":
            if result.returncode != 0:
                raise RuntimeError(
                    "runner entrypoint command failed: "
                    f"container={lease.container_name or lease.container_id} "
                    f"returncode={result.returncode} "
                    f"stdout={tail(result.stdout)} stderr={tail(result.stderr)}"
                )
            return SimulationStartResult(
                session_id=str(request.session_id),
                status="succeeded",
                total_reward=None,
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
                "runner entrypoint failed: "
                f"container={lease.container_name or lease.container_id} "
                f"returncode={result.returncode} "
                f"stdout={tail(result.stdout)} stderr={tail(result.stderr)}"
            )
        return self._collect_json_result(result.stdout, request)

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
            prefix = next((item for item in RUNNER_DIAGNOSTIC_PREFIXES if line.startswith(item)), "")
            if not prefix:
                continue
            payload = line[len(prefix) :].strip()
            log.debug(
                "runner entrypoint diagnostic: env=%s agent_id=%s container=%s payload=%s",
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

    def _collect_json_result(
        self,
        stdout: str,
        request: SimulationStartRequest,
    ) -> SimulationStartResult:
        try:
            body = parse_result_output(stdout)
            result = normalize_result(body, session_id=request.session_id)
            result.metrics = dict(result.metrics or {})
            result.metrics.setdefault("result_source", "stdout")
            return result
        except Exception as stdout_exc:
            try:
                body, path = parse_result_artifact(request)
            except Exception as artifact_exc:
                raise RuntimeError(
                    "runner entrypoint produced no parseable SimulationStartResult: "
                    f"stdout_error={stdout_exc}; artifact_error={artifact_exc}"
                ) from artifact_exc
            result = normalize_result(body, session_id=request.session_id)
            result.metrics = dict(result.metrics or {})
            result.metrics.update(
                {
                    "result_source": "artifact",
                    "result_artifact_path": str(path),
                    "stdout_parse_error": str(stdout_exc),
                }
            )
            return result

    def _timeout_result(
        self,
        lease: SimulationAgentLease,
        request: SimulationStartRequest,
        exc: subprocess.TimeoutExpired,
    ) -> SimulationStartResult:
        timeout_s = float(exc.timeout or self.timeout_s)
        return SimulationStartResult(
            session_id=str(request.session_id),
            status="truncated",
            total_reward=None,
            step_count=0,
            terminated=True,
            truncated=True,
            error_text=(
                f"docker exec timed out after {timeout_s:.1f}s "
                f"(inner agent timeout={float(request.agent_start_timeout_s):.1f}s)"
            ),
            metrics={
                "timeout_layer": "docker_exec",
                "outer_timeout_s": timeout_s,
                "inner_timeout_s": float(request.agent_start_timeout_s),
                "container_id": lease.container_id,
                "container_name": lease.container_name,
                "stdout_tail": self._timeout_stream_tail(getattr(exc, "stdout", None)),
                "stderr_tail": self._timeout_stream_tail(getattr(exc, "stderr", None)),
            },
        )

    @staticmethod
    def _cmd_for_log(cmd: List[str]) -> str:
        return shlex.join([str(part) for part in cmd])

    @staticmethod
    def _timeout_stream_tail(value: object, limit: int = 1000) -> str:
        if value is None:
            return ""
        if isinstance(value, bytes):
            text = value.decode("utf-8", errors="replace")
        else:
            text = str(value)
        return tail(text, limit=limit)
