from __future__ import annotations

import asyncio
import json
import logging
import shlex
import subprocess
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List

from .types import SimulationAgentLease, SimulationStartRequest, SimulationStartResult

log = logging.getLogger("manager.agent_start_client")
_RUNNER_DIAGNOSTIC_PREFIX = "SAFACTORY_OPENCLAW_DIAGNOSTIC "


class AgentStartClient:
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

        request_params = asdict(request)
        payload = json.dumps(request_params, ensure_ascii=False)
        cmd = self._docker_exec_cmd(lease, request)
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
            self._json_for_log(request_params),
        )
        result = await asyncio.to_thread(self._run, cmd, payload)
        self._log_runner_diagnostics(result.stderr, lease)
        if result.returncode != 0:
            raise RuntimeError(
                "OpenClaw run failed: "
                f"container={lease.container_name or lease.container_id} "
                f"returncode={result.returncode} "
                f"stdout={self._tail(result.stdout)} stderr={self._tail(result.stderr)}"
            )
        body = self._parse_output(result.stdout)
        return self._normalize_result(body, session_id=request.session_id)

    async def close(self) -> None:
        return

    def _docker_exec_cmd(self, lease: SimulationAgentLease, request: SimulationStartRequest) -> List[str]:
        return [
            lease.docker_bin or "docker",
            "exec",
            "-i",
            "-e",
            f"SAFACTORY_JOB_ID={request.job_id}",
            "-e",
            f"SAFACTORY_SESSION_ID={request.session_id}",
            "-e",
            f"SAFACTORY_AGENT_NAME={request.agent_name}",
            "-e",
            f"SAFACTORY_AGENT_ID={request.agent_id}",
            lease.container_id,
            "sh",
            "-lc",
            lease.run_command,
        ]

    @classmethod
    def _log_runner_diagnostics(cls, stderr: str, lease: SimulationAgentLease) -> None:
        for raw_line in (stderr or "").splitlines():
            line = raw_line.strip()
            if not line.startswith(_RUNNER_DIAGNOSTIC_PREFIX):
                continue
            payload = line[len(_RUNNER_DIAGNOSTIC_PREFIX) :].strip()
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

    @classmethod
    def _parse_output(cls, stdout: str) -> Dict[str, Any]:
        text = (stdout or "").strip()
        if not text:
            raise RuntimeError("OpenClaw run returned empty output; expected SimulationStartResult JSON")
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            for line in reversed(text.splitlines()):
                line = line.strip()
                if not line:
                    continue
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    continue
        raise RuntimeError(f"OpenClaw run returned non-JSON output: {cls._tail(text)}")

    @classmethod
    def _normalize_result(cls, result: Any, *, session_id: str) -> SimulationStartResult:
        body = cls._to_dict(result)
        metrics = body.get("metrics")
        if not isinstance(metrics, dict):
            metrics = {}
        return SimulationStartResult(
            session_id=str(body.get("session_id") or session_id),
            status=str(body.get("status") or "succeeded"),
            total_reward=float(body.get("total_reward", 0.0) or 0.0),
            step_count=int(body.get("step_count", 0) or 0),
            terminated=bool(body.get("terminated", False)),
            truncated=bool(body.get("truncated", False)),
            error_text=None if body.get("error_text") is None else str(body.get("error_text")),
            metrics=metrics,
        )

    @staticmethod
    def _to_dict(result: Any) -> Dict[str, Any]:
        if isinstance(result, dict):
            return result
        if is_dataclass(result):
            return asdict(result)
        if hasattr(result, "model_dump"):
            return result.model_dump(mode="json")
        raise TypeError(f"Unsupported OpenClaw result type: {type(result).__name__}")

    @staticmethod
    def _cmd_for_log(cmd: List[str]) -> str:
        return shlex.join([str(part) for part in cmd])

    @staticmethod
    def _json_for_log(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, default=str)

    @staticmethod
    def _tail(value: str, limit: int = 1000) -> str:
        return (value or "").strip()[-int(limit):]
