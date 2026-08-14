#!/usr/bin/env python3
"""Run one Harbor trial and emit one SAfactory SimulationStartResult."""

from __future__ import annotations

import json
import math
import os
import signal
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

REQUEST_ENV = "SAFACTORY_START_REQUEST_JSON"
RESULT_PATH_ENV = "SAFACTORY_RESULT_PATH"
RESULT_JSON_PREFIX = "SAFACTORY_RESULT_JSON "
HARBOR_BIN = "/opt/harbor-env/bin/harbor"
RUNTIME_DIR = Path("/tmp/safactory-harbor")
MODEL_ENV_NAMES = {
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_BASE_URL",
    "CLAUDE_CODE_OAUTH_TOKEN",
    "OPENAI_API_BASE",
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
}


@dataclass(frozen=True)
class RunSpec:
    session_id: str
    task_id: str
    task_path: Path
    gateway_url: str
    agent: str
    model: str | None
    reward_key: str
    timeout_s: int
    result_path: Path
    jobs_root: Path
    harbor_job_name: str

    @property
    def episode_dir(self) -> Path:
        return self.result_path.parent

    @property
    def harbor_job_dir(self) -> Path:
        return self.jobs_root / self.harbor_job_name

    @property
    def docker_socket(self) -> Path:
        return RUNTIME_DIR / "docker.sock"

    @property
    def dockerd_log_path(self) -> Path:
        return self.episode_dir / "infra" / "dockerd.log"

    @property
    def harbor_log_path(self) -> Path:
        return self.episode_dir / "harbor" / "harbor-run.log"


@dataclass
class NestedDocker:
    process: subprocess.Popen[bytes] | None = None
    harbor_process: subprocess.Popen[bytes] | None = None
    env: dict[str, str] = field(default_factory=dict)

    def start(self, spec: RunSpec) -> str:
        if os.geteuid() != 0:
            raise RuntimeError("nested Docker requires root")
        if not Path("/dev/fuse").exists():
            raise RuntimeError(
                "nested Docker requires /dev/fuse; use privileged RJob with "
                "brainpp.cn/fuse:1"
            )

        spec.dockerd_log_path.parent.mkdir(parents=True, exist_ok=True)
        spec.docker_socket.unlink(missing_ok=True)
        (RUNTIME_DIR / "exec").mkdir(parents=True, exist_ok=True)
        Path("/docker-data").mkdir(parents=True, exist_ok=True)

        with spec.dockerd_log_path.open("ab") as log:
            self.process = subprocess.Popen(
                [
                    "dockerd",
                    f"--host=unix://{spec.docker_socket}",
                    f"--pidfile={RUNTIME_DIR / 'dockerd.pid'}",
                    f"--exec-root={RUNTIME_DIR / 'exec'}",
                    "--data-root=/docker-data",
                    "--storage-driver=fuse-overlayfs",
                ],
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )

        self.env = {
            **os.environ,
            "DOCKER_HOST": f"unix://{spec.docker_socket}",
            "HARBOR_TELEMETRY": "0",
        }
        for _ in range(120):
            if self._docker_ready():
                break
            if self.process.poll() is not None:
                raise RuntimeError(
                    f"nested dockerd exited during startup; see {spec.dockerd_log_path}"
                )
            time.sleep(0.5)
        else:
            raise RuntimeError(
                f"nested dockerd did not become ready; see {spec.dockerd_log_path}"
            )

        driver = self._docker_output("info", "--format", "{{.Driver}}")
        if driver != "fuse-overlayfs":
            raise RuntimeError(
                f"nested dockerd uses {driver!r}, expected 'fuse-overlayfs'"
            )
        return driver

    def run_harbor(self, spec: RunSpec) -> tuple[int, bool]:
        spec.harbor_log_path.parent.mkdir(parents=True, exist_ok=True)
        timed_out = False
        harbor_env = {
            key: value for key, value in self.env.items() if key not in MODEL_ENV_NAMES
        }
        harbor_env.update(model_connection_env(spec))
        with spec.harbor_log_path.open("ab") as log:
            self.harbor_process = subprocess.Popen(
                harbor_command(spec),
                env=harbor_env,
                cwd=spec.episode_dir,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                return_code = self.harbor_process.wait(timeout=spec.timeout_s)
            except subprocess.TimeoutExpired:
                timed_out = True
                self._stop(self.harbor_process)
                return_code = 124
            finally:
                self.harbor_process = None
        return return_code, timed_out

    def cleanup(self) -> None:
        if self.harbor_process is not None:
            self._stop(self.harbor_process)
            self.harbor_process = None
        if self.process is not None:
            self._stop(self.process)
            self.process = None

    def _docker_ready(self) -> bool:
        if not self.env:
            return False
        try:
            return (
                subprocess.run(
                    ["docker", "info"],
                    env=self.env,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                    timeout=5,
                ).returncode
                == 0
            )
        except (OSError, subprocess.TimeoutExpired):
            return False

    def _docker_output(self, *args: str) -> str:
        completed = subprocess.run(
            ["docker", *args],
            env=self.env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=20,
        )
        if completed.returncode != 0:
            raise RuntimeError(completed.stdout.strip() or "docker command failed")
        return completed.stdout.strip()

    @staticmethod
    def _stop(process: subprocess.Popen[bytes]) -> None:
        if process.poll() is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass


def read_request() -> dict[str, Any]:
    raw = os.environ.get(REQUEST_ENV, "").strip()
    if not raw:
        raise RuntimeError(f"{REQUEST_ENV} is required")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise TypeError(f"{REQUEST_ENV} must contain a JSON object")
    return value


def resolve_run_spec(request: dict[str, Any]) -> RunSpec:
    """Parse the small request subset needed to run exactly one Harbor trial."""
    params = request.get("env_params")
    params = params if isinstance(params, dict) else {}
    dataset = params.get("dataset")
    dataset = dataset if isinstance(dataset, dict) else {}

    session_id = _required_text(request.get("session_id"), "session_id")
    task_path = Path(
        _required_text(
            _param(dataset, params, "task_path", "/tmp/safactory-harbor-task"),
            "task_path",
        )
    ).resolve()
    if not task_path.is_dir():
        raise RuntimeError(f"Harbor task directory does not exist: {task_path}")

    gateway_url = (
        os.environ.get("SAFACTORY_GATEWAY_SESSION_URL_CONTAINER")
        or os.environ.get("SAFACTORY_GATEWAY_SESSION_URL")
        or f"{_required_text(request.get('gateway_base_url'), 'gateway_base_url').rstrip('/')}/{session_id}"
    ).rstrip("/")
    parsed_gateway = urlsplit(gateway_url)
    if parsed_gateway.scheme not in {"http", "https"} or not parsed_gateway.hostname:
        raise RuntimeError(f"invalid SAfactory gateway URL: {gateway_url!r}")

    agent = _required_text(_param(dataset, params, "agent", "oracle"), "agent")
    model = str(
        _param(dataset, params, "model", request.get("model") or "") or ""
    ).strip()
    if agent in {"oracle", "nop"}:
        model = ""
    reward_key = _required_text(
        _param(dataset, params, "reward_key", "reward"), "reward_key"
    )
    timeout_s = int(_param(dataset, params, "timeout_s", 900))
    if timeout_s <= 0:
        raise RuntimeError("timeout_s must be positive")

    result_path_text = os.environ.get(RESULT_PATH_ENV, "").strip()
    if not result_path_text:
        raise RuntimeError(f"{RESULT_PATH_ENV} is required")
    result_path = Path(result_path_text)
    job_name = ("safactory-" + _safe_name(session_id).lower())[:63].strip("-")
    return RunSpec(
        session_id=session_id,
        task_id=str(dataset.get("task_id") or task_path.name).strip() or task_path.name,
        task_path=task_path,
        gateway_url=gateway_url,
        agent=agent,
        model=model or None,
        reward_key=reward_key,
        timeout_s=timeout_s,
        result_path=result_path,
        jobs_root=result_path.parent / "harbor" / "jobs",
        harbor_job_name=job_name,
    )


def harbor_command(spec: RunSpec) -> list[str]:
    command = [
        HARBOR_BIN,
        "run",
        "--path",
        str(spec.task_path),
        "--agent",
        spec.agent,
    ]
    if spec.model:
        command.extend(["--model", spec.model])
    command.extend(
        [
            "--env",
            "docker",
            "--n-attempts",
            "1",
            "--n-concurrent",
            "1",
            "--jobs-dir",
            str(spec.jobs_root),
            "--job-name",
            spec.harbor_job_name,
            "--quiet",
        ]
    )
    return command


def model_connection_env(spec: RunSpec) -> dict[str, str]:
    if spec.agent == "claude-code":
        return {
            "ANTHROPIC_BASE_URL": spec.gateway_url,
            "ANTHROPIC_API_KEY": "EMPTY",
        }
    if spec.agent in {"codex", "opencode"}:
        return {
            "OPENAI_BASE_URL": spec.gateway_url,
            "OPENAI_API_KEY": "EMPTY",
        }
    return {}


def parse_harbor_result(
    spec: RunSpec,
    *,
    return_code: int,
    timed_out: bool,
    docker_driver: str,
    duration_ms: float,
) -> dict[str, Any]:
    job_result_path = spec.harbor_job_dir / "result.json"
    if not job_result_path.is_file():
        raise RuntimeError(
            f"Harbor did not write {job_result_path}; return_code={return_code}; "
            f"log_tail={_tail(spec.harbor_log_path)}"
        )
    _read_object(job_result_path)

    candidates = sorted(spec.harbor_job_dir.glob("*/result.json"))
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected exactly one Harbor trial result under {spec.harbor_job_dir}, "
            f"got {len(candidates)}: {[str(path) for path in candidates]}"
        )
    trial_result_path = candidates[0]
    trial_dir = trial_result_path.parent
    trial = _read_object(trial_result_path)
    verifier = trial.get("verifier_result")
    verifier = verifier if isinstance(verifier, dict) else {}
    rewards = verifier.get("rewards")
    rewards = rewards if isinstance(rewards, dict) else {}
    raw_reward = rewards.get(spec.reward_key)
    reward = _numeric_reward(raw_reward)

    errors = _trial_errors(trial)
    if return_code != 0:
        errors.append(f"harbor process exited with status {return_code}")
    if timed_out:
        errors.append(f"harbor run timed out after {spec.timeout_s}s")
    if reward is None:
        errors.append(
            f"verifier did not produce numeric reward {spec.reward_key!r}; "
            f"available rewards={sorted(rewards)}"
        )

    trajectories = sorted(
        str(path)
        for path in trial_dir.rglob("*trajectory*.json*")
        if path.is_file()
    )
    succeeded = not errors
    metrics = {
        "bench": "harbor",
        "task_id": spec.task_id,
        "task_path": str(spec.task_path),
        "harbor_agent": spec.agent,
        "harbor_model": spec.model,
        "reward_key": spec.reward_key,
        "harbor_reward": reward,
        "harbor_rewards": rewards,
        "harbor_errors": errors,
        "harbor_return_code": return_code,
        "harbor_job_result_path": str(job_result_path),
        "harbor_trial_result_path": str(trial_result_path),
        "harbor_log_path": str(spec.harbor_log_path),
        "dockerd_log_path": str(spec.dockerd_log_path),
        "trajectory_paths": trajectories,
        "docker_driver": docker_driver,
        "duration_ms": round(duration_ms, 3),
    }
    return {
        "session_id": spec.session_id,
        "status": "succeeded" if succeeded else "failed",
        "total_reward": reward if succeeded and reward is not None else 0.0,
        "step_count": 1,
        "terminated": not timed_out,
        "truncated": timed_out,
        "error_text": "; ".join(errors) if errors else None,
        "metrics": metrics,
    }


def _trial_errors(trial: dict[str, Any]) -> list[str]:
    errors: list[str] = []

    def add(value: Any, location: str) -> None:
        if not isinstance(value, dict):
            return
        kind = str(value.get("exception_type") or "Exception")
        message = str(value.get("exception_message") or "").strip()
        errors.append(f"{location}: {kind}: {message}".rstrip())

    add(trial.get("exception_info"), "trial")
    steps = trial.get("step_results")
    if isinstance(steps, list):
        for index, step in enumerate(steps):
            if isinstance(step, dict):
                add(step.get("exception_info"), f"step {step.get('step_name') or index}")
    return errors


def _failure(
    session_id: str,
    error: BaseException,
    *,
    started: float,
    spec: RunSpec | None,
    timed_out: bool,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "bench": "harbor",
        "harbor_errors": [str(error)],
        "duration_ms": round((time.perf_counter() - started) * 1000, 3),
    }
    if spec is not None:
        metrics.update(
            {
                "task_id": spec.task_id,
                "task_path": str(spec.task_path),
                "harbor_agent": spec.agent,
                "harbor_model": spec.model,
                "reward_key": spec.reward_key,
                "harbor_job_result_path": str(spec.harbor_job_dir / "result.json"),
                "harbor_log_path": str(spec.harbor_log_path),
                "dockerd_log_path": str(spec.dockerd_log_path),
                "trajectory_paths": [],
            }
        )
    return {
        "session_id": session_id,
        "status": "failed",
        "total_reward": 0.0,
        "step_count": 0,
        "terminated": not timed_out,
        "truncated": timed_out,
        "error_text": str(error),
        "metrics": metrics,
    }


def _param(dataset: dict[str, Any], params: dict[str, Any], name: str, default: Any) -> Any:
    if dataset.get(name) is not None:
        return dataset[name]
    if params.get(name) is not None:
        return params[name]
    return default


def _required_text(value: Any, name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise RuntimeError(f"SimulationStartRequest missing {name}")
    return text


def _safe_name(value: str) -> str:
    text = "".join(
        character if character.isalnum() or character in "-_" else "-"
        for character in value
    ).strip("-_")
    return text or "episode"


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON result is not an object: {path}")
    return value


def _numeric_reward(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _tail(path: Path, limit: int = 2000) -> str:
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")[-limit:].strip()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    started = time.perf_counter()
    nested = NestedDocker()
    spec: RunSpec | None = None
    session_id = os.environ.get("SAFACTORY_SESSION_ID", "")
    result_path_text = os.environ.get(RESULT_PATH_ENV, "").strip()
    result_path = Path(result_path_text) if result_path_text else None
    timed_out = False

    def stop(signum: int, _frame: Any) -> None:
        nested.cleanup()
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        spec = resolve_run_spec(read_request())
        session_id = spec.session_id
        result_path = spec.result_path
        spec.episode_dir.mkdir(parents=True, exist_ok=True)
        docker_driver = nested.start(spec)
        return_code, timed_out = nested.run_harbor(spec)
        result = parse_harbor_result(
            spec,
            return_code=return_code,
            timed_out=timed_out,
            docker_driver=docker_driver,
            duration_ms=(time.perf_counter() - started) * 1000,
        )
    except Exception as error:
        result = _failure(
            session_id,
            error,
            started=started,
            spec=spec,
            timed_out=timed_out,
        )
    finally:
        nested.cleanup()

    if result_path is not None:
        _write_json(result_path, result)
    print(RESULT_JSON_PREFIX + json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
