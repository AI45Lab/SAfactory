#!/usr/bin/env python3
"""Run one Harbor trial and emit one SAfactory SimulationStartResult."""

from __future__ import annotations

import json
import math
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

REQUEST_ENV = "SAFACTORY_START_REQUEST_JSON"
RESULT_PATH_ENV = "SAFACTORY_RESULT_PATH"
RESULT_JSON_PREFIX = "SAFACTORY_RESULT_JSON "
HARBOR_BIN = "/opt/harbor-env/bin/harbor"
RUNTIME_DIR = Path("/tmp/safactory-harbor")
CANCELLATION_GRACE_S = 30.0
CANCELLATION_POLL_S = 1.0
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
    bundle_package_dir: Path | None = None
    bundle_task: str | None = None
    bundle_variant: str | None = None

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

    def materialize_bundle(self, spec: RunSpec) -> RunSpec:
        if spec.bundle_package_dir is None:
            return spec
        output_dir = RUNTIME_DIR / f"bundle-{_safe_name(spec.session_id)}"
        subprocess.run(
            [
                sys.executable,
                str(spec.bundle_package_dir / "bin" / "vulhub_task.py"),
                "--package-dir",
                str(spec.bundle_package_dir),
                "materialize",
                "--output-dir",
                str(output_dir),
                "--task",
                str(spec.bundle_task),
                "--variant",
                str(spec.bundle_variant),
                "--load-images",
            ],
            env=self.env,
            check=True,
        )
        return replace(
            spec,
            task_path=output_dir / f"{spec.bundle_task}-{spec.bundle_variant}",
        )

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
    bundle_package_dir_text = str(
        _param(dataset, params, "bundle_package_dir", "") or ""
    ).strip()
    bundle_package_dir = (
        Path(bundle_package_dir_text).resolve() if bundle_package_dir_text else None
    )
    bundle_task = str(_param(dataset, params, "bundle_task", "") or "").strip()
    bundle_variant = str(
        _param(dataset, params, "bundle_variant", "") or ""
    ).strip()
    if bundle_package_dir is not None:
        if not bundle_task or bundle_variant not in {"zero-day", "one-day"}:
            raise RuntimeError(
                "bundle_task and bundle_variant=zero-day|one-day are required "
                "when bundle_package_dir is set"
            )
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
        bundle_package_dir=bundle_package_dir,
        bundle_task=bundle_task or None,
        bundle_variant=bundle_variant or None,
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


def wait_for_cancellation_cleanup(
    spec: RunSpec,
    *,
    grace_s: float = CANCELLATION_GRACE_S,
    poll_s: float = CANCELLATION_POLL_S,
) -> None:
    """Wait briefly for Harbor's asynchronous cancellation writes to settle."""
    deadline = time.monotonic() + grace_s
    previous: tuple[tuple[str, int, int], ...] = ()
    while time.monotonic() < deadline:
        job_dir = spec.harbor_job_dir
        trajectories = _trajectory_paths(job_dir)
        manifests = sorted(job_dir.glob("*/artifacts/manifest.json"))
        trial_results = sorted(job_dir.glob("*/result.json"))
        paths = [job_dir / "result.json", *trial_results]
        paths.extend([*trajectories, *manifests])
        snapshot = tuple(
            (str(path), path.stat().st_size, path.stat().st_mtime_ns)
            for path in sorted(paths)
            if path.is_file()
        )
        ready = (
            (job_dir / "result.json").is_file()
            and bool(manifests)
            and (bool(trajectories) or bool(trial_results))
        )
        if ready and snapshot == previous:
            return
        previous = snapshot
        time.sleep(min(poll_s, max(0.0, deadline - time.monotonic())))


def _trajectory_paths(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(path for path in root.rglob("*trajectory*.json*") if path.is_file())


def _trajectory_step_count(paths: list[Path]) -> int:
    total = 0
    for path in paths:
        try:
            trajectory = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError):
            continue
        if isinstance(trajectory, dict):
            steps = trajectory.get("steps")
            if isinstance(steps, list):
                total += len(steps)
        elif isinstance(trajectory, list):
            total += len(trajectory)
    return total


def parse_harbor_result(
    spec: RunSpec,
    *,
    return_code: int,
    timed_out: bool,
    docker_driver: str,
    duration_ms: float,
) -> dict[str, Any]:
    job_result_path = spec.harbor_job_dir / "result.json"
    try:
        job_result = _read_object(job_result_path)
    except (OSError, TypeError, ValueError):
        if not timed_out:
            raise RuntimeError(
                f"Harbor did not write {job_result_path}; return_code={return_code}; "
                f"log_tail={_tail(spec.harbor_log_path)}"
            )
        job_result = {}

    candidates = sorted(spec.harbor_job_dir.glob("*/result.json"))
    if len(candidates) == 1:
        trial_result_path: Path | None = candidates[0]
        trial_dir = trial_result_path.parent
    elif not candidates and timed_out:
        trial_dirs = (
            sorted(path for path in spec.harbor_job_dir.iterdir() if path.is_dir())
            if spec.harbor_job_dir.is_dir()
            else []
        )
        if len(trial_dirs) != 1:
            raise RuntimeError(
                f"expected exactly one Harbor trial under {spec.harbor_job_dir}, "
                f"got {len(trial_dirs)}: {[str(path) for path in trial_dirs]}"
            )
        trial_dir = trial_dirs[0]
        trial_result_path = None
    else:
        raise RuntimeError(
            f"expected exactly one Harbor trial result under {spec.harbor_job_dir}, "
            f"got {len(candidates)}: {[str(path) for path in candidates]}"
        )

    trial: dict[str, Any] = {}
    if trial_result_path is not None:
        try:
            trial = _read_object(trial_result_path)
        except (OSError, TypeError, ValueError):
            if not timed_out:
                raise

    verifier = trial.get("verifier_result")
    verifier = verifier if isinstance(verifier, dict) else {}
    rewards = verifier.get("rewards")
    rewards = rewards if isinstance(rewards, dict) else {}
    raw_reward = rewards.get(spec.reward_key)
    reward = _numeric_reward(raw_reward)

    errors, agent_timed_out = _trial_errors(trial)
    if return_code != 0 and not timed_out:
        errors.append(f"harbor process exited with status {return_code}")
    cancel_reason: str | None = None
    if timed_out:
        cancel_reason = (
            f"runner timeout after {spec.timeout_s}s; sent SIGTERM to Harbor"
        )
        stats = job_result.get("stats")
        stats = stats if isinstance(stats, dict) else {}
        cancelled_trials = stats.get("n_cancelled_trials")
        if cancelled_trials:
            cancel_reason += f"; Harbor reported {cancelled_trials} cancelled trial(s)"
        errors.append(cancel_reason)
    if reward is None and not timed_out:
        errors.append(
            f"verifier did not produce numeric reward {spec.reward_key!r}; "
            f"available rewards={sorted(rewards)}"
        )

    trajectory_paths = _trajectory_paths(trial_dir)
    trajectories = [str(path) for path in trajectory_paths]
    step_count = _trajectory_step_count(trajectory_paths)
    truncated = timed_out or agent_timed_out
    succeeded = not errors and not truncated
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
        "harbor_trial_result_path": (
            str(trial_result_path) if trial_result_path is not None else None
        ),
        "harbor_cancel_reason": cancel_reason,
        "harbor_log_path": str(spec.harbor_log_path),
        "dockerd_log_path": str(spec.dockerd_log_path),
        "trajectory_paths": trajectories,
        "docker_driver": docker_driver,
        "duration_ms": round(duration_ms, 3),
    }
    return {
        "session_id": spec.session_id,
        "status": (
            "truncated" if truncated else ("succeeded" if succeeded else "failed")
        ),
        "total_reward": reward if succeeded and reward is not None else 0.0,
        "step_count": step_count,
        "terminated": not truncated,
        "truncated": truncated,
        "error_text": "; ".join(errors) if errors else None,
        "metrics": metrics,
    }


def _trial_errors(trial: dict[str, Any]) -> tuple[list[str], bool]:
    errors: list[str] = []
    agent_timed_out = False

    def add(value: Any, location: str) -> None:
        nonlocal agent_timed_out
        if not isinstance(value, dict):
            return
        kind = str(value.get("exception_type") or "Exception")
        agent_timed_out = agent_timed_out or kind == "AgentTimeoutError"
        message = str(value.get("exception_message") or "").strip()
        errors.append(f"{location}: {kind}: {message}".rstrip())

    add(trial.get("exception_info"), "trial")
    steps = trial.get("step_results")
    if isinstance(steps, list):
        for index, step in enumerate(steps):
            if isinstance(step, dict):
                add(step.get("exception_info"), f"step {step.get('step_name') or index}")
    return errors, agent_timed_out


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
        spec = nested.materialize_bundle(spec)
        return_code, timed_out = nested.run_harbor(spec)
        if timed_out:
            wait_for_cancellation_cleanup(spec)
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
