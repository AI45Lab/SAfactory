#!/usr/bin/env python3
"""Run one Harbor trial and emit one SAfactory SimulationStartResult."""

from __future__ import annotations

import contextlib
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REQUEST_ENV = "SAFACTORY_START_REQUEST_JSON"
RESULT_PATH_ENV = "SAFACTORY_RESULT_PATH"
RESULT_JSON_PREFIX = "SAFACTORY_RESULT_JSON "
HARBOR_BIN = "/opt/harbor-env/bin/harbor"
RUNTIME_DIR = Path("/tmp/safactory-harbor")
CODEX_CONFIG_TARGET = "/opt/safactory-codex-config"
# Profile subdirectory under <bundle_root>/<task_id>/; configurable per run so an
# upstream layout change does not require touching the runner.
DEFAULT_PROFILE_SUBDIR = "harbor-runtime-profile"
# Shared-image directory under <bundle_root>/, passed to load_shared_images.py.
# Upstream bundles disagree on the name (_runtime vs shared-images), so it is
# configurable; the default matches the original self-describing layout.
DEFAULT_SHARED_SUBDIR = "_runtime"
# The bundle ships its own loaders; the runner delegates image loading to them.
BUNDLE_LOADERS = ("load_shared_images.py", "load_image.py")
CANCELLATION_GRACE_S = 30.0
CANCELLATION_POLL_S = 1.0
LOAD_TIMEOUT_S = 1800
INSPECT_TIMEOUT_S = 20
MODEL_ENV_NAMES = {
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_BASE_URL",
    "CLAUDE_CODE_OAUTH_TOKEN",
    "OPENAI_API_BASE",
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "LLM_API_KEY",
    "LLM_BASE_URL",
}
AGENT_RUNTIME_TARGETS = {
    "codex": "/usr/local/bin/codex",
    "claude-code": "/usr/local/bin/claude",
}
MODEL_ENV_PREFIXES = {
    "claude-code": "ANTHROPIC",
    "codex": "OPENAI",
    "opencode": "OPENAI",
    "openhands": "LLM",
}
OPENCODE_CHAT_AGENT_IMPORT_PATHS = {
    "agents.offline_agents:OfflineOpenCode",
    "agents.tracing_agents:TracingOfflineOpenCode",
}


@dataclass(frozen=True)
class TaskRecord:
    task_id: str
    task_path: Path
    bundle_root: Path


def _has_controls(text: str) -> bool:
    return any(unicodedata.category(char) in {"Cc", "Cs"} for char in text)


def _clean_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a non-empty string without outer whitespace")
    if _has_controls(value):
        raise ValueError(f"{name} must be valid UTF-8 without control characters")
    return value


def _path_segment(value: Any, name: str) -> str:
    """A single safe path segment: no separators, no traversal, no option-leading '-'."""
    text = _clean_text(value, name)
    if "/" in text or "\\" in text or text in {".", ".."} or text.startswith("-"):
        raise ValueError(f"{name} must be a single safe path segment")
    return text


def _runtime_env(value: Any) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, dict):
        raise ValueError("runtime_env must be an object")
    reserved = MODEL_ENV_NAMES | {"PATH", "HOME", "BASH_ENV", "ENV"}
    # HARBOR_* is allowed so tasks whose Harbor verifier reads a public oracle
    # (e.g. HARBOR_EXPECTED_SECRET for propentbench live-flag capture) can pass
    # it through runtime_env; SAFACTORY_/DOCKER_/LD_/PYTHON remain reserved.
    prefixes = ("DOCKER_", "SAFACTORY_", "LD_", "PYTHON")
    for name, item in value.items():
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            raise ValueError("runtime_env keys must be environment variable names")
        if name in reserved or name.startswith(prefixes):
            raise ValueError(f"runtime_env cannot override reserved variable {name}")
        if not isinstance(item, str) or "\x00" in item:
            raise ValueError(f"runtime_env.{name} must be a string without NUL")
    return tuple(sorted(value.items()))


def resolve_task_record(
    task_id: str,
    bundle_root: Path,
    profile_subdir: str,
) -> TaskRecord:
    """Locate one task by convention: <bundle_root>/<task_id>/<profile_subdir>."""
    task_id = _path_segment(task_id, "dataset.task_id")
    profile_subdir = _path_segment(profile_subdir, "profile_subdir")
    if not bundle_root.is_absolute():
        raise ValueError("bundle_root must be absolute")
    try:
        root = bundle_root.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise ValueError(f"bundle_root does not exist: {bundle_root}") from error
    if root == Path(root.anchor) or not root.is_dir():
        raise ValueError("bundle_root must be a directory other than filesystem root")
    for name in BUNDLE_LOADERS:
        if not (root / "bin" / name).is_file():
            raise ValueError(f"bundle_root missing loader: bin/{name}")
    # task_id/profile_subdir are single safe segments, so the path stays inside
    # bundle_root syntactically. The directory may still be a symlink pointing
    # outside (e.g. a symlink-farm bundle whose tasks link to a real upstream
    # delivery); such escapes are permitted on purpose.
    task_path = root / task_id / profile_subdir
    return TaskRecord(task_id=task_id, task_path=task_path, bundle_root=root)


def validate_task_record(record: TaskRecord) -> None:
    if not record.task_path.is_dir():
        raise ValueError("task_path must be a directory")
    for name in ("task.toml", "instruction.md", "tests/test.sh"):
        if not (record.task_path / name).is_file():
            raise ValueError(f"task missing required file: {name}")
    if not (record.task_path / "environment").is_dir():
        raise ValueError("task missing environment directory")


def _docker_env(env: dict[str, str]) -> dict[str, str]:
    # CLI contexts and TLS options must never redirect the episode's connection.
    overrides = {
        "DOCKER_CONTEXT",
        "DOCKER_TLS",
        "DOCKER_TLS_VERIFY",
        "DOCKER_CERT_PATH",
        *MODEL_ENV_NAMES,
    }
    return {key: value for key, value in env.items() if key not in overrides}


def _docker_command(env: dict[str, str], *args: str) -> list[str]:
    host = env.get("DOCKER_HOST", "")
    if not host.startswith("unix:///"):
        raise ValueError("nested Docker requires an explicit Unix DOCKER_HOST")
    return ["docker", f"--host={host}", *args]


def _docker(env: dict[str, str], *args: str, timeout: int = INSPECT_TIMEOUT_S) -> str:
    completed = subprocess.run(
        _docker_command(env, *args),
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
        timeout=timeout,
    )
    if completed.returncode:
        raise RuntimeError(
            f"docker {args[0]} failed: {completed.stdout[-2000:].strip()}"
        )
    return completed.stdout.strip()


def load_task_images(spec: RunSpec, docker_env: dict[str, str]) -> None:
    """Delegate image loading to the bundle's own loader scripts.

    The bundle ships bin/load_shared_images.py (shared <bundle>/_runtime archives)
    and bin/load_image.py (per-task <bundle>/<task>/images archives); they handle
    gzip streaming, pre.tar/post.tar inner archives, and overlay max-depth
    flatten-import. The runner only wires them to this episode's daemon via
    DOCKER_HOST (already set in docker_env) and runs them isolated (-I) because the
    bundle is untrusted data.
    """
    loader_args = {
        "load_shared_images.py": [
            "--runtime-dir",
            str(spec.bundle_root / spec.shared_images_subdir),
        ],
        "load_image.py": ["--root", str(spec.bundle_root), "--task", spec.task_id],
    }
    for name in loader_args:
        if not (spec.bundle_root / "bin" / name).is_file():
            raise ValueError(f"bundle loader missing: bin/{name}")
    spec.image_load_log_path.parent.mkdir(parents=True, exist_ok=True)
    with spec.image_load_log_path.open("ab") as log:
        for name, args in loader_args.items():
            script = spec.bundle_root / "bin" / name
            completed = subprocess.run(
                [sys.executable, "-I", str(script), *args],
                env=docker_env,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
                timeout=LOAD_TIMEOUT_S,
            )
            if completed.returncode:
                raise RuntimeError(
                    f"{name} failed with status {completed.returncode}; "
                    f"see {spec.image_load_log_path}"
                )


@dataclass(frozen=True)
class AgentRuntime:
    source: Path
    version: str
    target: str


@dataclass(frozen=True)
class RunSpec:
    session_id: str
    task_record: TaskRecord
    gateway_url: str
    agent: str
    model: str | None
    reasoning_effort: str | None
    reward_key: str
    timeout_s: int
    result_path: Path
    jobs_root: Path
    harbor_job_name: str
    agent_import_path: str | None = None
    capture_http: bool = False
    agent_runtime: AgentRuntime | None = None
    codex_config_dir: Path | None = None
    agent_env: dict[str, str] = field(default_factory=dict)
    runtime_env: tuple[tuple[str, str], ...] = ()
    shared_images_subdir: str = DEFAULT_SHARED_SUBDIR

    @property
    def task_id(self) -> str:
        return self.task_record.task_id

    @property
    def task_path(self) -> Path:
        return self.task_record.task_path

    @property
    def bundle_root(self) -> Path:
        return self.task_record.bundle_root

    @property
    def preparation_path(self) -> Path:
        return self.episode_dir / "infra" / "preparation.json"

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
    def image_load_log_path(self) -> Path:
        return self.episode_dir / "infra" / "image-load.log"

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
            **_docker_env(dict(os.environ)),
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

        driver = _docker(self.env, "info", "--format", "{{.Driver}}")
        if driver != "fuse-overlayfs":
            raise RuntimeError(
                f"nested dockerd uses {driver!r}, expected 'fuse-overlayfs'"
            )
        return driver

    def run_harbor(self, spec: RunSpec) -> tuple[int, bool]:
        spec.harbor_log_path.parent.mkdir(parents=True, exist_ok=True)
        timed_out = False
        # self.env is already stripped of inherited model credentials by _docker_env.
        harbor_env = {
            **self.env,
            **model_connection_env(spec),
            **dict(spec.runtime_env),
        }
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
        try:
            _docker(self.env, "info", timeout=5)
        except (OSError, RuntimeError, subprocess.TimeoutExpired):
            return False
        return True

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
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            with contextlib.suppress(subprocess.TimeoutExpired):
                process.wait(timeout=5)


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
    params = _dict(request.get("env_params"))
    dataset = _dict(params.get("dataset"))

    session_id = _required_text(request.get("session_id"), "session_id")
    task_id = _path_segment(dataset.get("task_id"), "dataset.task_id")
    bundle_root = Path(_clean_text(params.get("bundle_root"), "bundle_root"))
    if not bundle_root.is_absolute():
        raise ValueError("bundle_root must be absolute")
    profile_subdir = _path_segment(
        params.get("profile_subdir", DEFAULT_PROFILE_SUBDIR), "profile_subdir"
    )
    shared_images_subdir = _path_segment(
        params.get("shared_images_subdir", DEFAULT_SHARED_SUBDIR),
        "shared_images_subdir",
    )
    task_record = resolve_task_record(task_id, bundle_root, profile_subdir)
    # Optional public run-time variables for the Harbor process (no per-task
    # catalog any more); applies to every task of this run.
    runtime_env = _runtime_env(params.get("runtime_env", {}))

    # The SAfactory launcher/cluster code has already resolved and containerized this URL.
    # Keep only a manual-run fallback; URL validation belongs to the launcher.
    gateway_url = (
        os.environ.get("SAFACTORY_GATEWAY_SESSION_URL_CONTAINER")
        or os.environ.get("SAFACTORY_GATEWAY_SESSION_URL")
        or f"{str(request.get('gateway_base_url') or '').rstrip('/')}/{session_id}"
    ).rstrip("/")

    agent = _optional_text(params.get("agent")) or "oracle"
    reasoning_effort = _optional_text(params.get("reasoning_effort"))
    agent_import_path = _optional_text(params.get("agent_import_path"))
    capture_http = params.get("capture_http") is True
    agent_runtime = _agent_runtime(params.get("agent_runtime"), agent)
    codex_config_dir = _codex_config_dir(params.get("codex_config_dir"), agent)
    agent_env = _dict(params.get("agent_env"))
    model = str(params.get("model", request.get("model") or "") or "").strip()
    if agent in {"oracle", "nop"}:
        model = ""
    reward_key = _required_text(params.get("reward_key", "reward"), "reward_key")
    timeout_s = int(params.get("timeout_s", 900))
    if timeout_s <= 0:
        raise RuntimeError("timeout_s must be positive")
    result_path_text = os.environ.get(RESULT_PATH_ENV, "").strip()
    if not result_path_text:
        raise RuntimeError(f"{RESULT_PATH_ENV} is required")
    result_path = Path(result_path_text)
    job_name = ("safactory-" + _safe_name(session_id).lower())[:63].strip("-")
    return RunSpec(
        session_id=session_id,
        task_record=task_record,
        gateway_url=gateway_url,
        agent=agent,
        model=model or None,
        reasoning_effort=reasoning_effort,
        reward_key=reward_key,
        timeout_s=timeout_s,
        result_path=result_path,
        jobs_root=result_path.parent / "harbor" / "jobs",
        harbor_job_name=job_name,
        agent_import_path=agent_import_path,
        capture_http=capture_http,
        agent_runtime=agent_runtime,
        codex_config_dir=codex_config_dir,
        agent_env=agent_env,
        runtime_env=runtime_env,
        shared_images_subdir=shared_images_subdir,
    )


def harbor_command(spec: RunSpec) -> list[str]:
    command = [
        HARBOR_BIN,
        "run",
        "--path",
        str(spec.task_path),
        "--agent",
        spec.agent_import_path or spec.agent,
    ]
    if spec.model:
        command.extend(["--model", spec.model])
    if spec.reasoning_effort is not None:
        if spec.agent == "opencode":
            opencode_config = _opencode_reasoning_config(
                spec.model,
                spec.reasoning_effort,
                chat_compatible=spec.agent_import_path
                in OPENCODE_CHAT_AGENT_IMPORT_PATHS,
            )
            command.extend(
                [
                    "--ak",
                    "opencode_config="
                    + json.dumps(opencode_config, separators=(",", ":")),
                ]
            )
        else:
            command.extend(["--ak", f"reasoning_effort={spec.reasoning_effort}"])
    mounts = []
    if spec.agent_runtime is not None:
        mounts.append(_bind_mount(spec.agent_runtime.source, spec.agent_runtime.target))
        command.extend(["--ak", f"version={spec.agent_runtime.version}"])
    if spec.codex_config_dir is not None:
        # Harbor merges the session URL into a writable CODEX_HOME copy of this file;
        # the read-only mount only serves companion files referenced from it.
        command.extend(["--ak", f"config={spec.codex_config_dir / 'config.toml'}"])
        mounts.append(_bind_mount(spec.codex_config_dir, CODEX_CONFIG_TARGET))
    if mounts:
        command.extend(["--mounts", json.dumps(mounts, separators=(",", ":"))])
    if spec.capture_http:
        command.extend(["--ak", "capture_http=true"])
    for name, value in sorted(spec.agent_env.items()):
        command.extend(["--ae", f"{name}={value}"])
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
            "--yes",
            "--quiet",
        ]
    )
    return command


def model_connection_env(spec: RunSpec) -> dict[str, str]:
    prefix = MODEL_ENV_PREFIXES.get(spec.agent)
    if prefix is None:
        return {}
    return {f"{prefix}_BASE_URL": spec.gateway_url, f"{prefix}_API_KEY": "EMPTY"}


def wait_for_cancellation_cleanup(spec: RunSpec) -> None:
    """Wait briefly for Harbor's asynchronous cancellation writes to settle."""
    deadline = time.monotonic() + CANCELLATION_GRACE_S
    job_dir = spec.harbor_job_dir
    job_result = job_dir / "result.json"
    previous: tuple[tuple[str, int, int], ...] = ()
    while time.monotonic() < deadline:
        trajectories = [
            path for path in job_dir.rglob("*trajectory*.json*") if path.is_file()
        ]
        manifests = list(job_dir.glob("*/artifacts/manifest.json"))
        trial_results = list(job_dir.glob("*/result.json"))
        snapshot = tuple(
            (str(path), (info := path.stat()).st_size, info.st_mtime_ns)
            for path in sorted({job_result, *trial_results, *trajectories, *manifests})
            if path.is_file()
        )
        ready = (
            job_result.is_file()
            and bool(manifests)
            and bool(trajectories or trial_results)
        )
        if ready and snapshot == previous:
            return
        previous = snapshot
        time.sleep(min(CANCELLATION_POLL_S, max(0.0, deadline - time.monotonic())))


def parse_harbor_result(
    spec: RunSpec,
    *,
    return_code: int,
    timed_out: bool,
) -> dict[str, Any]:
    job_result_path = spec.harbor_job_dir / "result.json"
    try:
        job_result = _read_object(job_result_path)
    except (OSError, TypeError, ValueError):
        if not timed_out:
            raise RuntimeError(
                f"Harbor did not write job result; return_code={return_code}"
            )
        job_result = {}

    job_dir = spec.harbor_job_dir
    candidates = list(job_dir.glob("*/result.json"))
    trial: dict[str, Any] = {}
    if len(candidates) == 1:
        try:
            trial = _read_object(candidates[0])
        except (OSError, TypeError, ValueError):
            if not timed_out:
                raise
    elif candidates or not timed_out or _subdir_count(job_dir) != 1:
        # Only a timed-out lone trial may be missing its result.json.
        raise RuntimeError("expected exactly one Harbor trial result")

    rewards = _dict(_dict(trial.get("verifier_result")).get("rewards"))
    reward = _numeric_reward(rewards.get(spec.reward_key))

    fatal_errors = _trial_errors(trial)
    if timed_out:
        cancel_reason = (
            f"runner timeout after {spec.timeout_s}s; sent SIGTERM to Harbor"
        )
        cancelled_trials = _dict(job_result.get("stats")).get("n_cancelled_trials")
        if cancelled_trials:
            cancel_reason += f"; Harbor reported {cancelled_trials} cancelled trial(s)"
        fatal_errors.append(cancel_reason)
    else:
        if reward is None:
            fatal_errors.append("verifier did not produce numeric reward")
        if return_code != 0:
            fatal_errors.append(f"harbor process exited with status {return_code}")

    # Without a timeout, a missing reward always adds a fatal error.
    succeeded = not timed_out and not fatal_errors
    return _result(
        spec.session_id,
        status="truncated" if timed_out else "succeeded" if succeeded else "failed",
        reward=reward if succeeded else 0.0,
        timed_out=timed_out,
        errors=fatal_errors,
        metrics={"task_id": spec.task_id, "harbor_reward": reward},
    )


def _subdir_count(path: Path) -> int:
    return sum(child.is_dir() for child in path.iterdir()) if path.is_dir() else 0


def _result(
    session_id: str,
    *,
    status: str,
    reward: float,
    timed_out: bool,
    errors: list[str],
    metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "session_id": session_id,
        "status": status,
        "total_reward": reward,
        "step_count": 0,
        "terminated": not timed_out,
        "truncated": timed_out,
        "error_text": "; ".join(errors) or None,
        "metrics": {"bench": "harbor", **metrics, "harbor_fatal_errors": errors},
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
                add(
                    step.get("exception_info"), f"step {step.get('step_name') or index}"
                )
    return errors


def _failure(
    session_id: str,
    error: BaseException,
    *,
    task_id: str | None,
    timed_out: bool,
) -> dict[str, Any]:
    return _result(
        session_id,
        status="failed",
        reward=0.0,
        timed_out=timed_out,
        errors=[str(error)],
        metrics={} if task_id is None else {"task_id": task_id},
    )


def _preparation_metrics(spec: RunSpec) -> dict[str, Any]:
    return {
        "task_id": spec.task_id,
        "bundle_root": str(spec.bundle_root),
        "task_path": str(spec.task_path),
        "image_load_log_path": str(spec.image_load_log_path),
        "preparation_receipt_path": str(spec.preparation_path),
    }


def _agent_runtime(value: Any, agent: str) -> AgentRuntime | None:
    """Translate already-normalized start config into Harbor mount arguments."""
    source = _optional_text(_dict(value).get("source"))
    if source is None:
        return None
    return AgentRuntime(
        source=Path(source),
        version=_optional_text(value.get("version")) or "",
        target=AGENT_RUNTIME_TARGETS.get(agent, "/usr/local/bin/agent"),
    )


def _codex_config_dir(value: Any, agent: str) -> Path | None:
    if value is None:
        return None
    source = Path(_required_text(value, "codex_config_dir"))
    if agent != "codex":
        raise RuntimeError("codex_config_dir requires agent='codex'")
    if not source.is_absolute() or not (source / "config.toml").is_file():
        raise RuntimeError("codex_config_dir must be an absolute dir with config.toml")
    return source


def _bind_mount(source: Path, target: str) -> dict[str, Any]:
    return {"type": "bind", "source": str(source), "target": target, "read_only": True}


def _opencode_reasoning_config(
    model: str | None,
    reasoning_effort: str,
    *,
    chat_compatible: bool,
) -> dict[str, Any]:
    provider, _, model_id = str(model or "").strip().partition("/")
    config_provider = (
        "openai-chat" if chat_compatible and provider == "openai" else provider
    )
    config_model = f"{config_provider}/{model_id}"
    return {
        "small_model": config_model,
        "provider": {
            config_provider: {
                "models": {
                    model_id: {
                        "options": {"reasoningEffort": reasoning_effort},
                    }
                }
            }
        },
    }


def _optional_text(value: Any) -> str | None:
    return str(value or "").strip() or None


def _required_text(value: Any, name: str) -> str:
    text = _optional_text(value)
    if text is None:
        raise RuntimeError(f"SimulationStartRequest missing {name}")
    return text


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


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


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    nested = NestedDocker()
    session_id = os.environ.get("SAFACTORY_SESSION_ID", "")
    result_path_text = os.environ.get(RESULT_PATH_ENV, "").strip()
    result_path = Path(result_path_text) if result_path_text else None
    timed_out = False
    task_id: str | None = None

    def stop(signum: int, _frame: Any) -> None:
        nested.cleanup()
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        request = read_request()
        session_id = _required_text(request.get("session_id"), "session_id")
        # Parsed before resolve_run_spec so resolve-stage failures still report it.
        dataset = _dict(request.get("env_params")).get("dataset")
        if isinstance(dataset, dict):
            task_id = _path_segment(dataset.get("task_id"), "dataset.task_id")
        spec = resolve_run_spec(request)
        spec.episode_dir.mkdir(parents=True, exist_ok=True)
        validate_task_record(spec.task_record)
        nested.start(spec)
        load_task_images(spec, nested.env)
        _write_json(
            spec.preparation_path,
            {**_preparation_metrics(spec), "status": "prepared"},
        )
        return_code, timed_out = nested.run_harbor(spec)
        if timed_out:
            wait_for_cancellation_cleanup(spec)
        result = parse_harbor_result(
            spec,
            return_code=return_code,
            timed_out=timed_out,
        )
    except Exception as error:
        result = _failure(
            session_id,
            error,
            task_id=task_id,
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
