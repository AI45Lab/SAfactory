#!/usr/bin/env python3
"""Run one Harbor trial and emit one SAfactory SimulationStartResult."""

from __future__ import annotations

import contextlib
import json
import math
import os
import re
import signal
import shutil
import subprocess
import sys
import time
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

REQUEST_ENV = "SAFACTORY_START_REQUEST_JSON"
RESULT_PATH_ENV = "SAFACTORY_RESULT_PATH"
RESULT_JSON_PREFIX = "SAFACTORY_RESULT_JSON "
HARBOR_BIN = "/opt/harbor-env/bin/harbor"
# Harbor job config template: the static half of `harbor run` lives in YAML so
# a new passthrough option is a template edit, not a runner change.
TEMPLATE_ENV = "SAFACTORY_HARBOR_TEMPLATE"
TEMPLATE_SUBDIR = "templates"
DEFAULT_TEMPLATE_NAME = "harbor_template.yaml"
# env_params keys that used to become `harbor run` flags. They now belong to
# the job template; keeping them accepted would silently configure nothing.
RETIRED_AGENT_PARAMS = frozenset(
    {
        "agent",
        "agent_import_path",
        "model",
        "reasoning_effort",
        "capture_http",
        "agent_runtime",
        "codex_config_dir",
        "agent_env",
        "agent_timeout_multiplier",
    }
)
# Template lookup order when no explicit path is configured: the copy shipped
# in this file's templates/ directory (the layout used by the runtime image),
# then the image-wide path used by batch start configs that embed it elsewhere.
TEMPLATE_FALLBACKS = (
    Path("/opt/safactory-harbor") / TEMPLATE_SUBDIR / DEFAULT_TEMPLATE_NAME,
)
RUNTIME_DIR = Path("/tmp/safactory-harbor")
DOCKER_MODE_ENV = "SAFACTORY_DOCKER_MODE"
DEFAULT_DOCKER_MODE = "rjob"
DOCKER_MODES = {"docker", "rjob", "sandbox"}
RJOB_DOCKER_SOCKET = RUNTIME_DIR / "docker.sock"
SANDBOX_DOCKER_SOCKET = Path("/var/run/docker.sock")
# Profile subdirectory under <bundle_root>/<task_id>/; configurable per run so an
# upstream layout change does not require touching the runner.
DEFAULT_PROFILE_SUBDIR = "harbor-runtime-profile"
# Shared-image directory under <bundle_root>/, passed to load_shared_images.py.
# Upstream bundles disagree on the name (_runtime vs shared-images), so it is
# configurable; the default matches the original self-describing layout.
DEFAULT_SHARED_SUBDIR = "_runtime"
# The bundle ships its own loaders; the runner delegates image loading to them.
# image_load_mode picks which of them run:
#   legacy - bin/load_shared_images.py (whole <shared_images_subdir>) then
#            bin/load_image.py; both scripts are required (original layout).
#   task   - only bin/load_image.py --task <task_id> --root <bundle_root>; the
#            bundle's single loader resolves everything the task needs (e.g. the
#            merged PJ collections, whose load_image.py dispatches per task to
#            the source batch loader).
IMAGE_LOAD_MODES = {
    "legacy": ("load_shared_images.py", "load_image.py"),
    "task": ("load_image.py",),
}
DEFAULT_IMAGE_LOAD_MODE = "legacy"
BUNDLE_LOADERS = IMAGE_LOAD_MODES[DEFAULT_IMAGE_LOAD_MODE]
CANCELLATION_GRACE_S = 30.0
CANCELLATION_POLL_S = 1.0
LOAD_TIMEOUT_S = 1800
INSPECT_TIMEOUT_S = 20
# Agent CLI output is stream-json: only the end of the file describes how the
# agent stopped. Reading a fixed 64 KiB tail keeps memory bounded on runs whose
# transcript reaches tens of MB (the design measured a 4.1 KB p99 for one row).
TAIL_BYTES = 64 * 1024
TAIL_LINES = 20
# A reward file written at (or before) verifier.started_at was produced by the
# agent, not by this verifier run; 2 s absorbs sub-second bind-mount clock skew.
REWARD_MTIME_TOLERANCE_S = 2.0
AGENT_LOG_NAMES = {
    "claude-code": "claude-code.txt",
    "codex": "codex.txt",
    # OpenCode has no exit parser; its log is still the proof that the agent
    # phase began when a runner timeout leaves no result.json (R2 vs R3).
    "opencode": "opencode.txt",
}
# Trial exception types that belong to the verifier phase. When the agent
# already raised, Harbor keeps only the first exception, so a missing verifier
# exception must not be read as "the agent ended badly".
VERIFIER_EXCEPTION_TYPES = frozenset(
    {
        "VerifierTimeoutError",
        "VerifierOutputParseError",
        "RewardFileNotFoundError",
        "RewardFileEmptyError",
        "AddTestsDirError",
        "DownloadVerifierDirError",
    }
)
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
MODEL_ENV_PREFIXES = {
    "claude-code": "ANTHROPIC",
    "codex": "OPENAI",
    "opencode": "OPENAI",
    "openhands": "LLM",
}


@dataclass(frozen=True)
class TaskRecord:
    task_id: str
    task_path: Path
    bundle_root: Path


@dataclass(frozen=True)
class AgentExit:
    """How the agent CLI itself says the run ended (see docs §4.3)."""

    parser: str | None
    source: str | None
    reason: str
    api_error_kind: str | None = None
    api_error_text: str | None = None
    api_error_status: int | None = None
    num_turns: int | None = None
    retrying_at_end: bool = False

    def as_metrics(self) -> dict[str, Any]:
        return {
            "parser": self.parser,
            "source": self.source,
            "reason": self.reason,
            "api_error_kind": self.api_error_kind,
            "api_error_text": self.api_error_text,
            "api_error_status": self.api_error_status,
            "num_turns": self.num_turns,
            "retrying_at_end": self.retrying_at_end,
        }


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
    loaders: tuple[str, ...] = BUNDLE_LOADERS,
) -> TaskRecord:
    """Locate one task by convention: <bundle_root>/<task_id>/<profile_subdir>.

    ``loaders`` are the bin/ scripts the selected image_load_mode will run.
    """
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
    for name in loaders:
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


def _docker_mode() -> str:
    """Resolve the outer runtime's Docker connection mode.

    ``rjob`` is the compatibility default.  Docker mode uses an already
    running daemon (normally a mounted host socket), while Sandbox mode starts
    the standard Docker-in-Sandbox daemon on ``/var/run/docker.sock``.
    """
    value = os.environ.get(DOCKER_MODE_ENV, DEFAULT_DOCKER_MODE).strip().lower()
    aliases = {"host": "docker", "dind": "rjob", "sandbox-dind": "sandbox"}
    mode = aliases.get(value, value)
    if mode not in DOCKER_MODES:
        allowed = ", ".join(sorted(DOCKER_MODES))
        raise RuntimeError(
            f"{DOCKER_MODE_ENV} must be one of {allowed}; got {value!r}"
        )
    return mode


def _docker_host_for_mode(mode: str) -> str:
    if mode == "rjob":
        return f"unix://{RJOB_DOCKER_SOCKET}"
    if mode == "sandbox":
        return f"unix://{SANDBOX_DOCKER_SOCKET}"
    value = os.environ.get("DOCKER_HOST", "").strip()
    return value or f"unix://{SANDBOX_DOCKER_SOCKET}"


def _docker_command(env: dict[str, str], *args: str) -> list[str]:
    host = env.get("DOCKER_HOST", "")
    if not host.startswith("unix:///"):
        raise ValueError("nested Docker requires an explicit Unix DOCKER_HOST")
    return ["docker", *args]


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

    legacy mode runs bin/load_shared_images.py (shared <bundle>/_runtime archives)
    and then bin/load_image.py (per-task archives); task mode runs only
    bin/load_image.py, which must load every image the task needs. The scripts
    handle gzip streaming, pre.tar/post.tar inner archives, and overlay max-depth
    flatten-import. The runner only wires them to this episode's daemon via
    DOCKER_HOST (already set in docker_env) and runs them isolated (-I) because the
    bundle is untrusted data.
    """
    all_args = {
        "load_shared_images.py": [
            "--runtime-dir",
            str(spec.bundle_root / spec.shared_images_subdir),
        ],
        "load_image.py": ["--root", str(spec.bundle_root), "--task", spec.task_id],
    }
    loader_args = {
        name: all_args[name] for name in IMAGE_LOAD_MODES[spec.image_load_mode]
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
class RunSpec:
    session_id: str
    task_record: TaskRecord
    gateway_url: str
    reward_key: str
    timeout_s: int
    result_path: Path
    jobs_root: Path
    harbor_job_name: str
    template_path: Path
    runtime_env: tuple[tuple[str, str], ...] = ()
    shared_images_subdir: str = DEFAULT_SHARED_SUBDIR
    image_load_mode: str = DEFAULT_IMAGE_LOAD_MODE

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
        # Kept for callers that inspect the legacy RJob endpoint.  The actual
        # endpoint is selected by ``SAFACTORY_DOCKER_MODE`` in NestedDocker.
        return RJOB_DOCKER_SOCKET

    @property
    def dockerd_log_path(self) -> Path:
        return self.episode_dir / "infra" / "dockerd.log"

    @property
    def image_load_log_path(self) -> Path:
        return self.episode_dir / "infra" / "image-load.log"

    @property
    def harbor_log_path(self) -> Path:
        return self.episode_dir / "harbor" / "harbor-run.log"

    @property
    def harbor_config_path(self) -> Path:
        return self.episode_dir / "harbor" / "harbor-config.yaml"


@dataclass
class NestedDocker:
    process: subprocess.Popen[bytes] | None = None
    harbor_process: subprocess.Popen[bytes] | None = None
    env: dict[str, str] = field(default_factory=dict)
    mode: str = ""
    docker_host: str = ""
    driver: str = ""
    agent: str = ""

    def start(self, spec: RunSpec) -> str:
        self.mode = _docker_mode()
        self.docker_host = _docker_host_for_mode(self.mode)
        if not self.docker_host.startswith("unix:///"):
            raise RuntimeError(
                f"{DOCKER_MODE_ENV}={self.mode!r} requires a Unix DOCKER_HOST; "
                f"got {self.docker_host!r}"
            )

        self.env = {
            **_docker_env(dict(os.environ)),
            "DOCKER_HOST": self.docker_host,
            "HARBOR_TELEMETRY": "0",
        }

        # Docker mode talks to an already running daemon (normally the mounted
        # /var/run/docker.sock) and must not start or stop that daemon.
        if self.mode == "docker":
            return self._wait_ready(spec)

        if os.geteuid() != 0:
            raise RuntimeError(f"{self.mode} Docker mode requires root")
        spec.dockerd_log_path.parent.mkdir(parents=True, exist_ok=True)
        socket = Path(self.docker_host.removeprefix("unix://"))
        socket.parent.mkdir(parents=True, exist_ok=True)
        socket.unlink(missing_ok=True)

        if self.mode == "rjob":
            for binary in ("docker", "dockerd", "fuse-overlayfs"):
                if shutil.which(binary) is None:
                    raise RuntimeError(f"{binary} missing from the RJob image")
            if not Path("/dev/fuse").exists():
                raise RuntimeError(
                    "rjob Docker mode requires /dev/fuse; use privileged RJob with "
                    "brainpp.cn/fuse:1"
                )
            (RUNTIME_DIR / "exec").mkdir(parents=True, exist_ok=True)
            Path("/docker-data").mkdir(parents=True, exist_ok=True)
            dockerd_args = [
                "dockerd",
                f"--host={self.docker_host}",
                f"--pidfile={RUNTIME_DIR / 'dockerd.pid'}",
                f"--exec-root={RUNTIME_DIR / 'exec'}",
                "--data-root=/docker-data",
                "--storage-driver=fuse-overlayfs",
            ]
        else:
            # Sandbox DIND follows sandbox/05-docker-in-sandbox.md.  Sysbox
            # supplies nested-container support; storage and runtime paths are
            # selected by the Sandbox image instead of RJob-specific flags.
            for binary in ("docker", "dockerd"):
                if shutil.which(binary) is None:
                    raise RuntimeError(f"{binary} missing from the Sandbox image")
            dockerd_args = ["dockerd", f"--host={self.docker_host}"]

        with spec.dockerd_log_path.open("ab") as log:
            self.process = subprocess.Popen(
                dockerd_args,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=self.env,
                start_new_session=True,
            )
        return self._wait_ready(spec)

    def _wait_ready(self, spec: RunSpec) -> str:
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            if self.process is not None and self.process.poll() is not None:
                raise RuntimeError(
                    f"nested dockerd exited during startup; see {spec.dockerd_log_path}"
                )
            try:
                driver = _docker(self.env, "info", "--format", "{{.Driver}}", timeout=5)
                if not driver:
                    raise RuntimeError("Docker daemon returned an empty storage driver")
                _docker(self.env, "compose", "version", timeout=20)
                self.driver = driver
                return driver
            except (OSError, RuntimeError, subprocess.TimeoutExpired):
                time.sleep(1)
        tail = ""
        if spec.dockerd_log_path.is_file():
            tail = spec.dockerd_log_path.read_text(errors="replace")[-6000:]
        raise RuntimeError(
            f"Docker daemon did not become ready in {self.mode} mode; "
            f"see {spec.dockerd_log_path}\n{tail}"
        )

    def run_harbor(self, spec: RunSpec) -> tuple[int, bool]:
        spec.harbor_log_path.parent.mkdir(parents=True, exist_ok=True)
        config_path = render_job_template(spec)
        # The template declares the agent, so the credential prefix follows it
        # instead of a second, possibly divergent request field.
        self.agent = rendered_template_agent(config_path)
        timed_out = False
        # self.env is already stripped of inherited model credentials by _docker_env.
        harbor_env = {
            **self.env,
            **model_connection_env(self.agent, spec.gateway_url),
            **dict(spec.runtime_env),
        }
        with spec.harbor_log_path.open("ab") as log:
            self.harbor_process = subprocess.Popen(
                harbor_command(config_path),
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

    # These used to be turned into `harbor run` flags. They now belong to the
    # job template; accepting them here would silently configure nothing.
    retired = sorted(name for name in RETIRED_AGENT_PARAMS if name in params)
    if retired:
        raise RuntimeError(
            "env_params no longer supports "
            + ", ".join(retired)
            + "; agent settings live in the Harbor job template "
            f"({DEFAULT_TEMPLATE_NAME}). Remove them from env_params, or set "
            "'harbor_template' / "
            f"{TEMPLATE_ENV} to the template that carries them."
        )

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
    image_load_mode = _clean_text(
        params.get("image_load_mode", DEFAULT_IMAGE_LOAD_MODE), "image_load_mode"
    )
    if image_load_mode not in IMAGE_LOAD_MODES:
        raise ValueError(
            "image_load_mode must be one of: " + ", ".join(sorted(IMAGE_LOAD_MODES))
        )
    task_record = resolve_task_record(
        task_id, bundle_root, profile_subdir, IMAGE_LOAD_MODES[image_load_mode]
    )
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

    reward_key = _required_text(params.get("reward_key", "reward"), "reward_key")
    timeout_s = int(params.get("timeout_s", 900))
    if timeout_s <= 0:
        raise RuntimeError("timeout_s must be positive")
    result_path_text = os.environ.get(RESULT_PATH_ENV, "").strip()
    if not result_path_text:
        raise RuntimeError(f"{RESULT_PATH_ENV} is required")
    result_path = Path(result_path_text)
    job_name = ("safactory-" + _safe_name(session_id).lower())[:63].strip("-")

    # Auto-select OpenCode offline template if requested
    harbor_template = params.get("harbor_template")
    if not harbor_template and params.get("use_opencode_offline"):
        harbor_template = str(
            Path(__file__).resolve().parent / TEMPLATE_SUBDIR / "harbor_config.opencode-offline.yaml"
        )

    return RunSpec(
        session_id=session_id,
        task_record=task_record,
        gateway_url=gateway_url,
        reward_key=reward_key,
        timeout_s=timeout_s,
        result_path=result_path,
        jobs_root=result_path.parent / "harbor" / "jobs",
        harbor_job_name=job_name,
        template_path=resolve_template_path(harbor_template),
        runtime_env=runtime_env,
        shared_images_subdir=shared_images_subdir,
        image_load_mode=image_load_mode,
    )


def resolve_template_path(value: Any) -> Path:
    """Locate the Harbor job template.

    Order: explicit ``harbor_template`` param, then ``SAFACTORY_HARBOR_TEMPLATE``,
    then the copy shipped in this file's templates/ directory, then the
    image-wide fallback.
    """
    text = _optional_text(value) or os.environ.get(TEMPLATE_ENV, "").strip()
    if text:
        path = Path(text).expanduser()
        if not path.is_absolute():
            raise RuntimeError(f"{TEMPLATE_ENV} must be an absolute path: {path}")
        if not path.is_file():
            raise RuntimeError(f"Harbor job template not found: {path}")
        return path
    candidates = (
        Path(__file__).resolve().parent / TEMPLATE_SUBDIR / DEFAULT_TEMPLATE_NAME,
        *TEMPLATE_FALLBACKS,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    tried = ", ".join(str(item) for item in candidates)
    raise RuntimeError(f"Harbor job template not found; tried {tried}")


def render_job_template(spec: RunSpec) -> Path:
    """Fill the session-scoped placeholders and write this episode's config."""
    text = spec.template_path.read_text(encoding="utf-8")
    values = {
        "task_path": str(spec.task_path),
        "job_name": spec.harbor_job_name,
        "gateway_url": spec.gateway_url,
    }
    leftover = re.search(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}", text)
    unknown = sorted(
        {
            match.group(1)
            for match in re.finditer(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}", text)
            if match.group(1) not in values
        }
    )
    if leftover and unknown:
        raise RuntimeError(
            f"unknown placeholder(s) {', '.join(unknown)} in {spec.template_path}; "
            "supported: " + ", ".join(sorted(values))
        )
    for name, value in values.items():
        # JSON scalars are valid YAML, so quoting/escaping is handled for us.
        text = text.replace("{{" + name + "}}", json.dumps(value))
    for required in ("task_path", "job_name"):
        if json.dumps(values[required]) not in text:
            raise RuntimeError(
                f"template {spec.template_path} does not use {{{{{required}}}}}"
            )
    config_path = spec.harbor_config_path
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(text, encoding="utf-8")
    return config_path


def rendered_template_agent(config_path: Path) -> str:
    """Return the agent name the rendered config will run.

    The template is the single source of truth: the Gateway credential prefix
    must follow whichever agent the config declares, so it is read back from
    the rendered file rather than from a separate request field.
    """
    try:
        import yaml
    except ImportError as error:  # pragma: no cover - Harbor venv ships pyyaml
        raise RuntimeError("pyyaml is required to read the Harbor job config") from error
    try:
        data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise RuntimeError(f"invalid YAML in rendered config {config_path}: {error}") from error
    if not isinstance(data, dict):
        raise RuntimeError(f"rendered config {config_path} is not a mapping")
    agents = data.get("agents")
    if not isinstance(agents, list) or not agents or not isinstance(agents[0], dict):
        raise RuntimeError(f"rendered config {config_path} must declare one agent")
    name = _optional_text(agents[0].get("name"))
    if name is None:
        raise RuntimeError(
            f"rendered config {config_path} must set agents[0].name "
            "(import_path agents cannot be routed by the runner)"
        )
    return name


def harbor_command(config_path: Path) -> list[str]:
    """The whole Harbor invocation: a rendered config plus an auto-confirm.

    Everything session-independent (agent, env, kwargs, mounts, timeout
    multipliers) lives in the template. ``--yes`` only skips the host
    environment access prompt. ``--agent`` must stay absent: passing it makes
    Harbor discard the template's entire ``agents`` block.
    """
    return [HARBOR_BIN, "run", "-c", str(config_path), "--yes"]


def model_connection_env(agent: str, gateway_url: str) -> dict[str, str]:
    """Provider credentials for the Harbor process itself.

    Some adapters read the endpoint from the Harbor process environment in
    addition to ``agents[].env``, so the per-session Gateway URL is exported
    here too. ``agents[].env`` in the template stays authoritative for the
    agent container.
    """
    prefix = MODEL_ENV_PREFIXES.get(agent)
    if prefix is None:
        return {}
    return {f"{prefix}_BASE_URL": gateway_url, f"{prefix}_API_KEY": "EMPTY"}


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


# --- agent exit parsing ------------------------------------------------------


def _read_tail_lines(
    path: Path, *, tail_bytes: int = TAIL_BYTES, tail_lines: int = TAIL_LINES
) -> list[str]:
    """Return the last non-empty lines within a byte-limited tail window."""
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            start = max(0, size - tail_bytes)
            handle.seek(start)
            raw = handle.read()
    except OSError:
        return []
    lines = raw.decode("utf-8", errors="replace").splitlines()
    if start > 0 and lines:
        # The first line is likely cut mid-record; dropping it is cheaper than
        # trying to repair it and cannot lose the terminating record.
        lines = lines[1:]
    lines = [line for line in lines if line.strip()]
    return lines[-tail_lines:]


def _json_records(lines: list[str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in lines:
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if isinstance(value, dict):
            records.append(value)
    return records


def _assistant_text(record: dict[str, Any]) -> str:
    message = _dict(record.get("message"))
    content = message.get("content")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(str(block.get("text") or ""))
    return "\n".join(parts)


# Ordered: the first matching pattern names the kind. Only used for statistics.
API_ERROR_KINDS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("model_unavailable", re.compile(r"API Error: 403|Model is not available")),
    (
        "connection",
        re.compile(r"Connection refused|ECONNRESET|ETIMEDOUT|socket hang up|ConnectionRefused"),
    ),
    (
        "rate_limit",
        re.compile(r"API Error: 429|Request rejected \(429\)|status: 429|429 Too Many Requests"),
    ),
    (
        "session_closed",
        re.compile(r"API Error: 409.*is closed|Connection closed|Connection lost"),
    ),
    ("quota", re.compile(r"API Error: 402|402 Payment Required|insufficient quota")),
    ("server_error", re.compile(r"API Error: 5\d\d|\b50[023]\b")),
)


def _clip(text: str | None, limit: int = 300) -> str | None:
    """Keep a short, single-use excerpt of the matched error text."""
    value = (text or "").strip()
    return value[:limit] if value else None


def _api_error_kind(text: str | None) -> str | None:
    if not text:
        return None
    for name, pattern in API_ERROR_KINDS:
        if pattern.search(text):
            return name
    return "other"


def parse_claude_code_exit(path: Path) -> AgentExit:
    """Classify how Claude Code ended from the tail of its stream-json output."""
    lines = _read_tail_lines(path)
    records = _json_records(lines)
    terminal = None
    for record in records:
        if record.get("type") == "result":
            # A max-turns exit may only carry subtype=error_max_turns, so a
            # record is terminal if it has either field.
            if record.get("terminal_reason") or record.get("subtype"):
                terminal = record
    reason = "unknown"
    status = None
    num_turns = None
    if terminal is not None:
        raw_reason = terminal.get("terminal_reason")
        value = str(raw_reason) if raw_reason is not None else None
        subtype = str(terminal.get("subtype") or "")
        is_error = terminal.get("is_error") is True
        if value == "completed":
            reason = "completed"
        elif value == "max_turns" or subtype == "error_max_turns":
            reason = "max_turns"
        elif value == "api_error":
            reason = "api_error"
        elif value is None and subtype == "success" and not is_error:
            # Older Claude Code builds emit no ``terminal_reason``; a plain
            # subtype=success without is_error is a normal completion.
            reason = "completed"
        elif value is None:
            reason = f"other:{subtype or 'unknown'}"
        else:
            reason = f"other:{value}"
        raw_status = terminal.get("api_error_status")
        status = raw_status if isinstance(raw_status, int) and not isinstance(raw_status, bool) else None
        raw_turns = terminal.get("num_turns")
        num_turns = raw_turns if isinstance(raw_turns, int) and not isinstance(raw_turns, bool) else None

    # The API error text lives either in the result record's ``result`` field or
    # in the preceding ``model="<synthetic>"`` assistant record.
    candidates = []
    if terminal is not None:
        candidates.append(str(terminal.get("result") or ""))
    for record in records:
        if record.get("type") != "assistant":
            continue
        if _dict(record.get("message")).get("model") == "<synthetic>":
            candidates.append(_assistant_text(record))
    text = next((item for item in candidates if "API Error:" in item), None)
    if text is not None and reason not in {"completed", "max_turns"}:
        # ``terminal_reason`` wins over ``subtype``: an API-error run still
        # reports subtype=success with is_error=true.
        reason = "api_error"

    last = records[-1] if records else {}
    retrying_at_end = last.get("type") == "system" and last.get("subtype") == "api_retry"
    return AgentExit(
        parser="claude-code",
        source=str(path),
        reason=reason,
        api_error_kind=_api_error_kind(text) if reason == "api_error" else None,
        api_error_text=_clip(text) if reason == "api_error" else None,
        api_error_status=status,
        num_turns=num_turns,
        retrying_at_end=retrying_at_end,
    )


def parse_codex_exit(path: Path) -> AgentExit:
    """Classify how Codex ended from the tail of its JSON event stream."""
    records = _json_records(_read_tail_lines(path))
    reason = "unknown"
    text: str | None = None
    for record in records:
        kind = record.get("type")
        if kind == "turn.completed":
            reason, text = "completed", None
        elif kind == "turn.failed":
            reason = "api_error"
            text = str(_dict(record.get("error")).get("message") or "") or None
        elif kind == "error" and reason != "completed":
            # A bare error record is not terminal by itself, but it is the only
            # signal when the turn never closed (e.g. the process was killed).
            reason = "api_error"
            text = str(record.get("message") or "") or None
    return AgentExit(
        parser="codex",
        source=str(path),
        reason=reason,
        api_error_kind=_api_error_kind(text) if reason == "api_error" else None,
        api_error_text=_clip(text) if reason == "api_error" else None,
    )


AGENT_EXIT_PARSERS = {
    "claude-code": parse_claude_code_exit,
    "codex": parse_codex_exit,
}


def read_agent_exit(spec: RunSpec, trial_dir: Path, agent: str | None) -> tuple[AgentExit, list[str]]:
    """Resolve the agent's end reason, falling back to ``unknown`` on any error."""
    parser = AGENT_EXIT_PARSERS.get(agent or "")
    name = AGENT_LOG_NAMES.get(agent or "")
    path = trial_dir / "agent" / name if name else None
    if parser is None:
        # No parser for this agent: the reason stays unknown, but an existing
        # log still proves the agent phase began.
        source = _relative(spec, path) if path is not None and path.is_file() else None
        return AgentExit(parser=None, source=source, reason="unknown"), []
    if path is None or not path.is_file():
        return (
            AgentExit(parser=None, source=None, reason="unknown"),
            [f"agent_exit_parser: no agent log for {agent!r}"],
        )
    try:
        result = parser(path)
    except Exception as error:  # noqa: BLE001 - parsers must never fail an episode
        return (
            AgentExit(parser=None, source=None, reason="unknown"),
            [f"agent_exit_parser: {type(error).__name__}: {error}"],
        )
    if result.source is not None:
        with contextlib.suppress(ValueError):
            result = AgentExit(
                parser=result.parser,
                source=_relative(spec, path),
                reason=result.reason,
                api_error_kind=result.api_error_kind,
                api_error_text=result.api_error_text,
                api_error_status=result.api_error_status,
                num_turns=result.num_turns,
                retrying_at_end=result.retrying_at_end,
            )
    return result, []


# --- verifier evidence -------------------------------------------------------


def _parse_time(value: Any) -> float | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text).timestamp()
    except ValueError:
        return None


def _read_reward_file(path: Path, reward_key: str) -> float | None:
    """Parse a reward file the way Harbor 0.25 does, then pick ``reward_key``.

    Harbor's rules (``harbor/verifier/verifier.py``): ``reward.json`` must be an
    object whose values are all finite numbers; ``reward.txt`` must hold one
    finite float and always yields the single key ``reward``. Anything else is
    a verifier failure, so every parse problem maps to ``None`` here and is
    reported as such by :func:`valid_reward` instead of escaping as an
    exception.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, ValueError):
        # OSError: unreadable file. ValueError covers UnicodeDecodeError, which
        # Harbor reports as VerifierOutputParseError.
        return None
    if not text.strip():
        # Harbor: RewardFileEmptyError.
        return None
    rewards: dict[str, float] = {}
    if path.suffix == ".json":
        try:
            value = json.loads(text)
        except ValueError:
            return None
        if not isinstance(value, dict):
            # Harbor's VerifierResult model requires an object; a bare scalar
            # fails validation there as well.
            return None
        for key, item in value.items():
            number = _numeric_reward(item)
            # Harbor rejects the whole file if any value is not a finite number.
            if number is None:
                return None
            rewards[str(key)] = number
    else:
        try:
            number = _numeric_reward(float(text))
        except ValueError:
            return None
        if number is None:
            return None
        # A text file never carries a custom key.
        rewards["reward"] = number
    return rewards.get(reward_key)


def valid_reward(
    spec: RunSpec, trial: dict[str, Any], trial_dir: Path
) -> tuple[float | None, Path | None, str | None]:
    """Apply the five validity checks; returns (reward, file, rejection reason)."""
    started = _parse_time(_dict(trial.get("verifier")).get("started_at"))
    if started is None:
        return None, None, "verifier did not start"
    verifier_dir = trial_dir / "verifier"
    for name in ("reward.json", "reward.txt"):
        path = verifier_dir / name
        if not path.is_file():
            continue
        try:
            modified = path.stat().st_mtime
        except OSError:
            continue
        if modified < started - REWARD_MTIME_TOLERANCE_S:
            # In shared verifier mode /logs/verifier is mounted into the agent
            # container, so a stale file here was written by the agent.
            return None, path, f"{name} predates the verifier run"
        value = _read_reward_file(path, spec.reward_key)
        if value is None:
            return None, path, f"{name} has no usable {spec.reward_key}"
        if not 0.0 <= value <= 1.0:
            return None, path, f"{name} reward {value} is outside [0, 1]"
        return value, path, None
    return None, None, "no reward file"


GRADE_SUMMARY_RE = re.compile(r"\[GRADE\]\s+summary\s+reward=(\S+)")
GRADE_ERROR_RE = re.compile(r"\[GRADE\]\s+error=(.*)$")
GRADE_FIELDS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("check", re.compile(r"(?:^|\s)check=([A-Za-z0-9_]+)")),
    ("status", re.compile(r"(?:^|\s)status=([a-z]+)")),
    ("score", re.compile(r"(?:^|\s)score=(\S+)")),
    # msg is always last and may contain spaces, so it consumes the rest.
    ("msg", re.compile(r"(?:^|\s)msg=(.*)$")),
)


def parse_grade_stdout(path: Path) -> dict[str, Any] | None:
    """Extract ``[GRADE]`` check lines from the verifier's captured stdout.

    The task side prints one line per check plus a summary line; see
    "细粒度判分项". Lines are printed one at a time, so a truncated run simply
    yields fewer checks. Returns ``None`` when the task prints no check lines.
    """
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    checks: list[dict[str, Any]] = []
    errors: list[str] = []
    summary: dict[str, Any] | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if "[GRADE]" not in line:
            continue
        body = line.split("[GRADE]", 1)[1]
        summary_match = GRADE_SUMMARY_RE.search(line)
        if summary_match:
            summary = {"reward": _numeric_reward(_safe_float(summary_match.group(1)))}
            summary["raw"] = line
            continue
        error_match = GRADE_ERROR_RE.search(line)
        if error_match:
            errors.append(error_match.group(1).strip())
            continue
        fields: dict[str, str] = {}
        for name, pattern in GRADE_FIELDS:
            match = pattern.search(body)
            if match:
                fields[name] = match.group(1).strip()
        if "check" not in fields:
            continue
        entry: dict[str, Any] = {
            "check": fields["check"],
            "status": fields.get("status"),
            "score": _numeric_reward(_safe_float(fields.get("score"))),
        }
        if fields.get("msg"):
            entry["msg"] = fields["msg"]
        checks.append(entry)
    if not checks and summary is None and not errors:
        return None
    result: dict[str, Any] = {"checks": checks}
    if summary is not None:
        result["summary"] = summary
    if errors:
        result["errors"] = errors
    return result


def _safe_float(value: Any) -> float | None:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


# --- status machine ---------------------------------------------------------


def parse_harbor_result(
    spec: RunSpec,
    *,
    return_code: int,
    timed_out: bool,
    agent: str | None = None,
) -> dict[str, Any]:
    """Derive status, reward and evidence from Harbor's own artifacts.

    The decision table follows docs/HARBOR_STATUS_REWARD_DESIGN.md §4.5
    (R1-R12), evaluated top to bottom with the first match winning. Return
    codes are never a status input: ``harbor run`` exits 0 for failed trials
    and the verifier's exit code is discarded by Harbor.
    """
    job_dir = spec.harbor_job_dir
    agent = agent if isinstance(agent, str) else None
    # Discover the trial by directory, not by result.json: a cancelled trial
    # keeps its agent log and verifier directory but may never get result.json
    # written, and that log is the only evidence the agent had started.
    trial_dirs = sorted(path for path in job_dir.iterdir() if path.is_dir()) if job_dir.is_dir() else []
    trial_dir = trial_dirs[0] if len(trial_dirs) == 1 else None
    trial = (
        _read_optional_object(trial_dir / "result.json") if trial_dir is not None else None
    )

    warnings: list[str] = []
    agent_exit = AgentExit(parser=None, source=None, reason="unknown")
    reward: float | None = None
    reward_file: Path | None = None
    reward_reject: str | None = None
    grade = None
    if trial_dir is not None:
        agent_exit, parse_warnings = read_agent_exit(spec, trial_dir, agent)
        warnings.extend(parse_warnings)
        if trial is not None:
            reward, reward_file, reward_reject = valid_reward(spec, trial, trial_dir)
        else:
            reward_reject = "trial result was never written"
        grade = parse_grade_stdout(trial_dir / "verifier" / "test-stdout.txt")
        if grade is not None:
            grade["source"] = _relative(spec, trial_dir / "verifier" / "test-stdout.txt")

    if trial is not None:
        exception = _dict(trial.get("exception_info"))
    else:
        exception = {}
    exception_type = str(exception.get("exception_type") or "")
    agent_started = bool(_dict(trial.get("agent_execution")).get("started_at")) if trial else False
    if not agent_started and trial_dir is not None:
        # Without a trial result, a non-empty agent log is the only proof that
        # the agent phase had begun (R2 vs R3).
        agent_started = bool(agent_exit.source)

    termination: str
    stage: str | None
    status: str
    detail = ""

    if timed_out and agent_started:
        # R2
        status, termination, stage = "truncated", "runner_timeout", None
        detail = f"runner timeout after {spec.timeout_s}s; sent SIGTERM to Harbor"
    elif timed_out:
        # R3: the timeout hit before the agent ran, so nothing was attempted.
        status, termination, stage = "failed", "runner_timeout", _current_stage(trial or {})
        detail = f"runner timeout after {spec.timeout_s}s before the agent started"
    elif trial is None:
        # R4: Harbor never produced a usable trial result.
        status, termination, stage = "failed", "harbor_error", "harbor_start"
        if len(trial_dirs) > 1:
            detail = f"{len(trial_dirs)} trial directories under {job_dir.name}"
        elif trial_dir is not None:
            detail = "Harbor wrote no trial result"
        else:
            detail = "Harbor wrote no trial directory"
    elif not _phase_finished(trial, "environment_setup") or not _dict(
        trial.get("agent_setup")
    ).get("started_at"):
        # R5
        status, termination, stage = "failed", "env_failed", "environment"
        detail = "environment setup did not finish"
    elif not _phase_finished(trial, "agent_setup") or not agent_started:
        # R6
        status, termination, stage = "failed", "agent_setup_failed", "agent_setup"
        detail = "agent setup did not finish"
    elif exception_type in VERIFIER_EXCEPTION_TYPES or (
        _dict(trial.get("verifier")).get("started_at") and not _phase_finished(trial, "verifier")
    ):
        # R6b: the verifier raised or never finished, so no reward file it may
        # have left behind is a valid score, whatever the agent did.
        status, termination, stage = "failed", "verifier_failed", "verifier"
        detail = str(exception.get("exception_message") or "verifier did not finish")
    elif agent_exit.reason == "completed":
        # R7 falls through to the reward check.
        status, termination, stage, detail = _reward_decision(reward_reject)
    elif agent_exit.reason == "max_turns":
        # R8
        status, termination, stage = "truncated", "max_turns", None
        detail = f"agent reached its turn limit after {agent_exit.num_turns} turns"
    elif exception_type == "AgentTimeoutError":
        # R9: Harbor's own timer, not a text match.
        status, termination, stage = "truncated", "agent_timeout", None
        detail = str(exception.get("exception_message") or "agent execution timed out")
    elif agent_exit.parser is not None and agent_exit.reason == "api_error":
        # R10a: the agent stopped because the upstream API failed, so the
        # verifier's 0 is not a statement about the model's ability.
        status, termination, stage = "failed", "api_error", "agent"
        detail = _api_error_detail(agent_exit)
    elif not exception_type:
        # R10b: the agent log gives no verdict and Harbor recorded no
        # exception, so the reward file decides.
        status, termination, stage, detail = _reward_decision(reward_reject)
    else:
        # R10c
        status, termination, stage = "failed", "agent_error", "agent"
        detail = (
            f"agent ended with {agent_exit.reason}"
            if agent_exit.parser is not None
            else f"agent phase raised {exception_type or 'an exception'}"
        )

    trial_errors = _trial_errors(trial or {})
    if status == "failed":
        fatal_errors = [f"{termination}: {detail}".rstrip(": ")]
        fatal_errors.extend(trial_errors)
    else:
        fatal_errors = []
        warnings.extend(trial_errors)
    if return_code != 0 and status != "failed":
        # Never a status input: Harbor exits 0 for failed trials and non-zero
        # only for configuration problems, which R4 already covers.
        warnings.append(f"harbor process exited with status {return_code}")
    if timed_out and status != "truncated":
        warnings.append(f"runner timeout after {spec.timeout_s}s")

    return _result(
        spec.session_id,
        status=status,
        # A truncated episode keeps its valid verifier reward; is_truncated
        # distinguishes it downstream (issue #127). Failed episodes stay null.
        reward=reward if status in {"succeeded", "truncated"} else None,
        timed_out=status == "truncated",
        errors=fatal_errors,
        warnings=warnings,
        metrics={
            "task_id": spec.task_id,
            "termination": termination,
            "failure_stage": stage,
            "harbor_reward": reward,
            "agent_exit": agent_exit.as_metrics() if trial is not None else None,
            "harbor_exception": exception or None,
            "stage_timing": _stage_timing(trial or {}),
            "evidence": {
                "trial_result": (
                    _relative(spec, trial_dir / "result.json") if trial is not None else None
                ),
                "agent_log": agent_exit.source if agent_exit.parser else None,
                "verifier_stdout": (
                    _relative(spec, trial_dir / "verifier" / "test-stdout.txt")
                    if trial_dir
                    else None
                ),
                "reward_file": _relative(spec, reward_file) if reward_file else None,
            },
            "grade": grade,
        },
    )


def _reward_decision(reward_reject: str | None) -> tuple[str, str, str | None, str]:
    if reward_reject is None:
        return "succeeded", "completed", None, ""
    # R12
    return "failed", "verifier_failed", "verifier", reward_reject


def _api_error_detail(agent_exit: AgentExit) -> str:
    kind = agent_exit.api_error_kind or "other"
    text = (agent_exit.api_error_text or "").strip()
    return f"upstream API error ({kind}): {text}" if text else f"upstream API error ({kind})"


def _current_stage(trial: dict[str, Any]) -> str:
    if not _phase_finished(trial, "environment_setup"):
        return "environment"
    if not _phase_finished(trial, "agent_setup"):
        return "agent_setup"
    return "agent"


def _phase_finished(trial: dict[str, Any], name: str) -> bool:
    phase = _dict(trial.get(name))
    return bool(phase.get("started_at")) and bool(phase.get("finished_at"))


def _stage_timing(trial: dict[str, Any]) -> dict[str, Any]:
    return {
        name: [_dict(trial.get(name)).get("started_at"), _dict(trial.get(name)).get("finished_at")]
        for name in ("environment_setup", "agent_setup", "agent_execution", "verifier")
    }


def _relative(spec: RunSpec, path: Path) -> str | None:
    try:
        return str(path.relative_to(spec.episode_dir))
    except ValueError:
        return str(path)


def _result(
    session_id: str,
    *,
    status: str,
    reward: float | None,
    timed_out: bool,
    errors: list[str],
    metrics: dict[str, Any],
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "session_id": session_id,
        "status": status,
        "total_reward": reward,
        "step_count": 0,
        "terminated": not timed_out,
        "truncated": timed_out,
        "error_text": "; ".join(errors) or None,
        "metrics": {
            "bench": "harbor",
            **metrics,
            "harbor_fatal_errors": errors,
            "harbor_warnings": warnings or [],
        },
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
    """R1: the runner itself failed before Harbor could decide anything."""
    message = str(error)
    return _result(
        session_id,
        status="failed",
        reward=None,
        timed_out=timed_out,
        errors=[f"prepare_failed: {message}".rstrip(": ")],
        metrics={
            **({} if task_id is None else {"task_id": task_id}),
            "termination": "prepare_failed",
            "failure_stage": "prepare",
            "harbor_reward": None,
            "agent_exit": None,
            "harbor_exception": None,
            "stage_timing": None,
            "evidence": {},
            "grade": None,
        },
    )


def _preparation_metrics(spec: RunSpec) -> dict[str, Any]:
    return {
        "task_id": spec.task_id,
        "bundle_root": str(spec.bundle_root),
        "task_path": str(spec.task_path),
        "image_load_mode": spec.image_load_mode,
        "image_load_log_path": str(spec.image_load_log_path),
        "preparation_receipt_path": str(spec.preparation_path),
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


def _read_optional_object(path: Path) -> dict[str, Any] | None:
    try:
        return _read_object(path)
    except (OSError, TypeError, ValueError):
        return None


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
            {
                **_preparation_metrics(spec),
                "status": "prepared",
                "docker_mode": nested.mode,
                "docker_host": nested.docker_host,
                "docker_driver": nested.driver,
            },
        )
        return_code, timed_out = nested.run_harbor(spec)
        if timed_out:
            wait_for_cancellation_cleanup(spec)
        result = parse_harbor_result(
            spec,
            return_code=return_code,
            timed_out=timed_out,
            agent=nested.agent or None,
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
