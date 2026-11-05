"""Owned process and nested Docker lifecycle for one quickenv episode.

RJob provides the network/PID namespace; each episode owns its Docker socket,
exec state and local image store.  No host daemon, global prune, or shared
GPFS Docker data directory is used.  Keep benchmark behavior in adapter.py.
"""

from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from typing import Any, Iterator


def _positive_seconds(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive number") from exc
    if not 0 < number < float("inf"):
        raise ValueError(f"{name} must be a positive finite number")
    return number


@contextmanager
def _cancel_on_signal() -> Iterator[None]:
    """Let scoped cleanup run on RJob cancellation before runner reports failure."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return

    def cancelled(signum: int, _frame: Any) -> None:
        raise RuntimeError(f"quickenv cancelled by signal {signum}")

    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        for sig in previous:
            signal.signal(sig, cancelled)
        yield
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def _stop_process_group(process: subprocess.Popen, grace: float = 10) -> None:
    """Stop only the new session created for this child, including its children."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        pass
    # A launcher may exit before its controller/proxy children.  Do not skip
    # the group cleanup just because the immediate process already exited.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def run_process(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    log_path: Path,
    timeout: int,
) -> int:
    """Run a native command without leaking stdout or subprocesses.

    Return the native exit code.  Timeout/cancellation raises an exception so
    the fixed protocol runner emits a controlled failure.  Native stdout and
    stderr are both retained in ``log_path``.
    """
    seconds = _positive_seconds(timeout, "launch_timeout")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with _cancel_on_signal(), log_path.open("ab") as log:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            return process.wait(timeout=seconds)
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError(
                f"quickenv native command exceeded {seconds:g}s; see {log_path}"
            ) from exc
        finally:
            _stop_process_group(process)


def _wait_ready(
    env: dict[str, str],
    process: subprocess.Popen | None,
    timeout: float,
    log_path: Path,
    *,
    expected_driver: str | None = None,
) -> str:
    deadline = time.monotonic() + timeout
    last_error = "no Docker response"
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError(f"nested dockerd exited during startup; see {log_path}")
        try:
            driver = ""
            for command in (
                ["docker", "info", "--format", "{{.Driver}}"],
                ["docker", "compose", "version"],
            ):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(command, timeout)
                completed = subprocess.run(
                    command,
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    timeout=min(5, remaining),
                    check=False,
                    text=True,
                )
                if completed.returncode or not completed.stdout.strip():
                    raise RuntimeError(completed.stdout.strip() or "empty Docker response")
                if command[1] == "info":
                    driver = completed.stdout.strip()
        except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
            last_error = str(exc)
            time.sleep(min(0.5, max(0, deadline - time.monotonic())))
        else:
            if expected_driver is not None and driver != expected_driver:
                raise RuntimeError(
                    f"nested Docker storage driver is {driver!r}, expected {expected_driver!r}; "
                    f"refusing to run the case; see {log_path}"
                )
            return driver
    raise RuntimeError(
        f"Docker/Compose did not become ready within {timeout:g}s: {last_error}; "
        f"see {log_path}"
    )


@contextmanager
def docker_runtime(
    config: dict[str, Any], episode_dir: Path, env: dict[str, str]
) -> Iterator[dict[str, str]]:
    """Yield a private Docker client environment and clean up owned resources.

    ``existing`` is an explicit local-debug option and never stops its daemon.
    Production RJob configs must use ``nested`` and distinct pod namespaces.
    """
    mode = config.get("docker_mode", "nested")
    if mode not in {"nested", "existing"}:
        raise ValueError("quickenv.docker_mode must be nested or existing")
    timeout = _positive_seconds(config.get("docker_start_timeout", 120), "docker_start_timeout")
    runtime_env = dict(env)
    # User Docker contexts/TLS settings can otherwise override DOCKER_HOST and
    # accidentally send image loads/cleanup to a different daemon.
    for key in ("DOCKER_CONTEXT", "DOCKER_TLS_VERIFY", "DOCKER_CERT_PATH", "DOCKER_TLS"):
        runtime_env.pop(key, None)
    runtime_env["HARBOR_TELEMETRY"] = "0"
    for binary in ("docker",):
        if shutil.which(binary, path=runtime_env.get("PATH")) is None:
            raise RuntimeError(f"quickenv runtime image is missing {binary}")
    log_path = episode_dir / "infra" / "dockerd.log"
    if mode == "existing":
        host = str(config.get("docker_host") or "")
        if not host.startswith("unix:///"):
            raise ValueError("existing mode requires an explicit quickenv.docker_host=unix:///...")
        runtime_env["DOCKER_HOST"] = host
        with _cancel_on_signal():
            _wait_ready(runtime_env, None, timeout, log_path)
            yield runtime_env
        return

    if os.geteuid() != 0:
        raise RuntimeError("nested Docker requires root in a privileged RJob")
    driver = config.get("docker_storage_driver", "fuse-overlayfs")
    if driver not in {"fuse-overlayfs", "overlay2", "vfs"}:
        raise ValueError("unsupported quickenv.docker_storage_driver")
    required = ["dockerd"] + (["fuse-overlayfs"] if driver == "fuse-overlayfs" else [])
    for binary in required:
        if shutil.which(binary, path=runtime_env.get("PATH")) is None:
            raise RuntimeError(f"quickenv runtime image is missing {binary}")
    if driver == "fuse-overlayfs" and not Path("/dev/fuse").exists():
        raise RuntimeError("nested Docker needs /dev/fuse; enable privileged RJob and brainpp.cn/fuse=1")

    log_path.parent.mkdir(parents=True, exist_ok=True)
    # Match Harbor's RJob storage layout. Large task image layers must not be
    # forced into /tmp, which can be a small tmpfs in the outer container.
    # The configured parent must be container-local disk; never use shared GPFS.
    data_base_value = config.get("docker_data_root", "/docker-data")
    if not isinstance(data_base_value, str) or not Path(data_base_value).is_absolute():
        raise ValueError("quickenv.docker_data_root must be an absolute container-local disk path")
    data_base = Path(data_base_value).resolve()
    data_base.mkdir(parents=True, exist_ok=True)
    # Socket/exec state stays local and short enough for a Unix socket name.
    runtime_root = Path(tempfile.mkdtemp(prefix="quickenv-docker-", dir="/tmp"))
    data_root = None
    try:
        data_root = Path(tempfile.mkdtemp(prefix="quickenv-", dir=data_base))
        runtime_env["DOCKER_HOST"] = f"unix://{runtime_root / 'docker.sock'}"
        runtime_env["DOCKER_CONFIG"] = str(runtime_root / "client")
        (runtime_root / "client").mkdir()
        daemon_config = runtime_root / "daemon.json"
        daemon_config.write_text("{}\n", encoding="utf-8")
        command = [
            "dockerd",
            f"--config-file={daemon_config}",
            f"--host={runtime_env['DOCKER_HOST']}",
            f"--pidfile={runtime_root / 'dockerd.pid'}",
            f"--exec-root={runtime_root / 'exec'}",
            f"--data-root={data_root}",
            f"--storage-driver={driver}",
        ]
        with _cancel_on_signal(), log_path.open("ab") as log:
            process = subprocess.Popen(
                command,
                env=runtime_env,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                actual_driver = _wait_ready(
                    runtime_env, process, timeout, log_path, expected_driver=driver,
                )
                usage = shutil.disk_usage(data_root)
                (log_path.parent / "docker-runtime.json").write_text(json.dumps({
                    "mode": mode,
                    "docker_host": runtime_env["DOCKER_HOST"],
                    "storage_driver": actual_driver,
                    "data_root": str(data_root),
                    "disk_total_bytes": usage.total,
                    "disk_free_bytes_before_case": usage.free,
                }, indent=2) + "\n", encoding="utf-8")
                yield runtime_env
            finally:
                _stop_process_group(process)
    finally:
        shutil.rmtree(runtime_root, ignore_errors=True)
        if data_root is not None:
            shutil.rmtree(data_root, ignore_errors=True)
