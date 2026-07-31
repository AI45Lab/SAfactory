#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


RESULT_JSON_PREFIX = "SAFACTORY_RESULT_JSON "
RESULT_PATH_ENV = "SAFACTORY_RESULT_PATH"
SUMMARY_FILENAME = "safactory-result.json"
DEFAULT_DATASET_PATH = "/benchmark/tasks/LiveCVEBench-verified"
DEFAULT_DATASET_PATHS = {
    "livecvebench": DEFAULT_DATASET_PATH,
    "patcheval": "/benchmark/tasks/PatchEval-verified",
}
DEFAULT_OUTPUT_ROOT = "/benchmark/runs"


def main() -> int:
    started_at = time.perf_counter()
    dockerd_proc: subprocess.Popen[str] | None = None
    request = _read_request()
    session_id = _required_text(request.get("session_id"), "session_id")
    env_params = request.get("env_params") if isinstance(request.get("env_params"), dict) else {}
    dataset = env_params.get("dataset") if isinstance(env_params.get("dataset"), dict) else {}
    task_id = ""
    suite = ""
    output_dir: Path | None = None

    try:
        task_id = _required_text(
            dataset.get("task_id") or dataset.get("id") or env_params.get("task_id"),
            "env_params.dataset.task_id",
        )
        suite = _first_text(dataset.get("suite"), env_params.get("suite"), "livecvebench").lower()
        dataset_path = _resolve_dataset_path(dataset, env_params, suite)
        output_root = Path(
            _first_text(dataset.get("output_root"), env_params.get("output_root"), DEFAULT_OUTPUT_ROOT)
        )
        job_id = _safe_name(str(request.get("job_id") or "job"))
        output_dir = output_root / "safactory" / job_id / _safe_name(session_id)
        output_dir.mkdir(parents=True, exist_ok=True)
        if not (dataset_path / task_id).is_dir():
            raise RuntimeError(f"{suite} task does not exist: {dataset_path / task_id}")

        agent = _first_text(dataset.get("agent"), env_params.get("agent"), "oracle")
        n_concurrent = _positive_int(
            dataset.get("n_concurrent", env_params.get("n_concurrent", 1)), default=1
        )
        test_timeout = _positive_int(
            dataset.get(
                "global_test_timeout_sec",
                env_params.get("global_test_timeout_sec", 900),
            ),
            default=900,
        )
        timeout_s = _runner_timeout(request, env_params, dataset, test_timeout)
        cmd = [
            "tb",
            "run",
            "--dataset-path",
            str(dataset_path),
            "--task-id",
            task_id,
            "--agent",
            agent,
            "--n-concurrent",
            str(n_concurrent),
            "--global-test-timeout-sec",
            str(test_timeout),
            "--output-path",
            str(output_dir),
        ]

        log_path = output_dir / "tb-run.log"
        run_env = _docker_client_env(output_dir)
        dockerd_proc = _ensure_docker_daemon(output_dir, run_env)
        with log_path.open("w", encoding="utf-8") as log_file:
            log_file.write("command: " + shlex.join(cmd) + "\n")
            log_file.flush()
            proc = subprocess.run(
                cmd,
                cwd="/benchmark",
                env=run_env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout_s,
                check=False,
            )

        metadata_path = _find_metadata(output_dir)
        metadata = _read_json(metadata_path) if metadata_path else {}
        result_paths = _result_paths(metadata_path, task_id)
        trial_results = [_read_json(path) for path in result_paths]
        resolved = [bool(item.get("is_resolved")) for item in trial_results]
        score = (
            sum(1.0 for value in resolved if value) / len(resolved)
            if resolved
            else _float_or_default(metadata.get("accuracy"), 0.0)
        )

        error_text: str | None = None
        status = "succeeded"
        if proc.returncode != 0:
            status = "failed"
            error_text = f"tb run exited with code {proc.returncode}"
        elif not metadata_path:
            status = "failed"
            error_text = _with_log_tail("tb run produced no new run_metadata.json", log_path)
        elif not result_paths:
            status = "failed"
            error_text = _with_log_tail(
                f"tb run produced no results.json for {task_id}",
                log_path,
            )

        result = {
            "session_id": session_id,
            "status": status,
            "total_reward": score if status == "succeeded" else 0.0,
            "step_count": max(1, len(trial_results)),
            "terminated": True,
            "truncated": False,
            "error_text": error_text,
            "metrics": {
                "bench": "livecvebench",
                "suite": suite,
                "task_id": task_id,
                "agent": agent,
                "score": score,
                "is_resolved": resolved[0] if len(resolved) == 1 else None,
                "resolved_trials": sum(1 for value in resolved if value),
                "trial_count": len(trial_results),
                "failure_modes": [item.get("failure_mode") for item in trial_results],
                "dataset_path": str(dataset_path),
                "output_root": str(output_root),
                "output_dir": str(output_dir),
                "safactory_result_path": str(output_dir / SUMMARY_FILENAME),
                "run_metadata_path": str(metadata_path) if metadata_path else None,
                "result_paths": [str(path) for path in result_paths],
                "log_path": str(log_path),
                "returncode": proc.returncode,
                "duration_ms": round((time.perf_counter() - started_at) * 1000, 3),
            },
        }
        _write_summary(output_dir, result)
        _write_result(result)
        return 0
    except subprocess.TimeoutExpired as exc:
        result = _failure_result(
            session_id,
            f"tb run timed out after {float(exc.timeout or 0):.1f}s",
            started_at,
            suite=suite,
            task_id=task_id,
            truncated=True,
        )
        _write_summary(output_dir, result)
        _write_result(result)
        return 0
    except Exception as exc:
        result = _failure_result(
            session_id,
            str(exc),
            started_at,
            suite=suite,
            task_id=task_id,
        )
        _write_summary(output_dir, result)
        _write_result(result)
        return 0
    finally:
        _stop_docker_daemon(dockerd_proc)


def _read_request() -> dict[str, Any]:
    raw = sys.stdin.read().strip() or os.environ.get("SAFACTORY_START_REQUEST_JSON", "").strip()
    if not raw:
        raise RuntimeError("SimulationStartRequest JSON was not provided on stdin")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise RuntimeError("SimulationStartRequest must be a JSON object")
    return value


def _required_text(value: Any, name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise RuntimeError(f"SimulationStartRequest missing {name}")
    return text


def _first_text(*values: Any) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _resolve_dataset_path(
    dataset: dict[str, Any],
    env_params: dict[str, Any],
    suite: str,
) -> Path:
    explicit_path = _first_text(dataset.get("dataset_path"), env_params.get("dataset_path"))
    if explicit_path:
        return Path(explicit_path)

    configured_paths = env_params.get("dataset_paths")
    if isinstance(configured_paths, dict):
        configured_path = _first_text(configured_paths.get(suite))
        if configured_path:
            return Path(configured_path)

    default_path = DEFAULT_DATASET_PATHS.get(suite)
    if default_path:
        return Path(default_path)
    supported = ", ".join(sorted(DEFAULT_DATASET_PATHS))
    raise RuntimeError(f"unsupported suite {suite!r}; expected one of: {supported}")


def _positive_int(value: Any, *, default: int) -> int:
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return int(default)


def _runner_timeout(
    request: dict[str, Any],
    env_params: dict[str, Any],
    dataset: dict[str, Any],
    test_timeout: int,
) -> float:
    outer = _float_or_default(request.get("agent_start_timeout_s"), float(test_timeout + 60))
    configured = _float_or_default(
        dataset.get("timeout_s", env_params.get("timeout_s")),
        float(test_timeout + 60),
    )
    return max(1.0, min(outer, configured))


def _docker_client_env(output_dir: Path) -> dict[str, str]:
    env = dict(os.environ)
    http_proxy = _first_text(env.get("HTTP_PROXY"), env.get("http_proxy"))
    https_proxy = _first_text(env.get("HTTPS_PROXY"), env.get("https_proxy"), http_proxy)
    if not http_proxy and not https_proxy:
        return env

    no_proxy = _first_text(env.get("NO_PROXY"), env.get("no_proxy"))
    docker_config_dir = output_dir / ".docker"
    docker_config_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "proxies": {
            "default": {
                "httpProxy": http_proxy,
                "httpsProxy": https_proxy,
                "noProxy": no_proxy,
            }
        }
    }
    (docker_config_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False),
        encoding="utf-8",
    )
    env["DOCKER_CONFIG"] = str(docker_config_dir)
    return env


def _ensure_docker_daemon(
    output_dir: Path,
    env: dict[str, str],
) -> subprocess.Popen[str] | None:
    if subprocess.run(
        ["docker", "info"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0:
        return None

    for legacy_binary, generic_binary in (
        ("/usr/sbin/iptables-legacy", "iptables"),
        ("/usr/sbin/ip6tables-legacy", "ip6tables"),
    ):
        if Path(legacy_binary).is_file():
            subprocess.run(
                ["update-alternatives", "--set", generic_binary, legacy_binary],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )

    log_path = output_dir / "dockerd.log"
    log_file = log_path.open("w", encoding="utf-8")
    try:
        proc = subprocess.Popen(
            ["dockerd", "--host=unix:///var/run/docker.sock"],
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
    finally:
        log_file.close()

    for _ in range(60):
        if proc.poll() is not None:
            raise RuntimeError(
                _with_log_tail("the inner Docker daemon failed to start", log_path)
            )
        if subprocess.run(
            ["docker", "info"],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        ).returncode == 0:
            return proc
        time.sleep(1)

    proc.terminate()
    raise RuntimeError(
        _with_log_tail("timed out waiting for the inner Docker daemon", log_path)
    )


def _stop_docker_daemon(proc: subprocess.Popen[str] | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def _find_metadata(output_dir: Path) -> Path | None:
    candidates = []
    for path in output_dir.glob("*/run_metadata.json"):
        try:
            candidates.append((path.stat().st_mtime, path))
        except OSError:
            continue
    return max(candidates, default=(0.0, None), key=lambda item: item[0])[1]


def _result_paths(metadata_path: Path | None, task_id: str) -> list[Path]:
    if metadata_path is None:
        return []
    return sorted((metadata_path.parent / task_id).glob("*/results.json"))


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def _float_or_default(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _with_log_tail(message: str, log_path: Path, *, max_chars: int = 6000) -> str:
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return message
    if not text:
        return message
    return f"{message}\n\nLast tb output:\n{text[-max_chars:]}"


def _safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "-" for char in value)[:100]


def _failure_result(
    session_id: str,
    error_text: str,
    started_at: float,
    *,
    suite: str = "",
    task_id: str = "",
    truncated: bool = False,
) -> dict[str, Any]:
    return {
        "session_id": session_id,
        "status": "failed",
        "total_reward": 0.0,
        "step_count": 0,
        "terminated": True,
        "truncated": truncated,
        "error_text": error_text,
        "metrics": {
            "bench": "livecvebench",
            "suite": suite or None,
            "task_id": task_id or None,
            "duration_ms": round((time.perf_counter() - started_at) * 1000, 3),
        },
    }


def _write_summary(output_dir: Path | None, result: dict[str, Any]) -> None:
    if output_dir is None:
        return
    path = output_dir / SUMMARY_FILENAME
    metrics = result.get("metrics")
    if isinstance(metrics, dict):
        metrics.setdefault("safactory_result_path", str(path))
    path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_result(result: dict[str, Any]) -> None:
    encoded = json.dumps(result, ensure_ascii=False, separators=(",", ":"))
    artifact = str(os.environ.get(RESULT_PATH_ENV) or "").strip()
    if artifact:
        path = Path(artifact)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(encoded + "\n", encoding="utf-8")
    print(RESULT_JSON_PREFIX + encoded, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
