"""Invoke one delivered Mining case using the fixed SAfactory runner contract.

Native solving, image loading and verification remain in run_mining_400.sh and
its delivery dependencies. This module only maps a row, isolates its execution,
routes the model upstream through its session, and collects the documented JSON.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any
from urllib.parse import urlsplit

from runtime import docker_runtime, run_process


def run_case(request: dict[str, Any], task: dict[str, Any], session_url: str
             ) -> tuple[dict[str, Any], int]:
    params = request.get("env_params")
    if not isinstance(params, dict) or not isinstance(params.get("quickenv"), dict):
        raise ValueError("env_params.quickenv must contain the delivery runtime configuration")
    config = params["quickenv"]
    if task.get("example"):
        raise ValueError("example dataset row is not runnable; generate real rows with build_dataset.py")
    task_id = _text(task.get("task_id"), "dataset.task_id")
    # The delivered 400-row catalog contains only task_id. Keep explicit
    # task_name support for rows exported by build_dataset.py.
    task_name = _text(task.get("task_name", task_id), "dataset.task_name (or task_id)")
    if not re.fullmatch(r"arvo-[A-Za-z0-9_.-]+-lv0", task_name):
        raise ValueError("dataset.task_name must be a single ARVO lv0 case name (arvo-...-lv0)")
    session_id = _text(request.get("session_id"), "session_id")
    job_id = _text(request.get("job_id"), "job_id")
    model = _text(request.get("model"), "model")
    gateway = urlsplit(session_url)
    if (gateway.scheme not in {"http", "https"} or not gateway.hostname
            or gateway.query or gateway.fragment or gateway.username or gateway.password):
        raise ValueError("Gateway session URL must be an HTTP(S) session endpoint without credentials/query")

    delivery = _absolute(config.get("delivery_root"), "quickenv.delivery_root")
    offline = _absolute(config.get("offline_root"), "quickenv.offline_root")
    batch = _resolve_batch_dir(task, config, task_name)
    results = _absolute(params.get("results_root"), "results_root")
    _preflight(delivery, offline, batch, task_name)

    turns = _positive_int(config.get("max_turns", 1), "max_turns")
    max_steps = request.get("max_steps")
    if max_steps is not None:
        if type(max_steps) is not int or max_steps < 0:
            raise ValueError("max_steps must be a nonnegative integer")
        if max_steps:
            turns = min(turns, max_steps)
    timeout = _positive_int(config.get("timeout", 2700), "timeout")
    launch_timeout = _positive_int(config.get("launch_timeout", timeout + 900), "launch_timeout")
    budget = config.get("budget", 20)
    if type(budget) not in (int, float) or not math.isfinite(budget) or budget <= 0:
        raise ValueError("budget must be a finite positive number")

    # An attempt-specific directory also prevents retries of the same session
    # from reading old result.json files. Never share the native launcher's lock.
    job_dir = results / "quickenv" / _path_key(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    episode_dir = Path(tempfile.mkdtemp(prefix=_path_key(session_id) + "-", dir=job_dir))
    output_dir = episode_dir / "native"
    output_dir.mkdir()
    log_path = episode_dir / "native-launcher.log"
    command = ["bash", str(delivery / "run_mining_400.sh"), task_name]
    env = _native_env()
    env.update({
        "EXPGYM_OFFLINE_ROOT": str(offline),
        "MINING_BATCH_DIR": str(batch),
        "PJ_USER_BATCH_DIR": str(batch),
        "OUT_DIR": str(output_dir),
        "WORKERS": "1",
        "RESUME": "0",
        "SETUP": "0",
        "MODEL": model,
        "MAX_TURNS": str(turns),
        "TIMEOUT": str(timeout),
        "BUDGET": str(budget),
        # The gateway implements <session_url>/v1/messages. The delivered
        # LLM proxy remains in the native path but uses this as its upstream.
        "ANTHROPIC_BASE_URL": session_url.rstrip("/"),
        "ANTHROPIC_API_KEY": "EMPTY",
    })
    (episode_dir / "invocation.json").write_text(json.dumps({
        "task_id": task_id, "task_name": task_name, "session_id": session_id,
        "command": command, "cwd": str(delivery), "batch_dir": str(batch),
        "model": model, "gateway_session_url": session_url.rstrip("/"),
        "max_turns": turns, "timeout": timeout, "launch_timeout": launch_timeout,
        "native_output_dir": str(output_dir), "workers": 1, "resume": False,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    with docker_runtime(config, episode_dir, env) as native_env:
        returncode = run_process(command, cwd=delivery, env=native_env,
                                 log_path=log_path, timeout=launch_timeout)
    if returncode != 0:
        raise RuntimeError(f"native single-case launcher exited {returncode}; see {log_path}")

    result_path = output_dir / task_name / "result.json"
    if not result_path.is_file():
        raise RuntimeError(f"native launcher produced no single-case result: {result_path}; see {log_path}")
    try:
        native_result = json.loads(result_path.read_text(encoding="utf-8"), parse_constant=_invalid_json_constant)
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"invalid native result JSON at {result_path}: {exc}") from exc
    if not isinstance(native_result, dict):
        raise RuntimeError(f"native result must be a JSON object: {result_path}")
    # No success/reward normalization: a low/zero native reward is still a
    # completed execution. Only an explicit runtime failure status is an error.
    if native_result.get("status") in ("failed", "error"):
        raise RuntimeError(f"native result reports {native_result['status']}; see {result_path}")
    raw_steps = native_result.get("step_count")
    steps = raw_steps if type(raw_steps) is int and raw_steps >= 0 else 0
    return {
        "bench": "quickenv", "task_id": task_id, "task_name": task_name,
        "session_id": session_id, "batch_dir": str(batch),
        "native_output_dir": str(output_dir), "native_result_path": str(result_path),
        "native_log_path": str(log_path), "native_result": native_result,
        "step_count_source": "native_result.step_count" if steps == raw_steps and type(raw_steps) is int else "unavailable",
        "evaluation_enabled": False,
    }, steps


def _resolve_batch_dir(task: dict[str, Any], config: dict[str, Any], task_name: str) -> Path:
    """Find this case's partition without assuming catalog order is partition order.

    SAfactory already selected one JSONL row. This lookup reads only delivery
    manifests; it never loops over the scheduling dataset or executes a batch.
    Explicit per-row batch_dir remains supported for exported/relocated rows.
    """
    if "batch_dir" in task:
        return _absolute(task["batch_dir"], "dataset.batch_dir")
    candidates = config.get("batch_dirs")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("task_id-only rows require a nonempty env_params.quickenv.batch_dirs list; "
                         "alternatively set dataset.batch_dir explicitly")
    matches: list[Path] = []
    visited: set[Path] = set()
    for index, value in enumerate(candidates):
        batch = _absolute(value, f"quickenv.batch_dirs[{index}]")
        if batch in visited:
            continue
        visited.add(batch)
        manifest = batch / "selected_tasks.txt"
        try:
            names = {line.strip() for line in manifest.read_text(encoding="utf-8").splitlines()
                     if line.strip()}
        except OSError as exc:
            raise FileNotFoundError(f"cannot resolve case partition: cannot read {manifest}; "
                                    "mount all configured batch_dirs or set dataset.batch_dir") from exc
        if task_name in names:
            matches.append(batch)
    if not matches:
        raise ValueError(f"task {task_name} was not found in any configured batch_dirs selected_tasks.txt")
    if len(matches) != 1:
        raise ValueError(f"task {task_name} appears in multiple partitions: "
                         + ", ".join(str(path) for path in matches)
                         + "; set dataset.batch_dir explicitly")
    return matches[0]


def _preflight(delivery: Path, offline: Path, batch: Path, task_name: str) -> None:
    files = [delivery / rel for rel in (
        "run_mining_400.sh", "env_cn.sh", "scripts/run_pj_exploit.sh",
        "examples/run_agent.py", "scripts/setup/pre_run.py",
    )]
    files += [offline / rel for rel in ("activate.sh", "relocate.py", "firewall-image.tar")]
    files += [batch / rel for rel in ("selected_tasks.txt", "tasks.json", "bin/load_image.py")]
    case_dir = batch / task_name
    files += [case_dir / rel for rel in ("harbor-runtime-profile/task.toml", "image-lock.json")]
    agent = delivery.parent / "claude-code-2.1.228-linux-amd64/claude"
    files.append(agent)
    missing = [str(path) for path in files if not path.is_file()]
    if not (batch / "shared-images").is_dir():
        missing.append(str(batch / "shared-images"))
    if not any((case_dir / "images").glob("*.tar*")):
        missing.append(str(case_dir / "images/*.tar*"))
    if missing:
        raise FileNotFoundError("quickenv delivery dependencies missing: " + "; ".join(missing)
                                + ". See env/quickenv/README.safactory.md")
    if not os.access(agent, os.X_OK):
        raise PermissionError(f"delivery Claude binary is not executable: {agent}")
    names = [line.strip() for line in (batch / "selected_tasks.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
    if task_name not in names:
        raise ValueError(f"task {task_name} is not in {batch / 'selected_tasks.txt'}")


def _native_env() -> dict[str, str]:
    # Avoid carrying provider credentials or a previous episode's proxy into
    # the delivered launcher. The native env_cn.sh must preserve these explicit
    # overrides; that file is an external delivery dependency.
    removed = {
        "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL",
        "CLAUDE_CODE_OAUTH_TOKEN", "OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_API_BASE",
        "OPENROUTER_API_KEY", "OPENROUTER_BASE_URL", "LLM_API_KEY", "LLM_BASE_URL",
        "PROXY_URL", "CYBERGYM_ADMIN_KEY", "SUDO_USER",
        "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy",
    }
    env = {key: value for key, value in os.environ.items() if key not in removed}
    env.update({"NO_PROXY": "*", "no_proxy": "*", "PYTHONDONTWRITEBYTECODE": "1"})
    return env


def _positive_int(value: Any, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{name} must be a nonempty string without outer whitespace")
    if any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise ValueError(f"{name} must not contain control characters")
    return value


def _absolute(value: Any, name: str) -> Path:
    path = Path(_text(value, name))
    if not path.is_absolute():
        raise ValueError(f"{name} must be an absolute path inside the runtime container")
    return path.resolve()


def _path_key(value: str) -> str:
    readable = re.sub(r"[^A-Za-z0-9_.-]", "_", value).strip("._")[:60] or "episode"
    return readable + "-" + hashlib.sha256(value.encode()).hexdigest()[:12]


def _invalid_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")
