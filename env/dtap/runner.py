#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit


DEFAULT_AGENT_TYPE = "openclaw"
DEFAULT_API_KEY = "dummy"
DEFAULT_DTAP_ROOT = "/workspace/DecodingTrust-Agent"
DEFAULT_RESULTS_ROOT = "/workspace/Safactory/results/dtap"
RESULT_JSON_PREFIX = "SAFACTORY_RESULT_JSON "
RESULT_PATH_ENV = "SAFACTORY_RESULT_PATH"


def main() -> int:
    started_at = time.perf_counter()
    request = _read_request()
    session_id = _required_text(request.get("session_id"), "session_id")
    env_params = request.get("env_params") if isinstance(request.get("env_params"), dict) else {}
    dataset = env_params.get("dataset") if isinstance(env_params.get("dataset"), dict) else {}

    try:
        task_row = _resolve_task_row(dataset)
        dtap_root = Path(_first_text(dataset.get("dtap_root"), env_params.get("dtap_root"), os.environ.get("DTAP_ROOT"), DEFAULT_DTAP_ROOT))
        if not dtap_root.is_dir():
            raise RuntimeError(f"DTAP root does not exist: {dtap_root}")

        results_dir = _results_dir(request, env_params, dataset)
        results_dir.mkdir(parents=True, exist_ok=True)
        task_list = _write_single_task_list(results_dir, task_row)
        log_path = results_dir / "dtap_eval.log"

        model_ref = _first_text(
            dataset.get("model_ref"),
            dataset.get("model"),
            env_params.get("model_ref"),
            env_params.get("model"),
            request.get("model"),
        )
        if not model_ref:
            raise RuntimeError("DTAP runner could not resolve model_ref")

        route_model = _first_text(
            dataset.get("route_model"),
            env_params.get("route_model"),
            request.get("model"),
            model_ref,
        )
        api_key = _first_text(
            dataset.get("api_key"),
            env_params.get("api_key"),
            os.environ.get("OPENAI_API_KEY"),
            DEFAULT_API_KEY,
        )
        gateway_session_root = _first_text(
            dataset.get("gateway_base_url"),
            env_params.get("gateway_base_url"),
            dataset.get("base_url"),
            env_params.get("base_url"),
            os.environ.get("SAFACTORY_GATEWAY_SESSION_ROOT_CONTAINER"),
            _session_root_from_session_url(os.environ.get("SAFACTORY_GATEWAY_SESSION_URL_CONTAINER"), session_id),
            os.environ.get("SAFACTORY_GATEWAY_BASE_URL_CONTAINER"),
            _gateway_session_root_url(request),
            os.environ.get("OPENAI_BASE_URL"),
        )
        if not gateway_session_root:
            raise RuntimeError("DTAP runner could not resolve Safactory gateway session root URL")
        gateway_session_root = gateway_session_root.rstrip("/")
        gateway_episode_url = _first_text(
            os.environ.get("SAFACTORY_GATEWAY_SESSION_URL_CONTAINER"),
            f"{gateway_session_root}/{session_id}",
        )

        cmd = _build_dtap_cmd(
            dtap_root=dtap_root,
            task_list=task_list,
            model=model_ref,
            gateway_session_root=gateway_session_root,
            env_params=env_params,
            dataset=dataset,
        )
        timeout_s = _timeout_s(request, env_params, dataset)
        env = _build_env(
            base=os.environ,
            dtap_root=dtap_root,
            results_dir=results_dir,
            session_id=session_id,
            gateway_session_root=gateway_session_root,
            gateway_episode_url=gateway_episode_url,
            api_key=api_key,
            route_model=route_model,
            model_ref=model_ref,
            env_params=env_params,
            dataset=dataset,
        )

        with log_path.open("w", encoding="utf-8") as log_file:
            log_file.write("command: " + shlex.join(cmd) + "\n")
            log_file.write("dtap_root: " + str(dtap_root) + "\n")
            log_file.write("results_dir: " + str(results_dir) + "\n")
            log_file.write("gateway_session_root: " + gateway_session_root + "\n")
            log_file.write("gateway_episode_url: " + gateway_episode_url + "\n")
            log_file.write("route_model: " + route_model + "\n")
            log_file.write("model_ref: " + model_ref + "\n")
            log_file.flush()
            proc = subprocess.run(
                cmd,
                cwd=str(dtap_root),
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout_s,
                check=False,
            )

        artifacts = _find_dtap_artifacts(results_dir)
        summary = _summarize_dtap_result(artifacts)
        has_task_artifacts = bool(artifacts.get("judge_result_paths") or artifacts.get("trajectory_paths"))
        result_status = "succeeded"
        error_text: str | None = None
        if proc.returncode != 0 and not has_task_artifacts:
            result_status = "failed"
            error_text = f"DTAP evaluation.py exited with code {proc.returncode}"
        elif not artifacts.get("task_log_paths"):
            result_status = "failed"
            error_text = "DTAP evaluation.py produced no task.log under the session results directory"

        safactory_result = {
            "session_id": session_id,
            "status": result_status,
            "total_reward": _reward(summary) if result_status == "succeeded" else 0.0,
            "step_count": max(1, int(summary.get("trajectory_step_count") or summary.get("task_log_count") or 1)),
            "terminated": True,
            "truncated": False,
            "error_text": error_text,
            "metrics": {
                "bench": "dtap",
                "task_id": task_row.get("task_id"),
                "domain": task_row.get("domain"),
                "task_type": task_row.get("type"),
                "threat_model": task_row.get("threat_model"),
                "risk_category": task_row.get("risk_category"),
                "dtap_root": str(dtap_root),
                "results_dir": str(results_dir),
                "task_list_path": str(task_list),
                "log_path": str(log_path),
                "gateway_session_root": gateway_session_root,
                "gateway_episode_url": gateway_episode_url,
                "route_model": route_model,
                "model_ref": model_ref,
                "returncode": proc.returncode,
                "duration_ms": round((time.perf_counter() - started_at) * 1000, 3),
                **artifacts,
                **summary,
            },
        }
        _persist_result(results_dir, safactory_result)
        _fix_result_ownership(results_dir)
        _write_result(safactory_result)
        return 0
    except subprocess.TimeoutExpired as exc:
        results_dir = _safe_results_dir_for_failure(request, env_params, dataset)
        result = _failure_result(
            session_id,
            f"DTAP evaluation.py timed out after {float(exc.timeout or 0):.1f}s",
            started_at,
            truncated=True,
        )
        if results_dir is not None:
            _fix_result_ownership(results_dir)
        _write_result(result)
        return 0
    except Exception as exc:
        results_dir = _safe_results_dir_for_failure(request, env_params, dataset)
        if results_dir is not None:
            _fix_result_ownership(results_dir)
        _write_result(_failure_result(session_id, str(exc), started_at))
        return 0


def _read_request() -> dict[str, Any]:
    raw = sys.stdin.read().strip() or os.environ.get("SAFACTORY_START_REQUEST_JSON", "").strip()
    if not raw:
        raise RuntimeError("SimulationStartRequest JSON was not provided on stdin")
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise RuntimeError("SimulationStartRequest must be a JSON object")
    return data


def _required_text(value: Any, name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise RuntimeError(f"SimulationStartRequest missing {name}")
    return text


def _resolve_task_row(dataset: dict[str, Any]) -> dict[str, Any]:
    domain = _first_text(dataset.get("domain"))
    task_type = _first_text(dataset.get("type"), dataset.get("task_type"))
    task_id = _first_text(dataset.get("task_id"))
    if not domain or not task_type or not task_id:
        raise RuntimeError("DTAP dataset row requires domain, type/task_type, and task_id")

    row = dict(dataset)
    row["domain"] = domain
    row["type"] = task_type
    row.pop("task_type", None)
    row["task_id"] = task_id
    if task_type == "malicious":
        threat_model = _first_text(row.get("threat_model"))
        risk_category = _first_text(row.get("risk_category"))
        if not threat_model or not risk_category:
            raise RuntimeError("Malicious DTAP task requires threat_model and risk_category")
        row["threat_model"] = threat_model
        row["risk_category"] = risk_category
    return row


def _write_single_task_list(results_dir: Path, task_row: dict[str, Any]) -> Path:
    path = results_dir / "task_list.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(task_row, ensure_ascii=False) + "\n")
    return path


def _build_dtap_cmd(
    *,
    dtap_root: Path,
    task_list: Path,
    model: str,
    gateway_session_root: str,
    env_params: dict[str, Any],
    dataset: dict[str, Any],
) -> list[str]:
    cmd = [
        sys.executable or "python",
        str(dtap_root / "eval" / "evaluation.py"),
        "--task-list",
        str(task_list),
        "--agent-type",
        _first_text(dataset.get("agent_type"), env_params.get("agent_type"), DEFAULT_AGENT_TYPE),
        "--model",
        model,
        "--max-parallel",
        _first_text(dataset.get("max_parallel"), env_params.get("max_parallel"), dataset.get("native_parallel"), env_params.get("native_parallel"), "1"),
    ]

    if _truthy(dataset.get("use_inner_gateway_sessions"), env_params.get("use_inner_gateway_sessions")):
        cmd.extend([
            "--safactory-gateway-session-root",
            gateway_session_root.rstrip("/"),
            "--safactory-session-prefix",
            _first_text(dataset.get("safactory_session_prefix"), env_params.get("safactory_session_prefix"), "dtap"),
            "--safactory-gateway-api-key",
            _first_text(dataset.get("api_key"), env_params.get("api_key"), os.environ.get("OPENAI_API_KEY"), DEFAULT_API_KEY),
        ])

    for key, cli_key in (
        ("port_range", "--port-range"),
        ("thinking_level", "--thinking-level"),
    ):
        value = _first_text(dataset.get(key), env_params.get(key))
        if value:
            cmd.extend([cli_key, value])

    temperature = _first_text(dataset.get("temperature"), env_params.get("temperature"))
    if temperature:
        cmd.extend(["--temperature", temperature])

    for key, cli_key in (
        ("skip_mcp", "--skip-mcp"),
        ("skip_judge", "--skip-judge"),
        ("debug", "--debug"),
        ("direct_prompt", "--direct-prompt"),
        ("verbose", "--verbose"),
        ("keep_envs", "--keep-envs"),
        ("skip_existing", "--skip-existing"),
    ):
        if _truthy(dataset.get(key), env_params.get(key)):
            cmd.append(cli_key)

    disallowed_tools = _text_list(dataset.get("disallowed_tools"), env_params.get("disallowed_tools"), default=[])
    if disallowed_tools:
        cmd.extend(["--disallowed-tools", *disallowed_tools])

    return cmd


def _results_dir(request: dict[str, Any], env_params: dict[str, Any], dataset: dict[str, Any]) -> Path:
    root = Path(_first_text(dataset.get("results_root"), env_params.get("results_root"), DEFAULT_RESULTS_ROOT))
    job_id = _safe_name(str(request.get("job_id") or "job"))
    session_id = _safe_name(str(request.get("session_id") or "session"))
    return root / job_id / session_id


def _safe_results_dir_for_failure(request: dict[str, Any], env_params: dict[str, Any], dataset: dict[str, Any]) -> Path | None:
    try:
        path = _results_dir(request, env_params, dataset)
    except Exception:
        return None
    return path if path.exists() else None


def _timeout_s(request: dict[str, Any], env_params: dict[str, Any], dataset: dict[str, Any]) -> float:
    raw = dataset.get("timeout_s", env_params.get("timeout_s", request.get("agent_start_timeout_s", 3600.0)))
    try:
        return max(1.0, float(raw))
    except (TypeError, ValueError):
        return 3600.0


def _gateway_session_root_url(request: dict[str, Any]) -> str:
    base = str(request.get("gateway_base_url") or "").rstrip("/")
    if not base:
        return ""
    return _containerize_local_gateway_url(base)


def _session_root_from_session_url(url: Any, session_id: str) -> str:
    text = str(url or "").strip().rstrip("/")
    if not text:
        return ""
    suffix = "/" + str(session_id).strip().strip("/")
    if suffix != "/" and text.endswith(suffix):
        return text[: -len(suffix)].rstrip("/")
    return text.rsplit("/", 1)[0] if "/" in text else text


def _containerize_local_gateway_url(url: str) -> str:
    try:
        parts = urlsplit(str(url))
    except Exception:
        return str(url)
    if parts.hostname not in {"127.0.0.1", "localhost", "::1"}:
        return str(url)
    netloc = "host.docker.internal"
    if parts.port is not None:
        netloc = f"{netloc}:{parts.port}"
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))


def _build_env(
    *,
    base: Any,
    dtap_root: Path,
    results_dir: Path,
    session_id: str,
    gateway_session_root: str,
    gateway_episode_url: str,
    api_key: str,
    route_model: str,
    model_ref: str,
    env_params: dict[str, Any],
    dataset: dict[str, Any],
) -> dict[str, str]:
    env = dict(base)
    env["DTAP_ROOT"] = str(dtap_root)
    dataset_root = _first_text(dataset.get("dataset_root"), env_params.get("dataset_root"), os.environ.get("DTAP_DATASET_ROOT"))
    if dataset_root:
        env["DTAP_DATASET_ROOT"] = dataset_root
    env["EVAL_RESULTS_ROOT"] = str(results_dir)
    env["OPENAI_BASE_URL"] = gateway_episode_url
    env["BAILIAN_BASE_URL"] = gateway_episode_url
    env["SAFACTORY_GATEWAY_SESSION_ROOT_CONTAINER"] = gateway_session_root
    env["OPENAI_API_KEY"] = api_key
    env["BAILIAN_API_KEY"] = api_key
    env["SAFACTORY_ROUTE_MODEL"] = route_model
    env["SAFACTORY_MODEL_REF"] = model_ref
    env["DTAP_PROJECT_PREFIX"] = _safe_name(
        _first_text(dataset.get("project_prefix"), env_params.get("project_prefix"), f"safactory_{session_id[:8]}")
    )
    env["PYTHONUNBUFFERED"] = "1"
    if not _truthy(dataset.get("prefer_default_ports"), env_params.get("prefer_default_ports")):
        env["DT_PREFER_DEFAULT_PORTS"] = "0"
    pythonpath = env.get("PYTHONPATH", "")
    if str(dtap_root) not in pythonpath.split(":"):
        env["PYTHONPATH"] = f"{dtap_root}:{pythonpath}" if pythonpath else str(dtap_root)
    if _truthy(dataset.get("allow_partial_dataset"), env_params.get("allow_partial_dataset")):
        env["DTAP_ALLOW_PARTIAL_DATASET"] = "1"
    env["NO_PROXY"] = _merge_no_proxy(env.get("NO_PROXY"), "host.docker.internal,localhost,127.0.0.1,::1")
    env["no_proxy"] = _merge_no_proxy(env.get("no_proxy"), "host.docker.internal,localhost,127.0.0.1,::1")
    return env


def _find_dtap_artifacts(results_dir: Path) -> dict[str, Any]:
    judge_paths = _paths(results_dir.rglob("judge_result.json"))
    task_log_paths = _paths(results_dir.rglob("task.log"))
    trajectory_paths = _paths(
        path
        for path in results_dir.rglob("*.json")
        if path.name != "judge_result.json" and _looks_like_trajectory(path)
    )
    runtime_trace_paths = _paths(results_dir.rglob("*.trajectory.jsonl"))
    return {
        "judge_result_paths": judge_paths,
        "task_log_paths": task_log_paths,
        "trajectory_paths": trajectory_paths,
        "runtime_trace_paths": runtime_trace_paths,
    }


def _looks_like_trajectory(path: Path) -> bool:
    try:
        body = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return isinstance(body, dict) and ("trajectory" in body or "traj_info" in body or "task_info" in body)


def _summarize_dtap_result(artifacts: dict[str, Any]) -> dict[str, Any]:
    judge_payload = _load_first_json(artifacts.get("judge_result_paths"))
    trajectory_payload = _load_first_json(artifacts.get("trajectory_paths"))
    trajectory = None
    if isinstance(trajectory_payload, dict):
        trajectory = trajectory_payload.get("trajectory")
        if trajectory is None and isinstance(trajectory_payload.get("data"), dict):
            trajectory = trajectory_payload["data"].get("trajectory")
    if not isinstance(trajectory, list):
        trajectory = []

    action_tool_calls = _count_action_tool_calls(trajectory)
    tool_calls = _count_tool_calls(trajectory)
    attack_success = _extract_bool(judge_payload, "attack_success")
    task_success = _extract_bool(judge_payload, "task_success")

    return {
        "judge_result": judge_payload if isinstance(judge_payload, dict) else None,
        "attack_success": attack_success,
        "task_success": task_success,
        "trajectory_step_count": len(trajectory),
        "tool_call_count": tool_calls,
        "action_tool_call_count": action_tool_calls,
        "judge_result_count": len(artifacts.get("judge_result_paths") or []),
        "trajectory_count": len(artifacts.get("trajectory_paths") or []),
        "task_log_count": len(artifacts.get("task_log_paths") or []),
    }


def _extract_bool(payload: Any, key: str) -> bool | None:
    if not isinstance(payload, dict) or key not in payload:
        return None
    value = payload.get(key)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lower = value.strip().lower()
        if lower in {"true", "1", "yes"}:
            return True
        if lower in {"false", "0", "no"}:
            return False
    return None


def _count_tool_calls(trajectory: list[Any]) -> int:
    count = 0
    for item in trajectory:
        if not isinstance(item, dict):
            continue
        kind = str(item.get("type") or item.get("role") or item.get("source") or "").lower()
        if "tool" in kind or item.get("tool_name") or item.get("name"):
            count += 1
    return count


def _count_action_tool_calls(trajectory: list[Any]) -> int:
    count = 0
    for item in trajectory:
        if not isinstance(item, dict):
            continue
        kind = str(item.get("type") or item.get("role") or item.get("source") or "").lower()
        name = str(item.get("tool_name") or item.get("name") or "").lower()
        if "tool" not in kind and not name:
            continue
        if any(token in name for token in ("create", "update", "delete", "send", "post", "write", "move", "copy", "share", "transfer", "submit", "upload")):
            count += 1
    return count


def _reward(summary: dict[str, Any]) -> float:
    attack_success = summary.get("attack_success")
    if attack_success is True:
        return 10.0
    if attack_success is False:
        return 0.0
    task_success = summary.get("task_success")
    if task_success is True:
        return 10.0
    return 0.0


def _load_first_json(paths: Any) -> Any:
    if isinstance(paths, str):
        paths = [paths]
    if not isinstance(paths, (list, tuple)):
        return None
    for raw in paths:
        try:
            return json.loads(Path(str(raw)).read_text(encoding="utf-8"))
        except Exception:
            continue
    return None


def _paths(paths: Any) -> list[str]:
    resolved: list[Path] = []
    for path in paths:
        try:
            resolved.append(Path(path))
        except TypeError:
            continue
    resolved = sorted(resolved, key=lambda item: item.stat().st_mtime if item.exists() else 0.0, reverse=True)
    return [str(path) for path in resolved]


def _persist_result(results_dir: Path, result: dict[str, Any]) -> None:
    path = results_dir / "safactory_result.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def _fix_result_ownership(results_dir: Path) -> None:
    uid = _int_or_none(os.environ.get("SAFACTORY_HOST_UID"))
    gid = _int_or_none(os.environ.get("SAFACTORY_HOST_GID"))
    if uid is None and gid is None:
        return
    for path in [results_dir, *results_dir.rglob("*")]:
        try:
            os.chown(path, -1 if uid is None else uid, -1 if gid is None else gid)
        except PermissionError:
            continue
        except OSError:
            continue


def _failure_result(session_id: str, error_text: str, started_at: float, *, truncated: bool = False) -> dict[str, Any]:
    return {
        "session_id": session_id,
        "status": "failed",
        "total_reward": 0.0,
        "step_count": 0,
        "terminated": True,
        "truncated": truncated,
        "error_text": error_text,
        "metrics": {
            "bench": "dtap",
            "duration_ms": round((time.perf_counter() - started_at) * 1000, 3),
        },
    }


def _write_result(result: dict[str, Any]) -> None:
    _persist_result_artifact(result)
    print(RESULT_JSON_PREFIX + json.dumps(result, ensure_ascii=False), flush=True)


def _persist_result_artifact(result: dict[str, Any]) -> None:
    raw_path = str(os.environ.get(RESULT_PATH_ENV) or "").strip()
    if not raw_path:
        return
    try:
        path = Path(raw_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(path.name + ".tmp")
        tmp_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        tmp_path.replace(path)
    except Exception as exc:
        print(f"SAFACTORY_RUNNER_DIAGNOSTIC result_artifact_write_failed: {exc}", file=sys.stderr, flush=True)


def _first_text(*values: Any) -> str:
    for value in values:
        if isinstance(value, (list, tuple)):
            if value:
                value = value[0]
            else:
                continue
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _text_list(*values: Any, default: list[str] | None = None) -> list[str]:
    for value in values:
        if value is None or value == "":
            continue
        if isinstance(value, str):
            items = [item.strip() for item in value.replace(",", " ").split() if item.strip()]
        elif isinstance(value, (list, tuple)):
            items = [str(item).strip() for item in value if str(item).strip()]
        else:
            items = [str(value).strip()]
        if items:
            return items
    return list(default or [])


def _truthy(*values: Any) -> bool:
    for value in values:
        if value is None or value == "":
            continue
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        return str(value).strip().lower() in {"1", "true", "yes", "on"}
    return False


def _merge_no_proxy(*values: Any) -> str:
    merged: list[str] = []
    for value in values:
        for item in str(value or "").split(","):
            item = item.strip()
            if item and item not in merged:
                merged.append(item)
    return ",".join(merged)


def _safe_name(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(value))
    return safe.strip("_") or "item"


def _int_or_none(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    raise SystemExit(main())
