#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


DEFAULT_RESULTS_ROOT = "/workspace/Safactory/results/clawenvkit"
DEFAULT_TASK_YAML = "/opt/clawenvkit/task.yaml"
DEFAULT_LOGS_DIR = "/logs"
DEFAULT_CLAWENVKIT_ROOT = "/opt/clawenvkit"
DEFAULT_HARNESS_ENTRYPOINT = "/opt/clawenvkit/entrypoint_openclaw.sh"
DEFAULT_WRITABLE_PLUGIN_DIR = "/tmp/clawenvkit-eval-plugin"
DEFAULT_MODEL_REF = "openai/dsv4pro"
DEFAULT_API_KEY = "dummy"
RESULT_JSON_PREFIX = "SAFACTORY_RESULT_JSON "
RESULT_PATH_ENV = "SAFACTORY_RESULT_PATH"


def main() -> int:
    started_at = time.perf_counter()
    request = _read_request()
    session_id = _required_text(request.get("session_id"), "session_id")
    env_params = request.get("env_params") if isinstance(request.get("env_params"), dict) else {}
    dataset = env_params.get("dataset") if isinstance(env_params.get("dataset"), dict) else {}
    results_dir: Path | None = None

    try:
        task_path = _resolve_task_path(dataset, env_params)
        if not task_path.is_file():
            raise RuntimeError(f"ClawEnvKit task yaml does not exist: {task_path}")
        task_config = _load_yaml_metadata(task_path)
        task_id = _first_text(dataset.get("task_id"), task_config.get("task_id"), task_path.stem)
        category = _first_text(dataset.get("category"), task_config.get("category"))

        results_dir = _results_dir(request, env_params, dataset, task_id)
        results_dir.mkdir(parents=True, exist_ok=True)
        container_logs_dir = Path(DEFAULT_LOGS_DIR)
        _prepare_container_logs(container_logs_dir)

        gateway_session_url = _gateway_session_url(request)
        model_ref = _first_text(
            dataset.get("model_ref"),
            env_params.get("model_ref"),
            request.get("model"),
            os.environ.get("SAFACTORY_MODEL_REF"),
            DEFAULT_MODEL_REF,
        )
        api_key = _first_text(
            dataset.get("api_key"),
            env_params.get("api_key"),
            os.environ.get("OPENAI_API_KEY"),
            DEFAULT_API_KEY,
        )
        timeout_s = _timeout_s(request, env_params, dataset)
        error_rate = _first_text(dataset.get("error_rate"), env_params.get("error_rate"), os.environ.get("ERROR_RATE"), "0.25")
        clawenvkit_root = Path(_first_text(
            dataset.get("clawenvkit_root"),
            env_params.get("clawenvkit_root"),
            os.environ.get("CLAWENVKIT_ROOT"),
            DEFAULT_CLAWENVKIT_ROOT,
        ))
        harness_entrypoint = Path(_first_text(
            dataset.get("harness_entrypoint"),
            env_params.get("harness_entrypoint"),
            os.environ.get("CLAWENVKIT_HARNESS_ENTRYPOINT"),
            DEFAULT_HARNESS_ENTRYPOINT,
        ))
        if not clawenvkit_root.is_dir():
            raise RuntimeError(f"ClawEnvKit root does not exist: {clawenvkit_root}")
        if not harness_entrypoint.is_file():
            raise RuntimeError(f"ClawEnvKit harness entrypoint does not exist: {harness_entrypoint}")

        log_path = results_dir / "clawenvkit_eval.log"
        writable_plugin_dir = _prepare_writable_plugin_dir(clawenvkit_root)
        env = _build_env(
            base=os.environ,
            task_path=task_path,
            logs_dir=container_logs_dir,
            clawenvkit_root=clawenvkit_root,
            eval_plugin_dir=writable_plugin_dir,
            model_ref=model_ref,
            api_key=api_key,
            gateway_session_url=gateway_session_url,
            error_rate=error_rate,
        )
        cmd = [str(harness_entrypoint)]

        proc = _run_harness(cmd, env=env, timeout_s=timeout_s, log_path=log_path)
        _copy_artifacts(container_logs_dir, results_dir)
        artifacts = _artifact_paths(results_dir)
        summary = _summarize_result(results_dir)

        result_status = "succeeded"
        error_text: str | None = None
        if proc.returncode != 0 and not artifacts.get("grading_path") and not artifacts.get("reward_path"):
            result_status = "failed"
            error_text = f"ClawEnvKit harness exited with code {proc.returncode}"
        elif not artifacts.get("grading_path") and not artifacts.get("reward_path"):
            result_status = "failed"
            error_text = "ClawEnvKit harness produced neither grading.json nor reward.txt"

        safactory_result = {
            "session_id": session_id,
            "status": result_status,
            "total_reward": _reward(summary) if result_status == "succeeded" else 0.0,
            "step_count": max(1, int(summary.get("num_tool_calls") or summary.get("llm_trajectory_line_count") or 1)),
            "terminated": True,
            "truncated": False,
            "error_text": error_text,
            "metrics": {
                "bench": "clawenvkit",
                "harness": _first_text(dataset.get("harness"), env_params.get("harness"), "openclaw"),
                "task_id": task_id,
                "category": category,
                "task_path": str(task_path),
                "clawenvkit_root": str(clawenvkit_root),
                "harness_entrypoint": str(harness_entrypoint),
                "eval_plugin_dir": str(writable_plugin_dir),
                "results_dir": str(results_dir),
                "log_path": str(log_path),
                "gateway_session_url": gateway_session_url,
                "model_ref": model_ref,
                "returncode": proc.returncode,
                "duration_ms": round((time.perf_counter() - started_at) * 1000, 3),
                **artifacts,
                **summary,
            },
        }
        _persist_result(results_dir, safactory_result)
        _write_result(safactory_result)
        return 0
    except subprocess.TimeoutExpired as exc:
        if results_dir is not None:
            _copy_artifacts(Path(DEFAULT_LOGS_DIR), results_dir)
        _write_result(_failure_result(session_id, f"ClawEnvKit harness timed out after {float(exc.timeout or 0):.1f}s", started_at, truncated=True))
        return 0
    except Exception as exc:
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


def _resolve_task_path(dataset: dict[str, Any], env_params: dict[str, Any]) -> Path:
    task_path = _first_text(dataset.get("task_path"), dataset.get("path"), dataset.get("yaml_path"))
    if task_path:
        return Path(task_path)
    dataset_root = Path(_first_text(dataset.get("dataset_root"), env_params.get("dataset_root"), "/datasets/Auto-ClawEval-mini"))
    task_id = _first_text(dataset.get("task_id"))
    category = _first_text(dataset.get("category"))
    if task_id and category:
        return dataset_root / "tasks" / category / f"{task_id}.yaml"
    if task_id:
        matches = sorted(dataset_root.rglob(f"{task_id}.yaml"))
        if matches:
            return matches[0]
    raise RuntimeError("ClawEnvKit dataset row requires task_path, or task_id with category/dataset_root")


def _load_yaml_metadata(path: Path) -> dict[str, Any]:
    try:
        import yaml

        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _results_dir(request: dict[str, Any], env_params: dict[str, Any], dataset: dict[str, Any], task_id: str) -> Path:
    root = Path(_first_text(dataset.get("results_root"), env_params.get("results_root"), os.environ.get("SAFACTORY_OUTPUT_SUBDIR"), DEFAULT_RESULTS_ROOT))
    job_id = _safe_part(_first_text(request.get("job_id"), "job"))
    session_id = _safe_part(_first_text(request.get("session_id"), "session"))
    task_part = _safe_part(task_id or "task")
    return root / job_id / session_id / task_part


def _prepare_container_logs(path: Path) -> None:
    if path.exists():
        for child in path.iterdir():
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
            else:
                child.unlink(missing_ok=True)
    path.mkdir(parents=True, exist_ok=True)


def _prepare_writable_plugin_dir(clawenvkit_root: Path) -> Path:
    src = clawenvkit_root / "extensions" / "clawenvkit-eval"
    if not src.is_dir():
        return src
    dst = Path(DEFAULT_WRITABLE_PLUGIN_DIR)
    if dst.exists():
        shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(src, dst)
    image_typebox = Path("/app/node_modules/typebox")
    if image_typebox.exists():
        node_modules = dst / "node_modules"
        node_modules.mkdir(exist_ok=True)
        shutil.copytree(image_typebox, node_modules / "typebox", dirs_exist_ok=True)
    return dst


def _gateway_session_url(request: dict[str, Any]) -> str:
    value = _first_text(os.environ.get("SAFACTORY_GATEWAY_SESSION_URL_CONTAINER"))
    if value:
        return value.rstrip("/")
    base = _first_text(request.get("gateway_base_url"), os.environ.get("SAFACTORY_GATEWAY_BASE_URL"))
    if not base:
        raise RuntimeError("gateway_base_url is missing")
    return f"{base.rstrip('/')}/{_required_text(request.get('session_id'), 'session_id')}"


def _build_env(
    *,
    base: os._Environ[str],
    task_path: Path,
    logs_dir: Path,
    clawenvkit_root: Path,
    eval_plugin_dir: Path,
    model_ref: str,
    api_key: str,
    gateway_session_url: str,
    error_rate: str,
) -> dict[str, str]:
    env = dict(base)
    env.update(
        {
            "TASK_YAML": str(task_path),
            "LOGS_DIR": str(logs_dir),
            "CLAWENVKIT_ROOT": str(clawenvkit_root),
            "EVAL_PLUGIN_DIR": str(eval_plugin_dir),
            "MODEL": model_ref,
            "OPENAI_API_KEY": api_key,
            "OPENAI_BASE_URL": gateway_session_url.rstrip("/"),
            "ERROR_RATE": error_rate,
            "HOME": "/home/node",
            "PYTHONPATH": str(clawenvkit_root),
            "HTTP_PROXY": "",
            "HTTPS_PROXY": "",
            "ALL_PROXY": "",
            "http_proxy": "",
            "https_proxy": "",
            "all_proxy": "",
            "NO_PROXY": _merge_no_proxy(base.get("NO_PROXY", "")),
            "no_proxy": _merge_no_proxy(base.get("no_proxy", "")),
        }
    )
    return env


def _run_harness(cmd: list[str], *, env: dict[str, str], timeout_s: float, log_path: Path) -> subprocess.CompletedProcess[str]:
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write("command: " + " ".join(cmd) + "\n")
        log_file.write("task_yaml: " + env.get("TASK_YAML", "") + "\n")
        log_file.write("logs_dir: " + env.get("LOGS_DIR", "") + "\n")
        log_file.write("clawenvkit_root: " + env.get("CLAWENVKIT_ROOT", "") + "\n")
        log_file.write("openai_base_url: " + env.get("OPENAI_BASE_URL", "") + "\n")
        log_file.write("model: " + env.get("MODEL", "") + "\n")
        log_file.flush()
        return subprocess.run(
            cmd,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout_s,
            check=False,
        )


def _copy_artifacts(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.mkdir(parents=True, exist_ok=True)
    for child in src.iterdir():
        target = dst / child.name
        if child.is_dir():
            shutil.copytree(child, target, dirs_exist_ok=True)
        else:
            shutil.copy2(child, target)


def _artifact_paths(results_dir: Path) -> dict[str, str]:
    names = {
        "grading_path": "grading.json",
        "reward_path": "reward.txt",
        "audit_path": "audit.json",
        "llm_trajectory_path": "llm_trajectory.jsonl",
        "eval_tools_path": "eval-tools.json",
        "openclaw_config_path": "openclaw.json",
        "openclaw_plugin_manifest_path": "openclaw.plugin.json",
    }
    out: dict[str, str] = {}
    for key, name in names.items():
        path = results_dir / name
        if path.exists():
            out[key] = str(path)
    return out


def _summarize_result(results_dir: Path) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    grading_path = results_dir / "grading.json"
    if grading_path.exists():
        try:
            grading = json.loads(grading_path.read_text(encoding="utf-8"))
            if isinstance(grading, dict):
                for key in (
                    "completion",
                    "robustness",
                    "safety",
                    "final_score",
                    "num_tool_calls",
                    "audit_data_source",
                    "raw_audit_num_tool_calls",
                    "trajectory_fallback_reason",
                ):
                    if key in grading:
                        summary[key] = grading[key]
                components = grading.get("components")
                if isinstance(components, list):
                    summary["component_count"] = len(components)
                    summary["passed_component_count"] = sum(1 for c in components if isinstance(c, dict) and c.get("passed"))
                violations = grading.get("safety_violations")
                if isinstance(violations, list):
                    summary["safety_violation_count"] = len(violations)
        except Exception as exc:
            summary["grading_parse_error"] = str(exc)

    reward_path = results_dir / "reward.txt"
    if reward_path.exists():
        try:
            summary.setdefault("final_score", float(reward_path.read_text(encoding="utf-8").strip()))
        except Exception:
            pass

    trajectory_path = results_dir / "llm_trajectory.jsonl"
    if trajectory_path.exists():
        try:
            with trajectory_path.open(encoding="utf-8", errors="ignore") as handle:
                summary["llm_trajectory_line_count"] = sum(1 for line in handle if line.strip())
        except Exception:
            pass
    return summary


def _reward(summary: dict[str, Any]) -> float:
    try:
        return float(summary.get("final_score", 0.0) or 0.0)
    except Exception:
        return 0.0


def _persist_result(results_dir: Path, result: dict[str, Any]) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / "safactory_result.json"
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def _write_result(result: dict[str, Any]) -> None:
    _persist_result_artifact(result)
    print(RESULT_JSON_PREFIX + json.dumps(result, ensure_ascii=False))
    sys.stdout.flush()


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
            "bench": "clawenvkit",
            "duration_ms": round((time.perf_counter() - started_at) * 1000, 3),
        },
    }


def _timeout_s(request: dict[str, Any], env_params: dict[str, Any], dataset: dict[str, Any]) -> float:
    value = _first_text(dataset.get("timeout_s"), env_params.get("timeout_s"), request.get("agent_start_timeout_s"), "600")
    try:
        return max(1.0, float(value))
    except Exception:
        return 600.0


def _first_text(*values: Any) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _merge_no_proxy(existing: str) -> str:
    required = ["host.docker.internal", "localhost", "127.0.0.1", "::1"]
    parts: list[str] = []
    seen: set[str] = set()
    for item in required + [p.strip() for p in str(existing or "").split(",")]:
        if item and item not in seen:
            parts.append(item)
            seen.add(item)
    return ",".join(parts)


def _safe_part(value: str) -> str:
    text = "".join(ch if ch.isalnum() or ch in "._-" else "-" for ch in str(value or "").strip())
    text = text.strip("-._")
    return text[:120] or "unknown"


if __name__ == "__main__":
    raise SystemExit(main())
