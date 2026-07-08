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
from urllib.parse import urlsplit, urlunsplit


DEFAULT_API_KEY = "EMPTY"
DEFAULT_DATASET = "harmbench"
DEFAULT_MODEL = "dsv4pro"
DEFAULT_RESULTS_ROOT = "/app/results"
RESULT_JSON_PREFIX = "SAFACTORY_RESULT_JSON "
RESULT_PATH_ENV = "SAFACTORY_RESULT_PATH"


def main() -> int:
    started_at = time.perf_counter()
    request = _read_request()
    session_id = _required_text(request.get("session_id"), "session_id")
    env_params = request.get("env_params") if isinstance(request.get("env_params"), dict) else {}
    dataset = env_params.get("dataset") if isinstance(env_params.get("dataset"), dict) else {}

    try:
        attack = _single_attack(dataset, env_params)
        attacker_model = _first_text(
            env_params.get("attacker_model"),
            env_params.get("default_attacker_model"),
            request.get("model"),
            DEFAULT_MODEL,
        )
        judge_model = _first_text(
            env_params.get("judge_model"),
            env_params.get("default_judge_model"),
            request.get("model"),
            DEFAULT_MODEL,
        )
        target_models = _text_list(
            env_params.get("target_models"),
            env_params.get("default_target_models"),
            request.get("model"),
            default=[DEFAULT_MODEL],
        )
        openrt_dataset = _first_text(
            dataset.get("openrt_dataset"),
            dataset.get("dataset"),
            env_params.get("openrt_dataset"),
            env_params.get("default_openrt_dataset"),
            DEFAULT_DATASET,
        )
        api_key = _first_text(
            dataset.get("api_key"),
            env_params.get("api_key"),
            os.environ.get("OPENAI_API_KEY"),
            DEFAULT_API_KEY,
        )
        base_url = _first_text(
            dataset.get("base_url"),
            env_params.get("base_url"),
            os.environ.get("SAFACTORY_GATEWAY_SESSION_URL_CONTAINER"),
            _gateway_session_url(request, session_id),
            os.environ.get("OPENAI_BASE_URL"),
        )
        if not base_url:
            raise RuntimeError("OpenRT runner could not resolve an OpenAI-compatible base URL")

        results_dir = _results_dir(request, env_params)
        results_dir.mkdir(parents=True, exist_ok=True)
        log_path = results_dir / "openrt_eval.log"

        cmd = _build_openrt_cmd(
            attack=attack,
            attacker_model=attacker_model,
            judge_model=judge_model,
            target_models=target_models,
            openrt_dataset=openrt_dataset,
            api_key=api_key,
            base_url=base_url,
            results_dir=results_dir,
            env_params=env_params,
            dataset=dataset,
        )
        timeout_s = _timeout_s(request, env_params, dataset)
        env = dict(os.environ)
        env["OPENAI_API_KEY"] = api_key
        env["OPENAI_BASE_URL"] = base_url

        with log_path.open("w", encoding="utf-8") as log_file:
            log_file.write("command: " + shlex.join(cmd) + "\n")
            log_file.write("base_url: " + base_url + "\n")
            log_file.flush()
            proc = subprocess.run(
                cmd,
                cwd="/app",
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout_s,
                check=False,
            )

        metrics_files = _find_metrics_files(results_dir)
        error_files = _find_error_files(results_dir)
        metrics_summary = _summarize_metrics(metrics_files)
        status = "succeeded"
        error_text: str | None = None
        if proc.returncode != 0:
            status = "failed"
            error_text = f"OpenRT eval.py exited with code {proc.returncode}"
        elif not metrics_files:
            status = "failed"
            error_text = "OpenRT eval.py produced no metrics JSON files"

        _write_result(
            {
                "session_id": session_id,
                "status": status,
                "total_reward": metrics_summary.get("reward_10", 0.0) if status == "succeeded" else 0.0,
                "step_count": max(1, int(metrics_summary.get("metrics_file_count") or 1)),
                "terminated": True,
                "truncated": False,
                "error_text": error_text,
                "metrics": {
                    "bench": "openrt",
                    "task_id": dataset.get("task_id"),
                    "attack": attack,
                    "attacker_model": attacker_model,
                    "judge_model": judge_model,
                    "target_models": target_models,
                    "openrt_dataset": openrt_dataset,
                    "base_url": base_url,
                    "results_dir": str(results_dir),
                    "log_path": str(log_path),
                    "metrics_files": [str(path) for path in metrics_files],
                    "error_files": [str(path) for path in error_files],
                    "returncode": proc.returncode,
                    "duration_ms": round((time.perf_counter() - started_at) * 1000, 3),
                    **metrics_summary,
                },
            }
        )
        return 0
    except subprocess.TimeoutExpired as exc:
        _write_result(
            _failure_result(
                session_id,
                f"OpenRT eval.py timed out after {float(exc.timeout or 0):.1f}s",
                started_at,
                truncated=True,
            )
        )
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


def _single_attack(dataset: dict[str, Any], env_params: dict[str, Any]) -> str:
    raw = dataset.get("attack", dataset.get("attacks", env_params.get("attack", env_params.get("attacks"))))
    values = _text_list(raw, default=[])
    if not values:
        raise RuntimeError("OpenRT dataset row must define one attack, for example {'attack': 'PAIR'}")
    if len(values) != 1:
        raise RuntimeError(f"OpenRT runner schedules one attack per row; got attacks={values!r}")
    return values[0]


def _build_openrt_cmd(
    *,
    attack: str,
    attacker_model: str,
    judge_model: str,
    target_models: list[str],
    openrt_dataset: str,
    api_key: str,
    base_url: str,
    results_dir: Path,
    env_params: dict[str, Any],
    dataset: dict[str, Any],
) -> list[str]:
    cmd = [
        sys.executable or "python",
        "eval.py",
        "--api-key",
        api_key,
        "--base-url",
        base_url,
        "--attacker-model",
        attacker_model,
        "--judge-model",
        judge_model,
        "--target-models",
        *target_models,
        "--attacks",
        attack,
        "--dataset",
        openrt_dataset,
        "--results-dir",
        str(results_dir),
    ]
    embedding_model = _first_text(dataset.get("embedding_model"), env_params.get("embedding_model"))
    if embedding_model:
        cmd.extend(["--embedding-model", embedding_model])
    for source_key, cli_key in (("max_workers", "--max-workers"), ("evaluator_workers", "--evaluator-workers")):
        value = _first_text(dataset.get(source_key), env_params.get(source_key))
        if value:
            cmd.extend([cli_key, value])
    return cmd


def _results_dir(request: dict[str, Any], env_params: dict[str, Any]) -> Path:
    root = Path(str(env_params.get("results_root") or DEFAULT_RESULTS_ROOT))
    job_id = _safe_name(str(request.get("job_id") or "job"))
    session_id = _safe_name(str(request.get("session_id") or "session"))
    return root / job_id / session_id


def _timeout_s(request: dict[str, Any], env_params: dict[str, Any], dataset: dict[str, Any]) -> float:
    request_timeout = _float_or_default(request.get("agent_start_timeout_s"), 3600.0)
    raw = dataset.get("timeout_s", env_params.get("timeout_s", request_timeout))
    try:
        return max(1.0, min(float(raw), request_timeout))
    except (TypeError, ValueError):
        return request_timeout


def _float_or_default(value: Any, default: float) -> float:
    try:
        return max(1.0, float(value))
    except (TypeError, ValueError):
        return float(default)


def _gateway_session_url(request: dict[str, Any], session_id: str) -> str:
    base = str(request.get("gateway_base_url") or "").rstrip("/")
    if not base:
        return ""
    return _containerize_local_gateway_url(f"{base}/{session_id}")


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


def _find_metrics_files(results_dir: Path) -> list[Path]:
    return sorted(results_dir.rglob("*_metrics.json"), key=lambda path: path.stat().st_mtime)


def _find_error_files(results_dir: Path) -> list[Path]:
    return sorted(results_dir.rglob("*_error.json"), key=lambda path: path.stat().st_mtime)


def _summarize_metrics(paths: list[Path]) -> dict[str, Any]:
    loaded: list[dict[str, Any]] = []
    for path in paths:
        try:
            body = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(body, dict):
            loaded.append(body)
    asr_values = [_float_or_none(item.get("attack_success_rate")) for item in loaded]
    insucess_values = [
        _float_or_none(item.get("insucess_rate", item.get("insuccess_rate")))
        for item in loaded
    ]
    score_values = [_float_or_none(item.get("average_attack_score")) for item in loaded]
    asr = _mean([value for value in asr_values if value is not None])
    insucess = _mean([value for value in insucess_values if value is not None])
    avg_score = _mean([value for value in score_values if value is not None])
    return {
        "metrics_file_count": len(paths),
        "parsed_metrics_count": len(loaded),
        "attack_success_rate": asr,
        "insucess_rate": insucess,
        "average_attack_score": avg_score,
        "reward_10": round(insucess * 10.0, 6),
    }


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
            "bench": "openrt",
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


def _float_or_none(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _safe_name(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(value))
    return safe.strip("_") or "item"


if __name__ == "__main__":
    raise SystemExit(main())
