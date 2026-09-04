#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
import signal
import subprocess
import sys
import tempfile
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, TextIO
from urllib.parse import urlsplit, urlunsplit


RESULT_PATH_ENV = "SAFACTORY_RESULT_PATH"
REQUIRE_RESULTS_MOUNT_ENV = "AGENTCOMPASS_REQUIRE_RESULTS_MOUNT"
DEFAULT_RESULTS_ROOT = "/app/results"
DEFAULT_TIMEOUT_S = 1800.0
TERMINATION_GRACE_S = 10.0
DEFAULT_ALLOWED_ENVIRONMENTS = frozenset({"host_process"})


class AdapterError(RuntimeError):
    def __init__(self, message: str, *, error_type: str) -> None:
        super().__init__(message)
        self.error_type = error_type

# These mappings describe result schemas verified at the pinned AgentCompass
# revision. Unknown benchmark schemas are never silently normalized.
RESULT_NORMALIZERS = {
    "browsecomp": "llm_judge",
    "deepsearchqa": "deepsearchqa_judge",
    "frontierscience": "frontierscience",
    "hle": "llm_judge",
    "hle_verified": "llm_judge",
    "scicode": "scicode_fractional",
    "sealqa": "sealqa_judge",
    "sgi_deep_research": "llm_judge",
    "special_pattern_check": "special_pattern_correct",
    "swebench_verified": "swebench_resolved",
}

# AgentCompass harnesses use different names for their interaction budgets.
# Only verified mappings are injected. A one-shot OpenAI chat call inherently
# consumes at most one agent step.
HARNESS_STEP_LIMIT_KEYS = {
    "mini_swe_agent": "step_limit",
}


def _text(value: Any) -> str:
    return str(value or "").strip()


def _safe_part(value: Any) -> str:
    rendered = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in _text(value))
    return rendered.strip("._") or "item"


def _read_request() -> dict[str, Any]:
    raw = sys.stdin.read().strip() or _text(os.environ.get("SAFACTORY_START_REQUEST_JSON"))
    if not raw:
        raise RuntimeError("SimulationStartRequest JSON was not provided")
    body = json.loads(raw)
    if not isinstance(body, dict):
        raise RuntimeError("SimulationStartRequest must be a JSON object")
    return body


def _required(value: Any, name: str) -> str:
    rendered = _text(value)
    if not rendered:
        raise RuntimeError(f"SimulationStartRequest missing {name}")
    return rendered


def _dataset(request: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    env_params = request.get("env_params")
    if not isinstance(env_params, dict):
        raise RuntimeError("SimulationStartRequest env_params must be an object")
    dataset = env_params.get("dataset")
    if not isinstance(dataset, dict):
        raise RuntimeError("env_params.dataset must be one dataset row object")
    return env_params, dataset


def _object(value: Any, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise RuntimeError(f"{name} must be a JSON object")
    return dict(value)


def _sample_id(dataset: dict[str, Any]) -> str:
    raw = dataset.get("sample_id")
    if isinstance(raw, (list, tuple, set, dict)):
        raise RuntimeError("each SAfactory dataset row must select exactly one scalar sample_id")
    return _required(raw, "env_params.dataset.sample_id")


def _allowed_environments() -> set[str]:
    raw = _text(os.environ.get("AGENTCOMPASS_ALLOWED_ENVIRONMENTS"))
    if not raw:
        return set(DEFAULT_ALLOWED_ENVIRONMENTS)
    return {item.strip() for item in raw.split(",") if item.strip()}


@lru_cache(maxsize=1)
def _registered_components() -> tuple[frozenset[str], frozenset[str], frozenset[str]]:
    try:
        from agentcompass.runtime.registry import BENCHMARKS, ENVIRONMENTS, HARNESSES, load_builtin_components

        load_builtin_components()
    except ModuleNotFoundError as exc:
        raise AdapterError(
            f"AgentCompass registry dependency is unavailable: {exc.name or 'unknown module'}",
            error_type="dependency_missing",
        ) from exc
    return frozenset(BENCHMARKS.names()), frozenset(HARNESSES.names()), frozenset(ENVIRONMENTS.names())


def _validate_registered_selection(benchmark: str, harness: str, environment: str) -> None:
    benchmarks, harnesses, environments = _registered_components()
    for kind, value, registered in (
        ("benchmark", benchmark, benchmarks),
        ("harness", harness, harnesses),
        ("environment", environment, environments),
    ):
        if value not in registered:
            raise AdapterError(
                f"AgentCompass {kind} {value!r} is not registered at the pinned revision",
                error_type="component_unregistered",
            )


def _component_selection(dataset: dict[str, Any]) -> tuple[str, str, str]:
    benchmark = _required(dataset.get("benchmark"), "env_params.dataset.benchmark")
    harness = _required(dataset.get("harness"), "env_params.dataset.harness")
    environment = _text(dataset.get("environment")) or "host_process"
    _validate_registered_selection(benchmark, harness, environment)
    allowed = _allowed_environments()
    if environment not in allowed:
        raise AdapterError(
            f"AgentCompass environment {environment!r} is not allowed; allowed environments: {sorted(allowed)}",
            error_type="environment_not_allowed",
        )
    return benchmark, harness, environment


def _merge_params(
    request: dict[str, Any], dataset: dict[str, Any], *, sample_id: str, harness: str
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    benchmark_params = _object(dataset.get("benchmark_params"), "benchmark_params")
    harness_params = _object(dataset.get("harness_params"), "harness_params")
    environment_params = _object(dataset.get("environment_params"), "environment_params")
    model_params = _object(dataset.get("model_params"), "model_params")

    benchmark_params["sample_ids"] = [sample_id]
    benchmark_params["k"] = 1
    # Episode step limits belong to SAfactory and verified harness mappings;
    # a dataset row cannot inject a competing model-level value.
    model_params.pop("max_steps", None)
    temperature = request.get("temperature")
    if temperature is not None:
        model_params["temperature"] = float(temperature)

    max_steps = int(request.get("max_steps", -1))
    if max_steps == 0 or max_steps < -1:
        raise RuntimeError("SAfactory max_steps must be -1 or a positive integer")
    if max_steps > 0:
        limit_key = HARNESS_STEP_LIMIT_KEYS.get(harness)
        if limit_key:
            harness_params[limit_key] = max_steps
        elif harness == "openai_chat" and max_steps < 1:
            raise RuntimeError("openai_chat requires at least one SAfactory step")
    return benchmark_params, harness_params, environment_params, model_params


def _session_url(request: dict[str, Any], session_id: str) -> str:
    injected = _text(
        os.environ.get("SAFACTORY_GATEWAY_SESSION_URL_CONTAINER")
        or os.environ.get("SAFACTORY_GATEWAY_SESSION_URL")
    )
    if injected:
        return injected.rstrip("/")
    base = _required(request.get("gateway_base_url"), "gateway_base_url").rstrip("/")
    url = f"{base}/{session_id}"
    try:
        parts = urlsplit(url)
    except ValueError:
        return url
    if parts.hostname not in {"127.0.0.1", "localhost", "::1"}:
        return url
    netloc = "host.docker.internal"
    if parts.port is not None:
        netloc += f":{parts.port}"
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))


def _timeout_s(request: dict[str, Any], dataset: dict[str, Any]) -> float:
    requested = request.get("agent_start_timeout_s")
    row_timeout = dataset.get("timeout_seconds")
    values = [value for value in (requested, row_timeout) if value not in (None, "")]
    try:
        timeout = min(float(value) for value in values) if values else DEFAULT_TIMEOUT_S
    except (TypeError, ValueError) as exc:
        raise RuntimeError("timeout_seconds must be numeric") from exc
    if not math.isfinite(timeout) or timeout <= 0:
        raise RuntimeError("timeout_seconds must be positive")
    return timeout


def _episode_root(request: dict[str, Any], env_params: dict[str, Any], task_id: str) -> Path:
    root = Path(_text(env_params.get("results_root")) or DEFAULT_RESULTS_ROOT)
    return root / _safe_part(request.get("job_id")) / _safe_part(request.get("session_id")) / _safe_part(task_id)


def _agentcompass_executable() -> Path:
    executable = Path(sys.executable).with_name("agentcompass")
    if not executable.is_absolute():
        executable = executable.resolve()
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise AdapterError(
            f"AgentCompass CLI derived from the runner Python is missing or not executable: {executable}",
            error_type="dependency_missing",
        )
    return executable


def _build_command(
    *,
    agentcompass_executable: Path,
    benchmark: str,
    harness: str,
    environment: str,
    model: str,
    session_url: str,
    benchmark_params: dict[str, Any],
    harness_params: dict[str, Any],
    environment_params: dict[str, Any],
    model_params: dict[str, Any],
    result_root: Path,
) -> list[str]:
    if not agentcompass_executable.is_absolute():
        raise RuntimeError("AgentCompass CLI path must be absolute")
    command = [
        str(agentcompass_executable),
        "run",
        benchmark,
        harness,
        model,
        "--env",
        environment,
        "--benchmark-params",
        json.dumps(benchmark_params, separators=(",", ":"), ensure_ascii=False),
        "--harness-params",
        json.dumps(harness_params, separators=(",", ":"), ensure_ascii=False),
        "--env-params",
        json.dumps(environment_params, separators=(",", ":"), ensure_ascii=False),
        "--model-params",
        json.dumps(model_params, separators=(",", ":"), ensure_ascii=False),
        "--model-base-url",
        session_url,
        "--model-api-key",
        "EMPTY",
        "--model-api-protocol",
        "openai-chat",
        "--task-concurrency",
        "1",
        "--max-retries",
        "0",
        "--results-dir",
        str(result_root),
        "--run-name",
        "episode",
        "--run-id",
        "single-sample",
        "--progress",
        "none",
    ]
    return command


def _terminate_process_group(process: subprocess.Popen[Any], grace_s: float = TERMINATION_GRACE_S) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=max(0.0, grace_s))
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def _run_command(
    command: list[str],
    *,
    timeout_s: float,
    stdout_file: TextIO,
    stderr_file: TextIO,
) -> tuple[int, bool]:
    process = subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=stdout_file,
        stderr=stderr_file,
        text=True,
        start_new_session=True,
    )
    try:
        return process.wait(timeout=timeout_s), False
    except subprocess.TimeoutExpired:
        _terminate_process_group(process)
        return 124, True


def _find_detail(result_root: Path, sample_id: str) -> tuple[dict[str, Any], Path]:
    candidates = sorted(result_root.rglob("details/*.json"))
    if not candidates:
        raise RuntimeError("AgentCompass produced no details JSON")
    matching = [path for path in candidates if _safe_part(sample_id) in path.name]
    selected = matching[0] if len(matching) == 1 else candidates[0] if len(candidates) == 1 else None
    if selected is None:
        raise RuntimeError(f"AgentCompass produced ambiguous details for sample {sample_id!r}")
    body = json.loads(selected.read_text(encoding="utf-8"))
    if not isinstance(body, dict):
        raise RuntimeError("AgentCompass detail is not an object")
    return body, selected


def _attempt(detail: dict[str, Any]) -> dict[str, Any]:
    attempts = detail.get("attempts")
    if not isinstance(attempts, dict) or len(attempts) != 1:
        raise RuntimeError("AgentCompass detail must contain exactly one attempt")
    value = next(iter(attempts.values()))
    if not isinstance(value, dict):
        raise RuntimeError("AgentCompass detail attempt must be an object")
    return value


def _number(value: Any, name: str, *, lower: float, upper: float) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise RuntimeError(f"AgentCompass {name} must be numeric")
    rendered = float(value)
    if not math.isfinite(rendered):
        raise RuntimeError(f"AgentCompass {name} contained a non-finite score")
    if rendered < lower or rendered > upper:
        raise RuntimeError(f"AgentCompass {name} must be between {lower:g} and {upper:g}")
    return rendered


def _finite_number(value: Any, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise RuntimeError(f"AgentCompass {name} must be numeric")
    rendered = float(value)
    if not math.isfinite(rendered):
        raise RuntimeError(f"AgentCompass {name} contained a non-finite value")
    return rendered


def _consistent_bool(detail: dict[str, Any], attempt: dict[str, Any], name: str) -> bool:
    values = [value for value in (detail.get(name), attempt.get(name)) if value is not None]
    if not values or any(not isinstance(value, bool) for value in values):
        raise RuntimeError(f"AgentCompass detail did not contain boolean {name}")
    if any(value is not values[0] for value in values[1:]):
        raise RuntimeError(f"AgentCompass detail contained inconsistent {name} values")
    return values[0]


def _successful_attempt(benchmark: str, detail: dict[str, Any], attempt: dict[str, Any]) -> tuple[str, Any]:
    status = attempt.get("status", detail.get("status"))
    if not isinstance(status, str) or status.strip().lower() != "completed":
        raise RuntimeError(f"AgentCompass {benchmark} detail status was not completed")
    error = attempt.get("error", detail.get("error"))
    if error not in (None, ""):
        raise RuntimeError(f"AgentCompass {benchmark} detail contained an error")
    return status, error


def _object_field(parent: dict[str, Any], name: str, context: str) -> dict[str, Any]:
    value = parent.get(name)
    if not isinstance(value, dict):
        raise RuntimeError(f"AgentCompass {context}.{name} must be an object")
    return value


def _validate_native_identity(
    detail: dict[str, Any], *, sample_id: str
) -> None:
    task_id = detail.get("task_id")
    if not isinstance(task_id, str) or task_id != sample_id:
        raise RuntimeError("AgentCompass detail task_id did not match the selected sample_id")


def _normalized_result(
    *,
    correct: bool,
    reward: float,
    raw_score: float,
    status: str,
    error: Any,
    strategy: str,
    **extra: Any,
) -> dict[str, Any]:
    return {
        "correct": correct,
        "normalized_reward_10": reward,
        "raw_score": raw_score,
        "agentcompass_status": status,
        "agentcompass_error": error,
        "normalization_strategy": strategy,
        "schema_validated": True,
        **extra,
    }


def _normalize_scicode(detail: dict[str, Any], attempt: dict[str, Any]) -> dict[str, Any]:
    status, error = _successful_attempt("scicode", detail, attempt)
    correct = _consistent_bool(detail, attempt, "correct")
    score = _number(attempt.get("score", detail.get("score")), "scicode score", lower=0.0, upper=1.0)
    meta = _object_field(attempt, "meta", "scicode")
    evaluation = _object_field(meta, "evaluation", "scicode.meta")
    if evaluation.get("error") not in (None, ""):
        raise RuntimeError("AgentCompass scicode evaluation contained an error")
    problem_correct = evaluation.get("problem_correct")
    if type(problem_correct) is not int or problem_correct not in (0, 1):
        raise RuntimeError("AgentCompass scicode problem_correct must be integer 0 or 1")
    total_correct = evaluation.get("total_correct")
    total_steps = evaluation.get("total_steps")
    if type(total_correct) is not int or type(total_steps) is not int or total_steps < 0:
        raise RuntimeError("AgentCompass scicode step counts must be non-negative integers")
    if total_correct < 0 or total_correct > total_steps:
        raise RuntimeError("AgentCompass scicode total_correct was outside total_steps")
    subproblem_score = _number(
        evaluation.get("subproblem_correctness"),
        "scicode subproblem_correctness",
        lower=0.0,
        upper=1.0,
    )
    expected_score = total_correct / total_steps if total_steps else 0.0
    if not math.isclose(score, subproblem_score, abs_tol=1e-9) or not math.isclose(
        score, expected_score, abs_tol=1e-9
    ):
        raise RuntimeError("AgentCompass scicode score and step counts were inconsistent")
    expected_correct = bool(total_steps and total_correct == total_steps)
    if correct is not bool(problem_correct) or correct is not expected_correct:
        raise RuntimeError("AgentCompass scicode correct and evaluation fields were inconsistent")
    if not isinstance(evaluation.get("steps"), list):
        raise RuntimeError("AgentCompass scicode evaluation.steps must be a list")
    return _normalized_result(
        correct=correct,
        reward=score * 10.0,
        raw_score=score,
        status=status,
        error=error,
        strategy="scicode_fractional",
        agentcompass_score=score,
        scicode_problem_correct=problem_correct,
        scicode_total_correct=total_correct,
        scicode_total_steps=total_steps,
    )


def _scoring(attempt: dict[str, Any], benchmark: str) -> dict[str, Any]:
    extra = _object_field(attempt, "extra", benchmark)
    scoring = _object_field(extra, "scoring", f"{benchmark}.extra")
    if scoring.get("error") not in (None, ""):
        raise RuntimeError(f"AgentCompass {benchmark} scoring contained an error")
    return scoring


def _normalize_llm_judge(benchmark: str, detail: dict[str, Any], attempt: dict[str, Any]) -> dict[str, Any]:
    status, error = _successful_attempt(benchmark, detail, attempt)
    correct = _consistent_bool(detail, attempt, "correct")
    scoring = _scoring(attempt, benchmark)
    if scoring.get("evaluation_type") != "llm_judge" or scoring.get("correct") is not correct:
        raise RuntimeError(f"AgentCompass {benchmark} LLM judge fields were inconsistent")
    return _normalized_result(
        correct=correct,
        reward=10.0 if correct else 0.0,
        raw_score=1.0 if correct else 0.0,
        status=status,
        error=error,
        strategy="llm_judge",
    )


def _normalize_deepsearchqa(detail: dict[str, Any], attempt: dict[str, Any]) -> dict[str, Any]:
    status, error = _successful_attempt("deepsearchqa", detail, attempt)
    correct = _consistent_bool(detail, attempt, "correct")
    scoring = _scoring(attempt, "deepsearchqa")
    if scoring.get("evaluation_type") != "deepsearchqa_judge" or scoring.get("correct") is not correct:
        raise RuntimeError("AgentCompass deepsearchqa judge fields were inconsistent")
    return _normalized_result(
        correct=correct,
        reward=10.0 if correct else 0.0,
        raw_score=1.0 if correct else 0.0,
        status=status,
        error=error,
        strategy="deepsearchqa_judge",
    )


def _normalize_frontierscience(detail: dict[str, Any], attempt: dict[str, Any]) -> dict[str, Any]:
    status, error = _successful_attempt("frontierscience", detail, attempt)
    correct = _consistent_bool(detail, attempt, "correct")
    scoring = _scoring(attempt, "frontierscience")
    if scoring.get("correct") is not correct:
        raise RuntimeError("AgentCompass frontierscience correct and scoring.correct were inconsistent")
    evaluation_type = scoring.get("evaluation_type")
    if evaluation_type == "frontierscience_olympiad_judge":
        return _normalized_result(
            correct=correct,
            reward=10.0 if correct else 0.0,
            raw_score=1.0 if correct else 0.0,
            status=status,
            error=error,
            strategy="frontierscience_olympiad",
            frontierscience_mode="olympiad",
        )
    if evaluation_type != "frontierscience_research_rubric":
        raise RuntimeError("AgentCompass frontierscience evaluation_type was not recognized")
    total_score = _number(scoring.get("total_score"), "frontierscience total_score", lower=0.0, upper=10.0)
    threshold = _finite_number(scoring.get("passing_threshold"), "frontierscience passing_threshold")
    items = scoring.get("rubric_items")
    if not isinstance(items, list):
        raise RuntimeError("AgentCompass frontierscience rubric_items must be a list")
    awarded_sum = 0.0
    for item in items:
        if not isinstance(item, dict):
            raise RuntimeError("AgentCompass frontierscience rubric item was invalid")
        max_points = _finite_number(item.get("max_points"), "frontierscience rubric max_points")
        if max_points < 0.0:
            raise RuntimeError("AgentCompass frontierscience rubric max_points must be non-negative")
        awarded = _number(
            item.get("awarded_points"), "frontierscience rubric awarded_points", lower=0.0, upper=max_points
        )
        awarded_sum += awarded
    if not math.isclose(total_score, awarded_sum, abs_tol=1e-6):
        raise RuntimeError("AgentCompass frontierscience total_score did not match rubric items")
    if correct is not (total_score >= threshold):
        raise RuntimeError("AgentCompass frontierscience correct and threshold fields were inconsistent")
    return _normalized_result(
        correct=correct,
        reward=total_score,
        raw_score=total_score,
        status=status,
        error=error,
        strategy="frontierscience_research",
        frontierscience_mode="research",
        passing_threshold=threshold,
    )


def _normalize_sealqa(detail: dict[str, Any], attempt: dict[str, Any]) -> dict[str, Any]:
    status, error = _successful_attempt("sealqa", detail, attempt)
    correct = _consistent_bool(detail, attempt, "correct")
    scoring = _scoring(attempt, "sealqa")
    if scoring.get("evaluation_type") != "sealqa_official_llm_judge":
        raise RuntimeError("AgentCompass sealqa evaluation_type was not recognized")
    grade = scoring.get("grade")
    labels = {"A": "correct", "B": "incorrect", "C": "not_attempted"}
    if grade not in labels or scoring.get("label") != labels[grade]:
        raise RuntimeError("AgentCompass sealqa grade and label were inconsistent")
    if scoring.get("correct") is not correct or correct is not (grade == "A"):
        raise RuntimeError("AgentCompass sealqa grade and correct fields were inconsistent")
    return _normalized_result(
        correct=correct,
        reward=10.0 if correct else 0.0,
        raw_score=1.0 if correct else 0.0,
        status=status,
        error=error,
        strategy="sealqa_judge",
        sealqa_grade=grade,
        sealqa_label=scoring["label"],
    )


def _normalize_special_pattern(detail: dict[str, Any], attempt: dict[str, Any]) -> dict[str, Any]:
    status, error = _successful_attempt("special_pattern_check", detail, attempt)
    correct = _consistent_bool(detail, attempt, "correct")
    extra = attempt.get("extra")
    if extra is not None and not isinstance(extra, dict):
        raise RuntimeError("AgentCompass special_pattern_check extra must be an object")
    badcases = (extra or {}).get("badcase_analyzers")
    if badcases is not None:
        if not isinstance(badcases, dict):
            raise RuntimeError("AgentCompass special_pattern_check badcase_analyzers must be an object")
        if any(isinstance(item, dict) and item.get("error") for item in badcases.values()):
            raise RuntimeError("AgentCompass special_pattern_check analyzer failed")
    return _normalized_result(
        correct=correct, reward=10.0 if correct else 0.0, raw_score=1.0 if correct else 0.0,
        status=status, error=error, strategy="special_pattern_correct",
    )


def _normalize_swebench(detail: dict[str, Any], attempt: dict[str, Any]) -> dict[str, Any]:
    status, error = _successful_attempt("swebench_verified", detail, attempt)
    correct = _consistent_bool(detail, attempt, "correct")
    extra = _object_field(attempt, "extra", "swebench_verified")
    evaluation = _object_field(extra, "eval_raw_data", "swebench_verified.extra")
    if extra.get("status") != "completed" or evaluation.get("completed") is not True:
        raise RuntimeError("AgentCompass swebench_verified evaluation was not completed")
    if evaluation.get("resolved") is not correct:
        raise RuntimeError("AgentCompass swebench_verified resolved fields were inconsistent")
    return _normalized_result(
        correct=correct, reward=10.0 if correct else 0.0, raw_score=1.0 if correct else 0.0,
        status=status, error=error, strategy="swebench_resolved",
    )


def _normalize_detail(benchmark: str, detail: dict[str, Any]) -> dict[str, Any]:
    normalizer = RESULT_NORMALIZERS.get(benchmark)
    if normalizer is None:
        raise RuntimeError(
            f"unsupported AgentCompass result schema for benchmark {benchmark!r}; add an explicit normalization mapping"
        )
    attempt = _attempt(detail)
    if normalizer == "scicode_fractional":
        return _normalize_scicode(detail, attempt)
    if normalizer == "frontierscience":
        return _normalize_frontierscience(detail, attempt)
    if normalizer == "llm_judge":
        return _normalize_llm_judge(benchmark, detail, attempt)
    if normalizer == "deepsearchqa_judge":
        return _normalize_deepsearchqa(detail, attempt)
    if normalizer == "sealqa_judge":
        return _normalize_sealqa(detail, attempt)
    if normalizer == "special_pattern_correct":
        return _normalize_special_pattern(detail, attempt)
    if normalizer == "swebench_resolved":
        return _normalize_swebench(detail, attempt)
    raise RuntimeError(f"unknown AgentCompass normalizer {normalizer!r} for benchmark {benchmark!r}")


def _result_from_detail(
    *,
    session_id: str,
    task_id: str,
    benchmark: str,
    harness: str,
    environment: str,
    sample_id: str,
    detail: dict[str, Any],
    duration_ms: float,
) -> dict[str, Any]:
    _validate_native_identity(detail, sample_id=sample_id)
    normalized = _normalize_detail(benchmark, detail)
    return {
        "session_id": session_id,
        "status": "succeeded",
        "total_reward": normalized["normalized_reward_10"],
        "step_count": 1,
        "terminated": True,
        "truncated": False,
        "error_text": None,
        "metrics": {
            "bench": "agentcompass",
            "task_id": task_id,
            "benchmark": benchmark,
            "harness": harness,
            "environment": environment,
            "sample_id": sample_id,
            "duration_ms": round(duration_ms, 3),
            **normalized,
        },
    }


def _failure(
    session_id: str,
    error_text: str,
    *,
    started_at: float,
    truncated: bool = False,
    metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "session_id": session_id,
        "status": "truncated" if truncated else "failed",
        "total_reward": 0.0,
        "step_count": 0,
        "terminated": True,
        "truncated": truncated,
        "error_text": error_text,
        "metrics": {
            "bench": "agentcompass",
            "duration_ms": round((time.perf_counter() - started_at) * 1000, 3),
            **(metrics or {}),
        },
    }


def _subprocess_error_type(stderr_text: str) -> str:
    lowered = stderr_text.lower()
    dependency_markers = ("modulenotfounderror", "no module named", "missing optional dependency", "command not found")
    if any(marker in lowered for marker in dependency_markers):
        return "dependency_missing"
    if (
        "filenotfounderror" in lowered
        or "no such file or directory" in lowered
        or "does not exist" in lowered
        or ("dataset" in lowered and any(marker in lowered for marker in ("missing", "not found", "unavailable")))
        or ("repository" in lowered and any(marker in lowered for marker in ("missing", "not found")))
    ):
        return "asset_missing"
    return "agentcompass_failed"


def _exception_error_type(exc: Exception) -> str:
    return str(getattr(exc, "error_type", "adapter_error"))


def _write_result(result: dict[str, Any]) -> None:
    artifact = _text(os.environ.get(RESULT_PATH_ENV))
    if artifact:
        try:
            path = Path(artifact)
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
            temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            temporary.replace(path)
        except Exception as exc:
            print(f"SAFACTORY_RUNNER_DIAGNOSTIC result artifact write failed: {exc}", file=sys.stderr)
    print(json.dumps(result, ensure_ascii=False), flush=True)


def _rjob_results_preflight() -> None:
    artifact_text = _text(os.environ.get(RESULT_PATH_ENV))
    if not artifact_text:
        raise RuntimeError(f"{RESULT_PATH_ENV} is required for AgentCompass RJob result handoff")
    artifact = Path(artifact_text)
    mount_root = Path(DEFAULT_RESULTS_ROOT)
    if not artifact.is_absolute():
        raise RuntimeError(f"{RESULT_PATH_ENV} must be an absolute path under {mount_root}")
    try:
        artifact.relative_to(mount_root)
    except ValueError as exc:
        raise RuntimeError(f"{RESULT_PATH_ENV} must be located under {mount_root}") from exc
    if not os.path.ismount(mount_root):
        raise RuntimeError(
            f"AgentCompass RJob requires {mount_root} to be a real mount point; "
            "provide mount_config through the private/global --rjob-config"
        )
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", prefix=".safactory-write-probe-", dir=mount_root
        ) as probe:
            probe.write("write-probe\n")
            probe.flush()
    except OSError as exc:
        raise RuntimeError(
            f"results_mount_not_writable: AgentCompass RJob cannot write to {mount_root}: {exc}"
        ) from exc


def main() -> int:
    started_at = time.perf_counter()
    session_id = _text(os.environ.get("SAFACTORY_SESSION_ID"))
    if _text(os.environ.get(REQUIRE_RESULTS_MOUNT_ENV)) == "1":
        try:
            _rjob_results_preflight()
        except Exception as exc:
            print(f"SAFACTORY_RUNNER_DIAGNOSTIC RJob result mount preflight failed: {exc}", file=sys.stderr)
            return 2
    try:
        request = _read_request()
        session_id = _required(request.get("session_id"), "session_id")
        _required(request.get("job_id"), "job_id")
        model = _required(request.get("model"), "model")
        env_params, dataset = _dataset(request)
        task_id = _required(dataset.get("task_id"), "env_params.dataset.task_id")
        sample_id = _sample_id(dataset)
        benchmark, harness, environment = _component_selection(dataset)
        benchmark_params, harness_params, environment_params, model_params = _merge_params(
            request,
            dataset,
            sample_id=sample_id,
            harness=harness,
        )
        session_url = _session_url(request, session_id)
        result_root = _episode_root(request, env_params, task_id)
        result_root.mkdir(parents=True, exist_ok=True)

        if dataset.get("contract_only") is True:
            contract_normalizer = RESULT_NORMALIZERS.get(benchmark)
            contract_score = {
                "correct": False,
                "normalized_reward_10": 0.0,
                "raw_score": 0.0,
            } if contract_normalizer == "binary_correct" else {}
            _write_result(
                {
                    "session_id": session_id,
                    "status": "succeeded",
                    "total_reward": 0.0,
                    "step_count": 0,
                    "terminated": True,
                    "truncated": False,
                    "error_text": None,
                    "metrics": {
                        "bench": "agentcompass",
                        "task_id": task_id,
                        "benchmark": benchmark,
                        "harness": harness,
                        "environment": environment,
                        "sample_id": sample_id,
                        "contract_only": True,
                        "offline_assets_ready": False,
                        **contract_score,
                    },
                }
            )
            return 0

        command = _build_command(
            agentcompass_executable=_agentcompass_executable(),
            benchmark=benchmark,
            harness=harness,
            environment=environment,
            model=model,
            session_url=session_url,
            benchmark_params=benchmark_params,
            harness_params=harness_params,
            environment_params=environment_params,
            model_params=model_params,
            result_root=result_root,
        )
        timeout_s = _timeout_s(request, dataset)
        stdout_path = result_root / "agentcompass.stdout.log"
        stderr_path = result_root / "agentcompass.stderr.log"
        print(
            f"SAFACTORY_RUNNER_DIAGNOSTIC starting AgentCompass "
            f"benchmark={benchmark} harness={harness} sample={sample_id} timeout_s={timeout_s}",
            file=sys.stderr,
            flush=True,
        )
        with stdout_path.open("w", encoding="utf-8") as stdout_file, stderr_path.open(
            "w", encoding="utf-8"
        ) as stderr_file:
            returncode, timed_out = _run_command(
                command,
                timeout_s=timeout_s,
                stdout_file=stdout_file,
                stderr_file=stderr_file,
            )
        if timed_out:
            _write_result(
                _failure(
                    session_id,
                    f"AgentCompass timed out after {timeout_s:.1f}s",
                    started_at=started_at,
                    truncated=True,
                    metrics={
                        "timeout_layer": "agentcompass_process_group",
                        "task_id": task_id,
                        "benchmark": benchmark,
                        "harness": harness,
                        "sample_id": sample_id,
                    },
                )
            )
            return 0
        if returncode != 0:
            stderr_text = stderr_path.read_text(encoding="utf-8", errors="replace")[-16384:]
            raise AdapterError(
                f"AgentCompass {benchmark}/{harness}/{environment} exited with code {returncode}; see diagnostics artifact",
                error_type=_subprocess_error_type(stderr_text),
            )
        detail, _ = _find_detail(result_root, sample_id)
        _write_result(
            _result_from_detail(
                session_id=session_id,
                task_id=task_id,
                benchmark=benchmark,
                harness=harness,
                environment=environment,
                sample_id=sample_id,
                detail=detail,
                duration_ms=(time.perf_counter() - started_at) * 1000,
            )
        )
        return 0
    except Exception as exc:
        _write_result(
            _failure(
                session_id,
                str(exc),
                started_at=started_at,
                metrics={"error_type": _exception_error_type(exc)},
            )
        )
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
