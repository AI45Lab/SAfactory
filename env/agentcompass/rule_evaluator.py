from __future__ import annotations

import math
from typing import Any


NORMALIZATION_BY_BENCHMARK = {
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


def _failed(benchmark: str, reason: str) -> dict[str, Any]:
    return {
        "status": "failed",
        "score": 0.0,
        "reason": reason,
        "error_text": "invalid AgentCompass normalized score schema",
        "artifacts": {"bench": "agentcompass", "benchmark": benchmark},
    }


def _number(value: Any, *, lower: float, upper: float) -> float | None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    rendered = float(value)
    if not math.isfinite(rendered) or rendered < lower or rendered > upper:
        return None
    return rendered


def _finite_number(value: Any) -> float | None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    rendered = float(value)
    return rendered if math.isfinite(rendered) else None


def _validated_metrics(metrics: dict[str, Any], strategy: str) -> bool:
    return (
        metrics.get("schema_validated") is True
        and metrics.get("normalization_strategy") == strategy
        and isinstance(metrics.get("agentcompass_status"), str)
        and metrics["agentcompass_status"].strip().lower() == "completed"
        and isinstance(metrics.get("agentcompass_error"), str)
        and not metrics["agentcompass_error"].strip()
    )


def evaluate(request: Any, spec: Any, trajectory: Any) -> dict[str, Any]:
    del trajectory
    start_result = getattr(request, "start_result", None)
    request_session = getattr(request, "session_id", "")
    result_session = getattr(start_result, "session_id", request_session)
    if result_session != request_session:
        return _failed("", "AgentCompass start result session identity was inconsistent")
    if (
        getattr(start_result, "status", "succeeded") != "succeeded"
        or getattr(start_result, "terminated", True) is not True
        or getattr(start_result, "truncated", False) is not False
    ):
        return _failed("", "AgentCompass episode was not successfully completed and terminal")
    metrics = getattr(start_result, "metrics", None)
    metrics = dict(metrics) if isinstance(metrics, dict) else {}
    benchmark_value = metrics.get("benchmark")
    benchmark = benchmark_value.strip() if isinstance(benchmark_value, str) else ""
    normalizer = NORMALIZATION_BY_BENCHMARK.get(benchmark)
    if normalizer is None:
        return _failed(benchmark, f"unsupported AgentCompass score schema for benchmark {benchmark!r}")

    strategy = metrics.get("normalization_strategy")
    allowed_strategies = (
        {"frontierscience_olympiad", "frontierscience_research"}
        if normalizer == "frontierscience"
        else {normalizer}
    )
    if strategy not in allowed_strategies or not _validated_metrics(metrics, strategy):
        return _failed(benchmark, f"AgentCompass {benchmark} normalized metrics failed status/schema validation")
    for name in ("task_id", "harness", "environment", "sample_id"):
        if not isinstance(metrics.get(name), str) or not metrics[name].strip():
            return _failed(benchmark, f"AgentCompass metrics did not contain valid {name} identity")

    correct = metrics.get("correct")
    if not isinstance(correct, bool):
        return _failed(benchmark, f"AgentCompass {benchmark} metrics did not contain boolean correct")
    if normalizer in {
        "llm_judge", "deepsearchqa_judge", "sealqa_judge",
        "special_pattern_correct", "swebench_resolved",
    }:
        raw_score = _number(metrics.get("raw_score"), lower=0.0, upper=1.0)
        normalized = _number(metrics.get("normalized_reward_10"), lower=0.0, upper=10.0)
        expected = 1.0 if correct else 0.0
        if raw_score != expected or normalized != expected * 10.0:
            return _failed(benchmark, f"AgentCompass {benchmark} binary result fields were inconsistent")
        score = 10.0 if correct else 0.0
        reason = f"AgentCompass {benchmark} native correctness mapped to SAfactory reward 10/0"
    elif normalizer == "scicode_fractional":
        raw_score = _number(metrics.get("raw_score"), lower=0.0, upper=1.0)
        normalized = _number(metrics.get("normalized_reward_10"), lower=0.0, upper=10.0)
        if (
            raw_score is None
            or normalized is None
            or not math.isclose(normalized, raw_score * 10.0, abs_tol=1e-9)
            or correct is not math.isclose(raw_score, 1.0, abs_tol=1e-9)
        ):
            return _failed(benchmark, "AgentCompass scicode fractional score fields were inconsistent")
        score = normalized
        reason = "AgentCompass scicode subproblem correctness mapped to SAfactory reward 0-10"
    elif normalizer == "frontierscience":
        upper = 1.0 if strategy == "frontierscience_olympiad" else 10.0
        raw_score = _number(metrics.get("raw_score"), lower=0.0, upper=upper)
        normalized = _number(metrics.get("normalized_reward_10"), lower=0.0, upper=10.0)
        if raw_score is None or normalized is None:
            return _failed(benchmark, "AgentCompass frontierscience score fields were invalid")
        if strategy == "frontierscience_olympiad":
            expected = 1.0 if correct else 0.0
            if raw_score != expected or normalized != expected * 10.0:
                return _failed(benchmark, "AgentCompass frontierscience olympiad fields were inconsistent")
            reason = "AgentCompass FrontierScience olympiad correctness mapped to SAfactory reward 10/0"
        else:
            threshold = _finite_number(metrics.get("passing_threshold"))
            if (
                threshold is None
                or not math.isclose(normalized, raw_score, abs_tol=1e-9)
                or correct is not (raw_score >= threshold)
            ):
                return _failed(benchmark, "AgentCompass frontierscience research fields were inconsistent")
            reason = "AgentCompass FrontierScience research rubric score preserved on SAfactory reward 0-10"
        score = normalized
    else:
        return _failed(benchmark, f"unknown normalization strategy {normalizer!r}")

    return {
        "session_id": getattr(request, "session_id", ""),
        "eval_id": getattr(spec, "eval_id", "agentcompass_rule"),
        "status": "succeeded",
        "score": score,
        "raw_score": raw_score,
        "reason": reason,
        "artifacts": {
            "bench": "agentcompass",
            "benchmark": benchmark,
            "harness": metrics.get("harness"),
            "environment": metrics.get("environment"),
            "task_id": metrics.get("task_id"),
            "sample_id": metrics.get("sample_id"),
            "normalization_strategy": metrics.get("normalization_strategy", normalizer),
            "contract_only": bool(metrics.get("contract_only", False)),
        },
    }
