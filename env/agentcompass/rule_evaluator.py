from __future__ import annotations

import math
from typing import Any


NORMALIZATION_BY_BENCHMARK = {
    "browsecomp": "binary_correct",
    "deepsearchqa": "binary_correct",
    "frontierscience": "frontierscience",
    "hle": "binary_correct",
    "hle_verified": "binary_correct",
    "scicode": "scicode_fractional",
    "sealqa": "sealqa_judge",
    "sgi_deep_research": "sgi_binary_judge",
    "special_pattern_check": "binary_correct",
    "swebench_verified": "binary_correct",
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


def _validated_metrics(metrics: dict[str, Any], strategy: str) -> bool:
    return (
        metrics.get("schema_validated") is True
        and metrics.get("normalization_strategy") == strategy
        and isinstance(metrics.get("agentcompass_status"), str)
        and metrics["agentcompass_status"].strip().lower() == "completed"
        and metrics.get("agentcompass_error") in (None, "")
    )


def evaluate(request: Any, spec: Any, trajectory: Any) -> dict[str, Any]:
    del trajectory
    start_result = getattr(request, "start_result", None)
    metrics = getattr(start_result, "metrics", None)
    metrics = dict(metrics) if isinstance(metrics, dict) else {}
    benchmark = str(metrics.get("benchmark") or "").strip()
    normalizer = NORMALIZATION_BY_BENCHMARK.get(benchmark)
    if normalizer is None:
        return _failed(benchmark, f"unsupported AgentCompass score schema for benchmark {benchmark!r}")

    correct = metrics.get("correct")
    if not isinstance(correct, bool):
        return _failed(benchmark, f"AgentCompass {benchmark} metrics did not contain boolean correct")
    if normalizer == "binary_correct":
        score = 10.0 if correct else 0.0
        raw_score = 1.0 if correct else 0.0
        reason = f"AgentCompass {benchmark} correctness mapped to SAfactory reward 10/0"
    elif normalizer == "scicode_fractional":
        if not _validated_metrics(metrics, normalizer):
            return _failed(benchmark, "AgentCompass scicode normalized metrics failed status/schema validation")
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
        strategy = metrics.get("normalization_strategy")
        if strategy not in {"frontierscience_olympiad", "frontierscience_research"} or not _validated_metrics(
            metrics, strategy
        ):
            return _failed(benchmark, "AgentCompass frontierscience normalized metrics failed status/schema validation")
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
            threshold = _number(metrics.get("passing_threshold"), lower=0.0, upper=10.0)
            if (
                threshold is None
                or not math.isclose(normalized, raw_score, abs_tol=1e-9)
                or correct is not (raw_score >= threshold)
            ):
                return _failed(benchmark, "AgentCompass frontierscience research fields were inconsistent")
            reason = "AgentCompass FrontierScience research rubric score preserved on SAfactory reward 0-10"
        score = normalized
    elif normalizer in {"sgi_binary_judge", "sealqa_judge"}:
        if not _validated_metrics(metrics, normalizer):
            return _failed(benchmark, f"AgentCompass {benchmark} normalized metrics failed status/schema validation")
        raw_score = _number(metrics.get("raw_score"), lower=0.0, upper=1.0)
        normalized = _number(metrics.get("normalized_reward_10"), lower=0.0, upper=10.0)
        expected = 1.0 if correct else 0.0
        if raw_score != expected or normalized != expected * 10.0:
            return _failed(benchmark, f"AgentCompass {benchmark} binary judge fields were inconsistent")
        score = normalized
        reason = f"AgentCompass {benchmark} judge correctness mapped to SAfactory reward 10/0"
    else:
        return _failed(benchmark, f"unknown normalization strategy {normalizer!r}")

    return {
        "session_id": getattr(request, "session_id", ""),
        "eval_id": getattr(spec, "eval_id", "agentcompass_rule"),
        "status": "succeeded",
        "score": score,
        "raw_score": raw_score,
        "reason": reason,
        "ground_truth_answer": metrics.get("ground_truth_answer"),
        "evaluation_context": (
            dict(metrics["evaluation_context"])
            if isinstance(metrics.get("evaluation_context"), dict) else {}
        ),
        "artifacts": {
            "bench": "agentcompass",
            "benchmark": benchmark,
            "harness": metrics.get("harness"),
            "sample_id": metrics.get("sample_id"),
            "correct": correct,
            "normalization_strategy": metrics.get("normalization_strategy", normalizer),
            "contract_only": bool(metrics.get("contract_only", False)),
        },
    }
