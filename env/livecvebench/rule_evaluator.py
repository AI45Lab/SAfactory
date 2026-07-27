from __future__ import annotations

from typing import Any

from evaluator.eval_types import EvalRequest, EvalResult, EvalSpec, EvalStatus, Trajectory


async def evaluate_rule(
    *,
    request: EvalRequest,
    spec: EvalSpec,
    trajectory: Trajectory,
) -> EvalResult:
    metrics = _start_metrics(request)
    score = _float_or_none(metrics.get("score"))
    if score is None:
        resolved = metrics.get("is_resolved")
        if isinstance(resolved, bool):
            score = 1.0 if resolved else 0.0

    if score is None:
        return EvalResult.failed(
            session_id=request.session_id,
            eval_id=spec.eval_id,
            method=spec.method.value,
            reason="LiveCVEBench metrics did not contain score or is_resolved",
            artifacts={"bench": "livecvebench", "metrics": metrics},
        )

    score = max(0.0, min(1.0, score))
    return EvalResult(
        session_id=request.session_id,
        eval_id=spec.eval_id,
        method=spec.method.value,
        status=EvalStatus.SUCCEEDED.value,
        raw_score=score,
        normalized_score_10=round(score * 10.0, 6),
        reason="LiveCVEBench rule evaluator: reward = resolved rate * 10",
        artifacts={
            "bench": "livecvebench",
            "suite": metrics.get("suite"),
            "task_id": metrics.get("task_id"),
            "is_resolved": metrics.get("is_resolved"),
            "resolved_trials": metrics.get("resolved_trials"),
            "trial_count": metrics.get("trial_count"),
            "failure_modes": metrics.get("failure_modes"),
            "run_metadata_path": metrics.get("run_metadata_path"),
            "result_paths": metrics.get("result_paths"),
        },
    )


def _start_metrics(request: EvalRequest) -> dict[str, Any]:
    start_result = getattr(request, "start_result", None)
    metrics = getattr(start_result, "metrics", None)
    return dict(metrics) if isinstance(metrics, dict) else {}


def _float_or_none(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None
