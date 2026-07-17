from __future__ import annotations

from typing import Any

from evaluator.eval_types import EvalRequest, EvalResult, EvalSpec, EvalStatus, Trajectory


async def evaluate_rule(
    *,
    request: EvalRequest,
    spec: EvalSpec,
    trajectory: Trajectory,
) -> EvalResult:
    """Convert the geo3k runner's 0/1 correctness into a 0-10 Safactory reward.

    The runner already grades the answer with the sympy grader
    (``math_utils.grade_answer_verl``) and stores the correctness under
    ``metrics.score``. This evaluator only normalizes it: ``reward = score * 10``,
    so a correct answer scores 10 and a wrong answer scores 0 -- pass/fail is
    preserved exactly, matching the original geo3k judging.
    """
    metrics = _start_metrics(request)
    score = _float_or_none(metrics.get("score"))
    if score is None:
        passed = metrics.get("passed")
        if isinstance(passed, bool):
            score = 1.0 if passed else 0.0

    if score is None:
        return EvalResult.failed(
            session_id=request.session_id,
            eval_id=spec.eval_id,
            method=spec.method.value,
            reason="geo3k metrics did not contain a numeric score",
            artifacts={"bench": "geo3k", "metrics": metrics},
        )

    normalized_score = _clamp(float(score) * 10.0, 0.0, 10.0)
    return EvalResult(
        session_id=request.session_id,
        eval_id=spec.eval_id,
        method=spec.method.value,
        status=EvalStatus.SUCCEEDED.value,
        raw_score=float(score),
        normalized_score_10=round(normalized_score, 6),
        reason="geo3k rule evaluator: reward = score * 10",
        artifacts={
            "bench": "geo3k",
            "metric": "score",
            "score": float(score),
            "ground_truth": metrics.get("ground_truth"),
            "final_answer": metrics.get("final_answer"),
            "reward_source": metrics.get("reward_source"),
            "total_tool_calls": metrics.get("total_tool_calls"),
        },
    )


def _start_metrics(request: EvalRequest) -> dict[str, Any]:
    start_result = getattr(request, "start_result", None)
    metrics = getattr(start_result, "metrics", None)
    return dict(metrics) if isinstance(metrics, dict) else {}


def _float_or_none(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
