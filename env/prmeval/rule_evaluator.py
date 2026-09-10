from __future__ import annotations

import math
from numbers import Real
from typing import Any

from evaluator.eval_types import (
    EvalRequest,
    EvalResult,
    EvalSpec,
    EvalStatus,
    Trajectory,
)


async def evaluate_rule(
    *,
    request: EvalRequest,
    spec: EvalSpec,
    trajectory: Trajectory,
) -> EvalResult:
    """Normalize the stored native MSE without rerunning the benchmark."""
    metrics = _start_metrics(request)

    try:
        score = _float_or_none(metrics["metrics"]["progress"]["mse"])
        if score is None:
            return EvalResult.failed(
                session_id=request.session_id,
                eval_id=spec.eval_id,
                method=spec.method.value,
                reason="prmeval metrics did not contain a numeric score",
                artifacts={"bench": "prmeval", "metrics": metrics},
            )
        reward = mse_to_reward(score)
    except ValueError as exc:
        return EvalResult.failed(
            session_id=request.session_id,
            eval_id=spec.eval_id,
            method=spec.method.value,
            reason=str(exc),
        )
    return EvalResult(
        session_id=request.session_id,
        eval_id=spec.eval_id,
        method=spec.method.value,
        status=EvalStatus.SUCCEEDED.value,
        normalized_score_10=reward,
        raw_score=score,
        reason="progress_mse_log_v1: clip(7 * ln(0.18 / mse) / ln(4.5), 0, 10)",
        artifacts={
            "mse": score,
            "reward_10": reward,
            "successful_gateway_steps": len(trajectory.steps),
            "bench_output_path": metrics.get("bench_output_path"),
        },
    )


def mse_to_reward(mse: float) -> float:
    """0.18 -> 0, 0.04 -> 7; approximately 0.021 -> 10 (capped)."""
    if (
        isinstance(mse, bool)
        or not isinstance(mse, Real)
        or not math.isfinite(mse)
        or not 0 <= mse <= 1
    ):
        raise ValueError("progress MSE must be a finite number in [0, 1]")
    if mse == 0:
        return 10.0
    return round(
        max(0.0, min(10.0, 7 * (math.log(0.18) - math.log(mse)) / math.log(4.5))), 6
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
