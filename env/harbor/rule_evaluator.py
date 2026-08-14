from __future__ import annotations

import math
from typing import Any

from evaluator.eval_types import (
    EvalResult,
    EvalRequest,
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
    """Validate Harbor's native reward and normalize it onto SAfactory's 0-10 scale."""
    del trajectory
    start_result = request.start_result
    metrics = _start_metrics(request)
    artifacts = {
        "bench": "harbor",
        "task_id": metrics.get("task_id"),
        "agent": metrics.get("harbor_agent"),
        "model": metrics.get("harbor_model"),
        "reward_key": metrics.get("reward_key"),
        "rewards": metrics.get("harbor_rewards"),
        "errors": metrics.get("harbor_errors"),
        "job_result_path": metrics.get("harbor_job_result_path"),
        "trial_result_path": metrics.get("harbor_trial_result_path"),
        "trajectory_paths": metrics.get("trajectory_paths"),
    }

    if getattr(start_result, "status", None) != "succeeded":
        return EvalResult.failed(
            session_id=request.session_id,
            eval_id=spec.eval_id,
            method=spec.method.value,
            reason="Harbor trial did not complete successfully",
            error_text=getattr(start_result, "error_text", None),
            artifacts=artifacts,
        )
    if metrics.get("harbor_errors"):
        return EvalResult.failed(
            session_id=request.session_id,
            eval_id=spec.eval_id,
            method=spec.method.value,
            reason="Harbor trial reported errors",
            artifacts=artifacts,
        )

    raw_reward = _finite_float(metrics.get("harbor_reward"))
    if raw_reward is None:
        return EvalResult.failed(
            session_id=request.session_id,
            eval_id=spec.eval_id,
            method=spec.method.value,
            reason="Harbor verifier did not produce a numeric reward",
            artifacts=artifacts,
        )

    normalized = round(max(0.0, min(1.0, raw_reward)) * 10.0, 6)
    return EvalResult(
        session_id=request.session_id,
        eval_id=spec.eval_id,
        method=spec.method.value,
        status=EvalStatus.SUCCEEDED.value,
        raw_score=raw_reward,
        normalized_score_10=normalized,
        reason="Harbor rule evaluator: verifier reward [0,1] -> SAfactory reward [0,10]",
        artifacts=artifacts,
    )


def _start_metrics(request: EvalRequest) -> dict[str, Any]:
    metrics = getattr(request.start_result, "metrics", None)
    return dict(metrics) if isinstance(metrics, dict) else {}


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None
