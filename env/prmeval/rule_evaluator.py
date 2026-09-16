"""PRMEval score adapter for SAfactory's optional evaluation stage.

The native PRMEval run happens once in ``adapter.py``.  This file only reads
the stored metrics and normalizes progress MSE to SAfactory's 0--10 reward;
it must never invoke PRMEval or the model again.
"""

from __future__ import annotations

import math
from numbers import Real
from typing import Any

async def evaluate_rule(
    *,
    request: Any,
    spec: Any,
    trajectory: Any,
) -> Any:
    # Keep score helpers importable in lightweight local tooling.  The full
    # evaluator package is only needed when Launcher actually enables eval.
    from evaluator.eval_types import EvalResult, EvalStatus

    metrics = _start_metrics(request)
    try:
        raw_score = _numeric_progress_mse(metrics)
        reward = mse_to_reward(raw_score)
    except ValueError as exc:
        return EvalResult.failed(
            session_id=request.session_id,
            eval_id=spec.eval_id,
            method=spec.method.value,
            reason=str(exc),
            artifacts={"bench": "prmeval", "metrics": metrics},
        )

    return EvalResult(
        session_id=request.session_id,
        eval_id=spec.eval_id,
        method=spec.method.value,
        status=EvalStatus.SUCCEEDED.value,
        normalized_score_10=reward,
        raw_score=raw_score,
        reason="progress_mse_log_v1: clip(7 * ln(0.18 / mse) / ln(4.5), 0, 10)",
        artifacts={
            "bench": "prmeval",
            "mse": raw_score,
            "reward_10": reward,
            "successful_gateway_steps": len(trajectory.steps),
            "bench_output_path": metrics.get("bench_output_path")
            or metrics.get("native_output_dir"),
        },
    )


def mse_to_reward(mse: float) -> float:
    """Map PRMEval progress MSE in [0, 1] to SAfactory's [0, 10] scale."""
    if (
        isinstance(mse, bool)
        or not isinstance(mse, Real)
        or not math.isfinite(mse)
        or not 0 <= mse <= 1
    ):
        raise ValueError("progress MSE must be a finite number in [0, 1]")
    if mse == 0:
        return 10.0
    return round(max(0.0, min(10.0, 7 * math.log(0.18 / mse) / math.log(4.5))), 6)


def _start_metrics(request: Any) -> dict[str, Any]:
    start_result = getattr(request, "start_result", None)
    metrics = getattr(start_result, "metrics", None)
    return dict(metrics) if isinstance(metrics, dict) else {}


def _numeric_progress_mse(metrics: dict[str, Any]) -> float:
    progress = metrics.get("progress")
    if isinstance(progress, dict) and "mse" in progress:
        value = progress["mse"]
    else:
        native = metrics.get("native_summary")
        native_metrics = native.get("metrics") if isinstance(native, dict) else None
        native_progress = native_metrics.get("progress") if isinstance(native_metrics, dict) else None
        value = native_progress.get("mse") if isinstance(native_progress, dict) else metrics.get("mse")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("prmeval metrics did not contain a numeric progress MSE") from exc
