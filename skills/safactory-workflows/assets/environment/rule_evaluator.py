"""Optional evaluation shell; fill only score_metrics after agreeing on scoring.

Discovered by SAfactory at ``<env_root>/<env_name>/rule_evaluator.py`` only
when launcher.py runs with ``--enable-evaluation``.  WARNING: with that flag
set, a missing rule_evaluator.py marks the whole episode as failed — omit the
flag for integration-only runs.  This file must never rerun the benchmark or
call the model; it only maps stored metrics to SAfactory's 0-10 reward.
"""
from __future__ import annotations

import math
from typing import Any


def score_metrics(metrics: dict[str, Any], dataset: dict[str, Any], trajectory: Any):
    """Return (raw_score, score_0_to_10, reason) using the native scoring contract.

    This is the environment-specific hook.  Reject missing or invalid native
    scores; never treat successful adapter execution as a perfect score.
    """
    raise NotImplementedError("fill score_metrics using the agreed native score and scale")


async def evaluate_rule(*, request: Any, spec: Any, trajectory: Any) -> Any:
    # Keep score helpers importable in lightweight local tooling.  The full
    # evaluator package is only needed when Launcher actually enables eval.
    from evaluator.eval_types import EvalResult, EvalStatus

    metrics = _start_metrics(request)
    try:
        raw, score, reason = score_metrics(metrics, _dataset(request), trajectory)
        raw, score = float(raw), float(score)
        if not math.isfinite(raw) or not math.isfinite(score) or not 0 <= score <= 10:
            raise ValueError("scores must be finite and normalized score must be in 0-10")
    except Exception as exc:
        return EvalResult.failed(
            session_id=request.session_id,
            eval_id=spec.eval_id,
            reason=str(exc),
            artifacts={"metrics": metrics},
        )
    return EvalResult(
        session_id=request.session_id,
        eval_id=spec.eval_id,
        status=EvalStatus.SUCCEEDED.value,
        normalized_score_10=score,
        raw_score=raw,
        reason=str(reason),
        artifacts={"metrics": metrics},
    )


def _start_metrics(request: Any) -> dict[str, Any]:
    start_result = getattr(request, "start_result", None)
    metrics = getattr(start_result, "metrics", None)
    return dict(metrics) if isinstance(metrics, dict) else {}


def _dataset(request: Any) -> dict[str, Any]:
    env_params = getattr(request, "env_params", None)
    dataset = env_params.get("dataset") if isinstance(env_params, dict) else None
    return dataset if isinstance(dataset, dict) else {}
