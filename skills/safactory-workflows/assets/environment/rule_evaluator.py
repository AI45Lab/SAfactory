"""Optional evaluator shell; fill only score_metrics after agreeing on scoring.

Discovered by SAfactory only when --enable-evaluation is enabled.
No benchmark execution or runner/image dependencies belong in this file.
"""
import math


def score_metrics(metrics, dataset, trajectory):
    """Return (raw_score, score_0_to_10, reason) using the native scoring contract.

    This is the environment-specific hook. Reject missing/invalid native scores;
    never treat successful adapter execution as a perfect benchmark score.
    """
    raise NotImplementedError("fill score_metrics using the agreed native score and scale")


def evaluate_rule(*, request, spec, trajectory):
    metrics = getattr(request.start_result, "metrics", {}) or {}
    try:
        raw, score, reason = score_metrics(metrics, request.env_params.get("dataset", {}), trajectory)
        raw, score = float(raw), float(score)
        if not math.isfinite(raw) or not math.isfinite(score) or not 0 <= score <= 10:
            raise ValueError("scores must be finite and normalized score must be in 0–10")
        return {
            "session_id": request.session_id, "eval_id": spec.eval_id,
            "status": "succeeded", "raw_score": raw,
            "normalized_score_10": score, "reason": str(reason), "artifacts": {"metrics": metrics},
        }
    except Exception as exc:
        return {
            "session_id": request.session_id, "eval_id": spec.eval_id,
            "status": "failed", "normalized_score_10": 0.0,
            "reason": str(exc), "error_text": str(exc), "artifacts": {"metrics": metrics},
        }
