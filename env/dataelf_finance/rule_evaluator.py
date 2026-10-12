"""DataElf finance rule evaluator: normalize the in-container reward document.

The episode's adapter scored the rollout in-container (dataelf_eval reward
calculator via the read-only scoring mount, see ``adapter.py``
``_score_episode``) and wrote the unified ``dataelf_eval.reward.v1``
document into ``start_result.metrics["reward"]``.  This shell only maps that
document onto SAfactory's ``EvalResult`` — it never reruns the case, never
calls a model, and never touches the host filesystem, so the 60 s default
eval window is ample.  Judge execution, checklist matching and reward
shaping all live in the benchmark-side module, inside the episode's own
container and failure domain (benchmark_notes/
safactory_skill_incontainer_eval_migration.md, 2026-10-12).
"""

from __future__ import annotations

import logging
import math
from typing import Any

log = logging.getLogger("safactory.dataelf_finance.rule")


async def evaluate_rule(*, request: Any, spec: Any, trajectory: Any) -> Any:
    """Normalize ``metrics["reward"]`` (dataelf_eval.reward.v1) into an EvalResult."""
    from evaluator.eval_types import EvalResult, EvalStatus

    metrics = getattr(request.start_result, "metrics", None)
    metrics = dict(metrics) if isinstance(metrics, dict) else {}
    artifacts = {
        "bench": "dataelf_finance",
        "benchmark": str(metrics.get("benchmark") or ""),
    }
    score = metrics.get("reward")
    if not isinstance(score, dict):
        return _failed(
            request, spec,
            "metrics.reward is missing: the in-container tail scoring did not run "
            "(adapter._score_episode; see env/dataelf_finance/adapter.py)",
            artifacts=artifacts,
        )
    artifacts = {**artifacts, **_json_safe(_reward_audit(score))}
    if str(score.get("status") or "") != "succeeded":
        return _failed(
            request, spec,
            str(score.get("reason") or "in-container scoring did not succeed"),
            artifacts=artifacts,
        )
    try:
        score_10 = float(score["normalized_score_10"])
    except (KeyError, TypeError, ValueError) as exc:
        return _failed(
            request, spec, "reward document is missing numeric normalized_score_10",
            error_text=str(exc), artifacts=artifacts,
        )
    if not math.isfinite(score_10):
        # NaN would clamp to 10.0 (max/min comparisons are False for NaN)
        # and silently award a perfect reward.
        return _failed(
            request, spec, "reward normalized_score_10 is not finite",
            error_text=repr(score_10), artifacts=artifacts,
        )
    return EvalResult(
        session_id=request.session_id,
        eval_id=spec.eval_id,
        method=spec.method.value,
        status=EvalStatus.SUCCEEDED.value,
        normalized_score_10=max(0.0, min(10.0, score_10)),
        raw_score=_float_or_none(score.get("raw_score")),
        reason=str(score.get("reason") or "in-container dataelf_eval reward"),
        artifacts=artifacts,
    )


# ---------------------------------------------------------------- helpers


def _reward_audit(score: dict[str, Any]) -> dict[str, Any]:
    """Compact audit slice of the reward document for EvalResult artifacts.

    The full document already lives in ``metrics["reward"]`` and the
    episode's ``reward_debug.json``; keep only identity + judge
    observability here (fallbacks must be visible in the training ledger).
    """
    judge = score.get("judge") if isinstance(score.get("judge"), dict) else {}
    components = score.get("components") if isinstance(score.get("components"), dict) else {}
    return {
        "schema_version": score.get("schema_version"),
        "official": bool(score.get("official")),
        "question_id": score.get("question_id"),
        "entity_id": score.get("entity_id"),
        "judge_model": judge.get("model"),
        "judge_mode_effective": judge.get("mode_effective"),
        "judge_calls": judge.get("calls"),
        "judge_failed": judge.get("failed"),
        "component_scores": {
            name: (entry.get("score") if isinstance(entry, dict) else entry)
            for name, entry in components.items()
        },
    }


def _failed(request: Any, spec: Any, reason: str, *, error_text: Any = None, artifacts: dict[str, Any] | None = None) -> Any:
    from evaluator.eval_types import EvalResult

    return EvalResult.failed(
        session_id=request.session_id,
        eval_id=spec.eval_id,
        method=spec.method.value,
        reason=reason,
        error_text=None if error_text is None else str(error_text),
        artifacts=artifacts or {},
    )


def _float_or_none(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _json_safe(value: Any) -> Any:
    """Coerce reward audit artifacts to JSON-safe values (RewardCommitter
    ``json.dumps`` has no ``default=``; one numpy array would drop the
    reward of a finished episode)."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return str(value)
