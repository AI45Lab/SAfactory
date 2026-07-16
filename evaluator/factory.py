from __future__ import annotations

from typing import Any

from evaluator.eval_types import coerce_eval_spec
from evaluator.rule_evaluator import RuleEvaluatorBackend
from evaluator.service import EvaluationService
from evaluator.trajectory_reader import TrajectoryReader


def build_evaluation_service(
    *,
    config: dict[str, Any] | None = None,
    trajectory_reader: TrajectoryReader | None = None,
) -> EvaluationService:
    cfg = dict(config or {})
    default_specs = [coerce_eval_spec(item) for item in cfg.get("default_specs") or []]
    return EvaluationService(
        trajectory_reader=trajectory_reader,
        backend=RuleEvaluatorBackend(
            env_root=cfg.get("rule_evaluator_env_root") or "env",
        ),
        max_concurrency=int(cfg.get("max_concurrency") or 64),
        default_specs=default_specs,
    )
