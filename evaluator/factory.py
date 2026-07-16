from __future__ import annotations

from evaluator.rule_evaluator import RuleEvaluatorBackend
from evaluator.service import EvaluationService
from evaluator.trajectory_reader import TrajectoryReader


def build_evaluation_service(
    *,
    trajectory_reader: TrajectoryReader | None = None,
    max_concurrency: int = 64,
) -> EvaluationService:
    return EvaluationService(
        trajectory_reader=trajectory_reader,
        backend=RuleEvaluatorBackend(),
        max_concurrency=max_concurrency,
    )
