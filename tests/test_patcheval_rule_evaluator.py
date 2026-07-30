from __future__ import annotations

import asyncio
from types import SimpleNamespace

from env.patcheval import rule_evaluator
from evaluator.eval_types import EvalMethod, EvalRequest, EvalSpec, EvalStatus, Trajectory


def _request() -> tuple[EvalRequest, EvalSpec, Trajectory]:
    start_result = SimpleNamespace(
        metrics={
            "cve_id": "CVE-2024-0001",
            "setting": "s1.1",
            "protocol": "official_components_s1x",
            "patch": "diff --git a/a.py b/a.py\n",
        }
    )
    request = EvalRequest(
        job_id="job",
        session_id="session-1",
        start_result=start_result,
        env_params={
            "dataset": {
                "cve_id": "CVE-2024-0001",
                "official_record": {"programming_language": "Python"},
            }
        },
    )
    spec = EvalSpec(eval_id="patcheval_rule", method=EvalMethod.RULE_EVALUATOR)
    trajectory = Trajectory(session_id=request.session_id)
    return request, spec, trajectory


def test_official_success_is_committed_as_ten(monkeypatch) -> None:
    class FakeEvaluation:
        def __init__(self, **_kwargs) -> None:
            pass

        def run_evaluation(self, *_args):
            return True, "poc passed", True, "unit tests passed", "Repair Success"

    monkeypatch.setattr(rule_evaluator, "_load_official_evaluation", lambda _params: FakeEvaluation)
    request, spec, trajectory = _request()
    result = asyncio.run(
        rule_evaluator.evaluate_rule(request=request, spec=spec, trajectory=trajectory)
    )

    assert result.status == EvalStatus.SUCCEEDED.value
    assert result.raw_score == 1.0
    assert result.normalized_score_10 == 10.0
    assert result.artifacts["official_evaluator"] == "evaluation/run_evaluation.py:Evaluation"


def test_official_validation_failure_is_zero(monkeypatch) -> None:
    class FakeEvaluation:
        def __init__(self, **_kwargs) -> None:
            pass

        def run_evaluation(self, *_args):
            return False, "poc failed", None, None, "validation_fail"

    monkeypatch.setattr(rule_evaluator, "_load_official_evaluation", lambda _params: FakeEvaluation)
    request, spec, trajectory = _request()
    result = asyncio.run(
        rule_evaluator.evaluate_rule(request=request, spec=spec, trajectory=trajectory)
    )

    assert result.status == EvalStatus.SUCCEEDED.value
    assert result.raw_score == 0.0
    assert result.normalized_score_10 == 0.0
    assert result.artifacts["validation_type"] == "validation_fail"
