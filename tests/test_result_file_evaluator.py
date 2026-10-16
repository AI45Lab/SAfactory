from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from core.result_artifact import write_result_atomic
from evaluator.eval_types import EvalRequest, EvalSpec, Trajectory
from evaluator.factory import build_evaluation_service
from evaluator.result_file import ResultFileEvaluatorBackend
from evaluator.rule_evaluator import RuleEvaluatorBackend
from test_result_artifact import IDENTITY, artifact


class ResultFileEvaluatorTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "result.json"
        self.request = EvalRequest(
            job_id=IDENTITY["job_id"], session_id=IDENTITY["session_id"],
            lease=SimpleNamespace(runtime_config={"attempt_id": IDENTITY["attempt_id"]}),
            start_result=SimpleNamespace(result_artifact_path=str(self.path), total_reward=999, metrics={}),
            eval_spec=EvalSpec(eval_id="safactory_result_file"),
        )

    async def test_backend_uses_file_score_without_scaling_or_clamping(self):
        backend = ResultFileEvaluatorBackend()
        service = build_evaluation_service(backend=backend)
        self.assertIs(service.backend, backend)
        self.assertIsInstance(build_evaluation_service().backend, RuleEvaluatorBackend)
        for score in (0, 0.8, 8, 10):
            write_result_atomic(self.path, artifact(score=score))
            result = await service.evaluate(self.request)
            self.assertEqual(result.normalized_score_10, score)
            self.assertEqual(result.eval_id, "native_rule")
            self.assertEqual(result.artifacts["normalizer_id"], "native_v1")

    async def test_file_failure_and_timeout_stay_unscored(self):
        for status in ("failed", "timeout"):
            write_result_atomic(self.path, artifact(status=status, score=0))
            result = await ResultFileEvaluatorBackend().evaluate(
                request=self.request, spec=self.request.eval_spec, trajectory=Trajectory(session_id=IDENTITY["session_id"]),
            )
            self.assertEqual(result.status, status)

    async def test_gym_truncation_with_unterminated_flag_keeps_file_score(self):
        body = artifact(status="truncated", score=2.5)
        body["terminated"] = False
        write_result_atomic(self.path, body)
        result = await build_evaluation_service(backend=ResultFileEvaluatorBackend()).evaluate(self.request)
        self.assertEqual(result.status, "truncated")
        self.assertEqual(result.normalized_score_10, 2.5)

    async def test_missing_or_wrong_attempt_cannot_supply_success_score(self):
        service = build_evaluation_service(backend=ResultFileEvaluatorBackend())
        result = await service.evaluate(self.request)
        self.assertEqual(result.status, "failed")
        write_result_atomic(self.path, artifact())
        self.request.lease.runtime_config["attempt_id"] = "stale"
        result = await service.evaluate(self.request)
        self.assertEqual(result.status, "failed")

    async def test_valid_file_cannot_override_execution_error(self):
        write_result_atomic(self.path, artifact())
        self.request.start_result.metrics["execution_error"] = "runner crashed"
        result = await build_evaluation_service(backend=ResultFileEvaluatorBackend()).evaluate(self.request)
        self.assertEqual(result.status, "failed")


if __name__ == "__main__":
    unittest.main()
