from __future__ import annotations

import asyncio
import json
import re
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from core.data_manager.strategy import cloud_strategy_impl
from env.agentcompass import rule_evaluator, runner
from evaluator.eval_types import EvalRequest, EvalResult, EvalSpec
from evaluator.reward_committer import RewardCommitter, _merge_meta_json
from evaluator.rule_evaluator import RuleEvaluatorBackend


FIXTURE = Path(__file__).parent / "fixtures" / "sgi_details.json"


def details():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


class EvaluationContextTests(unittest.TestCase):
    def test_real_style_sgi_details_parse_complete_available_context(self):
        for expected_reward, detail in zip((0.0, 10.0), details()):
            normalized = runner._normalize_detail("sgi_deep_research", detail)
            context = normalized["evaluation_context"]
            self.assertEqual(normalized["normalized_reward_10"], expected_reward)
            self.assertIs(type(context["correct"]), bool)
            self.assertIsInstance(normalized["ground_truth_answer"], str)
            self.assertIsInstance(context["model_answer"], str)
            self.assertIsNone(context["judge_verdict"])
            self.assertIsNone(context["judge_reason"])
            self.assertNotIn("ground_truth_answer", context)
            self.assertNotIn("attempt_id", context)

    def test_sgi_missing_required_judge_input_fails(self):
        for key in ("ground_truth", "model_answer", "correct"):
            detail = details()[0]
            del detail["attempts"]["1"]["extra"]["scoring"][key]
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                runner._normalize_detail("sgi_deep_research", detail)

    def test_sgi_normalization_uses_attempt_one(self):
        detail = details()[0]
        detail["attempts"]["2"] = json.loads(json.dumps(detail["attempts"]["1"]))
        detail["attempts"]["2"]["extra"]["scoring"]["model_answer"] = "Wrong attempt."
        normalized = runner._normalize_detail("sgi_deep_research", detail)
        self.assertEqual(normalized["evaluation_context"]["model_answer"], "Synthetic candidate response.")

        detail = details()[0]
        detail["attempts"]["2"] = detail["attempts"].pop("1")
        with self.assertRaisesRegex(RuntimeError, "attempts\['1'\]"):
            runner._normalize_detail("sgi_deep_research", detail)

    def test_parser_context_survives_evaluator_coercion(self):
        normalized = runner._normalize_detail("sgi_deep_research", details()[1])
        env_value = rule_evaluator.evaluate(
            SimpleNamespace(session_id="session-test", start_result=SimpleNamespace(metrics={"benchmark": "sgi_deep_research", **normalized})),
            SimpleNamespace(eval_id="agentcompass", method="rule_evaluator"),
            None,
        )
        result = RuleEvaluatorBackend()._coerce_result(
            env_value,
            request=EvalRequest(job_id="job-test", session_id="session-test"),
            spec=EvalSpec(eval_id="agentcompass"),
        )
        self.assertEqual(result.ground_truth_answer, normalized["ground_truth_answer"])
        self.assertEqual(result.evaluation_context, normalized["evaluation_context"])

    def test_other_benchmark_without_context_remains_compatible(self):
        result = RuleEvaluatorBackend()._coerce_result(
            {"score": 5},
            request=EvalRequest(job_id="job", session_id="session"),
            spec=EvalSpec(eval_id="other"),
        )
        self.assertIsNone(result.ground_truth_answer)
        self.assertEqual(result.evaluation_context, {})


class FakeManager:
    job_id = "job-test"

    def __init__(self):
        self.rows = [{"record_id": "row-0", "job_id": "job-test", "session_id": "session-test", "step_id": 0, "is_terminal": False, "meta_json": {"eval": {"existing": True}}}]
        self.inserted = []

    async def init(self): pass
    async def list_session_steps(self, *args, **kwargs): return list(self.rows)
    async def insert_session_step_rows(self, rows): self.inserted.extend(rows); return [row["record_id"] for row in rows]


class PersistenceTests(unittest.TestCase):
    def test_reward_committer_writes_context_only_to_terminal_summary(self):
        manager = FakeManager()
        normalized = runner._normalize_detail("sgi_deep_research", details()[1])
        context = normalized["evaluation_context"]
        asyncio.run(RewardCommitter(db_url="", data_manager=manager).commit(
            session_id="session-test",
            eval_result=EvalResult(session_id="session-test", status="succeeded", normalized_score_10=10, ground_truth_answer=normalized["ground_truth_answer"], evaluation_context=context),
        ))
        self.assertNotIn("ground_truth_answer", manager.rows[0])
        summary = manager.inserted[0]
        self.assertTrue(summary["is_terminal"])
        self.assertEqual(summary["ground_truth_answer"], normalized["ground_truth_answer"])
        self.assertNotIn("reference_answer", summary)
        self.assertEqual(summary["meta_json"]["eval"]["context"], context)

    def test_reward_metadata_is_deep_merged(self):
        merged = json.loads(_merge_meta_json(
            {"eval": {"existing": True, "result": {"kept": True}}, "other": 1},
            json.dumps({"eval": {"status": "succeeded", "result": {"new": True}}}),
        ))
        self.assertTrue(merged["eval"]["existing"])
        self.assertEqual(merged["eval"]["result"], {"kept": True, "new": True})
        self.assertEqual(merged["other"], 1)

    def test_cloud_landing_writer_and_projection_preserve_fields(self):
        class Record:
            def __init__(self, **values): self.__dict__.update(values)
            def model_dump(self): return dict(self.__dict__)
        strategy = object.__new__(cloud_strategy_impl.CloudStrategy)
        strategy.job_id = "job-test"
        strategy._record_job_ids = {}
        with mock.patch.object(cloud_strategy_impl, "LandingRecord", Record, create=True):
            record, _ = strategy._landing_record_from_row({
                "record_id": "row", "session_id": "session", "step_id": 1, "job_id": "job-test",
                "ground_truth_answer": "synthetic truth", "reference_answer": None,
                "meta_json": {"eval": {"context": {"model_answer": "synthetic answer", "correct": True}}},
                "is_terminal": True, "is_session_completed": True,
            })
        serving = Record(**record.model_dump())
        self.assertEqual(serving.ground_truth_answer, "synthetic truth")
        self.assertIsNone(serving.reference_answer)
        self.assertIn("synthetic answer", serving.meta_json)

        updates = strategy._normalize_session_step_updates_for_cloud(
            {"ground_truth_answer": "replacement truth", "reference_answer": None},
            filter_query="id = 'row'",
            job_id="job-test",
        )
        self.assertEqual(updates["ground_truth_answer"], "replacement truth")
        self.assertIsNone(updates["reference_answer"])

    def test_fixture_and_artifacts_are_safe(self):
        text = FIXTURE.read_text(encoding="utf-8")
        self.assertNotRegex(text, r"https?://|AKIA|/home/|/app/results")
        value = rule_evaluator.evaluate(
            SimpleNamespace(session_id="s", start_result=SimpleNamespace(metrics={"benchmark": "sgi_deep_research", **runner._normalize_detail("sgi_deep_research", details()[0]), "detail_path": "/private/detail.json"})),
            SimpleNamespace(eval_id="e", method="rule_evaluator"), None,
        )
        self.assertNotIn("detail_path", value["artifacts"])
