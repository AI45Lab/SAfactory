from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from core.result_artifact import finalize_result, read_result_file, update_json_atomic, write_json_atomic, write_result_atomic

IDENTITY = {"job_id": "job-1", "session_id": "session-1", "attempt_id": "attempt-1"}


def artifact(*, score=8.0, status="succeeded"):
    return {
        "schema_version": "safactory.result.v2", **IDENTITY,
        "status": status, "total_reward": 0.8, "step_count": 12,
        "terminated": True, "truncated": status in {"truncated", "timeout"},
        "error_text": None, "metrics": {"native_score": 0.8},
        "evaluation": {
            "session_id": IDENTITY["session_id"], "status": status,
            "eval_id": "native_rule", "method": "rule_evaluator",
            "raw_score": 0.8, "normalized_score_10": score,
            "normalizer_id": "native_v1", "reason": "native verification",
            "artifacts": {"evidence": "checked"}, "error_text": None,
        },
    }


class ResultArtifactTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "safactory_result.json"

    def test_finalize_calls_scorer_once_and_preserves_normalized_score(self):
        body = artifact(score=0)
        evaluation = body.pop("evaluation")
        scorer = Mock(return_value=evaluation)
        result = finalize_result(body, env_params={"native": True}, scorer=scorer, identity=IDENTITY)
        scorer.assert_called_once()
        self.assertEqual(scorer.call_args.kwargs, {"env_params": {"native": True}})
        self.assertEqual(result["evaluation"]["normalized_score_10"], 0)
        self.assertEqual(result["total_reward"], 0.8)

    def test_scorer_exception_is_an_unscored_failure(self):
        scorer = Mock(side_effect=RuntimeError("verifier unavailable"))
        result = finalize_result(artifact(), env_params={}, scorer=scorer, identity=IDENTITY)
        scorer.assert_called_once()
        self.assertEqual(result["evaluation"]["status"], "failed")
        self.assertNotIn("normalized_score_10", result["evaluation"])
        self.assertIn("verifier unavailable", result["evaluation"]["error_text"])
        write_result_atomic(self.path, result)
        self.assertEqual(read_result_file(self.path, **IDENTITY), result)

    def test_atomic_roundtrip_and_invalid_replacement_preserves_previous_file(self):
        write_result_atomic(self.path, artifact())
        original = self.path.read_bytes()
        invalid = artifact(score=11)
        with self.assertRaises(ValueError):
            write_result_atomic(self.path, invalid)
        self.assertEqual(self.path.read_bytes(), original)
        self.assertEqual(read_result_file(self.path, **IDENTITY)["evaluation"]["normalized_score_10"], 8.0)
        with patch("core.result_artifact.os.replace", side_effect=OSError("disk failed")):
            with self.assertRaises(OSError):
                write_result_atomic(self.path, artifact(score=4))
        self.assertEqual(self.path.read_bytes(), original)
        self.assertEqual(list(self.path.parent.iterdir()), [self.path])

    def test_generic_json_writer_preserves_runtime_metadata_without_result_validation(self):
        reference = {**IDENTITY, "sandbox_id": "sandbox-1", "command_finished": False}
        write_json_atomic(self.path, reference)
        self.assertEqual(json.loads(self.path.read_text()), reference)
        with self.assertRaises(ValueError):
            write_json_atomic(self.path, {"invalid": float("nan")})
        self.assertEqual(json.loads(self.path.read_text()), reference)
        with patch("core.result_artifact.os.replace", side_effect=OSError("disk failed")):
            with self.assertRaises(OSError):
                write_json_atomic(self.path, {**reference, "command_finished": True})
        self.assertEqual(json.loads(self.path.read_text()), reference)
        self.assertEqual(list(self.path.parent.iterdir()), [self.path])

    def test_reference_update_keeps_existing_fields_and_durable_gateway_result(self):
        reference = {**IDENTITY, "sandbox_id": "sandbox-1", "config_digest": "original", "command_finished": True}
        write_json_atomic(self.path, reference)
        update_json_atomic(self.path, {"gateway_finalized": False}, identity=IDENTITY)
        self.assertEqual(json.loads(self.path.read_text()), {**reference, "gateway_finalized": False})
        update_json_atomic(self.path, {"command_id": "command-1"}, identity=IDENTITY)
        loaded = json.loads(self.path.read_text())
        self.assertFalse(loaded["gateway_finalized"])
        self.assertEqual(loaded["config_digest"], "original")
        self.assertEqual(loaded["command_id"], "command-1")
        update_json_atomic(self.path, {"gateway_finalized": True}, identity=IDENTITY)
        self.assertTrue(json.loads(self.path.read_text())["gateway_finalized"])

    def test_reference_update_rejects_invalid_or_changed_identity_without_replacing(self):
        write_json_atomic(self.path, {**IDENTITY, "gateway_finalized": False})
        original = self.path.read_bytes()
        for name in IDENTITY:
            with self.subTest(name=name, source="identity"), self.assertRaises(ValueError):
                update_json_atomic(self.path, {"gateway_finalized": True}, identity={**IDENTITY, name: "other"})
            with self.subTest(name=name, source="update"), self.assertRaises(ValueError):
                update_json_atomic(self.path, {name: "other", "gateway_finalized": True}, identity=IDENTITY)
            self.assertEqual(self.path.read_bytes(), original)
        with patch("core.result_artifact.os.replace", side_effect=OSError("disk failed")):
            with self.assertRaises(OSError):
                update_json_atomic(self.path, {"gateway_finalized": True}, identity=IDENTITY)
        self.assertEqual(self.path.read_bytes(), original)
        self.assertEqual(list(self.path.parent.iterdir()), [self.path])

    def test_reference_update_rejects_missing_or_ambiguous_json(self):
        with self.assertRaises(FileNotFoundError):
            update_json_atomic(self.path, {"gateway_finalized": True}, identity=IDENTITY)
        for source in ("[]", "null", '{"job_id":"job-1","job_id":"job-2"}'):
            self.path.write_text(source)
            with self.subTest(source=source), self.assertRaises(ValueError):
                update_json_atomic(self.path, {"gateway_finalized": True}, identity=IDENTITY)
            self.assertEqual(self.path.read_text(), source)

    def test_score_and_identity_fail_closed(self):
        for score in (True, False, None, "8", -0.01, 10.01, float("nan"), float("inf")):
            with self.subTest(score=score):
                self.path.write_text(json.dumps(artifact(score=score)), encoding="utf-8")
                with self.assertRaises(ValueError):
                    read_result_file(self.path, **IDENTITY)
        write_result_atomic(self.path, artifact())
        for field in IDENTITY:
            with self.subTest(field=field), self.assertRaises(ValueError):
                read_result_file(self.path, **{**IDENTITY, field: "another"})

    def test_missing_fields_and_ambiguous_json_are_rejected(self):
        for field in ("status", "terminated", "truncated", "step_count", "total_reward", "metrics", "evaluation"):
            body = artifact()
            del body[field]
            self.path.write_text(json.dumps(body), encoding="utf-8")
            with self.subTest(field=field), self.assertRaises(ValueError):
                read_result_file(self.path, **IDENTITY)
        for content in ('{"status":"failed","status":"succeeded"}', json.dumps(artifact()).replace('0.8', '1e999')):
            self.path.write_text(content, encoding="utf-8")
            with self.assertRaises(ValueError):
                read_result_file(self.path, **IDENTITY)

    def test_truncated_requires_explicit_valid_score_and_matching_status(self):
        write_result_atomic(self.path, artifact(status="truncated", score=0))
        self.assertEqual(read_result_file(self.path, **IDENTITY)["evaluation"]["status"], "truncated")
        for body in (artifact(status="truncated", score=None), artifact(status="truncated", score=True)):
            with self.assertRaises(ValueError):
                write_result_atomic(self.path, body)
        body = artifact()
        body["evaluation"]["status"] = "truncated"
        with self.assertRaises(ValueError):
            write_result_atomic(self.path, body)

    def test_gym_truncated_episode_does_not_require_terminated(self):
        body = artifact(status="truncated", score=0)
        body["terminated"] = False
        write_result_atomic(self.path, body)
        loaded = read_result_file(self.path, **IDENTITY)
        self.assertFalse(loaded["terminated"])
        self.assertTrue(loaded["truncated"])
        self.assertEqual(loaded["evaluation"]["normalized_score_10"], 0)
        body = artifact()
        body["terminated"] = False
        with self.assertRaisesRegex(ValueError, "terminated or truncated"):
            write_result_atomic(self.path, body)

    def test_module_is_portable_without_host_imports(self):
        source = Path(__file__).resolve().parents[1] / "core" / "result_artifact.py"
        spec = importlib.util.spec_from_file_location("standalone_result_artifact", source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.write_result_atomic(self.path, artifact())
        self.assertEqual(module.read_result_file(self.path, **IDENTITY), artifact())


if __name__ == "__main__":
    unittest.main()
