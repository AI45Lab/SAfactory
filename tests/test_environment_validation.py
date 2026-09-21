"""Checks for validate_environment.py and the check_environment.py orchestration."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

REPO_ROOT = Path(__file__).resolve().parents[1]
SKILL = REPO_ROOT / "skills" / "safactory-workflows"
sys.path.insert(0, str(SKILL / "scripts"))

from check_environment import check as check_environment
from scaffold_environment import scaffold
from validate_environment import SEV_ERROR, validate_environment


def load_yaml(path: Path) -> dict:
    import yaml
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def dump_yaml(path: Path, data: dict) -> None:
    import yaml
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


@unittest.skipUnless(importlib.util.find_spec("yaml"), "requires PyYAML for consistency checks")
class EnvironmentValidatorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = scaffold("mybench", self.root)

    def codes(self, report):
        return {f.code for f in report.findings}

    def test_standard_scaffold_passes_with_expected_warnings_only(self):
        report = validate_environment(self.path)
        self.assertTrue(report.ok(), [f.line() for f in report.errors])
        self.assertIn("cluster-storage", self.codes(report))
        self.assertNotIn("results-mount", self.codes(report))
        self.assertNotIn("dataset-path", self.codes(report))

    def test_agent_name_mismatch_is_a_config_error(self):
        start = load_yaml(self.path / "mybench_start.yaml")
        start["agent_name"] = "other"
        dump_yaml(self.path / "mybench_start.yaml", start)
        report = validate_environment(self.path)
        finding = next(f for f in report.errors if f.code == "agent-name")
        self.assertEqual(finding.side, "config")

    def test_results_root_not_mounted_is_a_config_error(self):
        config = load_yaml(self.path / "mybench_config.yaml")
        config["environments"][0]["env_params"]["results_root"] = "/elsewhere/results"
        dump_yaml(self.path / "mybench_config.yaml", config)
        report = validate_environment(self.path)
        finding = next(f for f in report.errors if f.code == "results-mount")
        self.assertEqual(finding.side, "config")

    def test_dataset_row_absolute_path_outside_mounts_is_env_error(self):
        dataset = self.path / "datasets" / "smoke.jsonl"
        dataset.write_text('{"task_id": "a", "frames": "/nowhere/trajectory.mp4"}\n', encoding="utf-8")
        report = validate_environment(self.path)
        finding = next(f for f in report.errors if f.code == "dataset-path")
        self.assertEqual(finding.side, "env")

    def test_missing_evaluator_with_expect_evaluation_is_config_error(self):
        report = validate_environment(self.path, expect_evaluation=True)
        finding = next(f for f in report.errors if f.code == "evaluator-missing")
        self.assertEqual(finding.side, "config")

    def test_adapter_container_path_drift_between_modes_is_config_error(self):
        start = load_yaml(self.path / "mybench_start.rjob.yaml")
        start["rjob"]["embedded_files"][0]["target"] = "/tmp/somewhere-else/adapter.py"
        dump_yaml(self.path / "mybench_start.rjob.yaml", start)
        report = validate_environment(self.path)
        finding = next(f for f in report.errors if f.code == "adapter-path-drift")
        self.assertEqual(finding.side, "config")

    def test_hand_edited_runner_shell_is_reported_as_drift_warning(self):
        runner = self.path / "runner.py"
        source = runner.read_text(encoding="utf-8").replace(
            'print(f"SAFACTORY_RUNNER_DIAGNOSTIC result_artifact_write_failed: {exc}", file=sys.stderr)',
            "print('custom diagnostics', file=sys.stderr)")
        runner.write_text(source, encoding="utf-8")
        report = validate_environment(self.path)
        self.assertIn("runner-drift", self.codes(report))
        self.assertTrue(report.ok(), "drift is a warning, not an error")

    def test_broken_dataset_row_is_env_error(self):
        dataset = self.path / "datasets" / "smoke.jsonl"
        dataset.write_text("not json\n", encoding="utf-8")
        report = validate_environment(self.path)
        finding = next(f for f in report.errors if f.code == "dataset-parse")
        self.assertEqual(finding.side, "env")


class CheckEnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = scaffold("mybench", self.root)

    def test_full_pipeline_passes_on_standard_scaffold(self):
        stages = check_environment(self.path)
        self.assertTrue(stages["static"]["ok"])
        self.assertTrue(stages["contract"]["ok"])
        self.assertTrue(stages["contract"]["outcome"]["model_calls"] >= 1)
        self.assertTrue(stages["live"]["skipped"])
        self.assertTrue(stages["summary"]["ok"])

    def test_static_failure_short_circuits_contract_stage(self):
        config = self.path / "mybench_config.yaml"
        text = config.read_text(encoding="utf-8").replace(
            "results_root: /workspace/Safactory/results", "results_root: /elsewhere")
        config.write_text(text, encoding="utf-8")
        stages = check_environment(self.path)
        self.assertFalse(stages["static"]["ok"])
        self.assertTrue(stages["contract"]["skipped"])
        self.assertFalse(stages["summary"]["ok"])

    def test_native_dependency_failure_is_labeled_with_fixture_hint(self):
        (self.path / "adapter.py").write_text(
            "import a_missing_native_module\n"
            "def run_case(request, task, session_url):\n"
            "    return {}, 1\n",
            encoding="utf-8",
        )
        stages = check_environment(self.path)
        self.assertTrue(stages["static"]["ok"])
        self.assertFalse(stages["contract"]["ok"])
        self.assertIn("fixture-adapter", stages["contract"].get("hint", ""))


if __name__ == "__main__":
    unittest.main()
