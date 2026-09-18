from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
from typing import Any
import unittest

try:
    import yaml
except ImportError:  # pragma: no cover - repository tests install PyYAML
    yaml = None


ENV = Path(__file__).resolve().parents[3] / "env" / "deepsafe_cyber"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DeepSafeCyberAdapterTest(unittest.TestCase):
    def setUp(self):
        self.adapter = load_module("deepsafe_adapter_under_test", ENV / "adapter.py")

    def test_provider_document_uses_session_model_and_profiles(self):
        document = self.adapter._provider_document(
            session_url="http://host.docker.internal:8000/v1/sessions/session-1",
            model="route/model",
            provider_type="openai-responses",
            provider_name="safactory-gateway",
            model_profile="safactory-gateway-responses",
        )
        self.assertIn('base_url: "http://host.docker.internal:8000/v1/sessions/session-1"', document)
        self.assertIn('model: "route/model"', document)
        self.assertIn("type: \"openai-responses\"", document)
        self.assertIn("model_profile: \"safactory-gateway-responses\"", document)
        self.assertGreaterEqual(len("safactory-gateway-session-scoped-placeholder-key"), 32)

    def test_write_root_private_yaml_permissions(self):
        path = Path(self.enterContext(_tempdir())) / "providers.yaml"
        self.adapter._write_root_private_yaml(path, "providers: {}\n")
        self.assertEqual(path.stat().st_mode & 0o777, 0o600)

    def test_run_case_maps_one_native_result(self):
        root = Path(self.enterContext(_tempdir()))
        template = root / "run.yaml"
        template.write_text(
            "\n".join(
                [
                    "schema: deepsafe-cyber.run-config.v1",
                    "benchmark:",
                    "  name: patcheval",
                    "  variant: verified",
                    "dataset_profile: patcheval-verified-v1",
                    "harness_profile: dsh-headless",
                    "model_profile: safactory-gateway-responses",
                    "provider_credential_profile: safactory-gateway",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        native_root = root / "runs"

        def fake_wait(timeout_s):
            self.assertEqual(timeout_s, 7)

        def fake_native(command, *, cwd, env, timeout_s):
            self.assertTrue(any(str(part).endswith("/deepsafe-cyber") for part in command))
            self.assertEqual(command[command.index("--task") + 1], "CVE-2021-23376")
            self.assertEqual(env["DEEPSAFE_CYBER_PROVIDER_CONFIG"].endswith("providers.yaml"), True)
            provider = Path(env["DEEPSAFE_CYBER_PROVIDER_CONFIG"]).read_text(encoding="utf-8")
            self.assertIn("base_url: \"http://gateway/session-1\"", provider)
            self.assertIn("model: \"test-model\"", provider)
            run_id = "native-run-1"
            result_dir = native_root / run_id
            public = result_dir / "public"
            public.mkdir(parents=True)
            envelope = _envelope(run_id, "MODEL_INCORRECT")
            (public / "result.json").write_text(json.dumps(envelope), encoding="utf-8")
            cli = {
                "schema": self.adapter.CLI_RESULT_SCHEMA,
                "run_id": run_id,
                "result_ref": run_id,
            }
            return subprocess.CompletedProcess(command, 0, json.dumps(cli), "")

        self.adapter._wait_for_nested_docker = fake_wait
        self.adapter._run_native = fake_native
        request = {
            "model": "test-model",
            "env_params": {
                "results_root": str(native_root),
                "dataset": {"task": "cve-2021-23376"},
                "deepsafe_cyber": {
                    "workspace": str(root),
                    "run_yaml_template": str(template),
                    "runs_subdir": ".",
                    "docker_health_timeout_s": 7,
                    "environment_preflight_timeout_s": 11,
                    "task_time_limit_s": 13,
                },
            },
        }
        metrics, steps = self.adapter.run_case(request, request["env_params"]["dataset"], "http://gateway/session-1/")
        self.assertEqual(steps, 1)
        self.assertEqual(metrics["bench"], "deepsafe-cyber")
        self.assertEqual(metrics["cve_id"], "CVE-2021-23376")
        self.assertEqual(metrics["terminal_code"], "MODEL_INCORRECT")
        self.assertEqual(metrics["model_verdict"], "INCORRECT")
        self.assertFalse(metrics["execution_failed"])

    def test_result_ref_cannot_escape_runs_root(self):
        with self.assertRaises(ValueError):
            self.adapter._resolve_result_directory(Path("/tmp/safe"), "../outside")

    def test_result_ref_may_point_at_published_envelope(self):
        root = Path("/tmp/safe")
        actual = self.adapter._resolve_result_directory(
            root, "dsc-001/public/result.json"
        )
        self.assertEqual(actual, root / "dsc-001")

    def test_run_yaml_profile_drift_is_rejected(self):
        document = "model_profile: wrong-profile\nprovider_credential_profile: safactory-gateway\n"
        with self.assertRaisesRegex(ValueError, "profiles are inconsistent"):
            self.adapter._validate_run_yaml(
                document,
                provider_name="safactory-gateway",
                model_profile="safactory-gateway-responses",
            )


class DeepSafeCyberEvaluatorTest(unittest.TestCase):
    def setUp(self):
        self.evaluator = load_module("deepsafe_evaluator_under_test", ENV / "rule_evaluator.py")

    def test_terminal_outcomes_are_binary(self):
        cases = {
            "PASS": (1.0, 10.0),
            "MODEL_INCORRECT": (0.0, 0.0),
        }
        for terminal_code, expected in cases.items():
            raw, normalized, reason = self.evaluator.score_metrics(
                {"terminal_code": terminal_code}, {"task": "CVE-2021-23376"}, None
            )
            self.assertEqual((raw, normalized), expected)
            self.assertIn(terminal_code, reason)

    def test_execution_failure_is_not_model_incorrect(self):
        with self.assertRaisesRegex(ValueError, "INVALID_CONFIGURATION"):
            self.evaluator.score_metrics(
                {"terminal_code": "INVALID_CONFIGURATION"}, {"task": "CVE-2021-23376"}, None
            )

    def test_missing_terminal_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "terminal.code"):
            self.evaluator.score_metrics({}, {"task": "CVE-2021-23376"}, None)


class DeepSafeCyberRJobConfigTest(unittest.TestCase):
    def test_result_roots_are_split_between_safactory_and_native_paths(self):
        if yaml is None:
            self.skipTest("PyYAML is required")
        for name in (
            "deepsafe_cyber_config.yaml",
            "deepsafe_cyber_config.smoke.yaml",
            "deepsafe_cyber_config.rjob.yaml",
            "deepsafe_cyber_config.rjob.smoke.yaml",
        ):
            with self.subTest(config=name):
                entry = yaml.safe_load((ENV / name).read_text())["environments"][0]
                params = entry["env_params"]
                native = params["deepsafe_cyber"]
                self.assertEqual(
                    params["results_root"],
                    "/mnt/shared-storage-user/adpc-share/chenxinquan/SAfactory/results/deepsafe_cyber",
                )
                self.assertEqual(native["results_root"], "/opt/deepsafe-cyber/runs")
                self.assertTrue(native["runs_subdir"].startswith("safactory"))

    def test_rjob_uses_cluster_mounts_and_dind_wrapper(self):
        if yaml is None:
            self.skipTest("PyYAML is required")
        start = yaml.safe_load((ENV / "deepsafe_cyber_start.rjob.yaml").read_text())
        rjob = start["rjob"]
        self.assertTrue(rjob["privileged"])
        self.assertEqual(rjob["resources"]["cpu"], 10)
        self.assertEqual(rjob["resources"]["memory_in_mb"], 20480)
        self.assertIn("brainpp.cn/fuse=1", rjob["resources"]["custom_resources"])
        mounts = "\n".join(rjob["mount_config"])
        self.assertIn("gpfs://gpfs2/trustcyberdata:", mounts)
        self.assertIn(
            "gpfs://gpfs2/adpc-share/chenxinquan/SAfactory/results/deepsafe_cyber:"
            "/mnt/shared-storage-user/adpc-share/chenxinquan/SAfactory/results/deepsafe_cyber",
            mounts,
        )
        self.assertNotIn(
            "gpfs://gpfs2/adpc-share/chenxinquan/SAfactory/results/deepsafe_cyber:/opt/deepsafe-cyber/runs",
            mounts,
        )
        self.assertIn("env/deepsafe_cyber/dind-data:/var/lib/docker", mounts)
        self.assertNotIn("CLUSTER_STORAGE", mounts)

        embedded = {Path(item["source"]).name: item["target"] for item in rjob["embedded_files"]}
        self.assertEqual(embedded["adapter.py"], "/tmp/safactory-deepsafe_cyber/adapter.py")
        self.assertEqual(embedded["run.yaml"], "/opt/operator/run.yaml")
        self.assertEqual(embedded["rjob_entrypoint.sh"], "/tmp/safactory-deepsafe_cyber/rjob_entrypoint.sh")

        wrapper = (ENV / "rjob_entrypoint.sh").read_text(encoding="utf-8")
        for marker in (
            "flock 9",
            "packaging/linux-amd64/entrypoint.sh",
            "remount,bind,ro",
            "safactory-dockerd.log",
            "ln -s \"$results_root\" /opt/deepsafe-cyber/runs",
            'python "$runtime_root/runner.py"',
        ):
            self.assertIn(marker, wrapper)


def _envelope(run_id: str, terminal_code: str) -> dict[str, Any]:
    return {
        "schema": "deepsafe-cyber.published-result.v2",
        "run_id": run_id,
        "terminal": {"code": terminal_code},
        "evaluation": {"model_verdict": "INCORRECT" if terminal_code != "PASS" else "CORRECT"},
        "cleanup": {"status": "COMPLETED"},
        "artifacts": {"eval_log": {"ref": f"{run_id}/private/eval/test.eval"}},
    }


class _tempdir:
    def __init__(self):
        import tempfile

        self._cm = tempfile.TemporaryDirectory(prefix="deepsafe-cyber-test-")

    def __enter__(self):
        return self._cm.__enter__()

    def __exit__(self, exc_type, exc, tb):
        return self._cm.__exit__(exc_type, exc, tb)


if __name__ == "__main__":
    unittest.main()
