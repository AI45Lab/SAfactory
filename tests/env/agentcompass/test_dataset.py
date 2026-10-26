from __future__ import annotations

import importlib.util
import inspect
import io
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import yaml

ROOT = Path(__file__).resolve().parents[3]
ENV = ROOT / "env/agentcompass"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


generator = load_module("agentcompass_dataset_test", ENV / "generate_dataset.py")
runner = load_module("agentcompass_runner_dataset_test", ENV / "runner.py")


class DatasetTests(unittest.TestCase):
    def test_selection_rejects_invented_or_duplicate_ids(self):
        tasks = [SimpleNamespace(task_id=str(i), metadata={"sub_steps": [1, 2, 3]})
                 for i in range(3)]  # Synthetic IDs only for selection unit tests.
        self.assertEqual(generator.select_tasks(tasks), tasks[:2])
        self.assertEqual(generator.select_tasks(tasks, sample_ids=["2", "0"]), [tasks[2], tasks[0]])
        for ids in (["unknown"], ["0", "0"]):
            with self.assertRaises(ValueError):
                generator.select_tasks(tasks, sample_ids=ids)
        with self.assertRaises(ValueError):
            generator.select_tasks(tasks, limit=0)
        with self.assertRaises(ValueError):
            generator.select_tasks([tasks[0], tasks[0]])

    def test_manifest_uses_existing_config_and_expands_independent_rows(self):
        try:
            from core.data_manager.load_yaml import load_yaml_configs
            from core.data_manager.yaml_aggregator import _expand_environment_rows
        except ModuleNotFoundError as exc:
            self.skipTest(f"SAfactory loader dependency is not installed: {exc.name}")

        tasks = [SimpleNamespace(task_id=str(i), metadata={"sub_steps": [1, 2, 3]})
                 for i in range(2)]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "tasks.jsonl"
            kwargs = dict(split="validation", container_data_dir="/app/data/agentcompass", assets={})
            report = generator.write_manifest(output, tasks, **kwargs)
            self.assertEqual(report["recommended_max_steps"], 4)
            self.assertEqual(list(root.iterdir()), [output])  # No second config/start entry.
            with self.assertRaises(FileExistsError):
                generator.write_manifest(output, tasks, **kwargs)
            config = yaml.safe_load((ENV / "agentcompass_config.rjob.yaml").read_text())
            config["environments"][0]["dataset"] = str(output)
            config_path = root / "config.local.yaml"
            config_path.write_text(yaml.safe_dump(config))
            rows = _expand_environment_rows("offline-batch", load_yaml_configs(str(config_path)))
            self.assertEqual(len(rows), 2)
            self.assertEqual(len({r["env_id"] for r in rows}), 2)
            self.assertEqual(len({r["group_id"] for r in rows}), 2)
            self.assertEqual([r["env_params"]["dataset"]["sample_id"] for r in rows], ["0", "1"])
            self.assertTrue(all(r["env_params"]["dataset"]["harness_params"]["mode"] == "naive"
                                for r in rows))

    def test_mount_missing_fails_and_workspaces_are_session_scoped(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "data"
            dataset = {"data_dir": str(data)}
            with self.assertRaisesRegex(RuntimeError, "Missing native"):
                runner._prepare_native_data("scicode", "scicode_tool_use", dataset, root / "s1", {}, {})
            (data / "scicode").mkdir(parents=True)
            for name in ("test_data.h5", "problems_dev.jsonl"):
                (data / "scicode" / name).touch()  # Path test only, not valid benchmark assets.
            workspaces = []
            for session in ("s1", "s2"):
                bp, hp = {"split": "validation"}, {}
                actual = runner._prepare_native_data(
                    "scicode", "scicode_tool_use", dataset, root / session, bp, hp)
                self.assertEqual(actual, str(data))
                self.assertEqual(bp["h5py_file"], str(data / "scicode/test_data.h5"))
                self.assertEqual(Path(hp["code_workdir"]).parent, root / session)
                workspaces.append(bp["workspace_root"])
            self.assertNotEqual(*workspaces)
            with self.assertRaisesRegex(RuntimeError, "explicit mounted"):
                runner._prepare_native_data("scicode", "scicode_tool_use", {}, root, {}, {})
            with self.assertRaisesRegex(RuntimeError, "absolute"):
                runner._prepare_native_data("scicode", "scicode_tool_use", {"data_dir": "../data"}, root, {}, {})

    def test_command_retains_real_model_and_session_url(self):
        command = runner._build_command(
            agentcompass_executable=Path("/opt/venv/bin/agentcompass"),
            benchmark="scicode", harness="scicode_tool_use", environment="host_process",
            model="glm-4.7-Flash", session_url="http://gateway.example/v1/sessions/test-session",
            benchmark_params={"sample_ids": ["native-id"], "k": 1}, harness_params={"mode": "naive"},
            environment_params={}, model_params={"max_retries": 0}, result_root=Path("/app/results/s1"),
            data_dir="/app/data/with spaces")
        self.assertEqual(command[4], "glm-4.7-Flash")
        self.assertEqual(command[command.index("--data-dir") + 1], "/app/data/with spaces")
        self.assertEqual(command[command.index("--model-base-url") + 1],
                         "http://gateway.example/v1/sessions/test-session")
        self.assertEqual(command[command.index("--task-concurrency") + 1], "1")

    def test_sample_80_requires_its_extra_dependency_before_writing(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            generator.importlib.util, "find_spec", return_value=None
        ), mock.patch.object(generator, "load_native_tasks", return_value=(
            [SimpleNamespace(task_id="80", metadata={"sub_steps": [1]})], {}
        )), mock.patch("sys.stderr", new_callable=io.StringIO) as stderr:
            output = Path(tmp) / "tasks.jsonl"
            with self.assertRaises(SystemExit):
                generator.main(["--benchmark", "scicode", "--data-dir", "/unused",
                                "--split", "test", "--sample-id", "80", "--output", str(output)])
            self.assertIn("matplotlib", stderr.getvalue())
            self.assertFalse(output.exists())

    def test_catalog_selection_needs_no_native_runtime_and_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            catalog = root / "catalog.jsonl"
            tasks = [SimpleNamespace(task_id=str(i), metadata={"sub_steps": [1, 2]}) for i in range(3)]
            report = generator.write_manifest(catalog, tasks, split="validation",
                                               container_data_dir="/app/data/agentcompass", assets={})
            report["task_manifest_sha256"] = generator.sha256(catalog)
            catalog.with_suffix(".metadata.json").write_text(json.dumps(report))
            with mock.patch.object(generator, "load_native_tasks", side_effect=AssertionError("Native import")), \
                    mock.patch("sys.stdout", new_callable=io.StringIO):
                self.assertEqual(generator.main([
                    "--benchmark", "scicode", "--catalog", str(catalog), "--sample-id", "1",
                    "--output", str(root / "selected.jsonl"),
                ]), 0)
            rows = [json.loads(line) for line in (root / "selected.jsonl").read_text().splitlines()]
            self.assertEqual([r["sample_id"] for r in rows], ["1"])
            catalog.write_text(catalog.read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "checksum"):
                generator.load_catalog(catalog)

    def test_official_hdf5_validation_rejects_a_fixture(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test_data.h5"
            path.write_bytes(b"fixture")
            with self.assertRaisesRegex(ValueError, "verified official"):
                generator.verify_h5(path)


@unittest.skipUnless(importlib.util.find_spec("agentcompass"), "Requires pinned AgentCompass runtime")
class NativeDatasetTests(unittest.TestCase):
    def test_pinned_loader_ids_and_asset_checks_without_download(self):
        from agentcompass.benchmarks.scicode import SciCodeBenchmark
        import agentcompass.benchmarks.scicode as native
        import h5py

        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            native, "ensure_wget_unzip", side_effect=AssertionError("Unexpected download")
        ):
            root = Path(tmp)
            data = root / "scicode"
            data.mkdir()
            packaged = Path(inspect.getfile(SciCodeBenchmark)).parent / "data"
            for name in ("problems_dev.jsonl", "problems_test.jsonl"):
                shutil.copyfile(packaged / name, data / name)
            with self.assertRaisesRegex(ValueError, "test_data.h5"):
                generator.load_native_tasks(root, "validation")
            # Synthetic HDF5 allows an offline loader test ONLY. No evaluation
            # is run and this fixture must never be presented as official data.
            with h5py.File(data / "test_data.h5", "w") as fixture:
                fixture.create_dataset("loader-test-only", data=[1])
            # Explicit test-only override. The production CLI rejects this file.
            with mock.patch.object(generator, "verify_h5"):
                tasks, assets = generator.load_native_tasks(root, "validation")
            self.assertEqual(len(tasks), 15)
            selected = generator.select_tasks(tasks, limit=2)
            generator.write_manifest(root / "tasks.jsonl", selected, split="validation",
                                     container_data_dir="/app/data/agentcompass", assets=assets)
            rows = [json.loads(line) for line in (root / "tasks.jsonl").read_text().splitlines()]
            self.assertEqual([r["sample_id"] for r in rows], [str(t.task_id) for t in selected])
            cli = subprocess.run([
                sys.executable, "-B", str(ENV / "generate_dataset.py"),
                "--benchmark", "scicode", "--data-dir", str(root),
                "--limit", "2", "--output", str(root / "cli-tasks.jsonl"),
            ], capture_output=True, text=True)
            self.assertNotEqual(cli.returncode, 0)
            self.assertIn("verified official", cli.stderr)
            self.assertFalse((root / "cli-tasks.jsonl").exists())
            with mock.patch.object(generator, "verify_h5"):
                self.assertEqual(len(generator.load_native_tasks(root, "all")[0]), 80)
            with (data / "problems_dev.jsonl").open("a") as stream:
                stream.write("\n")
            with self.assertRaisesRegex(ValueError, "pinned official"):
                generator.load_native_tasks(root, "validation")

    def test_shared_preparation_is_repeatable_without_download_or_overwrite(self):
        import h5py
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = root / "fixture.h5"
            with h5py.File(fixture, "w") as data:
                data.create_dataset("loader-test-only", data=[1])
            with mock.patch.object(generator, "verify_h5"), \
                    mock.patch.object(generator, "verify_h5_structure", return_value={"test_fixture": True}), \
                    mock.patch.object(generator.urllib.request, "urlopen", side_effect=AssertionError("Download")):
                first = generator.prepare_shared_data(root / "shared", fixture)
                asset = root / "shared/scicode/test_data.h5"
                original_mtime = asset.stat().st_mtime_ns
                second = generator.prepare_shared_data(root / "shared")
                self.assertFalse(first["reused_complete_data"])
                self.assertTrue(second["reused_complete_data"])
                self.assertEqual(asset.stat().st_mtime_ns, original_mtime)
                self.assertEqual(first["task_manifest_sha256"], second["task_manifest_sha256"])
                self.assertEqual(len(generator.load_catalog(root / "shared/manifests/scicode-validation.jsonl")[0]), 15)


if __name__ == "__main__":
    unittest.main()
