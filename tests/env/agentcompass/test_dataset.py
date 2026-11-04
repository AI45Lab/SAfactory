from __future__ import annotations

import hashlib
import importlib.util
import inspect
import io
import json
import os
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
# Optional: path to benchmarks/scicode/scicode.py in an AgentCompass checkout,
# used only to compare the pinned hashes with real source.
LOCAL_SCICODE_SOURCE = Path(os.environ.get("AGENTCOMPASS_SCICODE_SOURCE") or "/nonexistent")


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
            with self.assertRaisesRegex(RuntimeError, "does not exist"):
                runner._prepare_native_data("scicode", "scicode_tool_use", dataset, root / "s1", {}, {})
            self.assertFalse(data.exists())  # A missing mount is never created.
            data.mkdir()
            with self.assertRaisesRegex(RuntimeError, "Missing native"):
                runner._prepare_native_data("scicode", "scicode_tool_use", dataset, root / "s1", {}, {})
            (data / "scicode").mkdir()
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
        # Real pinned package definitions and native loader; synthetic HDF5.
        import h5py
        unified = generator.unified
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = root / "fixture.h5"
            with h5py.File(fixture, "w") as data:
                data.create_dataset("loader-test-only", data=[1])
            with mock.patch.object(unified, "verify_scicode_h5"), \
                    mock.patch.object(unified, "verify_scicode_h5_structure", return_value={"test_fixture": True}), \
                    mock.patch.object(unified, "_download", side_effect=AssertionError("Download")):
                first = generator.prepare_shared_data(root / "shared", fixture)
                asset = root / "shared/scicode/test_data.h5"
                original_mtime = asset.stat().st_mtime_ns
                second = generator.prepare_shared_data(root / "shared")
                self.assertFalse(first["reused_complete_data"])
                self.assertTrue(second["reused_complete_data"])
                self.assertEqual(asset.stat().st_mtime_ns, original_mtime)
                self.assertEqual(first["catalogs"], second["catalogs"])
                self.assertEqual(first["catalogs"]["validation"]["native_task_count"], 15)
                self.assertEqual(first["catalogs"]["test"]["native_task_count"], 65)
                self.assertEqual(len(unified.load_catalog(root / "shared/manifests/v2/scicode-validation.jsonl")), 15)


def _scicode_fixture(root: Path) -> tuple[Path, bytes]:
    """Shrunken SciCode inputs: synthetic definitions and a matching tiny HDF5.

    Test assets only; they exercise the preparation flow, never the official data.
    Returns (package dir standing in for the pinned module's data/, zip bytes
    standing in for the official archive).
    """
    import h5py
    import zipfile

    package = root / "package-data"
    package.mkdir()
    problems = {
        "problems_dev.jsonl": [{"problem_id": "1", "sub_steps": [{"step_number": "1.1", "test_cases": ["a", "b"]}]},
                               {"problem_id": "2", "sub_steps": [{"step_number": "2.1", "test_cases": ["a"]},
                                                                 {"step_number": "13.6", "test_cases": ["a"]}]}],
        "problems_test.jsonl": [{"problem_id": "3", "sub_steps": [{"step_number": "3.1", "test_cases": ["a"]}]}],
    }
    for name in generator.unified.SCICODE_ASSET_FILES:
        rows = problems.get(name)
        text = "".join(json.dumps(row) + "\n" for row in rows) if rows else f"fixture {name}\n"
        (package / name).write_text(text)
    h5 = root / "fixture.h5"
    with h5py.File(h5, "w") as data:
        for key in ("1.1/test1", "1.1/test2", "2.1/test1", "3.1/test1"):
            data.create_group(key)
    archive = root / "fixture.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.write(h5, generator.unified.SCICODE_H5_MEMBER)
        bundle.writestr("scicode/problems_dev.jsonl", "mutable archive copy; never used\n")
    return package, archive.read_bytes()


class UnifiedSciCodePrepareTests(unittest.TestCase):
    """SciCode through ``prepare_shared_data.py prepare`` from an empty directory.

    Test doubles (no official data, network or AgentCompass runtime involved):
    the download (``_download``) serves a shrunken archive; the pinned package
    directory is a synthetic one; the HDF5 size/SHA256 pins are those of the
    shrunken HDF5; the native loader is replaced by a reader of the staged
    definitions; and AgentCompass's zip validity helper is replaced by
    ``zipfile.testzip``. Real: staging, resume, records, catalogs, select, and
    the HDF5 structure check against the definitions.
    """

    def setUp(self):
        import zipfile

        self.prep = generator.unified
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.data = self.root / "shared"
        self.data.mkdir()
        package, self.zip_bytes = _scicode_fixture(self.root)
        self.downloads = []
        h5 = self.root / "fixture.h5"

        def fake_download(url, target):
            self.downloads.append(url)
            target.write_bytes(self.zip_bytes)

        def fake_loader(benchmark, data_dir, params=None):
            if not Path(params["h5py_file"]).is_file():
                raise ValueError("loader stand-in: missing HDF5")
            names = {"validation": "problems_dev.jsonl", "test": "problems_test.jsonl"}
            lines = (Path(data_dir) / "scicode" / names[params["split"]]).read_text().splitlines()
            return [SimpleNamespace(task_id=json.loads(line)["problem_id"]) for line in lines if line.strip()]

        def zip_is_valid(path):
            try:
                with zipfile.ZipFile(path) as bundle:
                    return bundle.testzip() is None
            except zipfile.BadZipFile:
                return False

        stub = SimpleNamespace(_zip_is_valid=zip_is_valid)
        for patch in (
            mock.patch.object(self.prep, "_download", side_effect=fake_download),
            mock.patch.object(self.prep, "scicode_package_dir", return_value=package),
            mock.patch.object(self.prep, "load_native_tasks", side_effect=fake_loader),
            mock.patch.object(self.prep, "SCICODE_H5_SIZE", h5.stat().st_size),
            mock.patch.object(self.prep, "SCICODE_H5_SHA256", self.prep.sha256(h5)),
            mock.patch.dict(sys.modules, {"agentcompass": SimpleNamespace(), "agentcompass.benchmarks": SimpleNamespace(),
                                          "agentcompass.benchmarks.utils": SimpleNamespace(),
                                          "agentcompass.benchmarks.utils.data": stub}),
        ):
            patch.start()
            self.addCleanup(patch.stop)

    def _cli(self, *argv) -> tuple[int, object]:
        with mock.patch("sys.stdout", new_callable=io.StringIO) as out, \
                mock.patch("sys.stderr", new_callable=io.StringIO):
            code = self.prep.main(list(argv))
        return code, json.loads(out.getvalue())

    def _prepare(self):
        return self._cli("prepare", "--benchmark", "scicode", "--data-dir", str(self.data))

    def test_empty_directory_prepares_publishes_and_reuses(self):
        code, (report,) = self._prepare()
        self.assertEqual(code, 0, report)
        self.assertNotIn("error", report)
        self.assertEqual(self.downloads, [self.prep.SCICODE_URL])
        self.assertFalse(report["reused_complete_data"])
        self.assertEqual(report["provenance"]["hdf5_structure"], {"subproblems": 3, "test_groups": 4})
        published = self.data / "scicode"
        self.assertEqual(sorted(p.name for p in published.iterdir()),
                         sorted([*self.prep.SCICODE_ASSET_FILES, "test_data.h5"]))
        # Definitions come from the pinned package, not from the archive.
        self.assertIn('"problem_id": "1"', (published / "problems_dev.jsonl").read_text())
        self.assertTrue((self.data / self.prep.STATE_DIR / "scicode.json").exists())
        self.assertFalse((self.data / self.prep.STAGING_DIR / "scicode").exists())
        self.assertEqual({k: v["native_task_count"] for k, v in report["catalogs"].items()},
                         {"validation": 2, "test": 1})
        mtime = (published / "test_data.h5").stat().st_mtime_ns

        code, (again,) = self._prepare()
        self.assertEqual(code, 0, again)
        self.assertTrue(again["reused_complete_data"])
        self.assertEqual(len(self.downloads), 1)
        self.assertEqual((published / "test_data.h5").stat().st_mtime_ns, mtime)

        for variant, expected in (("validation", ["2"]), ("test", ["3"])):
            output = self.root / f"{variant}-tasks.jsonl"
            code, summary = self._cli("select", "--catalog", str(self.data / f"manifests/v2/scicode-{variant}.jsonl"),
                                      "--harness", "scicode_tool_use", "--harness-params", '{"mode": "naive"}',
                                      "--sample-id", expected[0], "--output", str(output))
            self.assertEqual(code, 0)
            row = json.loads(output.read_text())
            self.assertEqual(row["sample_id"], expected[0])
            self.assertEqual(row["benchmark_params"], {"split": variant,
                                                      "h5py_file": "/app/data/agentcompass/scicode/test_data.h5"})
            self.assertEqual(row["harness"], "scicode_tool_use")
            self.assertEqual(row["data_dir"], "/app/data/agentcompass")

    def test_interruption_after_download_reuses_the_archive(self):
        for point in ("verify_scicode_h5", "_build_record"):  # HDF5 check, then record hashing
            with self.subTest(point=point):
                real = getattr(self.prep, point)
                calls = []

                def fail_once(*args, **kwargs):
                    if not calls:
                        calls.append(1)
                        raise KeyboardInterrupt(f"killed during {point}")
                    return real(*args, **kwargs)

                with mock.patch.object(self.prep, point, side_effect=fail_once), \
                        self.assertRaises(KeyboardInterrupt):
                    self.prep.prepare("scicode", self.data, "/app/data/agentcompass")
                self.assertFalse((self.data / "scicode").exists())
                self.assertEqual(len(list((self.data / ".staging/scicode/downloads").glob("*.zip"))), 1)
                report = self.prep.prepare("scicode", self.data, "/app/data/agentcompass")
                self.assertEqual(len(self.downloads), 1)
                self.assertTrue((self.data / "scicode/test_data.h5").is_file())
                if point == "verify_scicode_h5":
                    self.assertTrue(report["provenance"]["archive_reused_from_staging"])
                    # Start the second case from an empty directory again.
                    shutil.rmtree(self.data)
                    self.data.mkdir()
                    self.downloads.clear()

    def test_interruption_after_move_before_record_resumes(self):
        real = self.prep._write_once

        def fail_record(path, content):
            if path.parent.name == self.prep.STATE_DIR:
                raise KeyboardInterrupt("killed before the record was written")
            return real(path, content)

        with mock.patch.object(self.prep, "_write_once", side_effect=fail_record), \
                self.assertRaises(KeyboardInterrupt):
            self.prep.prepare("scicode", self.data, "/app/data/agentcompass")
        self.assertTrue((self.data / "scicode/test_data.h5").is_file())
        states = {row["benchmark"]: row["state"] for row in self.prep.status(self.data)}
        self.assertEqual(states["scicode"], "not_published")
        self.assertFalse((self.data / "manifests/v2").exists())
        report = self.prep.prepare("scicode", self.data, "/app/data/agentcompass")
        self.assertEqual(len(self.downloads), 1)
        self.assertFalse(report["reused_complete_data"])
        self.assertEqual({row["benchmark"]: row["state"] for row in self.prep.status(self.data)}["scicode"],
                         "published")

    def test_unverified_hdf5_is_never_published(self):
        bad = self.root / "other.h5"
        bad.write_bytes(b"not the pinned HDF5")
        with self.assertRaisesRegex(ValueError, "verified official"):
            self.prep.prepare("scicode", self.data, "/app/data/agentcompass", source=bad)
        self.assertFalse((self.data / "scicode").exists())
        self.assertFalse((self.data / self.prep.STATE_DIR / "scicode.json").exists())

    def test_legacy_prepare_forwards_to_the_unified_flow(self):
        report = generator.prepare_shared_data(self.data, self.root / "fixture.h5")
        self.assertEqual(self.downloads, [])  # --reuse-h5 maps to --source
        self.assertEqual(report["provenance"]["source"], "deployer-supplied official HDF5")
        self.assertTrue((self.data / self.prep.STATE_DIR / "scicode.json").exists())
        self.assertTrue(generator.prepare_shared_data(self.data)["reused_complete_data"])
        # Same through the legacy CLI flags.
        with mock.patch("sys.stdout", new_callable=io.StringIO) as out:
            self.assertEqual(generator.main(["--benchmark", "scicode", "--data-dir", str(self.data), "--prepare",
                                             "--reuse-h5", str(self.root / "fixture.h5")]), 0)
        self.assertTrue(json.loads(out.getvalue())["reused_complete_data"])

    def test_selected_v2_row_is_a_complete_runnable_task(self):
        self.prep.prepare("scicode", self.data, "/app/data/agentcompass")
        output = self.root / "tasks.jsonl"
        code, _ = self._cli("select", "--catalog", str(self.data / "manifests/v2/scicode-validation.jsonl"),
                            "--harness", "scicode_tool_use", "--harness-params", '{"mode": "naive"}',
                            "--all", "--output", str(output))
        self.assertEqual(code, 0)
        rows = [json.loads(line) for line in output.read_text().splitlines()]
        self.assertEqual(rows[0], {
            "task_id": "scicode-validation-1", "benchmark": "scicode", "harness": "scicode_tool_use",
            "environment": "host_process", "sample_id": "1", "data_dir": "/app/data/agentcompass",
            "benchmark_params": {"split": "validation", "h5py_file": "/app/data/agentcompass/scicode/test_data.h5"},
            "harness_params": {"mode": "naive"}, "environment_params": {}, "model_params": {"max_retries": 0},
            "timeout_seconds": 3600, "contract_only": False,
        })
        # The runner accepts the row as-is (data_dir pointed at this test's published tree).
        row = {**rows[0], "data_dir": str(self.data)}
        request = {"temperature": None, "max_steps": -1}
        bp, hp, ep, mp = runner._merge_params(request, row, sample_id=row["sample_id"], harness=row["harness"])
        data_dir = runner._prepare_native_data("scicode", row["harness"], row, self.root / "episode", bp, hp)
        command = runner._build_command(
            agentcompass_executable=Path("/opt/venv/bin/agentcompass"), benchmark="scicode",
            harness=row["harness"], environment=row["environment"], model="m", session_url="http://gw/s",
            benchmark_params=bp, harness_params=hp, environment_params=ep, model_params=mp,
            result_root=self.root / "episode", data_dir=data_dir)
        self.assertEqual(command[2:4], ["scicode", "scicode_tool_use"])
        sent = json.loads(command[command.index("--benchmark-params") + 1])
        self.assertEqual((sent["split"], sent["sample_ids"], sent["k"]), ("validation", ["1"], 1))
        self.assertEqual(sent["h5py_file"], str(self.data / "scicode/test_data.h5"))

    def test_existing_data_is_adopted_and_earlier_catalogs_are_kept(self):
        native = self.data / "scicode"
        native.mkdir()
        for path in self.prep.scicode_package_dir().iterdir():
            shutil.copyfile(path, native / path.name)
        shutil.copyfile(self.root / "fixture.h5", native / "test_data.h5")
        legacy = self.data / "manifests/scicode-validation.jsonl"
        legacy.parent.mkdir()
        legacy.write_text('{"legacy": true}\n')
        legacy.with_suffix(".metadata.json").write_text('{"legacy": true}\n')
        before = {p: p.read_bytes() for p in legacy.parent.iterdir()}
        h5_mtime = (native / "test_data.h5").stat().st_mtime_ns

        report = self.prep.prepare("scicode", self.data, "/app/data/agentcompass")
        self.assertEqual(self.downloads, [])
        self.assertTrue(report["provenance"]["source"].startswith("adopted"))
        self.assertEqual({p: p.read_bytes() for p in legacy.parent.iterdir() if p.is_file()}, before)
        self.assertEqual((native / "test_data.h5").stat().st_mtime_ns, h5_mtime)
        self.assertTrue((self.data / "manifests/v2/scicode-validation.jsonl").is_file())
        self.assertTrue(self.prep.prepare("scicode", self.data, "/app/data/agentcompass")["reused_complete_data"])

        # Adoption still rejects definitions that differ from the pinned package.
        other = self.root / "other"
        (other / "scicode").mkdir(parents=True)
        for path in native.iterdir():
            shutil.copyfile(path, other / "scicode" / path.name)
        (other / "scicode/problems_dev.jsonl").write_text("{}\n")
        with self.assertRaisesRegex(ValueError, "differs from the pinned package"):
            self.prep.prepare("scicode", other, "/app/data/agentcompass")


class SciCodeVersionCheckTests(unittest.TestCase):
    """The pinned-version check hashes the imported SciCode module source.

    The module is a stand-in written to a temp dir, so the real hashing code
    runs without an installed AgentCompass. The recorded commit string alone
    never satisfies the check.
    """

    def _with_module(self, source: bytes):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        package = Path(tmp.name) / "scicode"
        (package / "data").mkdir(parents=True)
        (package / "scicode.py").write_bytes(source)
        spec = importlib.util.spec_from_file_location("fixture_scicode", package / "scicode.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cls = type("SciCodeBenchmark", (), {"__module__": "fixture_scicode"})
        sys.modules["fixture_scicode"] = module
        self.addCleanup(sys.modules.pop, "fixture_scicode", None)
        fake = SimpleNamespace(SciCodeBenchmark=cls)
        patch = mock.patch.dict(sys.modules, {"agentcompass": SimpleNamespace(),
                                              "agentcompass.benchmarks": SimpleNamespace(),
                                              "agentcompass.benchmarks.scicode": fake})
        patch.start()
        self.addCleanup(patch.stop)
        return package, Path(tmp.name)

    def test_other_source_is_refused_before_any_data_is_written(self):
        unified = generator.unified
        source = b"# not the pinned SciCode module\n"
        package, root = self._with_module(source)
        with self.assertRaisesRegex(ValueError, "requires AgentCompass d2c3e148"):
            unified.scicode_package_dir()
        data = root / "shared"
        with self.assertRaisesRegex(ValueError, "requires AgentCompass"):
            unified.prepare("scicode", data, "/app/data/agentcompass")
        self.assertFalse(data.exists())
        with mock.patch.object(unified, "SCICODE_SOURCE_SHA256", hashlib.sha256(source).hexdigest()):
            self.assertEqual(unified.scicode_package_dir(), package / "data")

    @unittest.skipUnless(LOCAL_SCICODE_SOURCE.is_file(), "Set AGENTCOMPASS_SCICODE_SOURCE to a checkout's scicode.py")
    def test_pinned_hashes_match_the_real_scicode_source(self):
        self.assertEqual(generator.unified.sha256(LOCAL_SCICODE_SOURCE), generator.unified.SCICODE_SOURCE_SHA256)
        harness = LOCAL_SCICODE_SOURCE.parents[2] / "harnesses/scicode_tool_use.py"
        self.assertEqual(generator.sha256(harness), generator.SOURCE_HASHES["SciCodeToolUseHarness"])


class LegacyCatalogTests(unittest.TestCase):
    def test_catalog_written_by_the_previous_tool_still_loads(self):
        # Byte layout of manifests/scicode-validation.* as produced before this
        # change: rows from write_manifest (unchanged), metadata = its report
        # plus the extra fields the old --prepare added.
        with tempfile.TemporaryDirectory() as tmp:
            catalog = Path(tmp) / "manifests/scicode-validation.jsonl"
            tasks = [SimpleNamespace(task_id=str(i), metadata={"sub_steps": [1, 2, 3]}) for i in (1, 3)]
            report = generator.write_manifest(catalog, tasks, split="validation",
                                              container_data_dir="/app/data/agentcompass",
                                              assets={"test_data.h5": generator.H5_SHA256})
            report.update({"task_manifest_sha256": generator.sha256(catalog), "hdf5_structure": {},
                           "official_archive_url": generator.SCICODE_URL, "official_hdf5_crc32": generator.H5_CRC32})
            catalog.with_suffix(".metadata.json").write_text(json.dumps(report, indent=2) + "\n")
            output = Path(tmp) / "selected.jsonl"
            with mock.patch("sys.stdout", new_callable=io.StringIO):
                self.assertEqual(generator.main(["--benchmark", "scicode", "--catalog", str(catalog),
                                                 "--sample-id", "3", "--output", str(output)]), 0)
            row = json.loads(output.read_text())
            self.assertEqual((row["sample_id"], row["harness"], row["harness_params"]["mode"]),
                             ("3", "scicode_tool_use", "naive"))
            self.assertEqual(row["generation_calls_upper_bound"], 3)


if __name__ == "__main__":
    unittest.main()
