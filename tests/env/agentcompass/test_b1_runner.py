"""B1 common runner capabilities: judge routing, shared data and native caches."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("agentcompass_runner_b1_test", ROOT / "env/agentcompass/runner.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)

SESSION_URL = "http://gateway.example/v1/sessions/session-1"


class JudgeRoutingTests(unittest.TestCase):
    """Judge routing is pending framework guidance: refuse, never pick a route."""

    def test_every_judge_benchmark_is_rejected_before_any_model_call(self):
        for benchmark in runner.JUDGE_BENCHMARKS:
            for params in ({}, {"judge_model": {"id": "judge"}}):
                with self.subTest(benchmark=benchmark, params=params), \
                        self.assertRaisesRegex(runner.AdapterError, "no agreed routing") as caught:
                    runner._reject_unrouted_judge(benchmark, dict(params))
                self.assertEqual(caught.exception.error_type, "judge_routing_undefined")

    def test_judge_spec_on_other_benchmarks_is_rejected(self):
        with self.assertRaises(runner.AdapterError):
            runner._reject_unrouted_judge("scicode", {"judge_model": {"id": "judge"}})
        params = {}
        runner._reject_unrouted_judge("scicode", params)
        self.assertEqual(params, {})

    def test_runner_never_rewrites_judge_endpoints(self):
        source = (ROOT / "env/agentcompass/runner.py").read_text(encoding="utf-8")
        self.assertNotIn("/aux/", source)


class SharedDataTests(unittest.TestCase):
    def test_listed_benchmarks_need_mounted_native_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "data"
            for benchmark, files in runner.NATIVE_DATA_FILES.items():
                with self.subTest(benchmark=benchmark):
                    with self.assertRaisesRegex(RuntimeError, "explicit mounted data_dir"):
                        runner._prepare_native_data(benchmark, "naive_search_agent", {}, Path(tmp), {}, {})
                    with self.assertRaisesRegex(RuntimeError, "Missing native"):
                        runner._prepare_native_data(
                            benchmark, "naive_search_agent", {"data_dir": str(root)}, Path(tmp), {}, {})
                    for name in files:
                        (root / name).parent.mkdir(parents=True, exist_ok=True)
                        (root / name).touch()
                    self.assertEqual(runner._prepare_native_data(
                        benchmark, "naive_search_agent", {"data_dir": str(root)}, Path(tmp), {}, {}), str(root))

    def test_multimodal_hle_and_gaia_are_not_enabled(self):
        for benchmark in ("hle", "gaia"):
            with self.subTest(benchmark=benchmark), self.assertRaisesRegex(RuntimeError, "multimodal"):
                runner._prepare_native_data(
                    benchmark, "naive_search_agent", {"data_dir": "/data"}, Path("/tmp"), {"modality": "vlm"}, {})

    def test_sealqa_reads_shared_parquet_but_caches_per_episode(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "data"
            parquet = root / "sealqa/267b8197/longseal.parquet"
            parquet.parent.mkdir(parents=True)
            parquet.touch()
            episode = Path(tmp) / "results/job/session/task"
            dataset = {"data_dir": str(root)}
            with self.assertRaisesRegex(RuntimeError, "dataset_path"):
                runner._prepare_native_data("sealqa", "openai_chat", dataset, episode, {}, {})
            outside = Path(tmp) / "elsewhere.parquet"
            outside.touch()
            with self.assertRaisesRegex(RuntimeError, "dataset_path"):
                runner._prepare_native_data(
                    "sealqa", "openai_chat", dataset, episode, {"dataset_path": str(outside)}, {})
            data_dir = runner._prepare_native_data(
                "sealqa", "openai_chat", dataset, episode, {"dataset_path": str(parquet)}, {})
            self.assertEqual(Path(data_dir), episode.resolve() / "native-data")
            self.assertNotIn(root, Path(data_dir).parents)

    def test_native_library_caches_stay_in_episode_results(self):
        episode = Path("/app/results/job/session/task")
        with mock.patch.dict(os.environ, {"HF_HOME": "/shared/hf"}, clear=False):
            env = runner._native_env(episode)
        for key in ("HF_HOME", "HF_DATASETS_CACHE", "XDG_CACHE_HOME"):
            self.assertTrue(Path(env[key]).is_relative_to(episode), key)
        self.assertEqual(env["HF_HUB_OFFLINE"], "1")
        self.assertEqual(env["HF_DATASETS_OFFLINE"], "1")


class HarnessFreeTests(unittest.TestCase):
    registered = (
        frozenset({"taubench", "scicode"}),
        frozenset({"scicode_tool_use"}),
        frozenset({"host_process"}),
        frozenset({"taubench"}),
    )

    def test_none_harness_only_for_native_harness_free_benchmarks(self):
        with mock.patch.object(runner, "_registered_components", return_value=self.registered):
            runner._validate_registered_selection("taubench", "none", "host_process")
            with self.assertRaisesRegex(runner.AdapterError, "harness-free"):
                runner._validate_registered_selection("scicode", "none", "host_process")


class HarnessCredentialTests(unittest.TestCase):
    def test_search_harness_requires_keys_before_any_model_call(self):
        with mock.patch.dict(os.environ, {"SERPER_API_KEY": "", "JINA_API_KEY": "set"}, clear=False):
            with self.assertRaisesRegex(runner.AdapterError, "SERPER_API_KEY") as caught:
                runner._check_harness_credentials("naive_search_agent")
        self.assertEqual(caught.exception.error_type, "credential_missing")
        with mock.patch.dict(os.environ, {"SERPER_API_KEY": "set", "JINA_API_KEY": "set"}, clear=False):
            runner._check_harness_credentials("naive_search_agent")
        runner._check_harness_credentials("openai_chat")


prep_spec = importlib.util.spec_from_file_location(
    "agentcompass_prepare_b1_test", ROOT / "env/agentcompass/prepare_shared_data.py")
prepare = importlib.util.module_from_spec(prep_spec)
prep_spec.loader.exec_module(prepare)


class SharedCatalogTests(unittest.TestCase):
    RECORD = {"record_sha256": "r" * 64}

    def test_catalog_publication_is_idempotent_and_tamper_evident(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            params = prepare._container_params(
                {"category": "longseal", "dataset_path": str(root / "sealqa/rev/longseal.parquet")},
                root, "/app/data/agentcompass")
            self.assertEqual(params["dataset_path"], "/app/data/agentcompass/sealqa/rev/longseal.parquet")
            rows = [prepare._row("sealqa", "longseal", sid, params, "/app/data/agentcompass")
                    for sid in ("longseal-001", "longseal-002")]
            catalog = Path(prepare._publish_catalog(root, "sealqa", "longseal", rows, self.RECORD))
            self.assertEqual(prepare._publish_catalog(root, "sealqa", "longseal", rows, self.RECORD), str(catalog))
            with self.assertRaisesRegex(ValueError, "new version directory"):
                prepare._publish_catalog(root, "sealqa", "longseal", rows[:1], self.RECORD)
            loaded = prepare.load_catalog(catalog)
            self.assertEqual([r["task_id"] for r in loaded],
                             ["sealqa-longseal-longseal-001", "sealqa-longseal-longseal-002"])
            self.assertNotIn("harness", loaded[0])
            catalog.write_text(catalog.read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "checksum"):
                prepare.load_catalog(catalog)

    def test_selection_sets_harness_and_only_the_judge_id(self):
        rows = [prepare._row("hle", "llm", sid, {"modality": "llm"}, "/data") for sid in ("a", "b", "c")]
        picked = prepare.select(rows, harness="naive_search_agent", sample_ids=["c", "a"], judge_model="judge-x")
        self.assertEqual([r["sample_id"] for r in picked], ["c", "a"])
        self.assertEqual(picked[0]["harness"], "naive_search_agent")
        self.assertEqual(picked[0]["benchmark_params"], {"modality": "llm", "judge_model": {"id": "judge-x"}})
        self.assertNotIn("judge_model", rows[0]["benchmark_params"])
        self.assertEqual(len(prepare.select(rows, harness="x", limit=2)), 2)
        for bad in (["missing"], ["a", "a"]):
            with self.assertRaises(ValueError):
                prepare.select(rows, harness="x", sample_ids=bad)
        with self.assertRaises(ValueError):
            prepare.select(rows, harness="x", limit=0)

    def test_published_tree_is_verified_before_reuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "data"
            (root / "deepsearchqa").mkdir(parents=True)
            (root / "deepsearchqa/deepsearchqa.jsonl").write_text("{}\n")
            with self.assertRaisesRegex(ValueError, "Unrecorded"):
                prepare.prepare("deepsearchqa", root, "/app/data/agentcompass")
            files = prepare._tree_manifest(root / "deepsearchqa")
            (root / prepare.STATE_DIR).mkdir()
            (root / prepare.STATE_DIR / "deepsearchqa.json").write_text(json.dumps({"files": files}))
            (root / "deepsearchqa/deepsearchqa.jsonl").write_text("tampered\n")
            with self.assertRaisesRegex(ValueError, "differs from its record"):
                prepare.prepare("deepsearchqa", root, "/app/data/agentcompass")

    def test_external_sources_are_never_reported_as_prepared(self):
        with tempfile.TemporaryDirectory() as tmp:
            states = {row["benchmark"]: row for row in prepare.status(Path(tmp))}
        self.assertEqual(states["skillsbench"]["state"], "needs_deployer_data")
        self.assertEqual(states["special_pattern_check"]["state"], "needs_deployer_data")
        self.assertEqual(states["hle"]["state"], "not_published")

    def test_every_pinned_benchmark_has_a_profile(self):
        expected = {
            "browsecomp", "browsecomp_zh", "deepresearch_bench", "deepsearchqa", "deepswe",
            "frontier_engineering", "frontierscience", "gaia", "gdpval_ac", "hle", "hle_verified",
            "pinchbench", "researchclawbench", "scicode", "screenspot", "sealqa", "sgi_deep_research",
            "skillsbench", "special_pattern_check", "swebench_multilingual", "swebench_pro",
            "swebench_verified", "taubench", "terminal_bench_2", "terminal_bench_2_1",
            "terminal_bench_2_verified", "wildclawbench", "xbench_deepsearch",
        }
        self.assertEqual(set(prepare.PROFILES), expected)
        for benchmark, files in runner.NATIVE_DATA_FILES.items():
            for name in files:
                self.assertTrue(name.startswith(prepare.PROFILES[benchmark]["root"] + "/"), name)


class CatalogSafetyTests(unittest.TestCase):
    def test_catalog_rejects_model_configuration(self):
        row = prepare._row("hle", "llm", "a", {"modality": "llm", "judge_model": {"id": "catalog-only"}}, "/data")
        with tempfile.TemporaryDirectory() as tmp, self.assertRaisesRegex(ValueError, "judge_model"):
            prepare._publish_catalog(Path(tmp), "hle", "llm", [row], {"record_sha256": "x"})

    def test_external_profiles_document_the_native_layout(self):
        for name in ("skillsbench", "special_pattern_check"):
            profile = prepare.PROFILES[name]
            self.assertEqual(profile["kind"], "external")
            self.assertTrue(profile["layout"])
            with self.assertRaisesRegex(ValueError, "--source"):
                prepare._stage_external(profile, Path("/tmp"), None)


class TreeVerificationTests(unittest.TestCase):
    def test_lock_files_are_excluded_and_drift_is_classified(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.json").write_text("a")
            (root / "b.json").write_text("b")
            (root / "_cache.lock").write_text("")
            recorded = prepare._tree_manifest(root)
            self.assertEqual(set(recorded), {"a.json", "b.json"})
            (root / "a.json").write_text("changed")
            (root / "b.json").unlink()
            (root / "README.md").write_text("new")
            drift = prepare._verify_tree(root, recorded)
            self.assertEqual(drift, {"missing": ["b.json"], "unrecorded": ["README.md"], "changed": ["a.json"]})

    def test_new_catalogs_never_overwrite_earlier_ones(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "manifests").mkdir()
            (root / "manifests/scicode-validation.jsonl").write_text("old\n")
            row = prepare._row("scicode", "validation", "1", {"split": "validation"}, "/data")
            path = Path(prepare._publish_catalog(root, "scicode", "validation", [row], {"record_sha256": "x"}))
            self.assertEqual(path, root / prepare.CATALOG_DIR / "scicode-validation.jsonl")
            self.assertEqual((root / "manifests/scicode-validation.jsonl").read_text(), "old\n")

    def test_load_only_metrics_never_reach_the_catalog(self):
        profile = prepare.PROFILES["deepresearch_bench"]
        params = prepare._catalog_params("deepresearch_bench", profile, Path("/d"), {})
        self.assertNotIn("metrics", params)
        self.assertEqual(profile["load_only_params"], {"metrics": ["race"]})


class ReconcileTests(unittest.TestCase):
    def _published(self, tmp):
        data = Path(tmp)
        root = data / "deepsearchqa"
        root.mkdir()
        (root / "deepsearchqa.jsonl").write_text("asset\n")
        files = {"deepsearchqa.jsonl": prepare.sha256(root / "deepsearchqa.jsonl"), "cache_builder.lock": "0" * 64}
        (data / prepare.STATE_DIR).mkdir()
        record = {"root": "deepsearchqa", "files": files, "file_count": 2, "total_bytes": 6,
                  "provenance": {}, "record_sha256": "x"}
        (data / prepare.STATE_DIR / "deepsearchqa.json").write_text(json.dumps(record))
        return data, root

    def test_stale_lock_entries_are_dropped_only_when_assets_verify(self):
        with tempfile.TemporaryDirectory() as tmp:
            data, root = self._published(tmp)
            with self.assertRaisesRegex(ValueError, "differs from its record"):
                prepare.prepare("deepsearchqa", data, "/app/data", reconcile=False)
            with mock.patch.object(prepare, "_load_for_catalog", return_value=[]):
                prepare.prepare("deepsearchqa", data, "/app/data", reconcile=True)
            record = json.loads((data / prepare.STATE_DIR / "deepsearchqa.json").read_text())
            self.assertEqual(set(record["files"]), {"deepsearchqa.jsonl"})
            removed = [a["removed_runtime_locks"] for a in record["amendments"] if "removed_runtime_locks" in a]
            self.assertEqual(removed, [["cache_builder.lock"]])
            self.assertIn("metadata", record)

    def test_changed_asset_blocks_reconcile(self):
        with tempfile.TemporaryDirectory() as tmp:
            data, root = self._published(tmp)
            (root / "deepsearchqa.jsonl").write_text("tampered\n")
            with self.assertRaisesRegex(ValueError, "changed"):
                prepare.prepare("deepsearchqa", data, "/app/data", reconcile=True)


class ResumablePublishTests(unittest.TestCase):
    """Interruptions at each phase must resume without re-downloading or losing data."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.data = Path(self.tmp.name)
        self.stage_calls = 0
        self.patches = [
            mock.patch.object(prepare, "_stage", side_effect=self._fake_stage),
            mock.patch.object(prepare, "_load_for_catalog", return_value=[]),
        ]
        for patch in self.patches:
            patch.start()

    def tearDown(self):
        for patch in self.patches:
            patch.stop()
        self.tmp.cleanup()

    def _fake_stage(self, benchmark, profile, stage, source=None):
        self.stage_calls += 1
        (stage / "deepsearchqa").mkdir()
        (stage / "deepsearchqa/deepsearchqa.jsonl").write_text("asset\n")
        return {"source": "test"}

    def _record(self):
        return json.loads((self.data / prepare.STATE_DIR / "deepsearchqa.json").read_text())

    def test_hash_failure_keeps_staged_download(self):
        with mock.patch.object(prepare, "_build_record", side_effect=RuntimeError("killed while hashing")):
            with self.assertRaises(RuntimeError):
                prepare.prepare("deepsearchqa", self.data, "/app/data")
        self.assertFalse((self.data / "deepsearchqa").exists())
        self.assertTrue((self.data / prepare.STAGING_DIR / "deepsearchqa/work/deepsearchqa").exists())
        report = prepare.prepare("deepsearchqa", self.data, "/app/data")
        self.assertEqual(self.stage_calls, 1)
        self.assertFalse(report["reused_complete_data"])
        self.assertIn("hash_s", report["timings"])
        self.assertFalse((self.data / prepare.STAGING_DIR / "deepsearchqa").exists())

    def test_interruption_after_move_before_record_resumes(self):
        real_write_once = prepare._write_once

        def fail_on_record(path, content):
            if path.parent.name == prepare.STATE_DIR:
                raise RuntimeError("killed before the record was written")
            return real_write_once(path, content)

        with mock.patch.object(prepare, "_write_once", side_effect=fail_on_record):
            with self.assertRaises(RuntimeError):
                prepare.prepare("deepsearchqa", self.data, "/app/data")
        self.assertTrue((self.data / "deepsearchqa/deepsearchqa.jsonl").exists())
        self.assertFalse((self.data / prepare.STATE_DIR / "deepsearchqa.json").exists())
        prepare.prepare("deepsearchqa", self.data, "/app/data")
        self.assertEqual(self.stage_calls, 1)
        self.assertEqual(set(self._record()["files"]), {"deepsearchqa.jsonl"})

    def test_publish_rejects_tree_changed_after_hashing(self):
        real_build = prepare._build_record

        def build_then_tamper(benchmark, profile, base, provenance):
            result = real_build(benchmark, profile, base, provenance)
            (base / "deepsearchqa/extra.txt").write_text("appeared after hashing\n")
            return result

        with mock.patch.object(prepare, "_build_record", side_effect=build_then_tamper):
            with self.assertRaisesRegex(ValueError, "does not match its pending record"):
                prepare.prepare("deepsearchqa", self.data, "/app/data")
        self.assertFalse((self.data / prepare.STATE_DIR / "deepsearchqa.json").exists())

    def test_rerun_uses_metadata_check_and_labels_it(self):
        prepare.prepare("deepsearchqa", self.data, "/app/data")
        fast = prepare.prepare("deepsearchqa", self.data, "/app/data")
        self.assertTrue(fast["reused_complete_data"])
        self.assertIn("not content", fast["verification"]["level"])
        full = prepare.prepare("deepsearchqa", self.data, "/app/data", verify_content=True)
        self.assertIn("SHA256", full["verification"]["level"])

    def test_metadata_check_misses_same_size_same_mtime_edit_but_content_check_catches_it(self):
        prepare.prepare("deepsearchqa", self.data, "/app/data")
        asset = self.data / "deepsearchqa/deepsearchqa.jsonl"
        stat = asset.stat()
        asset.write_text("ASSET\n")
        os.utime(asset, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        prepare.prepare("deepsearchqa", self.data, "/app/data")  # documented limit
        with self.assertRaisesRegex(ValueError, "changed"):
            prepare.prepare("deepsearchqa", self.data, "/app/data", verify_content=True)

    def test_old_record_without_metadata_needs_content_check_before_backfill(self):
        prepare.prepare("deepsearchqa", self.data, "/app/data")
        record_path = self.data / prepare.STATE_DIR / "deepsearchqa.json"
        record = self._record()
        record.pop("metadata")
        record_path.write_text(json.dumps(record))
        asset = self.data / "deepsearchqa/deepsearchqa.jsonl"
        asset.write_text("tampered\n")
        with self.assertRaisesRegex(ValueError, "SHA256"):
            prepare.prepare("deepsearchqa", self.data, "/app/data")
        self.assertNotIn("metadata", self._record())
        asset.write_text("asset\n")
        report = prepare.prepare("deepsearchqa", self.data, "/app/data")
        self.assertIn("SHA256", report["verification"]["level"])
        self.assertIn("metadata", self._record())
        self.assertTrue(self._record()["amendments"][-1]["metadata_backfilled_after_content_verification"])


class ReviewRegressionTests(unittest.TestCase):
    """Regressions from review of d523833."""

    def test_extra_published_roots_are_verified_by_both_checks(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp)

            def fake_stage(benchmark, profile, stage, source=None):
                (stage / "gdpval_ac").mkdir()
                (stage / "gdpval_ac/train.parquet").write_text("rows\n")
                (stage / "gdpval_baseline").mkdir()
                (stage / "gdpval_baseline/b.xlsx").write_text("baseline\n")
                return {"source": "test"}

            with mock.patch.object(prepare, "_stage", side_effect=fake_stage), \
                    mock.patch.object(prepare, "_load_for_catalog", return_value=[]):
                prepare.prepare("gdpval_ac", data, "/app/data")
                (data / "gdpval_baseline/b.xlsx").unlink()
                for verify_content in (False, True):
                    with self.subTest(verify_content=verify_content), \
                            self.assertRaisesRegex(ValueError, "gdpval_baseline"):
                        prepare.prepare("gdpval_ac", data, "/app/data", verify_content=verify_content)

    def test_record_without_extra_root_checksums_is_not_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp)
            (data / "gdpval_ac").mkdir()
            (data / "gdpval_ac/train.parquet").write_text("rows\n")
            record = {"root": "gdpval_ac", "files": prepare._tree_manifest(data / "gdpval_ac"),
                      "file_count": 1, "total_bytes": 5, "provenance": {}, "record_sha256": "x"}
            (data / prepare.STATE_DIR).mkdir()
            (data / prepare.STATE_DIR / "gdpval_ac.json").write_text(json.dumps(record))
            with mock.patch.object(prepare, "_load_for_catalog", return_value=[]), \
                    self.assertRaisesRegex(ValueError, "no checksums for gdpval_baseline"):
                prepare.prepare("gdpval_ac", data, "/app/data", verify_content=True)

    @unittest.skipUnless(importlib.util.find_spec("agentcompass"), "Requires pinned AgentCompass runtime")
    def test_completed_archive_is_not_downloaded_again_after_hash_failure(self):
        import zipfile
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp)
            payload = data.parent / f"{data.name}-src.zip"
            with zipfile.ZipFile(payload, "w") as archive:
                archive.writestr("deepsearchqa/deepsearchqa.jsonl", "asset\n")
            downloads = []

            def fake_download(url, target):
                downloads.append(url)
                target.write_bytes(payload.read_bytes())

            real_sha = prepare.sha256
            calls = {"n": 0}

            def failing_first_archive_hash(path):
                if path.suffix == ".zip" and calls["n"] == 0:
                    calls["n"] += 1
                    raise RuntimeError("killed while hashing the archive")
                return real_sha(path)

            try:
                with mock.patch.object(prepare, "_download", side_effect=fake_download), \
                        mock.patch.object(prepare, "_load_for_catalog", return_value=[]), \
                        mock.patch.object(prepare, "_resolve", return_value="http://example/deepsearchqa.zip"):
                    with mock.patch.object(prepare, "sha256", side_effect=failing_first_archive_hash):
                        with self.assertRaises(RuntimeError):
                            prepare.prepare("deepsearchqa", data, "/app/data")
                    self.assertEqual(len(downloads), 1)
                    report = prepare.prepare("deepsearchqa", data, "/app/data")
                self.assertEqual(len(downloads), 1)
                self.assertTrue(report["provenance"]["archive_reused_from_staging"])
                self.assertFalse((data / prepare.STAGING_DIR / "deepsearchqa").exists())
            finally:
                payload.unlink()

    @unittest.skipUnless(importlib.util.find_spec("agentcompass"), "Requires pinned AgentCompass runtime")
    def test_corrupt_archive_in_staging_is_downloaded_again(self):
        with tempfile.TemporaryDirectory() as tmp:
            downloads_dir = Path(tmp) / "downloads"
            downloads_dir.mkdir()
            url = "http://example/x.zip"
            bad = downloads_dir / f"{hashlib.sha256(url.encode()).hexdigest()[:16]}.zip"
            bad.write_bytes(b"not a zip")
            fetched = []
            with mock.patch.object(prepare, "_download",
                                   side_effect=lambda u, t: (fetched.append(u), t.write_bytes(b"new"))):
                archive, reused = prepare._fetch_archive(url, downloads_dir)
            self.assertEqual(fetched, [url])
            self.assertFalse(reused)
            self.assertEqual(archive.read_bytes(), b"new")
