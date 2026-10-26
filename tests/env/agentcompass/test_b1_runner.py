"""B1 common runner capabilities: judge routing, shared data and native caches."""
from __future__ import annotations

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
            self.assertEqual(record["amendments"][-1]["removed_runtime_locks"], ["cache_builder.lock"])

    def test_changed_asset_blocks_reconcile(self):
        with tempfile.TemporaryDirectory() as tmp:
            data, root = self._published(tmp)
            (root / "deepsearchqa.jsonl").write_text("tampered\n")
            with self.assertRaisesRegex(ValueError, "changed"):
                prepare.prepare("deepsearchqa", data, "/app/data", reconcile=True)
