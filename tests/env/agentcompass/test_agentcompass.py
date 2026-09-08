from __future__ import annotations

import asyncio
import ast
import importlib.util
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import yaml


TEST_ROOT = Path(__file__).resolve().parent
REPO_ROOT = TEST_ROOT.parents[2]
ENV_ROOT = REPO_ROOT / "env" / "agentcompass"
FIXTURE_ROOT = TEST_ROOT / "fixtures"
APPROVED_RJOB_IMAGE = (
    "registry.h.pjlab.org.cn/ailab-evobox-evobox_cpu/benches:agentcompass-02"
)


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ENV_ROOT / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = _load("agentcompass_runner_test", "runner.py")
rule_evaluator = _load("agentcompass_evaluator_test", "rule_evaluator.py")


def _request(
    *,
    benchmark: str = "swebench_verified",
    harness: str = "mini_swe_agent",
    task_id: str = "task-1",
    sample_id: str = "sample-1",
    contract_only: bool = True,
) -> dict:
    return {
        "job_id": "job-1",
        "session_id": "session-1",
        "agent_name": "agentcompass",
        "agent_id": "row-1",
        "group_id": "group-1",
        "gateway_base_url": "http://127.0.0.1:8000/v1/sessions",
        "model": "test-route",
        "temperature": 0.25,
        "max_steps": 7,
        "storage_type": "sqlite",
        "agent_start_timeout_s": 30,
        "env_params": {
            "results_root": "/tmp/agentcompass-test-results",
            "dataset": {
                "task_id": task_id,
                "sample_id": sample_id,
                "benchmark": benchmark,
                "harness": harness,
                "environment": "host_process",
                "benchmark_params": {"sample_ids": ["must-be-overwritten"]},
                "harness_params": {},
                "environment_params": {},
                "model_params": {"temperature": 99},
                "timeout_seconds": 20,
                "contract_only": contract_only,
            },
        },
    }


def _command_for(request: dict) -> list[str]:
    _, dataset = runner._dataset(request)
    sample_id = runner._sample_id(dataset)
    registered = (
        frozenset({"special_pattern_check", "swebench_verified"}),
        frozenset({"mini_swe_agent", "openai_chat"}),
        frozenset({"host_process"}),
    )
    with mock.patch.object(runner, "_registered_components", return_value=registered):
        benchmark, harness, environment = runner._component_selection(dataset)
    benchmark_params, harness_params, environment_params, model_params = runner._merge_params(
        request,
        dataset,
        sample_id=sample_id,
        harness=harness,
    )
    return runner._build_command(
        agentcompass_executable=Path("/opt/venv/bin/agentcompass"),
        benchmark=benchmark,
        harness=harness,
        environment=environment,
        model=request["model"],
        session_url="http://gateway/v1/sessions/session-1",
        benchmark_params=benchmark_params,
        harness_params=harness_params,
        environment_params=environment_params,
        model_params=model_params,
        result_root=Path("/tmp/result"),
    )


class RunnerTests(unittest.TestCase):
    # Structural sources at Dockerfile commit d2c3e148902e948db3270fa34b2198fb1b10beb7:
    # - benchmarks/{browsecomp,deepsearchqa,frontierscience,hle,sgi_deep_research,
    #   sealqa,special_pattern,swebench_verified}.py
    # - benchmarks/scicode/scicode.py and benchmarks/scorers/{llm,deepsearchqa,
    #   frontierscience,sealqa}.py
    # - runtime/results/detail.py
    @staticmethod
    def _source_shaped_detail(benchmark: str, correct: bool) -> dict:
        """Synthetic values shaped from pinned src/agentcompass/benchmarks scorer outputs."""
        attempt = {
            "status": "completed", "correct": correct, "score": None, "error": "",
            "final_answer": "synthetic candidate", "ground_truth": "synthetic reference",
            "artifacts": {}, "extra": {},
        }
        if benchmark in {"browsecomp", "hle", "hle_verified", "sgi_deep_research"}:
            attempt["extra"] = {"scoring": {
                "evaluation_type": "llm_judge", "correct": correct,
                "model_answer": attempt["final_answer"], "ground_truth": attempt["ground_truth"],
            }}
        elif benchmark == "deepsearchqa":
            attempt["extra"] = {"scoring": {
                "evaluation_type": "deepsearchqa_judge", "correct": correct,
                "all_expected_correct": correct, "has_excessive_answers": False,
                "correctness_details": {"synthetic part": correct}, "excessive_answers": [],
                "explanation": "synthetic explanation",
            }}
        elif benchmark == "frontierscience":
            attempt["extra"] = {"scoring": {
                "evaluation_type": "frontierscience_olympiad_judge", "correct": correct,
                "reason": "synthetic judge reason",
            }}
        elif benchmark == "scicode":
            attempt["ground_truth"] = {"problem_id": "sample-1", "total_steps": 1}
            attempt["final_answer"] = {"step_codes": {"sample-1.1": "return 1"}}
            attempt["score"] = 1.0 if correct else 0.0
            attempt["meta"] = {"evaluation": {
                "problem_correct": 1 if correct else 0,
                "total_correct": 1 if correct else 0, "total_steps": 1,
                "subproblem_correctness": attempt["score"],
                "steps": [{"status": "pass" if correct else "fail", "correct": correct}],
                "error": "",
            }}
        elif benchmark == "sealqa":
            grade = "A" if correct else "B"
            attempt["extra"] = {"scoring": {
                "evaluation_type": "sealqa_official_llm_judge", "correct": correct,
                "grade": grade, "label": "correct" if correct else "incorrect",
                "raw_response": grade, "judge_model": "synthetic-judge", "api_protocol": "openai-chat",
            }}
        elif benchmark == "swebench_verified":
            attempt["extra"] = {
                "status": "completed", "eval_raw_data": {"completed": True, "resolved": correct}
            }
        elif benchmark != "special_pattern_check":
            raise AssertionError(f"missing synthetic schema for {benchmark}")
        return {"task_id": "sample-1", "correct": correct, "score": attempt["score"], "attempts": {"1": attempt}}

    def test_dispatch_has_only_source_proven_ids_and_no_aliases(self):
        expected = {
            "browsecomp": "llm_judge", "deepsearchqa": "deepsearchqa_judge",
            "frontierscience": "frontierscience", "hle": "llm_judge", "hle_verified": "llm_judge",
            "scicode": "scicode_fractional", "sealqa": "sealqa_judge",
            "sgi_deep_research": "llm_judge", "special_pattern_check": "special_pattern_correct",
            "swebench_verified": "swebench_resolved",
        }
        self.assertEqual(runner.RESULT_NORMALIZERS, expected)
        self.assertEqual(rule_evaluator.NORMALIZATION_BY_BENCHMARK, expected)
        for unregistered_alias in ("swebench", "sgi", "frontier_science", "hle-verified"):
            self.assertNotIn(unregistered_alias, runner.RESULT_NORMALIZERS)

    def test_every_supported_benchmark_high_zero_and_invalid_native_states(self):
        for benchmark in runner.RESULT_NORMALIZERS:
            for correct, expected in ((True, 10.0), (False, 0.0)):
                with self.subTest(benchmark=benchmark, correct=correct):
                    normalized = runner._normalize_detail(benchmark, self._source_shaped_detail(benchmark, correct))
                    self.assertEqual(normalized["normalized_reward_10"], expected)

            base = self._source_shaped_detail(benchmark, False)
            for status, error in (("run_error", "synthetic failure"), ("skipped", "")):
                invalid = json.loads(json.dumps(base))
                invalid["attempts"]["1"].update(status=status, error=error)
                with self.subTest(benchmark=benchmark, status=status), self.assertRaises(RuntimeError):
                    runner._normalize_detail(benchmark, invalid)

            missing = json.loads(json.dumps(base))
            missing.pop("correct")
            missing["attempts"]["1"].pop("correct")
            with self.subTest(benchmark=benchmark, case="missing"), self.assertRaises(RuntimeError):
                runner._normalize_detail(benchmark, missing)

            wrong_type = json.loads(json.dumps(base))
            wrong_type["correct"] = wrong_type["attempts"]["1"]["correct"] = "false"
            with self.subTest(benchmark=benchmark, case="type"), self.assertRaises(RuntimeError):
                runner._normalize_detail(benchmark, wrong_type)

            if benchmark == "scicode":
                nonfinite = json.loads(json.dumps(base))
                nonfinite["attempts"]["1"]["score"] = float("nan")
                with self.subTest(benchmark=benchmark, case="nonfinite"), self.assertRaises(RuntimeError):
                    runner._normalize_detail(benchmark, nonfinite)

            with self.subTest(benchmark=benchmark, case="identity"), self.assertRaisesRegex(
                RuntimeError, "task_id did not match"
            ):
                runner._result_from_detail(
                    session_id="session-1", task_id="task-1", benchmark=benchmark,
                    harness="synthetic", environment="host_process", sample_id="different",
                    detail=base, duration_ms=1.0,
                )

    def test_malformed_attempt_sets_fail_closed(self):
        base = self._source_shaped_detail("special_pattern_check", True)
        for attempts in ({}, {"1": {}, "2": {}}):
            malformed = {**base, "attempts": attempts}
            with self.assertRaisesRegex(RuntimeError, "exactly one attempt"):
                runner._normalize_detail("special_pattern_check", malformed)

    def test_pinned_registry_contains_public_diagnostic_components(self):
        try:
            registered = runner._registered_components()
        except runner.AdapterError as exc:
            self.skipTest(f"AgentCompass registry is not installed in the host test environment: {exc}")
        self.assertIn("special_pattern_check", registered[0])
        self.assertIn("openai_chat", registered[1])
        self.assertIn("host_process", registered[2])

    def test_unregistered_components_fail_before_cli_start(self):
        registered = (
            frozenset({"special_pattern_check"}),
            frozenset({"openai_chat"}),
            frozenset({"host_process"}),
        )
        for field in ("benchmark", "harness"):
            request = _request(benchmark="special_pattern_check", harness="openai_chat")
            request["env_params"]["dataset"][field] = "not_registered"
            stdout = io.StringIO()
            with mock.patch.object(runner, "_registered_components", return_value=registered), mock.patch.object(
                sys, "stdin", io.StringIO(json.dumps(request))
            ), mock.patch.object(sys, "stdout", stdout):
                self.assertEqual(runner.main(), 0)
            result = json.loads(stdout.getvalue())
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["metrics"]["error_type"], "component_unregistered")

    def test_public_diagnostic_dataset_is_real_single_episode(self):
        config = yaml.safe_load((ENV_ROOT / "agentcompass_config.yaml").read_text(encoding="utf-8"))
        environment = config["environments"][0]
        self.assertEqual(environment["dataset"], "./datasets/agentcompass_diagnostic.jsonl")

        rows = [
            json.loads(line)
            for line in (ENV_ROOT / "datasets" / "agentcompass_diagnostic.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertIs(row["contract_only"], False)
        self.assertEqual(row["sample_id"], "empty_content_gate4_0")
        self.assertEqual(
            (row["benchmark"], row["harness"], row["environment"]),
            ("special_pattern_check", "openai_chat", "host_process"),
        )
        self.assertNotIn("contract-only-not-a-real", json.dumps(row))

    def test_two_manifest_rows_expand_to_two_independent_episodes(self):
        try:
            from core.data_manager.load_yaml import load_yaml_configs
        except ModuleNotFoundError as exc:
            self.skipTest(f"SAfactory loader dependency is not installed: {exc.name}")

        rows = [
            {"task_id": "one", "sample_id": "sample-one"},
            {"task_id": "two", "sample_id": "sample-two"},
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset = root / "rows.jsonl"
            dataset.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            config = root / "config.yaml"
            config.write_text(
                yaml.safe_dump({
                    "environments": [{
                        "env_name": "agentcompass",
                        "env_num": 1,
                        "dataset": "./rows.jsonl",
                        "dataset_load_mode": "eager",
                    }]
                }),
                encoding="utf-8",
            )
            expanded = load_yaml_configs(str(config))
        self.assertEqual(len(expanded), 2)
        self.assertEqual([item["task_idx"] for item in expanded], [1, 2])
        self.assertEqual(
            [item["env_params"]["dataset"]["sample_id"] for item in expanded],
            ["sample-one", "sample-two"],
        )

    def test_native_gate4_dataset_and_mount_match_fixed_loader_contract(self):
        native_path = ENV_ROOT / "datasets" / "special_pattern_check" / "gate4.jsonl"
        native_rows = [json.loads(line) for line in native_path.read_text(encoding="utf-8").splitlines() if line]
        self.assertEqual(len(native_rows), 1)
        self.assertEqual(native_rows[0]["category"], "empty_content")
        self.assertIsInstance(native_rows[0]["messages"], list)
        generated_ids = [f"{row.get('category', '')}_gate4_{index}" for index, row in enumerate(native_rows)]
        self.assertEqual(generated_ids, ["empty_content_gate4_0"])

        public_row = json.loads(
            (ENV_ROOT / "datasets" / "agentcompass_diagnostic.jsonl").read_text(encoding="utf-8")
        )
        start = yaml.safe_load((ENV_ROOT / "agentcompass_start.yaml").read_text(encoding="utf-8"))
        matching_mounts = [
            mount
            for mount in start["container"]["mounts"]
            if mount["target"] == public_row["benchmark_params"]["dataset_dir"]
        ]
        self.assertEqual(
            matching_mounts,
            [{"source": "./datasets/special_pattern_check", "target": "/app/data/special_pattern_check", "mode": "ro"}],
        )
        self.assertEqual(public_row["benchmark_params"]["version"], "gate4")

    def test_installed_agentcompass_loader_recognizes_native_gate4_sample(self):
        try:
            from agentcompass.benchmarks.special_pattern import SpecialPatternCheckBenchmark
        except ModuleNotFoundError as exc:
            self.skipTest(f"AgentCompass is not installed in the host test environment: {exc}")

        request = types.SimpleNamespace(
            model=types.SimpleNamespace(id="fixture-route"),
            benchmark=types.SimpleNamespace(
                params={
                    "dataset_dir": str(ENV_ROOT / "datasets" / "special_pattern_check"),
                    "version": "gate4",
                }
            )
        )
        tasks = SpecialPatternCheckBenchmark().load_tasks(request)
        self.assertEqual([task.task_id for task in tasks], ["empty_content_gate4_0"])

    def test_contract_fixture_is_not_referenced_by_public_config(self):
        fixture = FIXTURE_ROOT / "contract_rows.jsonl"
        fixture_rows = [json.loads(line) for line in fixture.read_text(encoding="utf-8").splitlines() if line]
        self.assertTrue(fixture_rows)
        self.assertTrue(all(row["contract_only"] is True for row in fixture_rows))
        config_text = (ENV_ROOT / "agentcompass_config.yaml").read_text(encoding="utf-8")
        self.assertNotIn("tests/fixtures", config_text)
        self.assertNotIn(fixture.name, config_text)

    def test_two_component_combinations_generate_isolated_single_sample_commands(self):
        cases = (
            ("swebench_verified", "mini_swe_agent", "swe-sample"),
            ("special_pattern_check", "openai_chat", "pattern-sample"),
        )
        for benchmark, harness, sample_id in cases:
            request = _request(benchmark=benchmark, harness=harness, sample_id=sample_id)
            command = _command_for(request)
            self.assertEqual(
                command[:5],
                ["/opt/venv/bin/agentcompass", "run", benchmark, harness, "test-route"],
            )
            self.assertEqual(command[command.index("--env") + 1], "host_process")
            benchmark_params = json.loads(command[command.index("--benchmark-params") + 1])
            model_params = json.loads(command[command.index("--model-params") + 1])
            self.assertEqual(benchmark_params["sample_ids"], [sample_id])
            self.assertEqual(model_params["temperature"], 0.25)
            self.assertEqual(command[command.index("--task-concurrency") + 1], "1")
            self.assertEqual(command[command.index("--max-retries") + 1], "0")

    def test_safactory_step_limit_overrides_mini_swe_row(self):
        request = _request()
        request["env_params"]["dataset"]["harness_params"] = {"step_limit": 999}
        request["env_params"]["dataset"]["model_params"] = {"max_steps": 999}
        command = _command_for(request)
        harness_params = json.loads(command[command.index("--harness-params") + 1])
        model_params = json.loads(command[command.index("--model-params") + 1])
        self.assertEqual(harness_params["step_limit"], 7)
        self.assertNotIn("max_steps", model_params)

    def test_contract_smokes_are_independent_and_output_one_result_each(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            roots = []
            for index, (benchmark, harness) in enumerate(
                (("swebench_verified", "mini_swe_agent"), ("special_pattern_check", "openai_chat")),
                start=1,
            ):
                request = _request(
                    benchmark=benchmark,
                    harness=harness,
                    task_id=f"task-{index}",
                    sample_id=f"sample-{index}",
                )
                request["job_id"] = f"job-{index}"
                request["session_id"] = f"session-{index}"
                request["env_params"]["results_root"] = temp_dir
                stdout = io.StringIO()
                registered = (
                    frozenset({"special_pattern_check", "swebench_verified"}),
                    frozenset({"mini_swe_agent", "openai_chat"}),
                    frozenset({"host_process"}),
                )
                with mock.patch.object(runner, "_registered_components", return_value=registered), mock.patch.object(
                    sys, "stdin", io.StringIO(json.dumps(request))
                ), mock.patch.object(
                    sys, "stdout", stdout
                ), mock.patch.dict(
                    os.environ,
                    {"SAFACTORY_GATEWAY_SESSION_URL_CONTAINER": f"http://gateway/v1/sessions/session-{index}"},
                    clear=False,
                ):
                    self.assertEqual(runner.main(), 0)
                lines = [line for line in stdout.getvalue().splitlines() if line.strip()]
                self.assertEqual(len(lines), 1)
                result = json.loads(lines[0])
                self.assertEqual(result["status"], "succeeded")
                self.assertTrue(result["metrics"]["contract_only"])
                roots.append(
                    str(Path(temp_dir) / request["job_id"] / request["session_id"] / f"task-{index}")
                )
                self.assertNotIn("result_dir", result["metrics"])
            self.assertNotEqual(roots[0], roots[1])

    def test_unallowed_environment_and_invalid_request_are_controlled_failures(self):
        request = _request()
        request["env_params"]["dataset"]["environment"] = "docker"
        for payload in (request, {}):
            stdout = io.StringIO()
            registered = (
                frozenset({"swebench_verified"}),
                frozenset({"mini_swe_agent"}),
                frozenset({"host_process", "docker"}),
            )
            with mock.patch.object(runner, "_registered_components", return_value=registered), mock.patch.object(
                sys, "stdin", io.StringIO(json.dumps(payload))
            ), mock.patch.object(
                sys, "stdout", stdout
            ):
                self.assertEqual(runner.main(), 0)
            result = json.loads(stdout.getvalue())
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["total_reward"], 0.0)

    def test_cli_command_uses_non_secret_gateway_api_key_placeholder(self):
        command = runner._build_command(
            agentcompass_executable=Path("/opt/venv/bin/agentcompass"),
            benchmark="special_pattern_check;exit 91",
            harness="openai_chat && false",
            environment="host_process",
            model="public-route",
            session_url="http://gateway.invalid/v1/sessions/session-1",
            benchmark_params={"sample_ids": ["sample-$(false)"]},
            harness_params={},
            environment_params={},
            model_params={},
            result_root=Path("/tmp/result"),
        )
        self.assertIsInstance(command, list)
        self.assertEqual(command[2], "special_pattern_check;exit 91")
        self.assertEqual(command[3], "openai_chat && false")
        params = json.loads(command[command.index("--benchmark-params") + 1])
        self.assertEqual(params["sample_ids"], ["sample-$(false)"])
        self.assertEqual(command.count("--model-api-key"), 1)
        self.assertEqual(command[command.index("--model-api-key") + 1], "EMPTY")
        self.assertEqual(command.count("--model-base-url"), 1)
        self.assertEqual(
            command[command.index("--model-base-url") + 1],
            "http://gateway.invalid/v1/sessions/session-1",
        )
        with mock.patch.dict(
            os.environ,
            {
                "AGENTCOMPASS_MODEL_API_KEY": "do-not-copy-agentcompass-key",
                "OPENAI_API_KEY": "do-not-copy-openai-key",
            },
            clear=False,
        ):
            rebuilt = runner._build_command(
                agentcompass_executable=Path("/opt/venv/bin/agentcompass"),
                benchmark="special_pattern_check;exit 91",
                harness="openai_chat && false",
                environment="host_process",
                model="public-route",
                session_url="http://gateway.invalid/v1/sessions/session-1",
                benchmark_params={"sample_ids": ["sample-$(false)"]},
                harness_params={},
                environment_params={},
                model_params={},
                result_root=Path("/tmp/result"),
            )
        self.assertEqual(rebuilt, command)
        self.assertNotIn("do-not-copy-agentcompass-key", rebuilt)
        self.assertNotIn("do-not-copy-openai-key", rebuilt)
        self.assertNotIn("OPENAI_API_KEY", (ENV_ROOT / "runner.py").read_text(encoding="utf-8"))

    def test_cli_path_is_derived_from_runner_python_without_path_lookup(self):
        with mock.patch.object(runner.sys, "executable", "/opt/venv/bin/python"), mock.patch.object(
            runner.Path, "is_file", return_value=True
        ), mock.patch.object(runner.os, "access", return_value=True) as access:
            executable = runner._agentcompass_executable()
        self.assertEqual(executable, Path("/opt/venv/bin/agentcompass"))
        access.assert_called_once_with(Path("/opt/venv/bin/agentcompass"), os.X_OK)

    def test_cli_path_with_spaces_remains_one_argv_element(self):
        command = runner._build_command(
            agentcompass_executable=Path("/opt/venv with spaces/bin/agentcompass"),
            benchmark="special_pattern_check",
            harness="openai_chat",
            environment="host_process",
            model="public-route",
            session_url="http://gateway.invalid/v1/sessions/session-1",
            benchmark_params={"sample_ids": ["sample-1"]},
            harness_params={},
            environment_params={},
            model_params={},
            result_root=Path("/tmp/result"),
        )
        self.assertEqual(command[0], "/opt/venv with spaces/bin/agentcompass")
        self.assertEqual(command[1], "run")

    def test_missing_cli_is_a_controlled_failure(self):
        request = _request(benchmark="special_pattern_check", harness="openai_chat", contract_only=False)
        registered = (
            frozenset({"special_pattern_check"}),
            frozenset({"openai_chat"}),
            frozenset({"host_process"}),
        )
        stdout = io.StringIO()
        with tempfile.TemporaryDirectory() as temp_dir:
            request["env_params"]["results_root"] = temp_dir
            missing_python = str(Path(temp_dir) / "venv" / "bin" / "python")
            with mock.patch.object(runner, "_registered_components", return_value=registered), mock.patch.object(
                runner.sys, "executable", missing_python
            ), mock.patch.object(sys, "stdin", io.StringIO(json.dumps(request))), mock.patch.object(
                sys, "stdout", stdout
            ):
                self.assertEqual(runner.main(), 0)

        result = json.loads(stdout.getvalue())
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["metrics"]["error_type"], "dependency_missing")
        self.assertIn("missing or not executable", result["error_text"])
        self.assertIn("agentcompass", result["error_text"])

    def test_subprocess_failure_categories_are_explicit(self):
        self.assertEqual(runner._subprocess_error_type("No module named optional_x"), "dependency_missing")
        self.assertEqual(runner._subprocess_error_type("dataset directory does not exist"), "asset_missing")
        self.assertEqual(runner._subprocess_error_type("invalid combination"), "agentcompass_failed")

    def test_real_path_controlled_failure_has_one_stdout_result_and_stderr_diagnostics(self):
        request = _request(benchmark="special_pattern_check", harness="openai_chat", contract_only=False)
        registered = (
            frozenset({"special_pattern_check"}),
            frozenset({"openai_chat"}),
            frozenset({"host_process"}),
        )
        stdout = io.StringIO()
        stderr = io.StringIO()
        with tempfile.TemporaryDirectory() as temp_dir:
            request["env_params"]["results_root"] = temp_dir

            def fail_command(command, *, timeout_s, stdout_file, stderr_file):
                del command, timeout_s, stdout_file
                stderr_file.write("dataset directory does not exist\n")
                return 2, False

            with mock.patch.object(runner, "_registered_components", return_value=registered), mock.patch.object(
                runner, "_run_command", side_effect=fail_command
            ), mock.patch.object(
                runner, "_agentcompass_executable", return_value=Path("/opt/venv/bin/agentcompass")
            ), mock.patch.object(sys, "stdin", io.StringIO(json.dumps(request))), mock.patch.object(
                sys, "stdout", stdout
            ), mock.patch.object(sys, "stderr", stderr):
                self.assertEqual(runner.main(), 0)

        lines = [line for line in stdout.getvalue().splitlines() if line.strip()]
        self.assertEqual(len(lines), 1)
        result = json.loads(lines[0])
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["metrics"]["error_type"], "asset_missing")
        self.assertIn("SAFACTORY_RUNNER_DIAGNOSTIC", stderr.getvalue())

    def test_timeout_terminates_process_group(self):
        process = mock.Mock()
        process.pid = 1234
        process.poll.return_value = None
        process.wait.side_effect = [subprocess.TimeoutExpired("agentcompass", 1), 0]
        with mock.patch.object(runner.os, "killpg") as killpg:
            runner._terminate_process_group(process, grace_s=0.01)
        self.assertEqual(
            killpg.call_args_list,
            [mock.call(1234, runner.signal.SIGTERM), mock.call(1234, runner.signal.SIGKILL)],
        )

    def test_timeout_rejects_non_finite_values_and_preserves_subsecond_values(self):
        for value in ("nan", "inf", "-inf", float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value), self.assertRaisesRegex(RuntimeError, "must be positive"):
                runner._timeout_s({}, {"timeout_seconds": value})
        self.assertEqual(runner._timeout_s({}, {"timeout_seconds": 0.25}), 0.25)

    def test_process_timeout_is_terminal_truncated_and_has_no_internal_paths(self):
        result = runner._failure(
            "session-1", "AgentCompass timed out", started_at=runner.time.perf_counter(),
            truncated=True, metrics={"timeout_layer": "agentcompass_process_group"},
        )
        self.assertEqual(result["status"], "truncated")
        self.assertIs(result["terminated"], True)
        self.assertIs(result["truncated"], True)
        self.assertEqual(result["total_reward"], 0.0)
        serialized = json.dumps(result)
        for forbidden in ("detail_path", "result_dir", "stdout_path", "stderr_path", "/app/results"):
            self.assertNotIn(forbidden, serialized)

    def test_public_detail_normalization_is_explicit(self):
        for benchmark in ("swebench_verified", "special_pattern_check"):
            extra = (
                {"extra": {"status": "completed", "eval_raw_data": {"completed": True, "resolved": True}}}
                if benchmark == "swebench_verified" else {}
            )
            normalized = runner._normalize_detail(
                benchmark,
                {"attempts": {"1": {
                    "correct": True, "status": "completed", "score": None,
                    "final_answer": "synthetic patch", "ground_truth": "synthetic gold patch", **extra,
                }}},
            )
            self.assertEqual(normalized["normalized_reward_10"], 10.0)
        with self.assertRaisesRegex(RuntimeError, "unsupported AgentCompass result schema"):
            runner._normalize_detail("unknown_benchmark", {"score": 0.75})
        normalized = runner._normalize_detail(
            "special_pattern_check",
            {"attempts": {"1": {"correct": True, "status": "completed", "score": 0.25}}},
        )
        self.assertEqual(normalized["normalized_reward_10"], 10.0)

    def test_scicode_fractional_detail_normalization(self):
        detail = {
            "attempts": {
                "1": {
                    "status": "completed",
                    "correct": False,
                    "score": 0.5,
                    "error": "",
                    "final_answer": {"step_codes": {"sample-1.1": "return 1"}},
                    "ground_truth": {"problem_id": "sample-1", "total_steps": 2},
                    "meta": {
                        "evaluation": {
                            "problem_correct": 0,
                            "total_correct": 1,
                            "total_steps": 2,
                            "subproblem_correctness": 0.5,
                            "steps": [{"correct": True}, {"correct": False}],
                            "error": "",
                        }
                    },
                }
            }
        }
        normalized = runner._normalize_detail("scicode", detail)
        self.assertEqual(normalized["raw_score"], 0.5)
        self.assertEqual(normalized["normalized_reward_10"], 5.0)
        self.assertFalse(normalized["correct"])

        detail["attempts"]["1"]["meta"]["evaluation"]["subproblem_correctness"] = 0.75
        with self.assertRaisesRegex(RuntimeError, "inconsistent"):
            runner._normalize_detail("scicode", detail)

        # Pinned source: scicode.py::_evaluate_answer explicitly emits 0.0
        # when total_steps is zero; that remains a valid completed reward 0.
        zero_steps = json.loads(json.dumps(detail))
        attempt = zero_steps["attempts"]["1"]
        attempt["score"] = 0.0
        attempt["meta"]["evaluation"].update(
            problem_correct=0, total_correct=0, total_steps=0,
            subproblem_correctness=0.0, steps=[], error="",
        )
        normalized = runner._normalize_detail("scicode", zero_steps)
        self.assertEqual(normalized["normalized_reward_10"], 0.0)
        self.assertFalse(normalized["correct"])

    def test_frontierscience_split_specific_detail_normalization(self):
        olympiad = {
            "attempts": {
                "1": {
                    "status": "completed",
                    "correct": True,
                    "final_answer": "candidate",
                    "ground_truth": "reference",
                    "error": "",
                    "extra": {
                        "scoring": {
                            "evaluation_type": "frontierscience_olympiad_judge",
                            "correct": True,
                            "reason": "equivalent",
                        }
                    },
                }
            }
        }
        normalized = runner._normalize_detail("frontierscience", olympiad)
        self.assertEqual(normalized["normalization_strategy"], "frontierscience_olympiad")
        self.assertEqual(normalized["normalized_reward_10"], 10.0)

        research = {
            "attempts": {
                "1": {
                    "status": "completed",
                    "correct": True,
                    "final_answer": "candidate",
                    "ground_truth": "rubric",
                    "error": "",
                    "extra": {
                        "scoring": {
                            "evaluation_type": "frontierscience_research_rubric",
                            "correct": True,
                            "total_score": 7.5,
                            "passing_threshold": 7.0,
                            "rubric_items": [
                                {"item": "method", "max_points": 5.0, "awarded_points": 4.0, "reason": "ok"},
                                {"item": "result", "max_points": 5.0, "awarded_points": 3.5, "reason": "ok"},
                            ],
                            "summary": "passes",
                        }
                    },
                }
            }
        }
        normalized = runner._normalize_detail("frontierscience", research)
        self.assertEqual(normalized["raw_score"], 7.5)
        self.assertEqual(normalized["normalized_reward_10"], 7.5)

        research["attempts"]["1"]["extra"]["scoring"]["correct"] = False
        with self.assertRaisesRegex(RuntimeError, "inconsistent"):
            runner._normalize_detail("frontierscience", research)

        zero = json.loads(json.dumps(research))
        scoring = zero["attempts"]["1"]["extra"]["scoring"]
        zero["correct"] = zero["attempts"]["1"]["correct"] = False
        scoring.update(correct=False, total_score=0.0, rubric_items=[], summary="empty_model_response")
        normalized = runner._normalize_detail("frontierscience", zero)
        self.assertEqual(normalized["normalized_reward_10"], 0.0)

        for mutation in (
            {"total_score": float("nan")},
            {"total_score": "0"},
            {"rubric_items": None},
            {"error": "synthetic judge failure"},
        ):
            invalid = json.loads(json.dumps(zero))
            invalid["attempts"]["1"]["extra"]["scoring"].update(mutation)
            with self.subTest(frontierscience_research=mutation), self.assertRaises(RuntimeError):
                runner._normalize_detail("frontierscience", invalid)

    def test_sgi_and_sealqa_judge_detail_normalization(self):
        sgi = {
            "attempts": {
                "1": {
                    "status": "completed",
                    "correct": True,
                    "error": "",
                    "final_answer": "answer",
                    "ground_truth": "reference",
                    "extra": {
                        "scoring": {
                            "evaluation_type": "llm_judge",
                            "correct": True,
                            "model_answer": "answer",
                            "ground_truth": "reference",
                        }
                    },
                }
            }
        }
        self.assertEqual(runner._normalize_detail("sgi_deep_research", sgi)["normalized_reward_10"], 10.0)

        sealqa = {
            "attempts": {
                "1": {
                    "status": "completed",
                    "correct": False,
                    "error": "",
                    "final_answer": "answer",
                    "ground_truth": "reference",
                    "extra": {
                        "scoring": {
                            "evaluation_type": "sealqa_official_llm_judge",
                            "correct": False,
                            "grade": "C",
                            "label": "not_attempted",
                            "raw_response": "C",
                            "judge_model": "judge",
                            "api_protocol": "openai-chat",
                        }
                    },
                }
            }
        }
        self.assertEqual(runner._normalize_detail("sealqa", sealqa)["normalized_reward_10"], 0.0)
        sealqa["attempts"]["1"]["extra"]["scoring"]["label"] = "incorrect"
        with self.assertRaisesRegex(RuntimeError, "inconsistent"):
            runner._normalize_detail("sealqa", sealqa)

    def test_sgi_uses_pinned_llm_scorer_boolean_without_inventing_verdict_fields(self):
        # Pinned source: benchmarks/scorers/llm.py::LLMJudgeScorer.score and
        # benchmarks/sgi_deep_research.py::SGIDeepResearchBenchmark.evaluate.
        scoring = {
            "evaluation_type": "llm_judge",
            "correct": True,
            "model_answer": "synthetic answer",
            "ground_truth": "synthetic reference",
            "reason": "equivalent",
        }

        def detail_with(fields):
            return {
                "sample_id": "sample-1",
                "attempts": {"1": {
                    "status": "completed",
                    "correct": True,
                    "error": "",
                    "final_answer": fields.get("model_answer"),
                    "ground_truth": fields.get("ground_truth"),
                    "extra": {"scoring": fields},
                }},
            }

        normalized = runner._normalize_detail("sgi_deep_research", detail_with(scoring))
        self.assertEqual(normalized["raw_score"], 1.0)
        self.assertNotIn("judge_verdict", normalized)
        self.assertNotIn("judge_reason", normalized)
        minimal_scoring = {"evaluation_type": "llm_judge", "correct": True}
        self.assertEqual(
            runner._normalize_detail("sgi_deep_research", detail_with(minimal_scoring))["raw_score"],
            1.0,
        )

    def test_result_contract_has_identity_terminal_semantics_and_no_internal_paths(self):
        detail = {
            "task_id": "sample-1", "correct": False, "score": None,
            "attempts": {"1": {"status": "completed", "correct": False, "score": None, "error": ""}},
        }
        result = runner._result_from_detail(
            session_id="session-1",
            task_id="task-1",
            benchmark="special_pattern_check",
            harness="openai_chat",
            environment="host_process",
            sample_id="sample-1",
            detail=detail,
            duration_ms=1.0,
        )
        self.assertEqual(result["total_reward"], 0.0)
        self.assertIs(result["terminated"], True)
        self.assertIs(result["truncated"], False)
        serialized = json.dumps(result)
        for forbidden in ("detail_path", "result_dir", "stdout_path", "stderr_path", "/app/results"):
            self.assertNotIn(forbidden, serialized)

        with self.assertRaisesRegex(RuntimeError, "task_id did not match"):
            runner._result_from_detail(
                session_id="session-1", task_id="task-1", benchmark="special_pattern_check",
                harness="openai_chat", environment="host_process", sample_id="other-sample",
                detail=detail, duration_ms=1.0,
            )

    def test_science_detail_errors_fail_closed(self):
        cases = {
            "scicode": {
                "status": "eval_error",
                "correct": False,
                "score": 0.0,
                "meta": {"evaluation": {}},
            },
            "frontierscience": {
                "status": "completed",
                "correct": False,
                "error": "judge failed",
                "extra": {"scoring": {}},
            },
            "sgi_deep_research": {
                "status": "run_error",
                "correct": False,
                "extra": {"scoring": {}},
            },
            "sealqa": {
                "status": "eval_error",
                "correct": False,
                "extra": {"scoring": {"error": "judge_failed"}},
            },
        }
        for benchmark, attempt in cases.items():
            with self.subTest(benchmark=benchmark), self.assertRaises(RuntimeError):
                runner._normalize_detail(benchmark, {"attempts": {"1": attempt}})

    def test_docker_and_rjob_yaml_expand_the_same_diagnostic_episode(self):
        try:
            from core.data_manager.load_yaml import load_yaml_configs
            from manager.simulation_config import load_agent_start_config
        except ModuleNotFoundError as exc:
            self.skipTest(f"SAfactory parser dependency is not installed: {exc.name}")

        docker_rows = load_yaml_configs(str(ENV_ROOT / "agentcompass_config.yaml"))
        rjob_rows = load_yaml_configs(str(ENV_ROOT / "agentcompass_config.rjob.yaml"))
        self.assertEqual(len(docker_rows), 1)
        self.assertEqual(len(rjob_rows), 1)
        self.assertEqual(docker_rows[0]["env_params"]["dataset"], rjob_rows[0]["env_params"]["dataset"])
        self.assertEqual(rjob_rows[0]["env_params"]["dataset"]["sample_id"], "empty_content_gate4_0")
        self.assertIn("agentcompass", load_agent_start_config(str(ENV_ROOT / "agentcompass_start.yaml")))
        self.assertIn("agentcompass", load_agent_start_config(str(ENV_ROOT / "agentcompass_start.rjob.yaml")))

    def test_local_docker_has_one_loopback_gateway_alias_and_rjob_does_not(self):
        try:
            from clusters.docker_clusters import DockerContainerBackend
            from manager.simulation_config import load_agent_start_config
        except ModuleNotFoundError as exc:
            self.skipTest(f"SAfactory Docker parser dependency is not installed: {exc.name}")

        docker_start = load_agent_start_config(str(ENV_ROOT / "agentcompass_start.yaml"))
        docker_cfg = docker_start["agentcompass"]["docker"]
        expected_arg = "--add-host=host.docker.internal:127.0.0.1"
        self.assertEqual(docker_cfg["network"], "host")
        self.assertEqual(docker_cfg["extra_args"], [expected_arg])

        backend = DockerContainerBackend(cluster_cfg={"env_types": docker_start})
        resolved = backend._docker_cfg_for_env("agentcompass")
        command = backend._build_run_command(
            image="safactory-agentcompass:h-dev",
            name="offline-dry-run",
            env_name="agentcompass",
            docker_cfg=resolved,
            idle_command=resolved["idle_command"],
            workdir=resolved["workdir"],
        )
        self.assertEqual(command.count(expected_arg), 1)
        self.assertEqual(command[command.index("--network") + 1], "host")

        rjob_raw = yaml.safe_load((ENV_ROOT / "agentcompass_start.rjob.yaml").read_text(encoding="utf-8"))
        self.assertNotIn("extra_args", rjob_raw["container"])
        self.assertNotIn("host.docker.internal", json.dumps(rjob_raw))

    def test_rjob_embedding_command_dry_run_stops_before_submit(self):
        try:
            from clusters.rjob_cluster import RJobClusterBackend
            from manager.simulation_config import load_agent_start_config
        except ModuleNotFoundError as exc:
            self.skipTest(f"SAfactory RJob parser dependency is not installed: {exc.name}")

        start = load_agent_start_config(str(ENV_ROOT / "agentcompass_start.rjob.yaml"))
        backend = RJobClusterBackend(cluster_cfg={"rjob": {"no_packaging": True}, "env_types": start})
        entry = asyncio.run(
            backend.allocate(
                row_id=1,
                env_name="agentcompass",
                env_id="dry-run-env",
                image=APPROVED_RJOB_IMAGE,
                env_params={},
                group_id="dry-run-group",
            )
        )
        runner_target = "/tmp/safactory-agentcompass/runner.py"
        gate4_target = "/app/data/special_pattern_check/gate4.jsonl"
        embedded = {item["target"]: Path(item["source"]) for item in entry.runtime_config["embedded_files"]}
        self.assertEqual(set(embedded), {runner_target, gate4_target})
        self.assertTrue(embedded[runner_target].samefile(ENV_ROOT / "runner.py"))
        self.assertTrue(embedded[gate4_target].samefile(ENV_ROOT / "datasets/special_pattern_check/gate4.jsonl"))
        row = json.loads((ENV_ROOT / "datasets/agentcompass_diagnostic.jsonl").read_text(encoding="utf-8"))
        self.assertEqual(row["benchmark_params"]["dataset_dir"], str(Path(gate4_target).parent))

        docker_cfg = backend._docker_cfg_for_env("agentcompass")
        rjob_cfg = backend._rjob_cfg_for_env("agentcompass")
        backend._validate_mounts("agentcompass", docker_cfg, rjob_cfg)
        self.assertTrue(all(Path(item["source"]).is_file() for item in docker_cfg["volumes"]))
        self.assertNotIn("mount", rjob_cfg)
        self.assertNotIn("mount_config", rjob_cfg)
        self.assertIs(rjob_cfg["privileged"], False)

        dry_run_cfg = {**entry.runtime_config, "dry_run": True}
        submit_kwargs = backend.submit_kwargs(dry_run_cfg)
        self.assertIs(submit_kwargs["dry_run"], True)
        self.assertIs(submit_kwargs["no_packaging"], True)
        command = backend._command_with_embedded_files(dry_run_cfg, entry.run_command)
        self.assertIn(repr(runner_target), command)
        self.assertIn(repr(gate4_target), command)
        self.assertIn("exec /opt/venv/bin/python /tmp/safactory-agentcompass/runner.py", command)

    def test_rjob_result_handoff_contract_is_shared_and_read_before_cleanup(self):
        # The launcher and child independently derive the same artifact path
        # below the shared /app/results mount for every benchmark.
        from manager.episode_common import result_artifact_path
        from manager.types import SimulationStartRequest

        for benchmark in runner.RESULT_NORMALIZERS:
            request = SimulationStartRequest(
                job_id="job-1", session_id="session-1", agent_name="agentcompass",
                agent_id="agent-1", group_id="group-1", gateway_base_url="http://gateway.invalid",
                model="model-1", temperature=0.0, max_steps=1, storage_type="sqlite",
                env_params={"results_root": "/app/results", "dataset": {
                    "task_id": "task-1", "benchmark": benchmark,
                }},
            )
            self.assertEqual(
                result_artifact_path(request),
                "/app/results/job-1/session-1/safactory_result.json",
            )

        with tempfile.TemporaryDirectory() as temp_dir:
            artifact = Path(temp_dir) / "job-1" / "session-1" / "safactory_result.json"
            payload = {"session_id": "session-1", "status": "succeeded"}
            with mock.patch.dict(os.environ, {runner.RESULT_PATH_ENV: str(artifact)}), mock.patch(
                "sys.stdout", new_callable=io.StringIO
            ):
                runner._write_result(payload)
            self.assertEqual(json.loads(artifact.read_text(encoding="utf-8")), payload)

        episode_runner_source = (
            REPO_ROOT / "manager" / "rjob_episode_runner.py"
        ).read_text(encoding="utf-8")
        self.assertLess(
            episode_runner_source.index("with trace.span(\"parse_result\""),
            episode_runner_source.index("with trace.span(\n                    \"cleanup_job\""),
        )

    def test_rjob_global_mount_is_not_overridden_by_per_agent_config(self):
        rjob_start = yaml.safe_load((ENV_ROOT / "agentcompass_start.rjob.yaml").read_text(encoding="utf-8"))
        self.assertNotIn("mount_config", rjob_start["rjob"])

        source = (REPO_ROOT / "clusters" / "rjob_cluster.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        merge_node = next(
            node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "merge_dicts"
        )
        namespace = {"Any": object, "Dict": dict}
        exec(compile(ast.Module(body=[merge_node], type_ignores=[]), "rjob_cluster.py", "exec"), namespace)
        fake_mount = "gpfs://example.invalid/shared-results:/app/results"
        merged = namespace["merge_dicts"](
            {"mount_config": [fake_mount]}, rjob_start["rjob"]
        )
        self.assertEqual(merged["mount_config"], [fake_mount])

    def test_rjob_mount_preflight_fails_before_benchmark_start_and_passes_when_mounted(self):
        artifact = "/app/results/job-1/session-1/safactory_result.json"
        rjob_env = {
            runner.REQUIRE_RESULTS_MOUNT_ENV: "1",
            runner.RESULT_PATH_ENV: artifact,
        }
        with mock.patch.dict(os.environ, rjob_env, clear=False), mock.patch(
            "os.path.ismount", return_value=False
        ), mock.patch.object(runner, "_build_command") as build_command, mock.patch.object(
            runner, "_run_command"
        ) as run_command, mock.patch("sys.stderr", new_callable=io.StringIO) as stderr:
            self.assertEqual(runner.main(), 2)
            self.assertIn("result mount preflight failed", stderr.getvalue())
            build_command.assert_not_called()
            run_command.assert_not_called()

        writable_probe = mock.MagicMock()
        writable_probe.__enter__.return_value = mock.MagicMock()
        with mock.patch.dict(os.environ, rjob_env, clear=False), mock.patch(
            "os.path.ismount", return_value=True
        ), mock.patch("tempfile.NamedTemporaryFile", return_value=writable_probe) as named_temporary_file:
            runner._rjob_results_preflight()
            named_temporary_file.assert_called_once_with(
                mode="w", encoding="utf-8", prefix=".safactory-write-probe-", dir=Path("/app/results")
            )

        with mock.patch.dict(os.environ, rjob_env, clear=False), mock.patch(
            "os.path.ismount", return_value=True
        ), mock.patch("tempfile.NamedTemporaryFile", side_effect=PermissionError("read-only file system")), mock.patch.object(
            runner, "_build_command"
        ) as build_command, mock.patch.object(runner, "_run_command") as run_command, mock.patch(
            "sys.stderr", new_callable=io.StringIO
        ) as stderr:
            self.assertEqual(runner.main(), 2)
            self.assertIn("results_mount_not_writable", stderr.getvalue())
            build_command.assert_not_called()
            run_command.assert_not_called()

        for invalid_path in ("", "results/result.json", "/tmp/safactory_result.json"):
            with self.subTest(result_path=invalid_path), mock.patch.dict(
                os.environ, {runner.RESULT_PATH_ENV: invalid_path}, clear=False
            ), mock.patch("os.path.ismount", return_value=True), self.assertRaises(RuntimeError):
                runner._rjob_results_preflight()

    def test_public_rjob_files_have_only_portable_structure(self):
        public_paths = (
            ENV_ROOT / "agentcompass_config.yaml",
            ENV_ROOT / "agentcompass_start.yaml",
            ENV_ROOT / "agentcompass_config.rjob.yaml",
            ENV_ROOT / "agentcompass_start.rjob.yaml",
            ENV_ROOT / "README.md",
        )
        rjob_config = yaml.safe_load((ENV_ROOT / "agentcompass_config.rjob.yaml").read_text(encoding="utf-8"))
        rjob_start = yaml.safe_load((ENV_ROOT / "agentcompass_start.rjob.yaml").read_text(encoding="utf-8"))
        docker_start = yaml.safe_load((ENV_ROOT / "agentcompass_start.yaml").read_text(encoding="utf-8"))
        self.assertEqual(set(rjob_start["container"]["env"]), {
            "PYTHONDONTWRITEBYTECODE", "PYTHONUNBUFFERED", "AGENTCOMPASS_ALLOWED_ENVIRONMENTS",
            "AGENTCOMPASS_REQUIRE_RESULTS_MOUNT",
        })
        self.assertIs(rjob_start["rjob"]["privileged"], False)
        self.assertNotIn("mount", rjob_start["rjob"])
        self.assertNotIn("mount_config", rjob_start["rjob"])
        self.assertNotIn("AGENTCOMPASS_REQUIRE_RESULTS_MOUNT", docker_start["container"]["env"])
        self.assertNotIn("cleanup_on_failure", rjob_start["rjob"])
        self.assertEqual(rjob_config["environments"][0]["env_image"], APPROVED_RJOB_IMAGE)

        forbidden_fields = {
            "cluster_entry", "namespace", "access_key", "secret_key", "charged_group", "gateway_base_url"
        }
        allowed_prefixes = ("/opt/", "/tmp/", "/app/")

        def walk(value):
            if isinstance(value, dict):
                for key, child in value.items():
                    self.assertNotIn(str(key), forbidden_fields)
                    yield from walk(child)
            elif isinstance(value, list):
                for child in value:
                    yield from walk(child)
            elif isinstance(value, str):
                yield value

        for payload in (rjob_config, rjob_start):
            for value in walk(payload):
                if value == APPROVED_RJOB_IMAGE:
                    continue
                self.assertNotIn("://", value)
                if not value.startswith(("/", "./", "../")):
                    self.assertNotRegex(value, r"^[a-z0-9-]+(?:\.[a-z0-9-]+)+(?:/|$)")
                if value.startswith("/"):
                    self.assertTrue(value.startswith(allowed_prefixes))
        combined = "\n".join(path.read_text(encoding="utf-8") for path in public_paths)
        docker_start_text = (ENV_ROOT / "agentcompass_start.yaml").read_text(encoding="utf-8")
        self.assertEqual(
            [line.strip() for line in docker_start_text.splitlines() if "--add-host" in line],
            ["- --add-host=host.docker.internal:127.0.0.1"],
        )
        self.assertNotIn("host.docker.internal", json.dumps(rjob_start))
        without_approved_values = (
            combined
            .replace(APPROVED_RJOB_IMAGE, "")
            .replace("host.docker.internal", "")
        )
        self.assertNotIn("/var/run/docker.sock", combined)
        self.assertNotRegex(without_approved_values, r"(?i)https?://")
        self.assertNotRegex(without_approved_values, r"(?i)\b(?:http|https|all)_proxy\b")
        self.assertNotRegex(
            without_approved_values,
            r"(?i)\b(?:access_key|secret_key|password|token)\s*[:=]\s*\S+",
        )
        self.assertNotRegex(without_approved_values, r"/(?:home|mnt/shared-storage-user)/[^\s`]+")
        domain_candidates = re.findall(
            r"(?i)\b[a-z0-9-]+(?:\.[a-z0-9-]+)*\.[a-z]{2,}\b",
            without_approved_values,
        )
        file_suffixes = {"py", "yaml", "yml", "json", "jsonl", "toml", "md"}
        unexpected_domains = [
            value for value in domain_candidates
            if value.rsplit(".", 1)[-1].lower() not in file_suffixes
        ]
        self.assertEqual(unexpected_domains, [])
        readme = (ENV_ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("Linux/AMD64", readme)
        self.assertIn("d2c3e148902e948db3270fa34b2198fb1b10beb7", readme)


class EvaluatorTests(unittest.TestCase):
    def test_registered_binary_benchmarks_map_to_ten_or_zero(self):
        for benchmark in ("swebench_verified", "special_pattern_check"):
            for correct, expected in ((True, 10.0), (False, 0.0)):
                request = types.SimpleNamespace(
                    session_id="session-1",
                    start_result=types.SimpleNamespace(
                        metrics={
                            "benchmark": benchmark,
                            "correct": correct,
                            "task_id": "task-1",
                            "harness": "openai_chat",
                            "environment": "host_process",
                            "sample_id": "sample-1",
                            "schema_validated": True,
                            "normalization_strategy": (
                                "swebench_resolved" if benchmark == "swebench_verified"
                                else "special_pattern_correct"
                            ),
                            "agentcompass_status": "completed",
                            "agentcompass_error": "",
                            "raw_score": 1.0 if correct else 0.0,
                            "normalized_reward_10": expected,
                        }
                    ),
                )
                result = rule_evaluator.evaluate(
                    request,
                    types.SimpleNamespace(eval_id="agentcompass_rule"),
                    types.SimpleNamespace(),
                )
                self.assertEqual(result["score"], expected)
                for semantic in ("correct", "ground_truth", "model_answer", "judge_verdict", "judge_reason"):
                    self.assertNotIn(semantic, result["artifacts"])

    def test_unknown_score_schema_fails_closed(self):
        request = types.SimpleNamespace(
            session_id="session-1",
            start_result=types.SimpleNamespace(metrics={"benchmark": "unknown", "score": 0.75}),
        )
        result = rule_evaluator.evaluate(request, types.SimpleNamespace(eval_id="rule"), None)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["score"], 0.0)

    def _evaluate_metrics(self, metrics):
        metrics = {
            "task_id": "task-1",
            "harness": "fixture_harness",
            "environment": "host_process",
            "sample_id": "sample-1",
            **metrics,
        }
        request = types.SimpleNamespace(
            session_id="session-1",
            start_result=types.SimpleNamespace(metrics=metrics),
        )
        return rule_evaluator.evaluate(request, types.SimpleNamespace(eval_id="rule"), None)

    def test_science_normalized_metrics_map_to_expected_rewards(self):
        common = {
            "schema_validated": True,
            "agentcompass_status": "completed",
            "agentcompass_error": "",
        }
        cases = (
            (
                {
                    **common,
                    "benchmark": "scicode",
                    "correct": False,
                    "normalization_strategy": "scicode_fractional",
                    "raw_score": 0.5,
                    "normalized_reward_10": 5.0,
                },
                5.0,
            ),
            (
                {
                    **common,
                    "benchmark": "frontierscience",
                    "correct": True,
                    "normalization_strategy": "frontierscience_olympiad",
                    "raw_score": 1.0,
                    "normalized_reward_10": 10.0,
                },
                10.0,
            ),
            (
                {
                    **common,
                    "benchmark": "frontierscience",
                    "correct": True,
                    "normalization_strategy": "frontierscience_research",
                    "raw_score": 7.5,
                    "normalized_reward_10": 7.5,
                    "passing_threshold": 7.0,
                },
                7.5,
            ),
            (
                {
                    **common,
                    "benchmark": "sgi_deep_research",
                    "correct": True,
                    "normalization_strategy": "llm_judge",
                    "raw_score": 1.0,
                    "normalized_reward_10": 10.0,
                },
                10.0,
            ),
            (
                {
                    **common,
                    "benchmark": "sealqa",
                    "correct": False,
                    "normalization_strategy": "sealqa_judge",
                    "raw_score": 0.0,
                    "normalized_reward_10": 0.0,
                },
                0.0,
            ),
        )
        for metrics, expected in cases:
            with self.subTest(benchmark=metrics["benchmark"], strategy=metrics["normalization_strategy"]):
                result = self._evaluate_metrics(metrics)
                self.assertEqual(result["status"], "succeeded")
                self.assertEqual(result["score"], expected)
                for semantic in ("correct", "ground_truth", "model_answer", "judge_verdict", "judge_reason"):
                    self.assertNotIn(semantic, result["artifacts"])

    def test_science_normalized_metrics_fail_closed(self):
        base = {
            "benchmark": "scicode",
            "correct": False,
            "schema_validated": True,
            "normalization_strategy": "scicode_fractional",
            "agentcompass_status": "completed",
            "agentcompass_error": "",
            "raw_score": 0.5,
            "normalized_reward_10": 5.0,
        }
        mutations = (
            {"agentcompass_status": "eval_error"},
            {"agentcompass_error": "failure"},
            {"raw_score": "0.5"},
            {"normalized_reward_10": 4.0},
            {"correct": True},
            {"schema_validated": False},
            {"normalization_strategy": "sealqa_judge"},
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                result = self._evaluate_metrics({**base, **mutation})
                self.assertEqual(result["status"], "failed")
                self.assertEqual(result["score"], 0.0)

    def test_binary_evaluator_fails_closed_before_reward(self):
        valid = {
            "benchmark": "special_pattern_check",
            "correct": True,
            "schema_validated": True,
            "normalization_strategy": "special_pattern_correct",
            "agentcompass_status": "completed",
            "agentcompass_error": "",
            "raw_score": 1.0,
            "normalized_reward_10": 10.0,
        }
        cases = (
            ({}, "succeeded", 10.0),
            ({"correct": False, "raw_score": 0.0, "normalized_reward_10": 0.0}, "succeeded", 0.0),
            ({"agentcompass_status": "eval_error", "agentcompass_error": "judge failed"}, "failed", 0.0),
            ({"schema_validated": False}, "failed", 0.0),
            ({"correct": "true"}, "failed", 0.0),
        )
        for mutation, expected_status, expected_score in cases:
            with self.subTest(mutation=mutation):
                result = self._evaluate_metrics({**valid, **mutation})
                self.assertEqual(result["status"], expected_status)
                self.assertEqual(result["score"], expected_score)

    def test_nonterminal_failed_or_identity_invalid_start_results_fail_closed(self):
        metrics = {
            "benchmark": "special_pattern_check", "task_id": "task-1",
            "harness": "openai_chat", "environment": "host_process", "sample_id": "sample-1",
            "correct": True, "schema_validated": True, "normalization_strategy": "binary_correct",
            "agentcompass_status": "completed", "agentcompass_error": "",
        }
        for mutation in (
            {"terminated": False},
            {"status": "failed"},
            {"truncated": True},
            {"session_id": "other-session"},
        ):
            start = types.SimpleNamespace(
                session_id=mutation.get("session_id", "session-1"),
                status=mutation.get("status", "succeeded"),
                terminated=mutation.get("terminated", True),
                truncated=mutation.get("truncated", False),
                metrics=metrics,
            )
            request = types.SimpleNamespace(session_id="session-1", start_result=start)
            result = rule_evaluator.evaluate(request, types.SimpleNamespace(eval_id="rule"), None)
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["score"], 0.0)


if __name__ == "__main__":
    unittest.main()
