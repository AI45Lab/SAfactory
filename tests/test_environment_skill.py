"""Behavioral checks for the portable onboarding templates and owned test services."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[1]
SKILL = REPO_ROOT / "skills" / "safactory-workflows"
sys.path.insert(0, str(SKILL / "scripts"))
from contract_smoke import run_smoke
from live_smoke import run_live
from scaffold_environment import scaffold
import live_smoke


class EnvironmentSkillTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def create(self, mode="docker", evaluation=False):
        path = scaffold("mybench", mode, self.root, evaluation)
        return path, json.loads((path / "request.smoke.json").read_text())

    def test_scaffold_preserves_existing_files_and_validates_name(self):
        path, _ = self.create()
        for name in (
            "runner.py", "adapter.py", "mybench_config.yaml", "mybench_start.yaml",
            "mybench_config.rjob.yaml", "mybench_start.rjob.yaml", "request.smoke.json",
        ):
            self.assertTrue((path / name).exists(), name)
        (path / "adapter.py").write_text("user code")
        with self.assertRaises(FileExistsError):
            scaffold("mybench", "rjob", self.root, True)
        self.assertEqual((path / "adapter.py").read_text(), "user code")
        self.assertFalse((path / "rule_evaluator.py").exists())
        rjob_config = (path / "mybench_config.rjob.yaml").read_text()
        self.assertIn("RJob cluster", rjob_config)
        self.assertIn("output_root: /app/results/mybench", rjob_config)
        docker_start = (path / "mybench_start.yaml").read_text()
        self.assertIn(f"{path / 'adapter.py'}", docker_start)
        self.assertIn(f"{path / 'datasets'}", docker_start)
        for name in ("../escape", "bad-name", "bad\nname"):
            with self.assertRaises(ValueError):
                scaffold(name, "docker", self.root)

    def test_both_input_transports_work_without_evaluation_or_cluster(self):
        path, request = self.create(mode="rjob")
        self.assertFalse((path / "rule_evaluator.py").exists())
        # Even a present evaluator must never be imported by integration-only execution.
        (path / "rule_evaluator.py").write_text("raise RuntimeError('evaluation was invoked')")
        for mode in ("stdin", "env"):
            with self.subTest(mode=mode):
                outcome = run_smoke(path / "runner.py", request, input_mode=mode, require_model_call=True)
                self.assertEqual(outcome["result"]["metrics"]["answer"], "fixture answer")
                self.assertEqual(outcome["result"]["total_reward"], 0.0)
                self.assertEqual(outcome["model_calls"], 1)

    def test_contract_helper_can_inject_an_explicit_adapter_fixture(self):
        path, request = self.create()
        fixture = self.root / "adapter_fixture.py"
        fixture.write_text(
            "def run_case(request, task, session_url):\n"
            "    return {'fixture': True, 'task_id': task['task_id']}, 2\n"
        )
        outcome = run_smoke(
            path / "runner.py", request, adapter=fixture, require_model_call=False
        )
        self.assertEqual(outcome["result"]["metrics"]["fixture"], True)
        self.assertEqual(outcome["result"]["step_count"], 2)

    def test_two_native_case_fixtures_map_outputs_and_isolate_stdout(self):
        path, request = self.create()
        native = self.root / "native.py"
        native.write_text(
            "import json, pathlib, sys\n"
            "case, output = sys.argv[1:]\n"
            "print('native diagnostic')\n"
            "pathlib.Path(output).write_text(json.dumps({'case_id': case, 'answer': case.upper()}))\n"
        )
        (path / "adapter.py").write_text(
            "import json, pathlib, subprocess, sys\n"
            "def run_case(request, task, session_url):\n"
            "    print('adapter diagnostic')\n"
            "    output = pathlib.Path(request['env_params']['output_root']) / (task['case_id'] + '.json')\n"
            "    subprocess.run([sys.executable, request['env_params']['native'], task['case_id'], str(output)],\n"
            "                   check=True, stdout=sys.stderr)\n"
            "    return {'native': json.loads(output.read_text()), 'output_path': str(output)}, 1\n"
        )
        for case in ("case-a", "case-b"):
            request["session_id"] = case
            request["env_params"] = {"dataset": {"case_id": case}, "native": str(native),
                                     "output_root": str(self.root)}
            with contextlib.redirect_stderr(io.StringIO()) as diagnostics:
                outcome = run_smoke(path / "runner.py", request)
            self.assertIn("native diagnostic", diagnostics.getvalue())
            metrics = outcome["result"]["metrics"]
            self.assertEqual(metrics["native"]["case_id"], case)
            self.assertEqual(json.loads(Path(metrics["output_path"]).read_text())["answer"], case.upper())

    def test_controlled_native_failure_preserves_session(self):
        path, request = self.create()
        (path / "adapter.py").write_text("def run_case(*args):\n    raise ValueError('native fixture failed')\n")
        outcome = run_smoke(path / "runner.py", request, expected_status="failed")
        self.assertEqual(outcome["result"]["session_id"], request["session_id"])
        self.assertEqual(outcome["result"]["error_text"], "native fixture failed")
        with self.assertRaisesRegex(ValueError, "unexpected result status"):
            run_smoke(path / "runner.py", request)

    def test_malformed_request_is_one_failed_json_with_zero_exit(self):
        path, _ = self.create()
        for raw in ("[1]", "bad json"):
            completed = subprocess.run(
                [sys.executable, str(path / "runner.py")], input=raw,
                capture_output=True, text=True,
                env={**os.environ, "SAFACTORY_SESSION_ID": "fallback", "SAFACTORY_RESULT_PATH": ""},
            )
            self.assertEqual(completed.returncode, 0)
            result = json.loads(completed.stdout)
            self.assertEqual(result["session_id"], "fallback")
            self.assertEqual(result["status"], "failed")

    def test_extra_stdout_is_rejected(self):
        path, request = self.create()
        runner = path / "runner.py"
        runner.write_text("print('unstructured log')\n" + runner.read_text().replace(
            "from __future__ import annotations", ""
        ))
        with self.assertRaises(json.JSONDecodeError):
            run_smoke(runner, request)

    def test_contract_timeout_stops_runner(self):
        path, request = self.create()
        pid_path = self.root / "runner.pid"
        (path / "adapter.py").write_text(
            "import os, pathlib, time\n"
            "def run_case(*args):\n"
            f"    pathlib.Path({str(pid_path)!r}).write_text(str(os.getpid()))\n"
            "    time.sleep(60)\n"
        )
        with self.assertRaises(subprocess.TimeoutExpired):
            run_smoke(path / "runner.py", request, timeout=0.5)
        self.assert_stopped(pid_path)

    def test_mock_rejects_wrong_session_route(self):
        path, request = self.create()
        adapter = path / "adapter.py"
        adapter.write_text(adapter.read_text().replace(
            'f"{session_url.rstrip(\'/\')}/chat/completions"',
            'f"{request[\'gateway_base_url\']}/wrong-session/chat/completions"'
        ))
        with self.assertRaisesRegex(ValueError, "unexpected model path"):
            run_smoke(path / "runner.py", request, expected_status="failed")

    def test_optional_evaluator_hook_rejects_missing_and_invalid_scores(self):
        path, _ = self.create(evaluation=True)
        spec = importlib.util.spec_from_file_location("fixture_evaluator", path / "rule_evaluator.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        request = SimpleNamespace(session_id="one", env_params={"dataset": {}},
                                  start_result=SimpleNamespace(metrics={"score": 0.7}))
        kwargs = dict(request=request, spec=SimpleNamespace(eval_id="rule"), trajectory=None)
        self.assertEqual(module.evaluate_rule(**kwargs)["status"], "failed")
        module.score_metrics = lambda metrics, dataset, trajectory: (metrics["score"], metrics["score"] * 10, "fixture scale")
        for raw in (0.0, 0.7, 1.0):
            request.start_result.metrics = {"score": raw}
            result = module.evaluate_rule(**kwargs)
            self.assertEqual(result["status"], "succeeded")
            self.assertEqual(result["normalized_score_10"], raw * 10)
        for metrics in ({}, {"score": float("nan")}, {"score": float("inf")}, {"score": 2}):
            request.start_result.metrics = metrics
            self.assertEqual(module.evaluate_rule(**kwargs)["status"], "failed")

    def live_fixture(self):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        pid_path = self.root / "gateway.pid"
        code = (
            "from http.server import BaseHTTPRequestHandler, HTTPServer\n"
            "import os, pathlib\n"
            f"pathlib.Path({str(pid_path)!r}).write_text(str(os.getpid()))\n"
            "class Handler(BaseHTTPRequestHandler):\n"
            "    def do_GET(self):\n"
            "        self.send_response(200)\n"
            "        self.end_headers()\n"
            "        self.wfile.write(b'{\"status\": \"ready\"}')\n"
            f"HTTPServer(('127.0.0.1', {port}), Handler).serve_forever()\n"
        )
        return [sys.executable, "-c", code], f"http://127.0.0.1:{port}/readyz", pid_path

    def assert_stopped(self, pid_path):
        with self.assertRaises(ProcessLookupError):
            os.kill(int(pid_path.read_text()), 0)

    def test_live_helper_propagates_exit_and_stops_gateway(self):
        gateway, url, pid_path = self.live_fixture()
        result = run_live(gateway, [sys.executable, "-c", "raise SystemExit(7)"],
                          url, self.root / "gateway.log", startup_timeout=3, run_timeout=3)
        self.assertEqual(result, 7)
        self.assert_stopped(pid_path)

    def test_live_timeout_stops_both_processes(self):
        gateway, url, pid_path = self.live_fixture()
        launcher_pid = self.root / "launcher.pid"
        launcher = [sys.executable, "-c", "import os, pathlib, time; "
                    f"pathlib.Path({str(launcher_pid)!r}).write_text(str(os.getpid())); time.sleep(60)"]
        with self.assertRaisesRegex(TimeoutError, "Launcher"):
            run_live(gateway, launcher, url, self.root / "gateway.log", startup_timeout=3, run_timeout=0.5)
        self.assert_stopped(pid_path)
        self.assert_stopped(launcher_pid)

    def test_gateway_startup_failure_does_not_launch(self):
        _, url, _ = self.live_fixture()
        marker = self.root / "launched"
        launcher = [sys.executable, "-c", f"open({str(marker)!r}, 'w').close()"]
        with self.assertRaisesRegex(RuntimeError, "before readiness"):
            run_live([sys.executable, "-c", "raise SystemExit(9)"], launcher,
                     url, self.root / "gateway.log", startup_timeout=1)
        self.assertFalse(marker.exists())

    def test_live_helper_does_not_take_over_an_existing_port(self):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            sock.listen()
            url = f"http://127.0.0.1:{sock.getsockname()[1]}/readyz"
            with self.assertRaisesRegex(ValueError, "already occupied"):
                run_live([], [], url, self.root / "gateway.log")
            self.assertGreaterEqual(sock.fileno(), 0)

    @unittest.skipUnless(importlib.util.find_spec("yaml"), "requires the live runtime's PyYAML dependency")
    def test_live_cli_uses_gateway_config_defaults_and_evaluation_is_opt_in(self):
        config = self.root / "gateway.json"
        config.write_text(json.dumps({
            "listen_port": 8123, "storage_type": "sqlite",
            "storage_config": {"db_url": "sqlite://fixture.db"},
            "llm_routes": {"test": {"base_url": "http://unused.invalid/v1"}},
        }))
        base = ["live_smoke.py", "--gateway-config", str(config), "--", "--llm-model", "test"]
        for evaluation in (False, True):
            argv = base + (["--enable-evaluation"] if evaluation else [])
            with patch.object(sys, "argv", argv), patch.object(live_smoke, "run_live", return_value=0) as run:
                self.assertEqual(live_smoke.main(), 0)
                command = run.call_args.args[1]
                self.assertEqual("--enable-evaluation" in command, evaluation)
                self.assertEqual(command[command.index("--db-path") + 1], "sqlite://fixture.db")
                self.assertEqual(command[command.index("--gateway-base-url") + 1],
                                 "http://127.0.0.1:8123/v1/sessions")
        for flags in (["--db-path", "sqlite://wrong.db"], ["--storage-type", "cloud"],
                      ["--mode", "rjob"], ["--gateway-base-url=http://127.0.0.1:9999/v1/sessions"]):
            with patch.object(sys, "argv", base + flags), patch.object(live_smoke, "run_live") as run:
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    live_smoke.main()
                run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
