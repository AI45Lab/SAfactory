from __future__ import annotations

import asyncio
import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from core.result_artifact import write_result_atomic
from manager.sandbox_episode_runner import SandboxEpisodeRunner
from manager.types import SimulationAgentLease, SimulationStartRequest
from test_result_artifact import IDENTITY, artifact


class FakeFiles:
    def __init__(self, content):
        self.content = content
        self.reads = 0
        self.fail_reads = 0

    async def get_file_info(self, paths):
        return {paths[0]: SimpleNamespace(size=len(self.content))}

    async def read_bytes(self, path):
        self.reads += 1
        if self.fail_reads:
            self.fail_reads -= 1
            raise OSError("download interrupted")
        return self.content


class SandboxResultRunnerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name).resolve()
        self.destination = self.root / "job-1" / "session-1" / "safactory_result.json"
        self.reference_path = self.destination.parent / "runtime.json"
        self.reference_path.parent.mkdir(parents=True)
        self.reference = {**IDENTITY, "sandbox_id": "sandbox-1", "config_digest": "original", "command_finished": False}
        self.reference_path.write_text(json.dumps(self.reference), encoding="utf-8")
        self.files = FakeFiles(json.dumps(artifact()).encode())
        self.handle = SimpleNamespace(id="sandbox-1", files=self.files, retain=False, execution_stopped=True)
        self.lease = SimulationAgentLease(
            agent_name="test", agent_id=IDENTITY["session_id"], group_id="group", image="image", row_id=1,
            runtime="sandbox", runtime_handle=self.handle, resource_id="sandbox-1",
            run_command="python /tmp/runner.py", workdir="/work",
            runtime_config={
                "attempt_id": IDENTITY["attempt_id"], "runtime_ref_path": str(self.reference_path),
                "results": {"local_root": str(self.root), "download_timeout_s": 1, "download_retries": 0},
            },
        )
        self.request = SimulationStartRequest(
            **{key: IDENTITY[key] for key in ("job_id", "session_id")},
            agent_name="test", agent_id=IDENTITY["session_id"], group_id="group",
            gateway_base_url="http://gateway/v1/sessions", model="test", temperature=0.3,
            max_steps=10, storage_type="sqlite", metadata={"attempt_id": IDENTITY["attempt_id"]},
        )
        self.runner = SandboxEpisodeRunner(timeout_s=10)
        self.commands = []
        self.exit_code = 0
        self.stdout = ""
        self.bad_checksum = False

        async def run_command(handle, command, *, timeout_s, on_init=None):
            self.commands.append(command)
            if command.startswith("sha256sum -- "):
                digest = "0" * 64 if self.bad_checksum else hashlib.sha256(self.files.content).hexdigest()
                return SimpleNamespace(exit_code=0, stdout=f"{digest}  remote\n", stderr="")
            if on_init:
                await on_init("command-123")
            return SimpleNamespace(exit_code=self.exit_code, stdout=self.stdout, stderr="runner diagnostic")

        self.command_patch = patch("manager.sandbox_episode_runner.run_sandbox_command", side_effect=run_command)
        self.command_patch.start()
        self.addCleanup(self.command_patch.stop)

    async def test_empty_stdout_downloads_the_only_score_source_and_retains_handle(self):
        result = await self.runner.start(self.lease, self.request)
        self.assertEqual(result.status, "succeeded")
        self.assertIsNone(result.total_reward)
        self.assertEqual(result.result_artifact_path, str(self.destination))
        self.assertEqual(json.loads(self.destination.read_text())["evaluation"]["normalized_score_10"], 8)
        self.assertTrue(self.handle.retain)
        self.assertEqual(self.files.reads, 1)
        self.assertIn("SAFACTORY_RUNTIME=sandbox", self.commands[0])
        self.assertIn("SAFACTORY_ATTEMPT_ID=attempt-1", self.commands[0])
        reference = json.loads(self.reference_path.read_text())
        self.assertEqual(reference["command_id"], "command-123")
        self.assertEqual(reference["config_digest"], "original")
        self.assertTrue(reference["command_finished"])
        self.assertEqual(reference["exit_code"], 0)

    async def test_nonzero_exit_preserves_file_but_rejects_execution_success(self):
        self.exit_code = 1
        self.stdout = json.dumps({"status": "succeeded", "total_reward": 999})
        result = await self.runner.start(self.lease, self.request)
        self.assertTrue(self.destination.is_file())
        self.assertEqual(result.status, "failed")
        self.assertIn("execution_error", result.metrics)
        self.assertIsNone(result.total_reward)

    async def test_execution_error_clears_truncation_without_modifying_file_evidence(self):
        body = artifact(status="truncated", score=0)
        body["terminated"] = False
        self.files.content = json.dumps(body).encode()
        self.exit_code = 2
        result = await self.runner.start(self.lease, self.request)
        self.assertEqual(result.status, "failed")
        self.assertFalse(result.truncated)
        self.assertTrue(json.loads(self.destination.read_text())["truncated"])
        self.assertIn("execution_error", result.metrics)

    async def test_failed_download_never_uses_stdout_or_old_local_file(self):
        write_result_atomic(self.destination, artifact(score=1))
        original = self.destination.read_bytes()
        self.files.fail_reads = 1
        self.stdout = json.dumps(artifact(score=10))
        with self.assertRaises(OSError):
            await self.runner.start(self.lease, self.request)
        self.assertEqual(self.destination.read_bytes(), original)
        self.assertTrue(self.handle.retain)
        self.assertEqual(len(self.commands), 1)

    async def test_download_retry_does_not_rerun_episode(self):
        self.files.fail_reads = 1
        self.lease.runtime_config["results"]["download_retries"] = 1
        await self.runner.start(self.lease, self.request)
        self.assertEqual(self.files.reads, 2)
        self.assertEqual(sum(not c.startswith("sha256sum -- ") for c in self.commands), 1)

    async def test_wrong_attempt_and_bad_checksum_do_not_replace_old_file(self):
        for bad_attempt in (True, False):
            write_result_atomic(self.destination, artifact(score=1))
            original = self.destination.read_bytes()
            body = artifact()
            if bad_attempt:
                body["attempt_id"] = "another"
            else:
                self.bad_checksum = True
            self.files.content = json.dumps(body).encode()
            with self.subTest(bad_attempt=bad_attempt), self.assertRaises(ValueError):
                await self.runner.start(self.lease, self.request)
            self.assertEqual(self.destination.read_bytes(), original)
            self.assertEqual(list(self.destination.parent.glob("*.tmp")), [])

    async def test_local_resume_has_no_instance_or_remote_commands(self):
        write_result_atomic(self.destination, artifact())
        self.lease = replace(self.lease, runtime_handle=None, run_command="")
        self.lease.runtime_config["recovery"] = {
            "kind": "local", "runtime_ref": {**self.reference, "command_finished": True, "exit_code": 0},
        }
        result = await self.runner.start(self.lease, self.request)
        self.assertEqual(result.status, "succeeded")
        self.assertEqual(result.metrics["result_source"], "local_recovery")
        self.assertEqual(self.commands, [])
        self.assertEqual(self.files.reads, 0)

    async def test_remote_resume_only_downloads_without_business_command(self):
        self.lease = replace(self.lease, run_command="")
        self.lease.runtime_config["recovery"] = {
            "kind": "remote", "runtime_ref": {**self.reference, "command_finished": True, "exit_code": 0},
        }
        result = await self.runner.start(self.lease, self.request)
        self.assertEqual(result.status, "succeeded")
        self.assertEqual(len(self.commands), 1)
        self.assertTrue(self.commands[0].startswith("sha256sum -- "))

    async def test_resume_cannot_hide_prior_execution_failure_or_unknown_writer(self):
        write_result_atomic(self.destination, artifact())
        recovery = {"kind": "local", "runtime_ref": {**self.reference, "command_finished": True, "execution_error": "SDK failure"}}
        self.lease.runtime_config["recovery"] = recovery
        result = await self.runner.start(self.lease, self.request)
        self.assertEqual(result.status, "failed")
        recovery["runtime_ref"]["command_finished"] = False
        with self.assertRaisesRegex(RuntimeError, "previous writer"):
            await self.runner.start(self.lease, self.request)

    async def test_unknown_execution_stop_prevents_collection(self):
        self.handle.execution_stopped = False
        error = RuntimeError("interrupt failed")
        error.execution_stopped = False
        with patch("manager.sandbox_episode_runner.run_sandbox_command", new=AsyncMock(side_effect=error)):
            with self.assertRaisesRegex(RuntimeError, "writer state is unknown"):
                await self.runner.start(self.lease, self.request)
        self.assertEqual(self.files.reads, 0)
        self.assertTrue(self.handle.retain)
        self.assertFalse(json.loads(self.reference_path.read_text())["command_finished"])

    async def test_confirmed_execution_error_collects_file_without_scoring(self):
        error = TimeoutError("server timeout")
        error.execution_stopped = True
        original = self.command_patch
        original.stop()

        async def fail_runner(handle, command, *, timeout_s, on_init=None):
            if command.startswith("sha256sum -- "):
                return SimpleNamespace(exit_code=0, stdout=hashlib.sha256(self.files.content).hexdigest(), stderr="")
            raise error

        with patch("manager.sandbox_episode_runner.run_sandbox_command", side_effect=fail_runner):
            result = await self.runner.start(self.lease, self.request)
        self.assertEqual(result.status, "failed")
        self.assertTrue(self.destination.is_file())
        self.assertIn("server timeout", result.metrics["execution_error"])

    async def test_cancelled_command_recovers_only_after_confirmed_stop(self):
        async def cancel_runner(handle, command, *, timeout_s, on_init=None):
            if command.startswith("sha256sum -- "):
                return SimpleNamespace(exit_code=0, stdout=hashlib.sha256(self.files.content).hexdigest(), stderr="")
            if on_init:
                await on_init("cancelled-command")
            handle.execution_stopped = True
            raise asyncio.CancelledError()

        with patch("manager.sandbox_episode_runner.run_sandbox_command", side_effect=cancel_runner):
            with self.assertRaises(asyncio.CancelledError):
                await self.runner.start(self.lease, self.request)
        self.assertTrue(self.destination.is_file())
        self.assertTrue(self.handle.retain)
        reference = json.loads(self.reference_path.read_text())
        self.assertTrue(reference["command_finished"])
        self.assertEqual(reference["command_id"], "cancelled-command")
        self.assertEqual(reference["execution_error"], "runner cancelled")

    async def test_missing_execution_reference_prevents_starting_remote_work(self):
        self.reference_path.unlink()
        with self.assertRaises(FileNotFoundError):
            await self.runner.start(self.lease, self.request)
        self.assertEqual(self.commands, [])


if __name__ == "__main__":
    unittest.main()
