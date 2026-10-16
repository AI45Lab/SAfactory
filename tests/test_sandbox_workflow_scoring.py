from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from core.data_manager.manager import DataManager
from evaluator.factory import build_evaluation_service
from evaluator.result_file import ResultFileEvaluatorBackend
from evaluator.reward_committer import RewardCommitter
from manager.sandbox_episode_runner import SandboxEpisodeRunner
from manager.simulation_worker import SimulationWorkerGroup
from manager.types import SimulationAgentLease
from test_result_artifact import IDENTITY, artifact
from test_sandbox_result_runner import FakeFiles
from test_sandbox_workflow_integrity import _SingleLeasePool


class SandboxWorkflowScoringTest(unittest.IsolatedAsyncioTestCase):
    async def run_episode(self, *, status="succeeded", score=8, drained=True, gateway_instance="gateway-1"):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name).resolve()
        db_url = f"sqlite://{root}/trajectory.db"
        manager = DataManager(job_id=IDENTITY["job_id"], storage_type="sqlite", db_url=db_url,
                              enable_buffer=False, enable_upsert_batching=False)
        await manager.init()
        self.addAsyncCleanup(manager.close)
        session = manager.create_session(env_id=IDENTITY["session_id"], env_name="example", llm_model="model")
        messages = [{"role": "assistant", "content": [{"type": "thinking", "thinking": "检查文件", "signature": "native-signature"}]}]
        response = 'event: content_block_delta\ndata: {"delta":{"text":"verified"}}\n'
        request_body = '{"model":"model","messages":[]}'
        await manager.record_step(session, 1, messages, response, 0, request=request_body,
                                  meta_json={"event_type": "gateway_inference", "status_code": 200})
        original_row = (await manager.list_session_steps(session.session_id))[0]
        manager.mark_environment_finished = AsyncMock(return_value=1)

        body = artifact(status=status, score=score)
        if status == "truncated":
            body["terminated"] = False
        content = json.dumps(body).encode()
        handle = SimpleNamespace(id="sandbox-1", files=FakeFiles(content), retain=False, execution_stopped=True)
        reference = root / IDENTITY["job_id"] / IDENTITY["session_id"] / "runtime.json"
        reference.parent.mkdir(parents=True)
        reference.write_text(json.dumps({**IDENTITY, "gateway_instance_id": "gateway-1"}))
        lease = SimulationAgentLease(agent_name="example", agent_id=IDENTITY["session_id"], group_id="", image="", row_id=1,
                                     runtime="sandbox", runtime_handle=handle, resource_id="sandbox-1", run_command="python /tmp/runner.py",
                                     runtime_config={"attempt_id": IDENTITY["attempt_id"], "runtime_ref_path": str(reference), "gateway_instance_id": "gateway-1",
                                                     "results": {"local_root": str(root), "download_retries": 0}})
        gateway = SimpleNamespace(
            close_session=AsyncMock(return_value={"session_id": session.session_id, "status": "closed", "drained": drained, "instance_id": gateway_instance,
                                                  "telemetry_status": "flushed" if drained else "failed"}),
            clean_session=AsyncMock(), get_latest_success_step=AsyncMock(return_value=1),
        )
        cfg = SimpleNamespace(job_id=IDENTITY["job_id"], agent_root="env", storage_type="sqlite", db_url=db_url,
                              gateway_base_url="http://gateway/v1/sessions", llm_model="model", llm_temperature=0,
                              max_steps=10, agent_start_timeout_s=10, max_workers=1, circuit_breaker_enabled=False,
                              circuit_breaker_window=1, circuit_breaker_min_samples=1,
                              circuit_breaker_failure_rate=1, circuit_breaker_timeout_rate=1,
                              circuit_breaker_consecutive_timeouts=1)
        workers = SimulationWorkerGroup(
            _SingleLeasePool(lease), manager, SandboxEpisodeRunner(timeout_s=10), cfg,
            gateway_client=gateway,
            evaluation_service=build_evaluation_service(backend=ResultFileEvaluatorBackend()),
            reward_committer=RewardCommitter(db_url=db_url, data_manager=manager, gateway_client=gateway, llm_model="model"),
        )
        commands = []

        async def command(runtime, text, *, timeout_s, on_init=None):
            commands.append(text)
            if on_init:
                await on_init("command-1")
            stdout = hashlib.sha256(content).hexdigest() + " remote" if text.startswith("sha256sum -- ") else "SAFACTORY_RESULT_JSON {\"total_reward\":999}"
            return SimpleNamespace(exit_code=0, stdout=stdout, stderr="")

        with patch("manager.sandbox_episode_runner.run_sandbox_command", side_effect=command):
            await workers._worker_loop(0)
        rows = await manager.list_session_steps(session.session_id)
        self.assertEqual(len(rows), 1)
        self.assertEqual(json.loads(rows[0]["messages"]), messages)
        self.assertEqual(rows[0]["messages"], original_row["messages"])
        self.assertEqual(rows[0]["request"], request_body)
        self.assertEqual(rows[0]["response"], response)
        self.assertEqual(sum(not value.startswith("sha256sum -- ") for value in commands), 1)
        self.assertTrue((reference.parent / "safactory_result.json").is_file())
        self.assertIs(json.loads(reference.read_text())["gateway_finalized"], drained and gateway_instance == "gateway-1")
        return rows[0], workers._results["example_session-1"], handle, manager, gateway

    async def test_sdk_json_score_commits_once_preserving_native_trajectory(self):
        row, result, handle, manager, gateway = await self.run_episode(score=8)
        self.assertEqual(row["reward"], 8)
        self.assertEqual(result.total_reward, 8)
        self.assertTrue(row["is_session_completed"])
        self.assertFalse(handle.retain)
        manager.mark_environment_finished.assert_awaited_once_with("session-1")
        gateway.clean_session.assert_awaited_once()

    async def test_truncated_json_keeps_its_score_instead_of_forcing_zero(self):
        row, result, *_ = await self.run_episode(status="truncated", score=2.5)
        self.assertEqual(row["reward"], 2.5)
        self.assertEqual(result.status, "truncated")
        self.assertTrue(row["is_truncated"])

    async def test_undrained_gateway_keeps_artifact_without_committing_score(self):
        row, result, handle, manager, gateway = await self.run_episode(score=10, drained=False)
        self.assertIsNone(result.total_reward)
        self.assertFalse(row["is_session_completed"])
        self.assertTrue(handle.retain)
        manager.mark_environment_finished.assert_not_awaited()
        gateway.clean_session.assert_not_awaited()

    async def test_gateway_restart_cannot_confirm_the_previous_process_writes(self):
        row, result, handle, manager, gateway = await self.run_episode(gateway_instance="restarted-gateway")
        self.assertIsNone(result.total_reward)
        self.assertFalse(row["is_session_completed"])
        self.assertTrue(handle.retain)
        manager.mark_environment_finished.assert_not_awaited()
        gateway.clean_session.assert_not_awaited()
