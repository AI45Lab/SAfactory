from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from manager.resume_cleanup import prepare_sandbox_resume


class SandboxResumeReceiptTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.reference_path = self.root / "job" / "session" / "runtime.json"
        self.reference_path.parent.mkdir(parents=True)
        self.identity = {
            "job_id": "job", "session_id": "session", "attempt_id": "attempt",
            "gateway_instance_id": "gateway-1",
        }
        self.manager_cfg = {"cluster": {"sandbox": {"gateway_instance_id": "gateway-1"}}}
        self.row = {"env_id": "session"}
        self.cfg = SimpleNamespace(
            job_id="job", gateway_base_url="http://gateway/v1/sessions",
            gateway_close_timeout_s=1, gateway_close_retry_backoff_s=0,
            sandbox_config={"results": {"local_root": str(self.root)}},
        )
        self.backend = SimpleNamespace(
            prepare_resume=AsyncMock(return_value=([self.row], {})), close=AsyncMock(),
        )
        self.client = SimpleNamespace(
            get_session_status=AsyncMock(return_value=None),
            close_session=AsyncMock(return_value={
                "session_id": "session", "status": "closed", "drained": True,
                "telemetry_status": "flushed", "instance_id": "gateway-1",
            }),
            aclose=AsyncMock(),
        )
        self.manager = SimpleNamespace(list_session_steps=AsyncMock(return_value=[]))
        self.backend_patch = patch("manager.resume_cleanup.SandboxClusterBackend", return_value=self.backend)
        self.client_patch = patch("manager.resume_cleanup.GatewayClient", return_value=self.client)
        self.backend_patch.start()
        self.client_patch.start()
        self.addCleanup(self.backend_patch.stop)
        self.addCleanup(self.client_patch.stop)

    def write_reference(self, **updates) -> None:
        self.reference_path.write_text(json.dumps({**self.identity, **updates}))

    async def resume(self):
        return await prepare_sandbox_resume(
            cfg=self.cfg, manager_cfg=self.manager_cfg, environment_rows=[self.row], data_manager=self.manager,
        )

    async def test_failed_receipt_blocks_before_backend_even_with_live_binding(self) -> None:
        self.write_reference(gateway_finalized=False)
        self.client.get_session_status.return_value = {"status": "active", "instance_id": "gateway-1"}
        with self.assertRaisesRegex(RuntimeError, "previously failed"):
            await self.resume()
        self.backend.prepare_resume.assert_not_awaited()
        self.client.get_session_status.assert_not_awaited()
        self.client.close_session.assert_not_awaited()
        self.manager.list_session_steps.assert_not_awaited()
        self.client.aclose.assert_awaited_once()
        self.backend.close.assert_awaited_once()

    async def test_unacknowledged_attempt_cannot_use_empty_gateway_after_restart(self) -> None:
        for updates in ({}, {"gateway_finalized": None}):
            with self.subTest(updates=updates):
                self.write_reference(**updates)
                with self.assertRaisesRegex(RuntimeError, "no durable Gateway finalization receipt or live binding"):
                    await self.resume()
        self.backend.prepare_resume.assert_not_awaited()
        self.client.close_session.assert_not_awaited()
        self.manager.list_session_steps.assert_not_awaited()

    async def test_success_receipt_allows_local_recovery_without_gateway_binding(self) -> None:
        self.write_reference(gateway_finalized=True, gateway_instance_id="retired-gateway")
        recovery = {"session": {"kind": "local", "runtime_ref": {**self.identity, "gateway_finalized": True}}}
        self.backend.prepare_resume.return_value = ([], recovery)
        self.assertEqual(await self.resume(), ([], recovery))
        self.backend.prepare_resume.assert_awaited_once()
        self.client.get_session_status.assert_not_awaited()
        self.client.close_session.assert_not_awaited()

    async def test_missing_receipt_uses_live_binding_and_reuses_status_for_rerun_drain(self) -> None:
        self.write_reference()
        self.client.get_session_status.return_value = {"status": "active", "instance_id": "gateway-1"}
        self.assertEqual(await self.resume(), ([self.row], {}))
        self.client.get_session_status.assert_awaited_once_with("session")
        self.client.close_session.assert_awaited_once()
        self.manager.list_session_steps.assert_not_awaited()

    async def test_success_receipt_still_drains_live_gateway_before_reset(self) -> None:
        self.write_reference(gateway_finalized=True)
        self.client.get_session_status.return_value = {"status": "active", "instance_id": "gateway-1"}
        self.client.close_session.return_value["telemetry_status"] = "failed"
        with self.assertRaisesRegex(RuntimeError, "undrained"):
            await self.resume()
        self.backend.prepare_resume.assert_awaited_once()
        self.client.close_session.assert_awaited_once()

    async def test_missing_receipt_rejects_recreated_binding_in_another_gateway_epoch(self) -> None:
        self.write_reference()
        self.client.get_session_status.return_value = {"status": "active", "instance_id": "gateway-2"}
        with self.assertRaisesRegex(RuntimeError, "live Gateway instance changed"):
            await self.resume()
        self.backend.prepare_resume.assert_not_awaited()
        self.client.close_session.assert_not_awaited()

    async def test_missing_receipt_requires_matching_reference_and_current_gateway_epoch(self) -> None:
        self.client.get_session_status.return_value = {"status": "active", "instance_id": "gateway-1"}
        for epoch in (None, "", "retired-gateway"):
            with self.subTest(epoch=epoch):
                self.write_reference(gateway_instance_id=epoch)
                with self.assertRaisesRegex(RuntimeError, "Gateway instance changed"):
                    await self.resume()
        self.backend.prepare_resume.assert_not_awaited()
        self.client.get_session_status.assert_not_awaited()

    async def test_gateway_restart_between_status_and_close_blocks_trajectory_reset(self) -> None:
        self.write_reference()
        self.client.get_session_status.return_value = {"status": "active", "instance_id": "gateway-1"}
        self.client.close_session.return_value["instance_id"] = "gateway-2"
        with self.assertRaisesRegex(RuntimeError, "after Gateway instance changed"):
            await self.resume()
        self.backend.prepare_resume.assert_awaited_once()
        self.client.close_session.assert_awaited_once()

    async def test_invalid_reference_is_rejected_before_backend(self) -> None:
        values = [
            "{", "[]",
            json.dumps({**self.identity, "job_id": "other", "gateway_finalized": True}),
            json.dumps({**self.identity, "session_id": "other", "gateway_finalized": True}),
            json.dumps({**self.identity, "attempt_id": "", "gateway_finalized": True}),
            json.dumps({**self.identity, "attempt_id": "../other", "gateway_finalized": True}),
            json.dumps({**self.identity, "attempt_id": 1, "gateway_finalized": True}),
            json.dumps({**self.identity, "gateway_finalized": "true"}),
            json.dumps({**self.identity, "gateway_finalized": True, "unexpected": float("nan")}),
            '{"job_id":"job","session_id":"session","attempt_id":"attempt",'
            '"gateway_finalized":false,"gateway_finalized":true}',
        ]
        for value in values:
            with self.subTest(value=value):
                self.reference_path.write_text(value)
                with self.assertRaises(RuntimeError):
                    await self.resume()
        self.backend.prepare_resume.assert_not_awaited()
        self.client.get_session_status.assert_not_awaited()

    async def test_unreferenced_history_remains_protected(self) -> None:
        self.manager.list_session_steps.return_value = [{"step_id": 1}]
        with self.assertRaisesRegex(RuntimeError, "trajectory exists without a runtime reference"):
            await self.resume()
        self.backend.prepare_resume.assert_awaited_once()
        self.client.close_session.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
