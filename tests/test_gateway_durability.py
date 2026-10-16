from __future__ import annotations

import asyncio
import json
import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any

from fastapi.testclient import TestClient

from gateway.app import _finalize_session_close, create_app
from gateway.config import GatewayConfig
from gateway.llm_router import LLMRouteTarget
from gateway.models import GatewayRequestContext, GatewaySessionBinding
from gateway.telemetry import TelemetryRecorder


def _binding(session_id: str = "session-1") -> GatewaySessionBinding:
    return GatewaySessionBinding(
        session_id=session_id,
        model="model",
        upstream_base_url=None,
        status="active",
        last_seen_at=datetime.now(timezone.utc),
    )


class _Storage:
    def __init__(self, *, failure: str | None = None, strategy: Any = None) -> None:
        self.failure = failure
        self.writes = 0
        self.data_manager = SimpleNamespace(
            job_id="gateway",
            _strategy=strategy,
            _upsert_batcher=None,
        )

    async def _write(self) -> None:
        if self.failure == "error":
            raise RuntimeError("injected storage failure")
        if self.failure == "timeout":
            await asyncio.sleep(60)
        self.writes += 1

    async def record_inference_step(self, *_args: Any) -> None:
        await self._write()

    async def record_telemetry_batch(self, *_args: Any) -> None:
        await self._write()

    async def close(self) -> None:
        return None


async def _enqueue(recorder: TelemetryRecorder, binding: GatewaySessionBinding) -> None:
    context = GatewayRequestContext(
        request_id="request-1",
        session_id=binding.session_id,
        requested_model="model",
        endpoint="chat/completions",
        is_stream=False,
        created_at=datetime.now(timezone.utc),
    )
    await recorder.enqueue_success(
        context,
        binding,
        LLMRouteTarget("model", "http://upstream", None),
        {"messages": []},
        {"choices": []},
        1.0,
    )


class _Resolver:
    def __init__(self) -> None:
        self.result: tuple[bool, str] | None = None

    async def finish_close_session(
        self, _session_id: str, *, drained: bool, telemetry_status: str
    ) -> None:
        self.result = drained, telemetry_status


class TelemetryFailureTest(unittest.IsolatedAsyncioTestCase):
    async def test_strict_write_error_stays_failed_after_successful_retry(self) -> None:
        storage = _Storage(failure="error")
        recorder = TelemetryRecorder(GatewayConfig(), storage)  # type: ignore[arg-type]
        binding = _binding()
        with self.assertRaisesRegex(RuntimeError, "injected storage failure"):
            await _enqueue(recorder, binding)
        storage.failure = None
        await _enqueue(recorder, binding)
        self.assertTrue(binding.telemetry_write_failed)
        self.assertEqual(storage.writes, 1)

        resolver = _Resolver()
        await _finalize_session_close(
            binding=binding,
            resolver=resolver,  # type: ignore[arg-type]
            telemetry=recorder,
            cfg=GatewayConfig(),
        )
        self.assertEqual(resolver.result, (True, "failed"))

    async def test_strict_write_timeout_is_not_a_successful_flush(self) -> None:
        storage = _Storage(failure="timeout")
        recorder = TelemetryRecorder(
            GatewayConfig(telemetry_write_timeout_s=0.01),
            storage,  # type: ignore[arg-type]
        )
        binding = _binding()
        with self.assertRaises(asyncio.TimeoutError):
            await _enqueue(recorder, binding)
        self.assertTrue(binding.telemetry_write_failed)
        self.assertEqual(recorder.flushed_total, 0)

    async def test_cancelled_write_remains_unconfirmed(self) -> None:
        storage = _Storage(failure="timeout")
        recorder = TelemetryRecorder(GatewayConfig(), storage)  # type: ignore[arg-type]
        binding = _binding()
        task = asyncio.create_task(_enqueue(recorder, binding))
        await asyncio.sleep(0)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(binding.telemetry_write_failed)

    async def test_async_failed_batch_cannot_be_hidden_by_later_barrier(self) -> None:
        storage = _Storage(failure="error")
        cfg = GatewayConfig(
            storage_type="cloud",
            telemetry_async_cloud_writes=True,
            telemetry_flush_interval_ms=1,
            telemetry_batch_size=1,
            telemetry_writer_count=1,
        )
        recorder = TelemetryRecorder(cfg, storage)  # type: ignore[arg-type]
        binding = _binding()
        await recorder.start()
        try:
            await _enqueue(recorder, binding)
            await asyncio.wait_for(recorder._queues[0].join(), timeout=1)
            self.assertEqual(recorder.dropped_by_reason["write_failed"], 1)
            storage.failure = None
            await _enqueue(recorder, binding)
            await recorder.wait_for_session_flush(binding)
            resolver = _Resolver()
            await _finalize_session_close(
                binding=binding,
                resolver=resolver,  # type: ignore[arg-type]
                telemetry=recorder,
                cfg=cfg,
            )
            self.assertEqual(resolver.result, (True, "failed"))
        finally:
            await recorder.stop()

    async def test_manual_async_flush_failure_marks_all_dropped_sessions(self) -> None:
        storage = _Storage(failure="error")
        cfg = GatewayConfig(storage_type="cloud", telemetry_writer_count=1)
        recorder = TelemetryRecorder(cfg, storage)  # type: ignore[arg-type]
        first, second = _binding("first"), _binding("second")
        await _enqueue(recorder, first)
        await _enqueue(recorder, second)
        with self.assertRaisesRegex(RuntimeError, "injected storage failure"):
            await recorder.flush_once(drain_all=True)
        self.assertTrue(first.telemetry_write_failed)
        self.assertTrue(second.telemetry_write_failed)

    def test_failure_after_close_updates_previous_flushed_status(self) -> None:
        binding = _binding()
        binding.finish_close(
            drained=True, telemetry_status="flushed", closed_at=datetime.now(timezone.utc)
        )
        binding.mark_telemetry_write_failed()
        self.assertEqual(binding.close_telemetry_status, "failed")
        binding.finish_close(
            drained=True, telemetry_status="flushed", closed_at=datetime.now(timezone.utc)
        )
        self.assertEqual(binding.close_telemetry_status, "failed")


class GatewayReadinessDurabilityTest(unittest.TestCase):
    def test_close_endpoint_reports_sticky_write_failure(self) -> None:
        storage = _Storage(failure="error")
        cfg = GatewayConfig(request_log_enabled=False)
        app = create_app(cfg, storage=storage)  # type: ignore[arg-type]
        with TestClient(app) as client:
            binding = _binding()
            app.state.gateway_resolver._bindings[binding.session_id] = binding
            with self.assertRaisesRegex(RuntimeError, "injected storage failure"):
                client.portal.call(_enqueue, app.state.gateway_telemetry, binding)
            storage.failure = None
            client.portal.call(_enqueue, app.state.gateway_telemetry, binding)
            client.post("/v1/sessions/session-1/close", json={})
            client.portal.call(asyncio.sleep, 0.01)
            body = client.post("/v1/sessions/session-1/close", json={}).json()
        self.assertEqual(body["status"], "closed")
        self.assertTrue(body["drained"])
        self.assertEqual(body["telemetry_status"], "failed")
        self.assertTrue(body["telemetry_write_failed"])

    def test_sqlite_reports_actual_buffer_flags_and_preserves_db_url(self) -> None:
        storage = _Storage(strategy=SimpleNamespace(
            _enable_buffer=False, db_url="sqlite:///actual/trajectory.db"
        ))
        cfg = GatewayConfig(
            storage_type="sqlite",
            storage_config={"db_url": "sqlite:///configured.db", "enable_buffer": True},
            request_log_enabled=False,
        )
        with TestClient(create_app(cfg, storage=storage)) as client:  # type: ignore[arg-type]
            response = client.get("/readyz")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["durability"], {
            "telemetry_mode": "strict",
            "async_cloud_writes": False,
            "enable_buffer": False,
            "enable_upsert_batching": False,
        })
        self.assertEqual(body["storage_identity"], {
            "namespace": "gateway", "db_url": "sqlite:///actual/trajectory.db"
        })
        self.assertEqual(body["storage_config"]["db_url"], "sqlite:///configured.db")

    def test_cloud_reports_resolved_tables_without_credentials(self) -> None:
        storage = _Storage(strategy=SimpleNamespace(
            _enable_buffer=True,
            landing_table="resolved_landing",
            env_config_table="s3://user:password@bucket/envs?token=secret#secret",
        ))
        storage.data_manager._upsert_batcher = object()
        cfg = GatewayConfig(
            storage_type="cloud",
            storage_config={"landing_table": "configured_landing"},
            request_log_enabled=False,
        )
        with TestClient(create_app(cfg, storage=storage)) as client:  # type: ignore[arg-type]
            body = client.get("/readyz").json()
        self.assertTrue(body["durability"]["async_cloud_writes"])
        self.assertTrue(body["durability"]["enable_buffer"])
        self.assertTrue(body["durability"]["enable_upsert_batching"])
        self.assertEqual(body["storage_identity"], {
            "namespace": "gateway",
            "landing_table": "resolved_landing",
            "env_config_table": "s3://bucket/envs",
        })
        self.assertNotIn("secret", json.dumps(body))
        self.assertNotIn("password", json.dumps(body))

    def test_unknown_custom_storage_does_not_claim_buffering_is_disabled(self) -> None:
        storage = _Storage()
        del storage.data_manager
        cfg = GatewayConfig(request_log_enabled=False)
        with TestClient(create_app(cfg, storage=storage)) as client:  # type: ignore[arg-type]
            body = client.get("/readyz").json()
        self.assertIsNone(body["durability"]["enable_buffer"])
        self.assertIsNone(body["durability"]["enable_upsert_batching"])


class GatewayCloseDrainTest(unittest.IsolatedAsyncioTestCase):
    async def test_all_modes_time_out_with_active_requests_or_streams(self) -> None:
        for mode in ("soft_close", "hard_close", "soft_clsoe"):
            for field in ("active_request_count", "active_stream_count"):
                with self.subTest(mode=mode, field=field):
                    binding = _binding()
                    setattr(binding, field, 1)
                    flushed: list[str] = []

                    async def flush(current: GatewaySessionBinding) -> None:
                        flushed.append(current.session_id)

                    resolver = _Resolver()
                    await _finalize_session_close(
                        binding=binding,
                        resolver=resolver,  # type: ignore[arg-type]
                        telemetry=SimpleNamespace(wait_for_session_flush=flush),
                        cfg=GatewayConfig(close_mode=mode, session_close_timeout_s=0.01),
                    )
                    self.assertEqual(resolver.result, (False, "timeout"))
                    self.assertEqual(flushed, [])

    async def test_flush_only_runs_after_real_drain_for_every_mode(self) -> None:
        for mode in ("soft_close", "hard_close", "soft_clsoe"):
            with self.subTest(mode=mode):
                binding = _binding()
                binding.active_request_count = 1
                binding.active_stream_count = 1
                observed_counts: list[tuple[int, int]] = []

                async def finish_requests() -> None:
                    await asyncio.sleep(0.01)
                    binding.active_request_count = 0
                    binding.active_stream_count = 0

                async def flush(current: GatewaySessionBinding) -> None:
                    observed_counts.append((current.active_request_count, current.active_stream_count))

                producer = asyncio.create_task(finish_requests())
                resolver = _Resolver()
                await _finalize_session_close(
                    binding=binding,
                    resolver=resolver,  # type: ignore[arg-type]
                    telemetry=SimpleNamespace(wait_for_session_flush=flush),
                    cfg=GatewayConfig(close_mode=mode, session_close_timeout_s=1),
                )
                await producer
                self.assertEqual(resolver.result, (True, "flushed"))
                self.assertEqual(observed_counts, [(0, 0)])


if __name__ == "__main__":
    unittest.main()
