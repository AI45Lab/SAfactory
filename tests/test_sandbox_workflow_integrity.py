from __future__ import annotations

import unittest
import json
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from gateway.config import GatewayConfig
from gateway.models import GatewaySessionBinding
from gateway.session_resolver import SessionResolver
from manager.simulation_worker import SimulationWorkerGroup
from manager.types import SimulationAgentLease


class _SingleLeasePool:
    pool_size = 1

    def __init__(self, lease: SimulationAgentLease) -> None:
        self.lease = lease
        self.done_count = 0

    async def acquire(self) -> SimulationAgentLease | None:
        lease, self.lease = self.lease, None
        return lease

    async def done(self, *_args: Any, **_kwargs: Any) -> None:
        self.done_count += 1


class _Gateway:
    def __init__(self, *, flushed: bool) -> None:
        self.flushed = flushed
        self.cleaned: list[str] = []
        self.binding = GatewaySessionBinding(
            session_id="session-1", model="model", upstream_base_url=None,
            status="active", last_seen_at=datetime.now(timezone.utc),
        )

    async def close_session(self, session_id: str, **_kwargs: Any) -> dict[str, Any]:
        self.binding.finish_close(
            drained=self.flushed,
            telemetry_status="flushed" if self.flushed else "timeout",
            closed_at=datetime.now(timezone.utc),
        )
        return {
            "session_id": session_id, "status": "closed", "drained": self.flushed,
            "telemetry_status": "flushed" if self.flushed else "timeout", "instance_id": "gateway-1",
        }

    async def clean_session(self, session_id: str) -> dict[str, str]:
        self.cleaned.append(session_id)
        return {"status": "cleaned"}


async def _run_failed_episode(
    *, runtime: str = "sandbox", handle: Any = None,
    runtime_config: dict | None = None, flushed: bool = True,
) -> tuple[_Gateway, _SingleLeasePool]:
    temporary = tempfile.TemporaryDirectory()
    reference_path = Path(temporary.name) / "runtime.json"
    runtime_cfg = dict(runtime_config or {})
    reference = {
        "job_id": "job-1", "session_id": "session-1", "attempt_id": "attempt-1",
        "command_finished": False, "gateway_instance_id": "gateway-1",
        **dict(runtime_cfg.get("runtime_ref") or {}),
    }
    reference_path.write_text(json.dumps(reference))
    runtime_cfg.setdefault("runtime_ref", reference)
    runtime_cfg.update(
        attempt_id="attempt-1", runtime_ref_path=str(reference_path), gateway_instance_id="gateway-1",
    )
    lease = SimulationAgentLease(
        agent_name="example", agent_id="session-1", group_id="", image="", row_id=1,
        runtime=runtime, runtime_handle=handle, runtime_config=runtime_cfg,
    )
    pool = _SingleLeasePool(lease)
    gateway = _Gateway(flushed=flushed)

    def create_session(**kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(session_id=kwargs["env_id"])

    async def start(*_args: Any) -> None:
        raise RuntimeError("writer state is unknown or result collection failed")

    cfg = SimpleNamespace(
        job_id="job-1", agent_root="env", storage_type="sqlite", db_url="sqlite://unused.db",
        gateway_base_url="http://gateway/v1/sessions", llm_model="model", llm_temperature=0.0,
        max_steps=1, agent_start_timeout_s=1.0, max_workers=1, circuit_breaker_enabled=False,
        circuit_breaker_window=1, circuit_breaker_min_samples=1,
        circuit_breaker_failure_rate=1.0, circuit_breaker_timeout_rate=1.0,
        circuit_breaker_consecutive_timeouts=1,
    )
    workers = SimulationWorkerGroup(
        pool, SimpleNamespace(create_session=create_session), SimpleNamespace(start=start), cfg,
        gateway_client=gateway,  # type: ignore[arg-type]
    )
    try:
        await workers._worker_loop(0)
        gateway.receipt = json.loads(reference_path.read_text())
    finally:
        temporary.cleanup()
    return gateway, pool


class SandboxGatewayCleanupTest(unittest.IsolatedAsyncioTestCase):
    async def test_unknown_writer_keeps_closed_session_even_after_gateway_flush(self) -> None:
        gateway, pool = await _run_failed_episode(
            handle=SimpleNamespace(execution_stopped=False, retain=True),
        )
        self.assertEqual(gateway.binding.status, "closed")
        self.assertEqual(gateway.cleaned, [])
        self.assertEqual(pool.done_count, 1)
        self.assertIs(gateway.receipt["gateway_finalized"], True)

    async def test_live_unknown_handle_takes_precedence_over_old_finished_reference(self) -> None:
        gateway, _ = await _run_failed_episode(
            handle=SimpleNamespace(execution_stopped=False),
            runtime_config={"runtime_ref": {"command_finished": True}},
        )
        self.assertEqual(gateway.cleaned, [])

    async def test_confirmed_stopped_writer_can_clean_after_collection_failure(self) -> None:
        gateway, _ = await _run_failed_episode(handle=SimpleNamespace(execution_stopped=True))
        self.assertEqual(gateway.cleaned, ["session-1"])

    async def test_local_recovery_uses_confirmed_runtime_reference(self) -> None:
        for config in (
            {"recovery": {"kind": "local", "runtime_ref": {"command_finished": True}}},
            {"runtime_ref": {"command_finished": True}},
        ):
            with self.subTest(config=config):
                gateway, _ = await _run_failed_episode(runtime_config=config)
                self.assertEqual(gateway.cleaned, ["session-1"])

    async def test_missing_or_unconfirmed_stop_evidence_keeps_session_closed(self) -> None:
        for config in ({}, {"runtime_ref": {"command_finished": False}}):
            with self.subTest(config=config):
                gateway, _ = await _run_failed_episode(runtime_config=config)
                self.assertEqual(gateway.cleaned, [])

    async def test_stopped_writer_cannot_bypass_gateway_flush_failure(self) -> None:
        gateway, _ = await _run_failed_episode(
            handle=SimpleNamespace(execution_stopped=True), flushed=False,
        )
        self.assertEqual(gateway.cleaned, [])
        self.assertIs(gateway.receipt["gateway_finalized"], False)

    async def test_other_runtimes_keep_existing_cleanup_behavior(self) -> None:
        gateway, _ = await _run_failed_episode(runtime="docker")
        self.assertEqual(gateway.cleaned, ["session-1"])


class GatewayClosedSessionRetentionTest(unittest.IsolatedAsyncioTestCase):
    async def test_ttl_evicts_idle_active_sessions_but_retains_closed_and_closing(self) -> None:
        resolver = SessionResolver(GatewayConfig(session_cache_ttl_s=1))
        aged = datetime.now(timezone.utc) - timedelta(seconds=60)
        for status in ("active", "closing", "closed"):
            resolver._bindings[status] = GatewaySessionBinding(
                session_id=status, model="model", upstream_base_url=None,
                status=status, last_seen_at=aged,
            )
        with patch("gateway.session_resolver.time.monotonic", return_value=120):
            await resolver._evict_expired()
        self.assertEqual(set(resolver._bindings), {"closing", "closed"})

    async def test_aged_closed_session_cannot_be_reactivated_by_an_inference_request(self) -> None:
        resolver = SessionResolver(GatewayConfig(session_cache_ttl_s=1))
        for status in ("closing", "closed"):
            with self.subTest(status=status):
                binding = GatewaySessionBinding(
                    session_id=status, model="model", upstream_base_url=None, status=status,
                    last_seen_at=datetime.now(timezone.utc) - timedelta(seconds=60),
                )
                resolver._bindings[status] = binding
                context = await resolver.resolve(
                    {"model": "model"}, endpoint="chat/completions", path_session_id=status,
                )
                resolver._last_eviction_at = 0
                with patch("gateway.session_resolver.time.monotonic", return_value=120):
                    resolved = await resolver.get_or_create_binding(context)
                self.assertIs(resolved, binding)
                self.assertEqual(resolved.status, status)

    async def test_explicit_clear_allows_a_new_binding(self) -> None:
        resolver = SessionResolver(GatewayConfig(session_cache_ttl_s=1))
        for status in ("closing", "closed"):
            with self.subTest(status=status):
                binding = GatewaySessionBinding(
                    session_id=status, model="model", upstream_base_url=None, status=status,
                    last_seen_at=datetime.now(timezone.utc) - timedelta(seconds=60),
                )
                binding.mark_telemetry_write_failed()
                resolver._bindings[status] = binding
                self.assertEqual(await resolver.clear_session_cache([status]), 1)
                context = await resolver.resolve(
                    {"model": "model"}, endpoint="chat/completions", path_session_id=status,
                )
                replacement = await resolver.get_or_create_binding(context)
                self.assertIsNot(replacement, binding)
                self.assertEqual(replacement.status, "active")
                self.assertFalse(replacement.telemetry_write_failed)


if __name__ == "__main__":
    unittest.main()
