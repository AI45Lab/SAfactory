from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from clusters.sandbox_cluster import SandboxClusterBackend
from manager.sandbox_episode_runner import SandboxEpisodeRunner
from manager.simulation_lease_pool import SimulationLeasePool
from manager.simulation_worker import SimulationWorkerGroup
from manager.types import SimulationAgentLease, SimulationStartRequest


def _execution() -> tuple[SimulationAgentLease, SimulationStartRequest]:
    lease = SimulationAgentLease(
        agent_name="example", agent_id="session-1", group_id="", image="image", row_id=1,
        runtime="sandbox", runtime_handle=SimpleNamespace(execution_stopped=False, retain=True),
        runtime_config={"attempt_id": "attempt-1"}, resource_id="sandbox-1", run_command="run-episode",
    )
    request = SimulationStartRequest(
        job_id="job-1", session_id="session-1", agent_name="example", agent_id="session-1",
        group_id="", gateway_base_url="http://gateway/v1/sessions", model="model",
        temperature=0.0, max_steps=1, storage_type="sqlite", metadata={"attempt_id": "attempt-1"},
    )
    return lease, request


class SandboxCreateCancellationTest(unittest.IsolatedAsyncioTestCase):
    async def test_create_cancel_preserves_signal_when_transport_close_fails(self) -> None:
        backend = SandboxClusterBackend()
        started = asyncio.Event()

        async def create(**_kwargs):
            started.set()
            await asyncio.Event().wait()

        connection = SimpleNamespace(close_transport_if_owned=AsyncMock(side_effect=OSError("transport close failed")))
        service = SimpleNamespace(create_sandbox=AsyncMock(side_effect=create), kill_sandbox=AsyncMock())
        with patch.object(backend, "_connection", return_value=(connection, service)):
            task = asyncio.create_task(backend._create_sandbox(image="image", environment_id="env-1", cfg={}))
            await asyncio.wait_for(started.wait(), timeout=1)
            task.cancel("shutdown")
            with self.assertRaises(asyncio.CancelledError) as raised:
                await task
        self.assertEqual(raised.exception.args, ("shutdown",))
        connection.close_transport_if_owned.assert_awaited_once()
        service.kill_sandbox.assert_not_awaited()

    async def test_bootstrap_cancel_attempts_kill_and_preserves_signal_when_both_cleanups_fail(self) -> None:
        backend = SandboxClusterBackend()
        started = asyncio.Event()

        async def connect(**_kwargs):
            started.set()
            await asyncio.Event().wait()

        connection = SimpleNamespace(close_transport_if_owned=AsyncMock(side_effect=OSError("transport close failed")))
        service = SimpleNamespace(
            create_sandbox=AsyncMock(return_value=SimpleNamespace(id="sandbox-1")),
            kill_sandbox=AsyncMock(side_effect=OSError("kill failed")),
        )
        with (
            patch.object(backend, "_connection", return_value=(connection, service)),
            patch.object(backend, "_connect_instance", new=connect),
        ):
            task = asyncio.create_task(backend._create_sandbox(image="image", environment_id="env-1", cfg={}))
            await asyncio.wait_for(started.wait(), timeout=1)
            task.cancel("shutdown")
            with self.assertRaises(asyncio.CancelledError) as raised:
                await task
        self.assertEqual(raised.exception.args, ("shutdown",))
        service.kill_sandbox.assert_awaited_once_with("sandbox-1")
        connection.close_transport_if_owned.assert_awaited_once()

    async def test_ordinary_create_failure_keeps_existing_transport_error_behavior(self) -> None:
        backend = SandboxClusterBackend()
        connection = SimpleNamespace(close_transport_if_owned=AsyncMock(side_effect=OSError("transport close failed")))
        service = SimpleNamespace(create_sandbox=AsyncMock(side_effect=RuntimeError("create failed")))
        with patch.object(backend, "_connection", return_value=(connection, service)):
            with self.assertRaisesRegex(OSError, "transport close failed"):
                await backend._create_sandbox(image="image", environment_id="env-1", cfg={})


class SandboxRecoveryCancellationTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.cfg = {"job_id": "job-1", "results": {"local_root": temporary.name}, "environment_id": "env-1"}
        self.backend = SandboxClusterBackend(cluster_cfg={"sandbox": self.cfg})
        self.row = {"env_name": "example", "env_id": "session-1", "env_image": "image"}
        self.reference = {
            "job_id": "job-1", "session_id": "session-1", "attempt_id": "attempt-1",
            "sandbox_id": "sandbox-1", "image": "image", "environment_id": "env-1",
            "config_digest": self.backend._config_digest(self.backend._config_for_env("example"), "image"),
            "remote_result_path": "/app/results/job-1/session-1/safactory_result.json",
            "command_finished": True,
        }
        self.path = self.backend._runtime_path(self.cfg, "session-1")

    def _write_reference(self) -> bytes:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.reference))
        return self.path.read_bytes()

    async def _cancel(self, task: asyncio.Task, started: asyncio.Event) -> None:
        await asyncio.wait_for(started.wait(), timeout=1)
        task.cancel("shutdown")
        with self.assertRaises(asyncio.CancelledError) as raised:
            await task
        self.assertEqual(raised.exception.args, ("shutdown",))

    async def test_recover_handle_cancel_closes_connection_without_creating_replacement(self) -> None:
        started = asyncio.Event()

        async def get_info(*_args):
            started.set()
            await asyncio.Event().wait()

        connection = SimpleNamespace(close_transport_if_owned=AsyncMock(side_effect=OSError("close failed")))
        service = SimpleNamespace(get_sandbox_info=AsyncMock(side_effect=get_info))
        with (
            patch.object(self.backend, "_connection", return_value=(connection, service)),
            patch.object(self.backend, "_create_sandbox", new=AsyncMock()) as create,
        ):
            task = asyncio.create_task(self.backend._recover_handle(self.reference, self.cfg, "example"))
            await self._cancel(task, started)
        connection.close_transport_if_owned.assert_awaited_once()
        create.assert_not_awaited()

    async def test_resume_metadata_cancel_preserves_signal_when_connection_close_fails(self) -> None:
        started = asyncio.Event()

        async def find(*_args):
            started.set()
            await asyncio.Event().wait()

        connection = SimpleNamespace(close_transport_if_owned=AsyncMock(side_effect=OSError("close failed")))
        with (
            patch.object(self.backend, "_connection", return_value=(connection, object())),
            patch.object(self.backend, "_find_instance", new=AsyncMock(side_effect=find)) as lookup,
            patch.object(self.backend, "_recover_handle", new=AsyncMock()) as recover,
        ):
            task = asyncio.create_task(self.backend.prepare_resume([self.row, dict(self.row, env_id="session-2")]))
            await self._cancel(task, started)
        connection.close_transport_if_owned.assert_awaited_once()
        lookup.assert_awaited_once()
        recover.assert_not_awaited()
        self.assertFalse(self.path.exists())

    async def test_resume_original_result_cancel_preserves_reference_when_handle_close_fails(self) -> None:
        before = self._write_reference()
        started = asyncio.Event()

        async def read(*_args):
            started.set()
            await asyncio.Event().wait()

        handle = SimpleNamespace(
            id="sandbox-1", execution_stopped=True, retain=True,
            files=SimpleNamespace(read_bytes=AsyncMock(side_effect=read)),
            close=AsyncMock(side_effect=OSError("close failed")), kill=AsyncMock(),
        )
        with patch.object(self.backend, "_recover_handle", new=AsyncMock(return_value=handle)) as recover:
            task = asyncio.create_task(self.backend.prepare_resume([self.row, dict(self.row, env_id="session-2")]))
            await self._cancel(task, started)
        recover.assert_awaited_once()
        handle.close.assert_awaited_once()
        handle.kill.assert_not_awaited()
        self.assertTrue(handle.retain)
        self.assertEqual(self.path.read_bytes(), before)

    async def test_resume_reader_cancel_survives_kill_and_close_failures(self) -> None:
        before = self._write_reference()
        started = asyncio.Event()

        async def read(*_args):
            started.set()
            await asyncio.Event().wait()

        handle = SimpleNamespace(
            id="recovery-reader", execution_stopped=True, retain=True, recovery_reader=True,
            files=SimpleNamespace(read_bytes=AsyncMock(side_effect=read)),
            close=AsyncMock(side_effect=OSError("close failed")), kill=AsyncMock(side_effect=OSError("kill failed")),
        )
        with patch.object(self.backend, "_recover_handle", new=AsyncMock(return_value=handle)):
            task = asyncio.create_task(self.backend.prepare_resume([self.row]))
            await self._cancel(task, started)
        handle.kill.assert_awaited_once()
        handle.close.assert_awaited_once()
        self.assertEqual(self.path.read_bytes(), before)

    async def test_terminate_cancel_preserves_signal_when_close_fails(self) -> None:
        started = asyncio.Event()

        async def kill():
            started.set()
            await asyncio.Event().wait()

        handle = SimpleNamespace(kill=AsyncMock(side_effect=kill), close=AsyncMock(side_effect=OSError("close failed")))
        task = asyncio.create_task(self.backend._terminate(handle))
        await self._cancel(task, started)
        handle.close.assert_awaited_once()

    async def test_ordinary_terminate_error_keeps_existing_close_error_behavior(self) -> None:
        handle = SimpleNamespace(
            kill=AsyncMock(side_effect=RuntimeError("kill failed")),
            close=AsyncMock(side_effect=OSError("close failed")),
        )
        with self.assertRaisesRegex(OSError, "close failed"):
            await self.backend._terminate(handle)


class SandboxRunnerCancellationTest(unittest.IsolatedAsyncioTestCase):
    async def _cancel_with_cleanup_error(self, *, stopped: bool, persistence_error: bool) -> None:
        lease, request = _execution()
        runner = SandboxEpisodeRunner(timeout_s=1)
        started = asyncio.Event()

        async def command(*_args, **_kwargs):
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                lease.runtime_handle.execution_stopped = stopped
                raise

        update = AsyncMock(side_effect=[None, OSError("reference unavailable") if persistence_error else None])
        collect = AsyncMock(side_effect=OSError("download unavailable"))
        with (
            patch("manager.sandbox_episode_runner.run_sandbox_command", new=command),
            patch.object(runner, "_update_reference", new=update),
            patch.object(runner, "_collect_result", new=collect),
        ):
            task = asyncio.create_task(runner.start(lease, request))
            await asyncio.wait_for(started.wait(), timeout=1)
            task.cancel("shutdown")
            with self.assertRaises(asyncio.CancelledError) as raised:
                await task
        self.assertEqual(raised.exception.args, ("shutdown",))
        self.assertTrue(lease.runtime_handle.retain)
        self.assertEqual(update.await_count, 2)
        self.assertEqual(update.await_args.args[2]["command_finished"], stopped)
        self.assertEqual(collect.await_count, int(stopped))

    async def test_unknown_writer_reference_error_preserves_cancellation(self) -> None:
        await self._cancel_with_cleanup_error(stopped=False, persistence_error=True)

    async def test_stopped_writer_reference_error_still_attempts_result_salvage(self) -> None:
        await self._cancel_with_cleanup_error(stopped=True, persistence_error=True)

    async def test_result_salvage_error_preserves_cancellation(self) -> None:
        await self._cancel_with_cleanup_error(stopped=True, persistence_error=False)

    async def test_repeated_cancel_during_reference_update_never_resumes_collection(self) -> None:
        lease, request = _execution()
        runner = SandboxEpisodeRunner(timeout_s=1)
        started, cleanup_started, release_cleanup, cleanup_done = (asyncio.Event() for _ in range(4))
        update_count = 0

        async def command(*_args, **_kwargs):
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                lease.runtime_handle.execution_stopped = True
                raise

        async def update(*_args):
            nonlocal update_count
            update_count += 1
            if update_count == 2:
                cleanup_started.set()
                try:
                    await release_cleanup.wait()
                finally:
                    cleanup_done.set()

        collect = AsyncMock()
        with (
            patch("manager.sandbox_episode_runner.run_sandbox_command", new=command),
            patch.object(runner, "_update_reference", new=update),
            patch.object(runner, "_collect_result", new=collect),
        ):
            task = asyncio.create_task(runner.start(lease, request))
            try:
                await asyncio.wait_for(started.wait(), timeout=1)
                task.cancel("shutdown")
                await asyncio.wait_for(cleanup_started.wait(), timeout=1)
                task.cancel("forced shutdown")
                with self.assertRaises(asyncio.CancelledError):
                    await task
                collect.assert_not_awaited()
            finally:
                release_cleanup.set()
                await asyncio.wait_for(cleanup_done.wait(), timeout=1)
        self.assertTrue(lease.runtime_handle.retain)

    async def test_repeated_cancel_during_result_salvage_never_returns_result(self) -> None:
        lease, request = _execution()
        runner = SandboxEpisodeRunner(timeout_s=1)
        started, cleanup_started, release_cleanup, cleanup_done = (asyncio.Event() for _ in range(4))

        async def command(*_args, **_kwargs):
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                lease.runtime_handle.execution_stopped = True
                raise

        async def collect(*_args):
            cleanup_started.set()
            try:
                await release_cleanup.wait()
                return object()
            finally:
                cleanup_done.set()

        with (
            patch("manager.sandbox_episode_runner.run_sandbox_command", new=command),
            patch.object(runner, "_update_reference", new=AsyncMock()),
            patch.object(runner, "_collect_result", new=collect),
        ):
            task = asyncio.create_task(runner.start(lease, request))
            try:
                await asyncio.wait_for(started.wait(), timeout=1)
                task.cancel("shutdown")
                await asyncio.wait_for(cleanup_started.wait(), timeout=1)
                task.cancel("forced shutdown")
                with self.assertRaises(asyncio.CancelledError):
                    await task
            finally:
                release_cleanup.set()
                await asyncio.wait_for(cleanup_done.wait(), timeout=1)
        self.assertTrue(lease.runtime_handle.retain)


class SandboxWorkerCancellationTest(unittest.IsolatedAsyncioTestCase):
    async def test_cancelled_runner_does_not_refill_evaluate_commit_or_acquire_next_lease(self) -> None:
        lease, _ = _execution()
        manager = SimpleNamespace(close_and_refill=AsyncMock(), is_data_exhausted=AsyncMock(return_value=True))
        pool = SimulationLeasePool(manager, pool_size=1)
        await pool._runtime.add_ready_lease((lease.agent_name, lease.agent_id), lease)
        await pool._runtime.mark_initial_load_done()
        pool.acquire = AsyncMock(wraps=pool.acquire)
        runner = SandboxEpisodeRunner(timeout_s=1)
        started = asyncio.Event()

        async def command(*_args, **_kwargs):
            started.set()
            await asyncio.Event().wait()

        data_manager = SimpleNamespace(
            create_session=lambda **kwargs: SimpleNamespace(session_id=kwargs["env_id"]),
            mark_environment_finished=AsyncMock(),
        )
        evaluator = SimpleNamespace(evaluate=AsyncMock())
        committer = SimpleNamespace(commit=AsyncMock())
        cfg = SimpleNamespace(
            job_id="job-1", mode="sandbox", agent_root="env", storage_type="sqlite", db_url="sqlite://unused.db",
            gateway_base_url="http://gateway/v1/sessions", llm_model="model", llm_temperature=0.0,
            max_steps=1, agent_start_timeout_s=1.0, max_workers=1, circuit_breaker_enabled=False,
            circuit_breaker_window=1, circuit_breaker_min_samples=1, circuit_breaker_failure_rate=1.0,
            circuit_breaker_timeout_rate=1.0, circuit_breaker_consecutive_timeouts=1,
        )
        workers = SimulationWorkerGroup(
            pool, data_manager, runner, cfg, evaluation_service=evaluator, reward_committer=committer,
        )
        with (
            patch("manager.sandbox_episode_runner.run_sandbox_command", new=command),
            patch.object(runner, "_update_reference", new=AsyncMock(side_effect=[None, OSError("disk unavailable")])),
            patch.object(runner, "_collect_result", new=AsyncMock()) as collect,
        ):
            task = asyncio.create_task(workers.run_all())
            await asyncio.wait_for(started.wait(), timeout=1)
            task.cancel("shutdown")
            with self.assertRaises(asyncio.CancelledError):
                await task
        manager.close_and_refill.assert_not_awaited()
        self.assertEqual(pool.acquire.await_count, 1)
        evaluator.evaluate.assert_not_awaited()
        committer.commit.assert_not_awaited()
        data_manager.mark_environment_finished.assert_not_awaited()
        collect.assert_not_awaited()
        self.assertEqual(workers._results, {})
        self.assertTrue(lease.runtime_handle.retain)


if __name__ == "__main__":
    unittest.main()
