import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from clusters.rjob_cluster import RJobClusterBackend
from manager.rjob_episode_runner import RJobEpisodeRunner
from manager.types import SimulationAgentLease, SimulationStartRequest


def test_cancellation_during_submit_waits_and_cleans_remote_job():
    async def scenario():
        runner = RJobEpisodeRunner(timeout_s=10)
        submit_started = asyncio.Event()
        finish_submit = asyncio.Event()
        job_name = "tmax-model-session"

        async def submit_job(_client, _job, _kwargs):
            submit_started.set()
            await finish_submit.wait()
            return job_name

        runner._cluster.client = Mock(return_value=object())
        runner._cluster.resolve_gateway_base_url = Mock(
            return_value="http://gateway/v1/sessions"
        )
        runner._cluster.build_job_name = Mock(return_value=job_name)
        runner._cluster.build_job = Mock(return_value=object())
        runner._cluster.submit_kwargs = Mock(return_value={})
        runner._cluster.submit_job = submit_job
        runner._cluster.cleanup_job = AsyncMock()

        lease = SimulationAgentLease(
            agent_name="tmax",
            agent_id="session-id",
            group_id="group-id",
            image="image:latest",
            row_id=1,
            runtime="rjob",
            runtime_config={"cleanup_on_finish": True},
            run_command="true",
        )
        request = SimulationStartRequest(
            job_id="job-id",
            session_id="session-id",
            agent_name="tmax",
            agent_id="session-id",
            group_id="group-id",
            gateway_base_url="http://gateway/v1/sessions",
            model="model",
            temperature=1.0,
            max_steps=1,
            storage_type="sqlite",
        )

        task = asyncio.create_task(runner.start(lease, request))
        await submit_started.wait()
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()

        finish_submit.set()
        with pytest.raises(asyncio.CancelledError):
            await task

        runner._cluster.cleanup_job.assert_awaited_once()
        assert runner._cluster.cleanup_job.await_args.args[1] == job_name

    asyncio.run(scenario())


def test_normal_cleanup_waits_for_remote_job_to_disappear():
    async def scenario():
        backend = RJobClusterBackend()
        backend.force_cleanup_job = AsyncMock(return_value=True)
        client = object()

        await backend.cleanup_job(
            client,
            "test-job",
            cfg={
                "cleanup_on_finish": True,
                "cleanup_timeout_s": 12,
                "cleanup_poll_interval_s": 0.5,
            },
            terminal_status="Succeeded",
        )

        backend.force_cleanup_job.assert_awaited_once_with(
            client,
            "test-job",
            timeout_s=12.0,
            poll_interval_s=0.5,
        )

    asyncio.run(scenario())
