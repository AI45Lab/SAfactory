import unittest
from unittest.mock import AsyncMock, patch

from clusters.rjob_cluster import RJobClusterBackend


class RJobPhaseTimeoutTests(unittest.IsolatedAsyncioTestCase):
    async def _wait_with_statuses(
        self,
        statuses,
        *,
        timeout_s=100,
        starting_timeout_s=None,
        running_timeout_s=None,
    ):
        clock = [0.0]

        async def advance(seconds):
            clock[0] += seconds

        backend = RJobClusterBackend()
        backend._get_status = AsyncMock(side_effect=statuses)
        with (
            patch("clusters.rjob_cluster.time.monotonic", side_effect=lambda: clock[0]),
            patch("clusters.rjob_cluster.asyncio.sleep", side_effect=advance),
        ):
            return await backend.wait_terminal(
                None,
                "test-job",
                timeout_s=timeout_s,
                poll_interval_s=5,
                starting_timeout_s=starting_timeout_s,
                running_timeout_s=running_timeout_s,
            )

    async def test_starting_timeout_is_independent(self):
        with self.assertRaisesRegex(
            TimeoutError, "stuck in Starting after 10.0s"
        ) as caught:
            await self._wait_with_statuses(
                ["Starting", "Starting", "Starting"],
                starting_timeout_s=10,
            )
        self.assertEqual(caught.exception.last_status, "Starting")

    async def test_running_timeout_is_independent(self):
        with self.assertRaisesRegex(
            TimeoutError, "exceeded Running limit after 10.0s"
        ) as caught:
            await self._wait_with_statuses(
                ["Running", "Running", "Running"],
                running_timeout_s=10,
            )
        self.assertEqual(caught.exception.last_status, "Running")

    async def test_starting_time_does_not_consume_running_budget(self):
        status, polls = await self._wait_with_statuses(
            ["Starting"] * 6 + ["Running", "Running", "Succeeded"],
            timeout_s=20,
            starting_timeout_s=60,
            running_timeout_s=10,
        )
        self.assertEqual((status, polls), ("Succeeded", 9))

    async def test_legacy_total_timeout_remains_default(self):
        with self.assertRaisesRegex(TimeoutError, "timed out after 10.0s"):
            await self._wait_with_statuses(
                ["Pending", "Pending", "Pending"],
                timeout_s=10,
            )
