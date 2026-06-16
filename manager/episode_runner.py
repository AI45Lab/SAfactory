from __future__ import annotations

import asyncio
from typing import Dict, Protocol

from .types import SimulationAgentLease, SimulationStartRequest, SimulationStartResult


class EpisodeRunner(Protocol):
    async def start(
        self,
        lease: SimulationAgentLease,
        request: SimulationStartRequest,
    ) -> SimulationStartResult:
        ...

    async def close(self) -> None:
        ...


class EpisodeRunnerDispatcher:
    def __init__(self, runners: Dict[str, EpisodeRunner]) -> None:
        self._runners = {str(name).strip().lower(): runner for name, runner in runners.items()}

    async def start(
        self,
        lease: SimulationAgentLease,
        request: SimulationStartRequest,
    ) -> SimulationStartResult:
        runtime = str(getattr(lease, "runtime", "docker") or "docker").strip().lower()
        runner = self._runners.get(runtime)
        if runner is None:
            raise RuntimeError(f"Unsupported episode runtime: {runtime!r}")
        return await runner.start(lease, request)

    async def close(self) -> None:
        await asyncio.gather(
            *[runner.close() for runner in self._runners.values()],
            return_exceptions=True,
        )
