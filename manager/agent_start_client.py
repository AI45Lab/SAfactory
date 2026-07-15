from __future__ import annotations

from .docker_episode_runner import DockerEpisodeRunner
from .episode_runner import EpisodeRunnerDispatcher
from .rjob_episode_runner import RJobEpisodeRunner
from .sandbox_episode_runner import SandboxEpisodeRunner


class AgentStartClient(EpisodeRunnerDispatcher):
    """
    Compatibility facade for episode execution.

    Older code imports AgentStartClient. It now dispatches by
    SimulationAgentLease.runtime while preserving the same start/close methods.
    """

    def __init__(self, *, timeout_s: float) -> None:
        super().__init__(
            {
                "docker": DockerEpisodeRunner(timeout_s=timeout_s),
                "rjob": RJobEpisodeRunner(timeout_s=timeout_s),
                "sandbox": SandboxEpisodeRunner(timeout_s=timeout_s),
            }
        )
