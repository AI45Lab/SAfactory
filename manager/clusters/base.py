from __future__ import annotations

from abc import ABC, abstractmethod

from ..binding_plan import BindingPlan
from ..types import ClusterRegistry


class ClusterBackend(ABC):
    """
    Strategy interface:
      - RemoteRayJobBackend: creates RayJobs and polls head_ip
      - LocalHTTPBackend: does not create RayJobs; routes to localhost
    """

    @abstractmethod
    async def start(self, plan: BindingPlan) -> ClusterRegistry:
        raise NotImplementedError

    @abstractmethod
    async def close(self) -> None:
        raise NotImplementedError
