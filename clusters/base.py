from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from manager.binding_plan import BindingPlan


class ClusterBackend(ABC):
    """
    Strategy interface for the current OpenClaw Docker workflow.
    """

    @abstractmethod
    async def start(self, plan: BindingPlan) -> None:
        raise NotImplementedError

    @abstractmethod
    async def close(self) -> None:
        raise NotImplementedError
