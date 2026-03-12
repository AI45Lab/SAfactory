from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from core.types.base import StepOutput

if TYPE_CHECKING:
    from env.qagym.qa_env import QAGym


class AttackStrategy(ABC):
    def __init__(self, env: "QAGym"):
        self.env = env

    @abstractmethod
    def setup(self) -> None:
        # REFACTOR: summary common code between different attack methods.
        raise NotImplementedError

    @abstractmethod
    def get_prompt(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def step(self, action: str) -> StepOutput:
        raise NotImplementedError
