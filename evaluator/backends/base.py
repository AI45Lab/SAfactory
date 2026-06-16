from __future__ import annotations

from typing import Any, Protocol

from evaluator.eval_types import EvalRequest, EvalResult, EvalSpec, Trajectory


class EvaluationBackend(Protocol):
    async def evaluate(
        self,
        *,
        request: EvalRequest,
        spec: EvalSpec,
        trajectory: Trajectory,
    ) -> EvalResult:
        ...


class LLMJudgeClient(Protocol):
    async def complete(
        self,
        *,
        messages: list[dict[str, Any]],
        model: str,
        timeout_s: float,
        request: EvalRequest | None = None,
        eval_id: str | None = None,
    ) -> str:
        ...


class ContainerExecutor(Protocol):
    async def exec(
        self,
        *,
        lease: Any,
        command: str,
        timeout_s: float,
    ) -> str:
        ...

    async def read_file(
        self,
        *,
        lease: Any,
        path: str,
        timeout_s: float,
    ) -> str:
        ...
