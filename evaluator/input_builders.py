from __future__ import annotations

from typing import Any, Protocol

from evaluator.eval_types import EvalRequest, EvalSpec, Trajectory


class JudgeInputBuilder(Protocol):
    def build(
        self,
        *,
        request: EvalRequest,
        spec: EvalSpec,
        trajectory: Trajectory,
    ) -> dict[str, Any]:
        ...


class TrajectoryFinalAnswerBuilder:
    def build(
        self,
        *,
        request: EvalRequest,
        spec: EvalSpec,
        trajectory: Trajectory,
    ) -> dict[str, Any]:
        env_params = request.env_params or {}
        dataset = env_params.get("dataset") if isinstance(env_params.get("dataset"), dict) else {}
        variables = dict(spec.variables or {})
        return {
            "task": (
                env_params.get("task")
                or env_params.get("instruction")
                or env_params.get("prompt")
                or dataset.get("task")
                or dataset.get("instruction")
                or dataset.get("prompt")
                or dataset.get("query")
                or ""
            ),
            "rubric": spec.rubric or env_params.get("rubric") or dataset.get("rubric") or {},
            "trajectory": trajectory.compact(),
            "final_response": trajectory.final_response or "",
            "variables": variables,
            **variables,
        }


class FinalAnswerOnlyBuilder(TrajectoryFinalAnswerBuilder):
    def build(
        self,
        *,
        request: EvalRequest,
        spec: EvalSpec,
        trajectory: Trajectory,
    ) -> dict[str, Any]:
        data = super().build(request=request, spec=spec, trajectory=trajectory)
        data["trajectory"] = ""
        return data


def default_input_builders() -> dict[str, JudgeInputBuilder]:
    builder = TrajectoryFinalAnswerBuilder()
    return {
        "trajectory_final_answer": builder,
        "code_task_trajectory": builder,
        "tool_use_trajectory": builder,
        "safety_compliance": builder,
        "final_answer_only": FinalAnswerOnlyBuilder(),
    }
