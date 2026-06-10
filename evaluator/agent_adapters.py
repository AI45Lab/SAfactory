from __future__ import annotations

from typing import Any, Protocol

from evaluator.eval_types import AgentEvalTask, EvaluatorAgentLease, EvaluatorAgentSpec, EvaluatorRunResult, safe_dumps
from evaluator.templating import render_template


class EvaluatorAgentRunner(Protocol):
    async def write_text(
        self,
        *,
        lease: Any,
        path: str,
        text: str,
        timeout_s: float,
    ) -> None:
        ...


class EvaluatorAgentAdapter(Protocol):
    base_agent: str

    async def prepare_inputs(
        self,
        *,
        lease: EvaluatorAgentLease,
        task: AgentEvalTask,
        rendered_prompt: str,
        runner: EvaluatorAgentRunner,
    ) -> None:
        ...

    def build_command(
        self,
        *,
        spec: EvaluatorAgentSpec,
        task: AgentEvalTask,
    ) -> str:
        ...

    def normalize_output(
        self,
        *,
        run_result: EvaluatorRunResult,
        spec: EvaluatorAgentSpec,
    ) -> EvaluatorRunResult:
        ...


class TemplateCommandEvaluatorAdapter:
    base_agent = "template"

    async def prepare_inputs(
        self,
        *,
        lease: EvaluatorAgentLease,
        task: AgentEvalTask,
        rendered_prompt: str,
        runner: EvaluatorAgentRunner,
    ) -> None:
        timeout_s = min(task.timeout_s, 30.0)
        await runner.write_text(
            lease=lease,
            path=lease.spec.task_input_path,
            text=safe_dumps(task),
            timeout_s=timeout_s,
        )
        await runner.write_text(
            lease=lease,
            path=lease.spec.prompt_path,
            text=rendered_prompt,
            timeout_s=timeout_s,
        )

    def build_command(
        self,
        *,
        spec: EvaluatorAgentSpec,
        task: AgentEvalTask,
    ) -> str:
        return render_template(
            spec.command_template,
            {
                "workdir": spec.workdir,
                "task_input_path": spec.task_input_path,
                "prompt_path": spec.prompt_path,
                "result_path": spec.result_path,
                "spec": spec,
                "model": spec.model or task.evaluation_model or "",
                "evaluation_model": spec.model or task.evaluation_model or "",
                "gateway_base_url": spec.gateway_base_url or task.gateway_base_url or "",
                "target": task.target,
                "task": task,
            },
        )

    def normalize_output(
        self,
        *,
        run_result: EvaluatorRunResult,
        spec: EvaluatorAgentSpec,
    ) -> EvaluatorRunResult:
        run_result.result_path = run_result.result_path or spec.result_path
        return run_result


class CodexEvaluatorAdapter(TemplateCommandEvaluatorAdapter):
    base_agent = "codex"


class ClaudeCodeEvaluatorAdapter(TemplateCommandEvaluatorAdapter):
    base_agent = "claude_code"


class EvaluatorAgentAdapterRegistry:
    def __init__(self, adapters: list[EvaluatorAgentAdapter] | None = None) -> None:
        adapters = adapters or [CodexEvaluatorAdapter(), ClaudeCodeEvaluatorAdapter()]
        self._adapters = {adapter.base_agent: adapter for adapter in adapters}

    def get(self, base_agent: str) -> EvaluatorAgentAdapter:
        try:
            return self._adapters[base_agent]
        except KeyError as exc:
            raise KeyError(f"unknown evaluator base_agent: {base_agent}") from exc
