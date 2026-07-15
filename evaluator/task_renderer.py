from __future__ import annotations

from evaluator.eval_types import AgentEvalTask, EvalSpec, safe_dumps, to_jsonable
from evaluator.templating import render_template


DEFAULT_AGENT_EVAL_TEMPLATE = """\
你是一个评测 agent。请根据下面的结构化评测任务，对目标 agent 的执行结果进行评测。

评测任务：
{{ task_json }}

被评测 agent 的位置：
- container_alias: {{ target.container_alias }}
- container_name: {{ target.container_name }}
- container_id: {{ target.container_id }}
- runtime: {{ target.runtime }}
- resource_id: {{ target.resource_id }}
- access_mode: {{ target.access_mode }}
- sandbox_endpoint: {{ target.sandbox_endpoint }}
- sandbox_headers: {{ target.sandbox_headers }}
- exec_hint: {{ target.exec_hint }}

请只输出 JSON，格式必须满足：
{{ output_contract_json }}
"""


class EvaluatorTaskRenderer:
    def __init__(self, *, template_registry: dict[str, str] | None = None) -> None:
        self.template_registry = dict(template_registry or {})

    def render(
        self,
        *,
        task: AgentEvalTask,
        spec: EvalSpec,
    ) -> str:
        template = (
            spec.evaluator_task_template
            or self.template_registry.get(str(spec.evaluator_task_template_id or ""))
            or DEFAULT_AGENT_EVAL_TEMPLATE
        )
        payload = {
            "task": task,
            "target": task.target,
            "task_json": safe_dumps(task),
            "output_contract_json": safe_dumps(task.output_contract),
            **task.variables,
            **task.task_input,
        }
        return render_template(template, to_jsonable(payload))
