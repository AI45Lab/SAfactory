from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from core.runtime_metadata import SAFACTORY_INTERNAL_ENV_KEY
from evaluator.eval_types import EvalMethod, EvalSpec

log = logging.getLogger("evaluator.markdown_eval_resolver")

TASK_NAME_KEYS = ("task_name", "task_id", "case_uid", "id", "name")

DEFAULT_MARKDOWN_JUDGE_TEMPLATE = """\
{{ eval_task_body }}

## Public Task
{{ task }}

## Rubric
{{ rubric }}

## Trajectory
{{ trajectory }}

## Final Response
{{ final_response }}

Please output JSON only: {"score": 0-10, "reason": "...", "passed": true/false}
"""


@dataclass(frozen=True)
class MarkdownEvalTask:
    path: Path
    metadata: dict[str, Any]
    body: str

    @property
    def eval_type(self) -> str:
        return str(self.metadata.get("eval_type") or self.metadata.get("type") or "llm").strip().lower()

    @property
    def score_min(self) -> float:
        return float(self.metadata.get("score_min", 0.0))

    @property
    def score_max(self) -> float:
        return float(self.metadata.get("score_max", 10.0))


class MarkdownEvalTaskResolver:
    """Resolve env-local markdown evaluation tasks after rollout.

    The resolver uses metadata injected by the YAML loader:
    env/<name>/<config>.yaml + datasets/<dataset>.jsonl
      -> env/<name>/eval_tasks/<dataset>/<task_name>.md
    """

    def __init__(
        self,
        *,
        eval_task_dir_name: str = "eval_tasks",
        strict: bool = False,
    ) -> None:
        self.eval_task_dir_name = str(eval_task_dir_name or "eval_tasks").strip() or "eval_tasks"
        self.strict = bool(strict)
        self._cache: dict[Path, MarkdownEvalTask] = {}

    def resolve_specs(self, env_params: dict[str, Any] | None) -> list[EvalSpec]:
        env_params = env_params if isinstance(env_params, dict) else {}
        resolved_path = self.resolve_path(env_params)
        if resolved_path is None:
            return []
        task = self._load(resolved_path)
        specs = self._task_to_specs(task, env_params=env_params)
        log.info(
            "EVAL RESOLVER markdown task loaded: path=%s eval_type=%s specs=%s",
            task.path,
            task.eval_type,
            [
                {
                    "eval_id": spec.eval_id,
                    "method": spec.method.value,
                    "weight": spec.weight,
                    "requires_container": spec.requires_container,
                }
                for spec in specs
            ],
        )
        return specs

    def resolve_path(self, env_params: dict[str, Any]) -> Path | None:
        metadata = env_params.get(SAFACTORY_INTERNAL_ENV_KEY)
        metadata = metadata if isinstance(metadata, dict) else {}
        config_dir = str(metadata.get("config_dir") or "").strip()
        dataset_name = str(metadata.get("dataset_name") or "").strip()
        if not config_dir or not dataset_name:
            log.info(
                "EVAL RESOLVER skipped markdown lookup: missing config_dir/dataset_name metadata"
            )
            return None

        dataset = env_params.get("dataset")
        dataset = dataset if isinstance(dataset, dict) else {}
        task_name = _first_present(dataset, TASK_NAME_KEYS) or _first_present(env_params, TASK_NAME_KEYS)
        if not task_name:
            log.info(
                "EVAL RESOLVER skipped markdown lookup: dataset=%s has no task_name/task_id/case_uid/id/name",
                dataset_name,
            )
            return None

        eval_dir = Path(config_dir) / self.eval_task_dir_name / dataset_name
        candidates = self._candidate_paths(eval_dir=eval_dir, task_name=str(task_name), dataset=dataset)
        log.info(
            "EVAL RESOLVER lookup: config_dir=%s dataset=%s task=%s candidates=%s",
            config_dir,
            dataset_name,
            task_name,
            [str(path) for path in candidates],
        )
        for path in candidates:
            if path.is_file():
                log.info("EVAL RESOLVER matched markdown eval task: %s", path)
                return path

        message = (
            "markdown eval task not found: "
            f"config_dir={config_dir} dataset={dataset_name} task={task_name} candidates={candidates}"
        )
        if self.strict:
            raise FileNotFoundError(message)
        log.info("%s; fallback to env_params/default specs", message)
        return None

    def _candidate_paths(self, *, eval_dir: Path, task_name: str, dataset: dict[str, Any]) -> list[Path]:
        candidates: list[Path] = []
        eval_dir_resolved = eval_dir.resolve(strict=False)
        eval_task_file = dataset.get("eval_task_file")
        if eval_task_file:
            path = (eval_dir / str(eval_task_file)).resolve(strict=False)
            if _is_relative_to(path, eval_dir_resolved):
                candidates.append(path)
        exact = (eval_dir / f"{task_name}.md").resolve(strict=False)
        if _is_relative_to(exact, eval_dir_resolved):
            candidates.append(exact)
        sanitized = _safe_filename(task_name)
        if sanitized != task_name:
            candidates.append((eval_dir / f"{sanitized}.md").resolve(strict=False))

        unique: list[Path] = []
        seen: set[str] = set()
        for candidate in candidates:
            key = str(candidate)
            if key not in seen:
                unique.append(candidate)
                seen.add(key)
        return unique

    def _load(self, path: Path) -> MarkdownEvalTask:
        path = path.resolve(strict=False)
        cached = self._cache.get(path)
        if cached is not None:
            return cached
        text = path.read_text(encoding="utf-8")
        metadata, body = parse_markdown_eval_task(text)
        task = MarkdownEvalTask(path=path, metadata=metadata, body=body)
        self._cache[path] = task
        return task

    def _task_to_specs(self, task: MarkdownEvalTask, *, env_params: dict[str, Any]) -> list[EvalSpec]:
        raw_specs = task.metadata.get("specs")
        if isinstance(raw_specs, dict):
            raw_specs = [raw_specs]
        if isinstance(raw_specs, list) and raw_specs:
            return [
                self._build_spec(raw_spec, task=task, env_params=env_params, index=index)
                for index, raw_spec in enumerate(raw_specs, start=1)
            ]

        eval_type = task.eval_type
        if eval_type == "agent":
            return [self._build_agent_spec(task=task, env_params=env_params, weight=1.0)]
        if eval_type in {"rule", "rule_eval", "rule_evaluator"}:
            return [self._build_rule_spec(task=task, env_params=env_params, weight=1.0)]
        if eval_type not in {"", "llm", "llm_judge", "judge"}:
            raise ValueError(f"unsupported markdown eval_type: {eval_type!r}")
        return [self._build_llm_spec(task=task, env_params=env_params, weight=1.0)]

    def _build_spec(
        self,
        raw_spec: Any,
        *,
        task: MarkdownEvalTask,
        env_params: dict[str, Any],
        index: int,
    ) -> EvalSpec:
        if not isinstance(raw_spec, dict):
            raise ValueError(f"markdown eval spec #{index} must be a mapping: {task.path}")
        data = dict(raw_spec)
        method = _normalize_method(data.pop("method", data.pop("type", None)))
        data.setdefault("eval_id", f"{task.path.stem}_{method.value}_{index}")
        return self._coerce_spec_data(data, method=method, task=task, env_params=env_params)

    def _build_llm_spec(
        self,
        *,
        task: MarkdownEvalTask,
        env_params: dict[str, Any],
        weight: float,
    ) -> EvalSpec:
        data = {
            "eval_id": f"{task.path.stem}_llm",
            "judge_model": task.metadata.get("judge_model"),
            "judge_id": task.metadata.get("judge_id"),
            "judge_prompt_template": task.metadata.get("judge_prompt_template") or DEFAULT_MARKDOWN_JUDGE_TEMPLATE,
            "input_builder": task.metadata.get("input_builder") or "trajectory_final_answer",
            "output_parser": task.metadata.get("output_parser") or "json_score_reason",
            "weight": weight,
        }
        return self._coerce_spec_data(data, method=EvalMethod.LLM_JUDGE, task=task, env_params=env_params)

    def _build_agent_spec(
        self,
        *,
        task: MarkdownEvalTask,
        env_params: dict[str, Any],
        weight: float,
    ) -> EvalSpec:
        data = {
            "eval_id": f"{task.path.stem}_agent",
            "evaluator_agent_id": task.metadata.get("evaluator_agent_id"),
            "evaluator_pool_id": task.metadata.get("evaluator_pool_id"),
            "evaluator_agent_type": task.metadata.get("evaluator_agent_type") or task.metadata.get("agent_type"),
            "evaluator_base_agents": task.metadata.get("evaluator_base_agents") or [],
            "evaluator_required_capabilities": set(task.metadata.get("evaluator_required_capabilities") or []),
            "evaluator_task_template": task.metadata.get("evaluator_task_template") or task.body,
            "evaluator_task_input": {
                "eval_task_body": task.body,
                "eval_task_metadata": task.metadata,
                "eval_task_path": str(task.path),
            },
            "target_access_mode": task.metadata.get("target_access_mode") or "snapshot",
            "weight": weight,
        }
        return self._coerce_spec_data(data, method=EvalMethod.AGENT_EVAL, task=task, env_params=env_params)

    def _build_rule_spec(
        self,
        *,
        task: MarkdownEvalTask,
        env_params: dict[str, Any],
        weight: float,
    ) -> EvalSpec:
        data = {
            "eval_id": f"{task.path.stem}_rule",
            "rule_evaluator": task.metadata.get("rule_evaluator"),
            "weight": weight,
        }
        return self._coerce_spec_data(data, method=EvalMethod.RULE_EVALUATOR, task=task, env_params=env_params)

    def _coerce_spec_data(
        self,
        data: dict[str, Any],
        *,
        method: EvalMethod,
        task: MarkdownEvalTask,
        env_params: dict[str, Any],
    ) -> EvalSpec:
        data = dict(data)
        data["method"] = method
        data.setdefault("score_min", task.score_min)
        data.setdefault("score_max", task.score_max)
        data.setdefault("rubric", task.metadata.get("rubric") or {})
        variables = dict(data.get("variables") or {})
        variables.update(
            {
                "eval_task_body": task.body,
                "eval_task_path": str(task.path),
                "eval_task_metadata": task.metadata,
                "dataset": env_params.get("dataset") if isinstance(env_params.get("dataset"), dict) else {},
            }
        )
        data["variables"] = variables
        if data.get("judge_id") and data.get("judge_prompt_template"):
            data.pop("judge_id", None)
        return EvalSpec(**_filter_eval_spec_fields(data))


def parse_markdown_eval_task(text: str) -> tuple[dict[str, Any], str]:
    if not text.startswith("---"):
        return {}, text.strip()
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", text, flags=re.DOTALL)
    if not match:
        return {}, text.strip()
    metadata = yaml.safe_load(match.group(1)) or {}
    if not isinstance(metadata, dict):
        raise ValueError("markdown eval frontmatter must be a mapping")
    return dict(metadata), match.group(2).strip()


def _normalize_method(value: Any) -> EvalMethod:
    text = str(value or "llm").strip().lower()
    aliases = {
        "llm": EvalMethod.LLM_JUDGE,
        "llm_judge": EvalMethod.LLM_JUDGE,
        "judge": EvalMethod.LLM_JUDGE,
        "agent": EvalMethod.AGENT_EVAL,
        "agent_eval": EvalMethod.AGENT_EVAL,
        "rule": EvalMethod.RULE_EVALUATOR,
        "rule_eval": EvalMethod.RULE_EVALUATOR,
        "rule_evaluator": EvalMethod.RULE_EVALUATOR,
    }
    try:
        return aliases[text]
    except KeyError as exc:
        raise ValueError(f"unsupported markdown eval method: {value!r}") from exc


def _filter_eval_spec_fields(data: dict[str, Any]) -> dict[str, Any]:
    allowed = set(EvalSpec.__dataclass_fields__)
    return {key: value for key, value in data.items() if key in allowed}


def _first_present(data: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = data.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.:-]+", "_", value).strip("._") or "task"


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False
