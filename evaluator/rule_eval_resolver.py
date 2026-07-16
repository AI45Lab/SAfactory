from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from core.runtime_metadata import SAFACTORY_INTERNAL_ENV_KEY
from evaluator.eval_types import EvalMethod, EvalSpec

log = logging.getLogger("evaluator.rule_eval_resolver")


def resolve_rule_eval_specs(
    env_params: dict[str, Any] | None,
    *,
    agent_name: str = "",
    env_root: str | Path = "env",
) -> list[EvalSpec]:
    env_params = env_params if isinstance(env_params, dict) else {}
    locator = resolve_rule_evaluator_locator(
        env_params,
        agent_name=agent_name,
        env_root=env_root,
    )
    if not locator:
        return []

    evaluation = env_params.get("evaluation")
    evaluation = evaluation if isinstance(evaluation, dict) else {}
    timeout_s = _float_or_default(
        evaluation.get("rule_evaluator_timeout_s", env_params.get("rule_evaluator_timeout_s")),
        60.0,
    )
    resolved_agent = str(agent_name or env_params.get("task_family") or "env").strip() or "env"
    spec = EvalSpec(
        eval_id=str(evaluation.get("rule_evaluator_eval_id") or f"{resolved_agent}_rule"),
        method=EvalMethod.RULE_EVALUATOR,
        timeout_s=timeout_s,
        rule_evaluator=locator,
    )
    log.info(
        "EVAL RESOLVER rule evaluator: agent=%s locator=%s eval_id=%s timeout_s=%.2f",
        resolved_agent,
        locator,
        spec.eval_id,
        spec.timeout_s,
    )
    return [spec]


def resolve_rule_evaluator_locator(
    env_params: dict[str, Any] | None,
    *,
    agent_name: str = "",
    env_root: str | Path = "env",
) -> str | None:
    env_params = env_params if isinstance(env_params, dict) else {}
    configured = _rule_evaluator_config(env_params)
    if configured is False:
        return None
    if isinstance(configured, str) and configured.lower() not in {
        "",
        "1",
        "true",
        "yes",
        "default",
        "auto",
    }:
        return configured

    default_path = _default_rule_evaluator_path(
        env_params,
        agent_name=agent_name,
        env_root=env_root,
    )
    return f"file:{default_path}" if default_path is not None else None


def _rule_evaluator_config(env_params: dict[str, Any]) -> str | bool:
    evaluation = env_params.get("evaluation")
    evaluation = evaluation if isinstance(evaluation, dict) else {}
    value = (
        evaluation.get("rule_evaluator")
        if "rule_evaluator" in evaluation
        else evaluation.get("rule_evaluator_path", env_params.get("rule_evaluator"))
    )
    if value is False:
        return False
    if value is True:
        return "auto"
    text = str(value or "").strip()
    if text.lower() in {"false", "0", "no", "off", "disabled"}:
        return False
    return text


def _default_rule_evaluator_path(
    env_params: dict[str, Any],
    *,
    agent_name: str,
    env_root: str | Path,
) -> Path | None:
    metadata = env_params.get(SAFACTORY_INTERNAL_ENV_KEY)
    metadata = metadata if isinstance(metadata, dict) else {}
    config_dir = str(metadata.get("config_dir") or "").strip()
    candidates: list[Path] = []
    if config_dir:
        candidates.append(Path(config_dir).expanduser() / "rule_evaluator.py")

    resolved_agent = str(agent_name or env_params.get("task_family") or "").strip()
    if resolved_agent:
        candidates.append(Path(env_root).expanduser() / resolved_agent / "rule_evaluator.py")

    seen: set[str] = set()
    for candidate in candidates:
        path = candidate.resolve(strict=False)
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        if path.is_file():
            return path
    return None


def _float_or_default(value: Any, default: float) -> float:
    try:
        if value is None or value == "":
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)
