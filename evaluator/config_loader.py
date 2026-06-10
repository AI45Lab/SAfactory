from __future__ import annotations

from pathlib import Path
from typing import Any
import os

import yaml

from evaluator.eval_types import (
    EvalSpec,
    EvaluatorAgentPoolSpec,
    EvaluatorAgentSpec,
    coerce_eval_spec,
    normalize_evaluator_agent_type,
)


def load_evaluation_config(path: str | Path) -> dict[str, Any]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return dict(data.get("evaluation") or data)


def parse_default_specs(config: dict[str, Any]) -> list[EvalSpec]:
    return [coerce_eval_spec(item) for item in config.get("default_specs") or []]


def parse_evaluator_pool_specs(config: dict[str, Any]) -> list[EvaluatorAgentPoolSpec]:
    pools: list[EvaluatorAgentPoolSpec] = []
    for raw_pool in config.get("evaluator_agent_pools") or []:
        pool_id = str(raw_pool["pool_id"])
        members = []
        for raw_member in raw_pool.get("members") or []:
            data = dict(raw_member)
            data.setdefault("pool_id", pool_id)
            data["agent_type"] = normalize_evaluator_agent_type(data.get("agent_type"))
            data["mounts"] = [_expand_mount(mount) for mount in data.get("mounts") or []]
            if data.get("runtime_dir"):
                data["runtime_dir"] = _expand_path(str(data["runtime_dir"]))
            if data.get("agent_type") == "codex_cli" and data.get("workdir"):
                data["workdir"] = _expand_path(str(data["workdir"]))
            if data.get("agent_type") == "docker_container":
                if not str(data.get("image") or "").strip():
                    raise ValueError(f"evaluator pool {pool_id!r} member requires image for docker_container")
                if not str(data.get("command_template") or "").strip():
                    raise ValueError(f"evaluator pool {pool_id!r} member requires command_template for docker_container")
            members.append(EvaluatorAgentSpec(**data))
        pools.append(
            EvaluatorAgentPoolSpec(
                pool_id=pool_id,
                members=members,
                selection_policy=str(raw_pool.get("selection_policy") or "least_busy"),
                acquire_timeout_s=float(raw_pool.get("acquire_timeout_s") or 60.0),
                max_queue_size=int(raw_pool.get("max_queue_size") or 1024),
            )
        )
    return pools


def parse_judge_definition_dirs(config: dict[str, Any]) -> list[str]:
    return [_expand_path(str(path)) for path in config.get("judge_definition_dirs") or []]


def _expand_mount(mount: dict[str, Any]) -> dict[str, str]:
    return {
        str(key): _expand_path(str(value)) if key == "source" else str(value)
        for key, value in dict(mount).items()
    }


def _expand_path(value: str) -> str:
    return os.path.expanduser(os.path.expandvars(value))
