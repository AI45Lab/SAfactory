from __future__ import annotations

import logging
import math
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from .types import SimulationRunConfig

log = logging.getLogger("manager.simulation_config")


def load_simulation_run_config(args: Any) -> SimulationRunConfig:
    pool_size, warm_pool_size, startup_submit_count, followup_submit_batch = derive_pool_sizing(
        configured_pool_size=int(args.pool_size),
        pool_size_override=0,
        multiplier=float(args.multiplier),
    )

    mode = str(args.mode or "docker").strip().lower()
    if mode != "docker":
        raise ValueError(f"Only docker mode is supported by the OpenClaw workflow; got {mode!r}")

    job_id = str(args.job_id or "").strip() or uuid.uuid4().hex
    max_workers = int(args.max_workers) if int(args.max_workers or 0) > 0 else None

    _validate_gateway_route_key(str(args.llm_model))
    agent_config = getattr(args, "agent_config", None)
    agent_start_config = getattr(args, "agent_start_config", None)

    return SimulationRunConfig(
        job_id=job_id,
        exp_config_path=str(args.exp_config),
        agent_root=str(args.agent_root),
        agent_config=None if agent_config is None else str(agent_config),
        agent_start_config=None if agent_start_config is None else str(agent_start_config),
        storage_type=str(args.storage_type),
        db_url=str(args.db_path),
        pool_size=pool_size,
        warm_pool_size=warm_pool_size,
        startup_submit_count=startup_submit_count,
        followup_submit_batch=followup_submit_batch,
        mode=mode,
        gateway_base_url=str(args.gateway_base_url).rstrip("/"),
        llm_model=str(args.llm_model),
        llm_temperature=float(args.llm_temperature),
        max_steps=int(args.max_steps),
        agent_start_timeout_s=float(args.agent_start_timeout_s),
        docker_bin=str(args.docker_bin or "docker"),
        docker_pull_policy=str(args.docker_pull_policy or "never").strip().lower(),
        docker_startup_concurrency=max(1, int(args.docker_startup_concurrency or 1)),
        max_workers=max_workers,
        agent_runtime=str(args.agent_runtime),
        rebuild_table=bool(args.rebuild_table),
        enable_buffer=bool(args.enable_buffer),
        buffer_size=int(args.buffer_size),
        flush_interval=float(args.flush_interval),
        rl_group_size=int(args.rl_group_size),
        rl_epoch=max(1, int(args.rl_epoch)),
        evaluation_enabled=bool(args.evaluation_enabled),
    )


def derive_pool_sizing(
    configured_pool_size: int,
    pool_size_override: int,
    multiplier: float,
) -> Tuple[int, int, int, int]:
    base_pool_size = int(configured_pool_size or 1)
    if int(pool_size_override) > 0:
        base_pool_size = int(pool_size_override)

    normalized_multiplier = float(multiplier) if float(multiplier) > 0.0 else 1.2
    warm_pool_size = math.ceil(base_pool_size * normalized_multiplier)
    startup_submit_count = max(base_pool_size * 2, warm_pool_size)
    followup_submit_batch = max(1, base_pool_size)
    return base_pool_size, warm_pool_size, startup_submit_count, followup_submit_batch


def build_manager_runtime_config(cfg: SimulationRunConfig) -> Dict[str, Any]:
    return {
        "mode": cfg.mode,
        "pool_size": int(cfg.warm_pool_size),
        "database": {
            "driver": cfg.storage_type,
            "sqlite_path": cfg.db_url,
        },
        "cluster": {
            "docker": {
                "bin": cfg.docker_bin,
                "pull_policy": cfg.docker_pull_policy,
                "startup_concurrency": int(cfg.docker_startup_concurrency),
            },
            "env_types": load_agent_start_config(cfg.agent_start_config),
        },
    }


def load_agent_start_config(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}

    cfg_path = Path(path).expanduser()
    if not cfg_path.is_absolute():
        cfg_path = Path.cwd() / cfg_path

    cfg = load_yaml_file(str(cfg_path))
    if "agents" in cfg:
        agents = cfg.get("agents")
        if not isinstance(agents, dict):
            raise ValueError(f"agents must be a mapping in {cfg_path}")
        return {
            str(agent_name): {"docker": _normalize_agent_start_docker(agent_name, spec, cfg_path)}
            for agent_name, spec in agents.items()
        }

    agent_name = str(cfg.get("agent_name") or "").strip()
    if not agent_name:
        raise ValueError(f"agent start config requires agent_name or agents mapping: {cfg_path}")
    return {
        agent_name: {
            "docker": _normalize_agent_start_docker(agent_name, cfg, cfg_path),
        }
    }


def _normalize_agent_start_docker(agent_name: Any, spec: Any, cfg_path: Path) -> Dict[str, Any]:
    if not isinstance(spec, dict):
        raise ValueError(f"agent start config for {agent_name!r} must be a mapping in {cfg_path}")

    container = spec.get("container")
    if container is None:
        container = spec
    if not isinstance(container, dict):
        raise ValueError(f"container config for {agent_name!r} must be a mapping in {cfg_path}")

    docker: Dict[str, Any] = {}
    _copy_non_empty(container, docker, "workdir")
    _copy_non_empty(container, docker, "idle_command")
    _copy_non_empty(container, docker, "run_command")
    _copy_non_empty(container, docker, "network")

    if "env" in container:
        env = container.get("env") or {}
        if not isinstance(env, dict):
            raise ValueError(f"container.env for {agent_name!r} must be a mapping in {cfg_path}")
        docker["env"] = {str(key): str(value) for key, value in env.items()}

    mounts = container.get("mounts", container.get("volumes", [])) or []
    if isinstance(mounts, (str, dict)):
        mounts = [mounts]
    if not isinstance(mounts, list):
        raise ValueError(f"container.mounts for {agent_name!r} must be a list in {cfg_path}")
    docker["volumes"] = [_normalize_mount(mount, cfg_path) for mount in mounts]

    extra_args = container.get("extra_args", []) or []
    if isinstance(extra_args, str):
        docker["extra_args"] = extra_args
    elif isinstance(extra_args, list):
        docker["extra_args"] = [str(item) for item in extra_args]
    else:
        raise ValueError(f"container.extra_args for {agent_name!r} must be a string or list in {cfg_path}")

    if "install_runner_script" in container:
        docker["install_runner_script"] = bool(container.get("install_runner_script"))
    else:
        docker["install_runner_script"] = False
    return docker


def _copy_non_empty(src: Dict[str, Any], dst: Dict[str, Any], key: str) -> None:
    value = src.get(key)
    if value is not None and str(value).strip():
        dst[key] = str(value)


def _normalize_mount(mount: Any, cfg_path: Path) -> Any:
    if isinstance(mount, str):
        return mount
    if not isinstance(mount, dict):
        raise ValueError(f"mount entries must be strings or mappings in {cfg_path}")

    normalized = dict(mount)
    source = normalized.get("source") or normalized.get("hostPath") or normalized.get("host_path")
    if source:
        source_path = Path(str(source)).expanduser()
        if not source_path.is_absolute():
            source_path = (Path.cwd() / source_path).resolve(strict=False)
        normalized["source"] = str(source_path)
    return normalized


def expand_rl_group_size(yaml_config_list: List[Dict[str, Any]], group_size: int) -> List[Dict[str, Any]]:
    if int(group_size) <= 0:
        return yaml_config_list
    expanded = [dict(item) for item in yaml_config_list]
    for item in expanded:
        item["env_num"] = int(group_size)
    log.info("Override agent parallelism env_num=%d for %d config(s)", int(group_size), len(expanded))
    return expanded


def expand_rl_epoch(yaml_config_list: List[Dict[str, Any]], epoch: int) -> List[Dict[str, Any]]:
    epoch = max(1, int(epoch))
    if epoch <= 1:
        return yaml_config_list

    expanded = list(yaml_config_list)
    base_configs = list(yaml_config_list)
    num_tasks = len(base_configs)
    for epoch_idx in range(1, epoch):
        for item in base_configs:
            epoch_item = dict(item)
            epoch_item["task_idx"] = item.get("task_idx", 1) + epoch_idx * num_tasks
            expanded.append(epoch_item)
    log.info("rl_epoch=%d: expanded %d configs to %d configs", epoch, num_tasks, len(expanded))
    return expanded


def load_yaml_file(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid yaml root (expected dict): {path}")
    return cfg


def set_nested(cfg: Dict[str, Any], path: List[str], value: Any) -> None:
    cur: Dict[str, Any] = cfg
    for key in path[:-1]:
        nxt = cur.get(key)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[key] = nxt
        cur = nxt
    cur[path[-1]] = value


def rebuild_sqlite_db(db_url: str) -> None:
    if not db_url.startswith("sqlite://"):
        return
    file_path = db_url[len("sqlite://") :].split("?", 1)[0]
    if file_path and os.path.exists(file_path):
        os.remove(file_path)
        log.info("Removed existing SQLite DB for rebuild: %s", file_path)


def _validate_gateway_route_key(model: str) -> None:
    normalized_model = str(model or "").strip()
    placeholder_models = {
        "YOUR_ROUTE",
        "YOUR_MODEL",
        "YOUR_GATEWAY_ROUTE_KEY",
        "YOUR_LLM_MODEL",
    }
    if normalized_model.upper() in placeholder_models:
        raise ValueError(
            f"--llm-model must be a real gateway llm_routes key, got placeholder {model!r}"
        )

    gateway_config_path = os.environ.get("AIEVOBOX_GATEWAY_CONFIG")
    if not gateway_config_path:
        log.debug("AIEVOBOX_GATEWAY_CONFIG is unset; skip gateway route-key validation")
        return

    try:
        from gateway.config import load_gateway_config
    except Exception:
        log.debug("gateway config module is unavailable; skip route-key validation", exc_info=True)
        return

    try:
        gateway_cfg = load_gateway_config(gateway_config_path)
    except Exception:
        log.debug("gateway config could not be loaded; skip route-key validation", exc_info=True)
        return

    routes = gateway_cfg.llm_routes or {}
    if routes and model not in routes:
        raise ValueError(
            f"--llm-model must be a gateway llm_routes key; got {model!r}, available={sorted(routes)}"
        )
