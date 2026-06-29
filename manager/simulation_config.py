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

DEFAULT_SQLITE_DB_URL = "sqlite://env_trajs.db"

_RJOB_DEFAULT_CONFIG: Dict[str, Any] = {
    "cluster_entry": "",
    "namespace": "",
    "access_key": "",
    "secret_key": "",
    "verifyssl": True,
    "retries": 3,
    "charged_group": "",
    "poll_interval_s": 5.0,
    "cleanup_on_finish": True,
    "gateway_base_url": "",
    "name_prefix": "safactory",
    "no_packaging": True,
    "auto_delete_duration": "",
    "keep_failed_jobs": False,
    "submit_concurrency": 0,
}


def load_rjob_global_config(path: str) -> Dict[str, Any]:
    path = str(path or "").strip()
    if not path:
        return _normalize_rjob_config({})
    cfg_path = Path(path).expanduser()
    if not cfg_path.is_absolute():
        cfg_path = (Path.cwd() / cfg_path).resolve(strict=False)
    cfg = load_yaml_file(str(cfg_path))
    return _normalize_rjob_config(_rjob_config_section(cfg))


def _rjob_config_section(cfg: Dict[str, Any]) -> Dict[str, Any]:
    cluster = cfg.get("cluster") if isinstance(cfg.get("cluster"), dict) else {}
    merged: Dict[str, Any] = {}
    if isinstance(cluster.get("rjob"), dict):
        merged.update(cluster.get("rjob") or {})
    if isinstance(cfg.get("rjob"), dict):
        merged.update(cfg.get("rjob") or {})
    for key in _RJOB_DEFAULT_CONFIG:
        if key in cfg and cfg.get(key) is not None:
            merged[key] = cfg.get(key)
    return merged


def _normalize_rjob_config(section: Dict[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = dict(_RJOB_DEFAULT_CONFIG)
    if isinstance(section, dict):
        merged.update(section)

    for key in (
        "cluster_entry",
        "namespace",
        "access_key",
        "secret_key",
        "charged_group",
        "gateway_base_url",
        "name_prefix",
        "auto_delete_duration",
    ):
        merged[key] = str(merged.get(key) or "").strip()

    merged["verifyssl"] = _as_bool(merged.get("verifyssl"), default=True)
    merged["cleanup_on_finish"] = _as_bool(merged.get("cleanup_on_finish"), default=True)
    merged["no_packaging"] = _as_bool(merged.get("no_packaging"), default=True)
    merged["keep_failed_jobs"] = _as_bool(merged.get("keep_failed_jobs"), default=False)
    merged["retries"] = max(0, int(merged.get("retries") or 0))
    merged["poll_interval_s"] = max(0.1, float(merged.get("poll_interval_s") or 5.0))
    merged["submit_concurrency"] = max(0, int(merged.get("submit_concurrency") or 0))
    merged["name_prefix"] = merged["name_prefix"] or "safactory"
    return merged


def _as_bool(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "y", "on"}:
            return True
        if text in {"0", "false", "no", "n", "off"}:
            return False
    return bool(value)


def load_simulation_run_config(args: Any) -> SimulationRunConfig:
    rjob_config_arg = str(getattr(args, "rjob_config", "") or "").strip()
    rjob_section = load_rjob_global_config(rjob_config_arg)

    pool_size, warm_pool_size, startup_submit_count, followup_submit_batch = derive_pool_sizing(
        configured_pool_size=int(args.pool_size),
        pool_size_override=0,
        multiplier=float(args.multiplier),
    )

    mode = str(args.mode or "docker").strip().lower()
    if mode not in {"docker", "rjob"}:
        raise ValueError(f"Unsupported simulation mode: {mode!r}")

    job_id = str(args.job_id or "").strip() or uuid.uuid4().hex
    max_workers = int(args.max_workers) if int(args.max_workers or 0) > 0 else None

    evaluation_config = load_evaluation_runtime_config(str(getattr(args, "evaluation_config", "") or ""))
    evaluation_model = str(
        getattr(args, "evaluation_model", "") or evaluation_config.get("evaluation_model") or ""
    ).strip()
    _validate_gateway_route_key(str(args.llm_model), arg_name="--llm-model")
    if evaluation_model:
        _validate_gateway_route_key(evaluation_model, arg_name="--evaluation-model")
    agent_config = getattr(args, "agent_config", None)
    agent_start_config = getattr(args, "agent_start_config", None)

    storage_type = str(args.storage_type or "sqlite").strip().lower()
    db_url = _db_url_for_storage(storage_type, getattr(args, "db_path", None))
    _validate_storage_db_url(storage_type, db_url)

    return SimulationRunConfig(
        job_id=job_id,
        agent_root=str(args.agent_root),
        agent_config=None if agent_config is None else str(agent_config),
        agent_start_config=None if agent_start_config is None else str(agent_start_config),
        storage_type=storage_type,
        db_url=db_url,
        pool_size=pool_size,
        warm_pool_size=warm_pool_size,
        startup_submit_count=startup_submit_count,
        followup_submit_batch=followup_submit_batch,
        mode=mode,
        gateway_base_url=str(args.gateway_base_url).rstrip("/"),
        llm_model=str(args.llm_model),
        llm_temperature=float(args.llm_temperature),
        evaluation_model=evaluation_model,
        max_steps=int(args.max_steps),
        agent_start_timeout_s=float(args.agent_start_timeout_s),
        docker_bin=str(args.docker_bin or "docker"),
        docker_pull_policy=str(args.docker_pull_policy or "never").strip().lower(),
        docker_startup_concurrency=max(1, int(args.docker_startup_concurrency or 1)),
        agent_start_timeout_grace_s=_float_at_least(
            getattr(args, "agent_start_timeout_grace_s", 120.0),
            default=120.0,
            minimum=0.0,
        ),
        container_refill_timeout_s=_float_at_least(
            getattr(args, "container_refill_timeout_s", 300.0),
            default=300.0,
            minimum=1.0,
        ),
        docker_command_timeout_s=_float_at_least(
            getattr(args, "docker_command_timeout_s", 300.0),
            default=300.0,
            minimum=1.0,
        ),
        docker_start_timeout_s=_float_at_least(
            getattr(args, "docker_start_timeout_s", 300.0),
            default=300.0,
            minimum=1.0,
        ),
        docker_remove_timeout_s=_float_at_least(
            getattr(args, "docker_remove_timeout_s", 120.0),
            default=120.0,
            minimum=1.0,
        ),
        docker_lifecycle_timeout_s=_float_at_least(
            getattr(args, "docker_lifecycle_timeout_s", 60.0),
            default=60.0,
            minimum=1.0,
        ),
        rjob_config=rjob_section,
        cleanup_docker_container=bool(getattr(args, "cleanup_docker_container", True)),
        max_workers=max_workers,
        rebuild_table=bool(args.rebuild_table),
        enable_buffer=bool(args.enable_buffer),
        buffer_size=int(args.buffer_size),
        flush_interval=float(args.flush_interval),
        rl_group_size=int(args.rl_group_size),
        rl_epoch=max(1, int(args.rl_epoch)),
        evaluation_enabled=bool(args.evaluation_enabled),
        evaluation_config=evaluation_config,
        strict_eval_tasks=bool(args.strict_eval_tasks),
        circuit_breaker_enabled=bool(getattr(args, "circuit_breaker", True)),
        circuit_breaker_window=_int_at_least(getattr(args, "circuit_breaker_window", 50), default=50, minimum=1),
        circuit_breaker_min_samples=_int_at_least(
            getattr(args, "circuit_breaker_min_samples", 20),
            default=20,
            minimum=1,
        ),
        circuit_breaker_failure_rate=_rate_or_default(
            getattr(args, "circuit_breaker_failure_rate", 0.8),
            default=0.8,
        ),
        circuit_breaker_timeout_rate=_rate_or_default(
            getattr(args, "circuit_breaker_timeout_rate", 0.5),
            default=0.5,
        ),
        circuit_breaker_consecutive_timeouts=_int_at_least(
            getattr(args, "circuit_breaker_consecutive_timeouts", 5),
            default=5,
            minimum=1,
        ),
    )


def _db_url_for_storage(storage_type: str, raw_db_path: Any) -> str:
    if storage_type == "sqlite":
        return str(raw_db_path or DEFAULT_SQLITE_DB_URL).strip()
    if storage_type == "cloud":
        if raw_db_path:
            log.warning("cloud storage ignores --db-path and uses wt-data-gateway default database URIs")
        return ""
    raise ValueError(f"Unsupported storage type: {storage_type!r}")


def _validate_storage_db_url(storage_type: str, db_url: str) -> None:
    if db_url.startswith("--db-path"):
        raise ValueError("--db-path value must be a URI only, for example sqlite://env_trajs.db")
    if storage_type == "sqlite" and not db_url.startswith("sqlite://"):
        raise ValueError(f"sqlite storage requires a sqlite:// db path, got {db_url!r}")


def _float_at_least(value: Any, *, default: float, minimum: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return max(float(minimum), number)


def _int_at_least(value: Any, *, default: int, minimum: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return int(default)
    return max(int(minimum), number)


def _rate_or_default(value: Any, *, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return min(1.0, max(0.0, number))


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
    rjob_cfg = dict(cfg.rjob_config or {})
    env_types = load_agent_start_config(cfg.agent_start_config)
    database_cfg: Dict[str, Any] = {"driver": cfg.storage_type}
    if cfg.storage_type == "sqlite":
        database_cfg["sqlite_path"] = cfg.db_url

    return {
        "mode": cfg.mode,
        "pool_size": int(cfg.warm_pool_size),
        "database": database_cfg,
        "cluster": {
            "docker": {
                "bin": cfg.docker_bin,
                "pull_policy": cfg.docker_pull_policy,
                "startup_concurrency": int(cfg.docker_startup_concurrency),
                "cleanup_container_on_finish": bool(cfg.cleanup_docker_container),
                "remove_on_close": bool(cfg.cleanup_docker_container),
                "command_timeout_s": float(cfg.docker_command_timeout_s),
                "start_timeout_s": float(cfg.docker_start_timeout_s),
                "remove_timeout_s": float(cfg.docker_remove_timeout_s),
                "lifecycle_timeout_s": float(cfg.docker_lifecycle_timeout_s),
            },
            "rjob": rjob_cfg,
            "env_types": env_types,
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
        return {str(agent_name): _normalize_agent_start_entry(agent_name, spec, cfg_path) for agent_name, spec in agents.items()}

    agent_name = str(cfg.get("agent_name") or "").strip()
    if not agent_name:
        raise ValueError(f"agent start config requires agent_name or agents mapping: {cfg_path}")
    return {
        agent_name: _normalize_agent_start_entry(agent_name, cfg, cfg_path)
    }


def load_evaluation_runtime_config(path: str) -> Dict[str, Any]:
    path = str(path or "").strip()
    if not path:
        return {}

    cfg_path = Path(path).expanduser()
    if not cfg_path.is_absolute():
        cfg_path = Path.cwd() / cfg_path
    cfg = load_yaml_file(str(cfg_path))
    data = cfg.get("evaluation") if isinstance(cfg.get("evaluation"), dict) else cfg
    if not isinstance(data, dict):
        raise ValueError(f"evaluation config root must be a mapping: {cfg_path}")
    return dict(data)


def _normalize_agent_start_entry(agent_name: Any, spec: Any, cfg_path: Path) -> Dict[str, Any]:
    return {
        "docker": _normalize_agent_start_docker(agent_name, spec, cfg_path),
        "rjob": _normalize_agent_start_rjob(agent_name, spec, cfg_path),
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
    _copy_non_empty(container, docker, "result_mode")
    _copy_non_empty(container, docker, "run_result_mode")
    _copy_non_empty(container, docker, "network")
    _copy_non_empty(container, docker, "platform")

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


def _normalize_agent_start_rjob(agent_name: Any, spec: Any, cfg_path: Path) -> Dict[str, Any]:
    if not isinstance(spec, dict):
        raise ValueError(f"agent start config for {agent_name!r} must be a mapping in {cfg_path}")

    rjob_raw = spec.get("rjob", {}) or {}
    if not isinstance(rjob_raw, dict):
        raise ValueError(f"rjob config for {agent_name!r} must be a mapping in {cfg_path}")

    rjob: Dict[str, Any] = {}
    for key in (
        "charged_group",
        "private_machine",
        "image_pull_policy",
        "auto_delete_duration",
        "packaging_dir",
        "yaml_file_dump_path",
        "gateway_base_url",
        "name_prefix",
        "task_name",
        "container_name",
        "user",
        "preemptible",
        "topo_group",
        "max_wait_duration",
        "max_running_duration",
        "embed_python_bin",
        "python_bin",
    ):
        _copy_non_empty(rjob_raw, rjob, key)

    for key in (
        "cleanup_on_finish",
        "cleanup_on_failure",
        "keep_failed_jobs",
        "no_packaging",
        "dry_run",
        "predict_only",
        "top",
        "host_network",
        "enable_sshd",
        "gang_start",
        "share_host_shm",
        "privileged",
        "daemon",
    ):
        if key in rjob_raw:
            rjob[key] = bool(rjob_raw.get(key))

    for key in ("replicas", "poll_interval_s", "termination_grace_period_seconds", "local_storage_in_mb"):
        if key in rjob_raw and rjob_raw.get(key) is not None:
            value = rjob_raw.get(key)
            rjob[key] = float(value) if key == "poll_interval_s" else int(value)

    for key in ("env", "labels", "annotations", "resources", "requests", "affinity"):
        if key in rjob_raw:
            value = rjob_raw.get(key) or {}
            if not isinstance(value, dict):
                raise ValueError(f"rjob.{key} for {agent_name!r} must be a mapping in {cfg_path}")
            rjob[key] = {str(k): v for k, v in value.items()}

    for key in ("mount_config", "mount", "before_script", "depends_on"):
        if key in rjob_raw:
            value = rjob_raw.get(key) or []
            if isinstance(value, str):
                rjob[key] = [value]
            elif isinstance(value, list):
                rjob[key] = [str(item) for item in value]
            else:
                raise ValueError(f"rjob.{key} for {agent_name!r} must be a string or list in {cfg_path}")

    if "embedded_files" in rjob_raw:
        value = rjob_raw.get("embedded_files") or []
        if not isinstance(value, list):
            raise ValueError(f"rjob.embedded_files for {agent_name!r} must be a list in {cfg_path}")
        rjob["embedded_files"] = [_normalize_embedded_file(item, cfg_path) for item in value]

    return rjob


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


def _normalize_embedded_file(item: Any, cfg_path: Path) -> Dict[str, str]:
    if not isinstance(item, dict):
        raise ValueError(f"embedded_files entries must be mappings in {cfg_path}")
    source = item.get("source") or item.get("hostPath") or item.get("host_path")
    target = item.get("target") or item.get("containerPath") or item.get("container_path")
    if not source or not target:
        raise ValueError(f"embedded_files entries require source and target in {cfg_path}")
    source_path = Path(str(source)).expanduser()
    if not source_path.is_absolute():
        source_path = (cfg_path.parent / source_path).resolve(strict=False)
    return {"source": str(source_path), "target": str(target)}


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


def _validate_gateway_route_key(model: str, *, arg_name: str = "--llm-model") -> None:
    normalized_model = str(model or "").strip()
    placeholder_models = {
        "YOUR_ROUTE",
        "YOUR_MODEL",
        "YOUR_GATEWAY_ROUTE_KEY",
        "YOUR_LLM_MODEL",
        "YOUR_EVALUATION_MODEL",
    }
    if normalized_model.upper() in placeholder_models:
        raise ValueError(
            f"{arg_name} must be a real gateway llm_routes key, got placeholder {model!r}"
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
            f"{arg_name} must be a gateway llm_routes key; got {model!r}, available={sorted(routes)}"
        )
