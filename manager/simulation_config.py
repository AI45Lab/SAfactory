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
PROJECT_ROOT = Path(__file__).resolve().parent.parent

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

_SANDBOX_DEFAULT_CONFIG: Dict[str, Any] = {
    "domain": "https://h.pjlab.org.cn/brainbox",
    "protocol": "https",
    "project": "",
    "api_key": "",
    "api_key_env": "OPEN_SANDBOX_API_KEY",
    "environment_id": "",
    "gateway_base_url": "",
    "command_port": 44772,
    "lifecycle_minutes": 120,
    "request_timeout_s": 60.0,
    "create_timeout_s": 600.0,
    "command_timeout_s": 720.0,
    "startup_concurrency": 8,
    "cleanup_on_finish": True,
    "use_server_proxy": True,
    "skip_health_check": False,
}


def load_rjob_global_config(path: str) -> Dict[str, Any]:
    path = str(path or "").strip()
    if not path:
        return _normalize_rjob_config({})
    cfg_path = _resolve_config_path(path)
    cfg = load_yaml_file(str(cfg_path))
    return _normalize_rjob_config(_rjob_config_section(cfg))


def _resolve_config_path(path: str) -> Path:
    cfg_path = Path(path).expanduser()
    if cfg_path.is_absolute():
        return cfg_path

    cwd_path = (Path.cwd() / cfg_path).resolve(strict=False)
    if cwd_path.exists():
        return cwd_path

    project_path = (PROJECT_ROOT / cfg_path).resolve(strict=False)
    if project_path.exists():
        return project_path

    return cwd_path


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


def load_sandbox_global_config(path: str) -> Dict[str, Any]:
    path = str(path or "").strip()
    if not path:
        return _normalize_sandbox_config({})
    cfg = load_yaml_file(str(_resolve_config_path(path)))
    cluster = cfg.get("cluster") if isinstance(cfg.get("cluster"), dict) else {}
    section: Dict[str, Any] = {}
    if isinstance(cluster.get("sandbox"), dict):
        section.update(cluster.get("sandbox") or {})
    if isinstance(cfg.get("sandbox"), dict):
        section.update(cfg.get("sandbox") or {})
    return _normalize_sandbox_config(section)


def _normalize_sandbox_config(section: Dict[str, Any]) -> Dict[str, Any]:
    merged = {**_SANDBOX_DEFAULT_CONFIG, **dict(section or {})}
    for key in (
        "domain",
        "protocol",
        "project",
        "api_key",
        "api_key_env",
        "environment_id",
        "gateway_base_url",
    ):
        merged[key] = str(merged.get(key) or "").strip()
    if not merged["api_key"] and merged["api_key_env"]:
        merged["api_key"] = str(os.getenv(merged["api_key_env"], "")).strip()
    merged["command_port"] = min(65535, max(1, int(merged.get("command_port") or 44772)))
    merged["lifecycle_minutes"] = min(1440, max(3, int(merged.get("lifecycle_minutes") or 120)))
    merged["startup_concurrency"] = max(1, int(merged.get("startup_concurrency") or 8))
    for key in ("request_timeout_s", "create_timeout_s", "command_timeout_s"):
        merged[key] = max(1.0, float(merged.get(key) or _SANDBOX_DEFAULT_CONFIG[key]))
    for key in ("cleanup_on_finish", "use_server_proxy", "skip_health_check"):
        merged[key] = _as_bool(merged.get(key), default=bool(_SANDBOX_DEFAULT_CONFIG[key]))
    merged["domain"] = merged["domain"].rstrip("/")
    if merged["domain"].endswith("/v1"):
        raise ValueError("sandbox.domain must not include the /v1 suffix")
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
    mode = str(args.mode or "docker").strip().lower()
    if mode not in {"docker", "rjob", "sandbox"}:
        raise ValueError(f"Unsupported simulation mode: {mode!r}")

    rjob_section = (
        load_rjob_global_config(str(getattr(args, "rjob_config", "") or ""))
        if mode == "rjob"
        else _normalize_rjob_config({})
    )
    sandbox_section = (
        load_sandbox_global_config(str(getattr(args, "sandbox_config", "") or ""))
        if mode == "sandbox"
        else _normalize_sandbox_config({})
    )

    pool_size, warm_pool_size, startup_submit_count, followup_submit_batch = derive_pool_sizing(
        configured_pool_size=int(args.pool_size),
        pool_size_override=0,
        multiplier=float(args.multiplier),
    )

    job_id = str(args.job_id or "").strip() or uuid.uuid4().hex
    max_workers = int(args.max_workers) if int(args.max_workers or 0) > 0 else None

    _validate_gateway_route_key(str(args.llm_model), arg_name="--llm-model")
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
        max_steps=int(args.max_steps),
        agent_start_timeout_s=float(args.agent_start_timeout_s),
        docker_bin=str(args.docker_bin or "docker"),
        docker_pull_policy=str(args.docker_pull_policy or "never").strip().lower(),
        docker_image_archive_dir=str(getattr(args, "docker_image_archive_dir", "") or "").strip(),
        cleanup_docker_image=bool(getattr(args, "cleanup_docker_image", False)),
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
        row_wait_timeout_s=_float_at_least(
            getattr(args, "row_wait_timeout_s", 60.0),
            default=60.0,
            minimum=1.0,
        ),
        row_fetch_timeout_s=_float_at_least(
            getattr(args, "row_fetch_timeout_s", 30.0),
            default=30.0,
            minimum=1.0,
        ),
        gateway_close_timeout_s=_float_at_least(
            getattr(args, "gateway_close_timeout_s", 15.0),
            default=15.0,
            minimum=1.0,
        ),
        gateway_close_retries=_int_at_least(
            getattr(args, "gateway_close_retries", 1),
            default=1,
            minimum=0,
        ),
        gateway_close_retry_backoff_s=_float_at_least(
            getattr(args, "gateway_close_retry_backoff_s", 1.0),
            default=1.0,
            minimum=0.0,
        ),
        shutdown_timeout_s=_float_at_least(
            getattr(args, "shutdown_timeout_s", 120.0),
            default=120.0,
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
        docker_stop_timeout_s=_float_at_least(
            getattr(args, "docker_stop_timeout_s", 10.0),
            default=10.0,
            minimum=1.0,
        ),
        docker_inspect_timeout_s=_float_at_least(
            getattr(args, "docker_inspect_timeout_s", 10.0),
            default=10.0,
            minimum=1.0,
        ),
        docker_remove_retries=_int_at_least(
            getattr(args, "docker_remove_retries", 3),
            default=3,
            minimum=1,
        ),
        docker_remove_retry_delay_s=_float_at_least(
            getattr(args, "docker_remove_retry_delay_s", 2.0),
            default=2.0,
            minimum=0.0,
        ),
        docker_lifecycle_timeout_s=_float_at_least(
            getattr(args, "docker_lifecycle_timeout_s", 60.0),
            default=60.0,
            minimum=1.0,
        ),
        rjob_config=rjob_section,
        sandbox_config=sandbox_section,
        cleanup_docker_container=bool(getattr(args, "cleanup_docker_container", True)),
        cleanup_stale_docker_containers=bool(getattr(args, "cleanup_stale_docker_containers", True)),
        max_workers=max_workers,
        rebuild_table=bool(args.rebuild_table),
        resume=bool(getattr(args, "resume", False)),
        enable_buffer=bool(args.enable_buffer),
        buffer_size=int(args.buffer_size),
        flush_interval=float(args.flush_interval),
        rl_group_size=int(args.rl_group_size),
        rl_epoch=max(1, int(args.rl_epoch)),
        evaluation_enabled=bool(args.evaluation_enabled),
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
        "row_wait_timeout_s": float(cfg.row_wait_timeout_s),
        "row_fetch_timeout_s": float(cfg.row_fetch_timeout_s),
        "database": database_cfg,
        "cluster": {
            "docker": {
                "bin": cfg.docker_bin,
                "pull_policy": cfg.docker_pull_policy,
                "image_archive_dir": cfg.docker_image_archive_dir,
                "cleanup_image_on_finish": bool(cfg.cleanup_docker_image),
                "startup_concurrency": int(cfg.docker_startup_concurrency),
                "cleanup_container_on_finish": bool(cfg.cleanup_docker_container),
                "remove_on_close": bool(cfg.cleanup_docker_container),
                "cleanup_stale_on_start": bool(
                    cfg.cleanup_docker_container and cfg.cleanup_stale_docker_containers
                ),
                "command_timeout_s": float(cfg.docker_command_timeout_s),
                "start_timeout_s": float(cfg.docker_start_timeout_s),
                "remove_timeout_s": float(cfg.docker_remove_timeout_s),
                "stop_timeout_s": float(cfg.docker_stop_timeout_s),
                "inspect_timeout_s": float(cfg.docker_inspect_timeout_s),
                "remove_retries": int(cfg.docker_remove_retries),
                "remove_retry_delay_s": float(cfg.docker_remove_retry_delay_s),
                "lifecycle_timeout_s": float(cfg.docker_lifecycle_timeout_s),
                "labels": {
                    "safactory.job_id": cfg.job_id,
                    "safactory.runtime": cfg.mode,
                },
            },
            "rjob": rjob_cfg,
            "sandbox": dict(cfg.sandbox_config or {}),
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


def _normalize_agent_start_entry(agent_name: Any, spec: Any, cfg_path: Path) -> Dict[str, Any]:
    docker = _normalize_agent_start_docker(agent_name, spec, cfg_path)
    return {
        "docker": docker,
        "rjob": _normalize_agent_start_rjob(agent_name, spec, cfg_path),
        "sandbox": _normalize_agent_start_sandbox(agent_name, spec, cfg_path, docker),
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

    runner_entrypoint = _normalize_runner_entrypoint(container.get("runner_entrypoint"), cfg_path)
    if runner_entrypoint:
        docker["runner_entrypoint"] = runner_entrypoint
        entrypoint_command = str(runner_entrypoint.get("command") or "").strip()
        if entrypoint_command:
            existing_run_command = str(docker.get("run_command") or "").strip()
            if existing_run_command and existing_run_command != entrypoint_command:
                raise ValueError(
                    f"container.runner_entrypoint.command conflicts with run_command for {agent_name!r} "
                    f"in {cfg_path}"
                )
            docker["run_command"] = entrypoint_command

    install_runner_script = bool(container.get("install_runner_script", False))
    mounts = container.get("mounts", container.get("volumes", [])) or []
    if isinstance(mounts, (str, dict)):
        mounts = [mounts]
    if not isinstance(mounts, list):
        raise ValueError(f"container.mounts for {agent_name!r} must be a list in {cfg_path}")
    normalized_mounts = [_normalize_mount(mount, cfg_path) for mount in mounts]
    _resolve_relative_mount_references(docker, mounts, normalized_mounts)
    runner_mount = _mount_from_runner_entrypoint(runner_entrypoint)
    if runner_mount and not install_runner_script:
        _append_mount_if_missing(normalized_mounts, runner_mount, agent_name=agent_name, cfg_path=cfg_path)
    docker["volumes"] = normalized_mounts

    extra_args = container.get("extra_args", []) or []
    if isinstance(extra_args, str):
        docker["extra_args"] = extra_args
    elif isinstance(extra_args, list):
        docker["extra_args"] = [str(item) for item in extra_args]
    else:
        raise ValueError(f"container.extra_args for {agent_name!r} must be a string or list in {cfg_path}")

    docker["install_runner_script"] = install_runner_script
    if install_runner_script and runner_entrypoint:
        runner_source = str(runner_entrypoint.get("source") or "").strip()
        runner_target = str(runner_entrypoint.get("target") or "").strip()
        if not runner_source or not Path(runner_source).is_file():
            raise ValueError(
                f"container.install_runner_script requires runner_entrypoint.source to be a file "
                f"for {agent_name!r} in {cfg_path}, got {runner_source!r}"
            )
        if not runner_target:
            raise ValueError(
                f"container.install_runner_script requires runner_entrypoint.target "
                f"for {agent_name!r} in {cfg_path}"
            )
        docker["runner_script_host_path"] = runner_source
        docker["runner_container_path"] = runner_target
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


def _normalize_agent_start_sandbox(
    agent_name: Any,
    spec: Any,
    cfg_path: Path,
    docker: Dict[str, Any],
) -> Dict[str, Any]:
    if not isinstance(spec, dict):
        raise ValueError(f"agent start config for {agent_name!r} must be a mapping in {cfg_path}")
    raw = spec.get("sandbox", {}) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"sandbox config for {agent_name!r} must be a mapping in {cfg_path}")

    sandbox: Dict[str, Any] = {}
    for key in (
        "environment_id",
        "workdir",
        "run_command",
        "result_mode",
        "gateway_base_url",
    ):
        _copy_non_empty(raw, sandbox, key)
    sandbox.setdefault("workdir", str(docker.get("workdir") or ""))
    sandbox.setdefault("run_command", str(docker.get("run_command") or ""))
    sandbox.setdefault("result_mode", str(docker.get("result_mode") or docker.get("run_result_mode") or "json"))

    env = {str(key): str(value) for key, value in dict(docker.get("env") or {}).items()}
    raw_env = raw.get("env", {}) or {}
    if not isinstance(raw_env, dict):
        raise ValueError(f"sandbox.env for {agent_name!r} must be a mapping in {cfg_path}")
    env.update({str(key): str(value) for key, value in raw_env.items()})
    sandbox["env"] = env

    resources = raw.get("resource", raw.get("resources", {})) or {}
    if not isinstance(resources, dict):
        raise ValueError(f"sandbox.resources for {agent_name!r} must be a mapping in {cfg_path}")
    sandbox["resource"] = {str(key): str(value) for key, value in resources.items()}
    extensions = raw.get("extensions", {}) or {}
    if not isinstance(extensions, dict):
        raise ValueError(f"sandbox.extensions for {agent_name!r} must be a mapping in {cfg_path}")
    sandbox["extensions"] = {str(key): str(value) for key, value in extensions.items()}

    for key in ("lifecycle_minutes", "command_port"):
        if raw.get(key) is not None:
            sandbox[key] = int(raw[key])
    if "lifecycle_minutes" in sandbox:
        sandbox["lifecycle_minutes"] = min(1440, max(3, sandbox["lifecycle_minutes"]))
    if "command_port" in sandbox:
        sandbox["command_port"] = min(65535, max(1, sandbox["command_port"]))
    for key in ("skip_health_check", "cleanup_on_finish"):
        if key in raw:
            sandbox[key] = _as_bool(raw[key], default=bool(_SANDBOX_DEFAULT_CONFIG[key]))

    raw_embedded = raw.get("embedded_files", []) or []
    if not isinstance(raw_embedded, list):
        raise ValueError(f"sandbox.embedded_files for {agent_name!r} must be a list in {cfg_path}")
    embedded = [_normalize_embedded_file(item, cfg_path) for item in raw_embedded]
    runner = docker.get("runner_entrypoint") or {}
    if runner.get("source") and runner.get("target"):
        runner_file = {"source": str(runner["source"]), "target": str(runner["target"])}
        if runner_file not in embedded:
            embedded.append(runner_file)
    sandbox["embedded_files"] = embedded

    required_paths = raw.get("required_mount_paths", []) or []
    if isinstance(required_paths, str):
        required_paths = [required_paths]
    if not isinstance(required_paths, list):
        raise ValueError(f"sandbox.required_mount_paths for {agent_name!r} must be a list in {cfg_path}")
    sandbox["required_mount_paths"] = [str(path) for path in required_paths]
    return sandbox


def _copy_non_empty(src: Dict[str, Any], dst: Dict[str, Any], key: str) -> None:
    value = src.get(key)
    if value is not None and str(value).strip():
        dst[key] = str(value)


def _normalize_runner_entrypoint(value: Any, cfg_path: Path) -> Dict[str, str]:
    if value is None:
        return {}
    if isinstance(value, str):
        command = value.strip()
        if not command:
            return {}
        return {"command": command}
    if not isinstance(value, dict):
        raise ValueError(f"container.runner_entrypoint must be a mapping or string in {cfg_path}")

    source = value.get("source") or value.get("hostPath") or value.get("host_path")
    target = value.get("target") or value.get("containerPath") or value.get("container_path")
    command = value.get("command") or value.get("run_command")
    mode = str(value.get("mode") or "ro").strip() or "ro"

    normalized: Dict[str, str] = {}
    if source:
        source_path = Path(str(source)).expanduser()
        if not source_path.is_absolute():
            source_path = (cfg_path.parent / source_path).resolve(strict=False)
        normalized["source"] = str(source_path)
    if target:
        normalized["target"] = str(target)
    if command:
        normalized["command"] = str(command)
    if mode:
        normalized["mode"] = mode

    if normalized.get("source") and not normalized.get("target"):
        raise ValueError(f"container.runner_entrypoint.source requires target in {cfg_path}")
    if normalized.get("target") and not normalized.get("command"):
        raise ValueError(f"container.runner_entrypoint.target requires command in {cfg_path}")
    if not normalized.get("source") and not normalized.get("command"):
        raise ValueError(f"container.runner_entrypoint requires source or command in {cfg_path}")
    return normalized


def _mount_from_runner_entrypoint(entrypoint: Dict[str, str]) -> Dict[str, str]:
    source = str(entrypoint.get("source") or "").strip()
    target = str(entrypoint.get("target") or "").strip()
    if not source or not target:
        return {}
    return {
        "source": source,
        "target": target,
        "mode": str(entrypoint.get("mode") or "ro").strip() or "ro",
    }


def _append_mount_if_missing(
    mounts: List[Any],
    mount: Dict[str, str],
    *,
    agent_name: Any,
    cfg_path: Path,
) -> None:
    source = str(mount.get("source") or "").strip()
    target = str(mount.get("target") or "").strip()
    for existing in mounts:
        if not isinstance(existing, dict):
            continue
        existing_source = str(
            existing.get("source") or existing.get("hostPath") or existing.get("host_path") or ""
        ).strip()
        existing_target = str(
            existing.get("target") or existing.get("containerPath") or existing.get("container_path") or ""
        ).strip()
        if existing_source == source and existing_target == target:
            return
        if existing_target == target:
            raise ValueError(
                f"container.runner_entrypoint target conflicts with an existing mount for {agent_name!r} "
                f"in {cfg_path}: {target}"
            )
    mounts.append(mount)


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


def _resolve_relative_mount_references(
    docker: Dict[str, Any],
    mounts: List[Any],
    normalized_mounts: List[Any],
) -> None:
    """Resolve repeated relative mount paths to the normalized host path.

    Some Docker-outside-of-Docker workloads must mount a host directory at the
    exact same absolute path inside their controller container. Start configs
    can express that portably by repeating one relative path as the mount
    source, mount target, workdir, and environment value. The source is
    normalized first; every exact reference to it then receives that value.
    """
    aliases: Dict[str, str] = {}
    for raw_mount, normalized_mount in zip(mounts, normalized_mounts):
        if not isinstance(raw_mount, dict) or not isinstance(normalized_mount, dict):
            continue
        raw_source = (
            raw_mount.get("source")
            or raw_mount.get("hostPath")
            or raw_mount.get("host_path")
        )
        normalized_source = normalized_mount.get("source")
        if not raw_source or not normalized_source:
            continue
        source_text = str(raw_source)
        if Path(source_text).expanduser().is_absolute():
            continue
        aliases[source_text] = str(normalized_source)

    if not aliases:
        return

    for normalized_mount in normalized_mounts:
        if not isinstance(normalized_mount, dict):
            continue
        target = normalized_mount.get("target")
        if target is None:
            target = normalized_mount.get("containerPath") or normalized_mount.get("container_path")
        resolved_target = aliases.get(str(target))
        if resolved_target:
            normalized_mount["target"] = resolved_target

    workdir = str(docker.get("workdir") or "")
    if workdir in aliases:
        docker["workdir"] = aliases[workdir]

    env = docker.get("env")
    if isinstance(env, dict):
        docker["env"] = {
            key: aliases.get(str(value), str(value))
            for key, value in env.items()
        }


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
    log.debug("Override agent parallelism env_num=%d for %d config(s)", int(group_size), len(expanded))
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
    log.debug("rl_epoch=%d: expanded %d configs to %d configs", epoch, num_tasks, len(expanded))
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
