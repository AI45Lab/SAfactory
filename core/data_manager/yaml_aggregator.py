"""Load environment YAML and synchronize it through :class:`DataManager`."""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Set, Union

from .load_yaml import load_yaml_configs

log = logging.getLogger("yaml_aggregator")

_insert_tasks: Set[asyncio.Task] = set()
_job_db_processing_done: Dict[str, bool] = {}


def set_job_db_processing_done(job_id: str, done: bool) -> None:
    normalized_job_id = str(job_id or "").strip()
    if normalized_job_id:
        _job_db_processing_done[normalized_job_id] = bool(done)


def is_job_db_processing_done(job_id: str) -> bool:
    normalized_job_id = str(job_id or "").strip()
    return bool(normalized_job_id and _job_db_processing_done.get(normalized_job_id, False))


def _schedule_insert_task(job_id: str, coro: Any, *, task_name: str) -> asyncio.Task:
    async def _runner() -> None:
        try:
            await coro
        finally:
            set_job_db_processing_done(job_id, True)

    task = asyncio.create_task(_runner(), name=f"{task_name}:{job_id}")
    _insert_tasks.add(task)
    task.add_done_callback(_insert_tasks.discard)
    return task


async def _do_bulk_insert(data_manager: Any, rows: List[Dict[str, Any]], batch_size: int) -> None:
    """Submit follow-up config batches through the public data-manager boundary."""
    total = len(rows)
    batch_size_raw = os.environ.get("AIEVOBOX_SQLITE_BULK_INSERT_BATCH_SIZE")
    pause_raw = os.environ.get("AIEVOBOX_SQLITE_BULK_INSERT_PAUSE_S")
    try:
        batch_size = max(1, int(batch_size_raw or batch_size))
    except (TypeError, ValueError):
        log.warning("Invalid bulk insert batch size %r; using %d", batch_size_raw, batch_size)
        batch_size = max(1, int(batch_size))
    try:
        pause_s = max(0.0, float(pause_raw or 0.0))
    except (TypeError, ValueError):
        log.warning("Invalid bulk insert pause %r; using 0.0", pause_raw)
        pause_s = 0.0

    for index in range(0, total, batch_size):
        await data_manager.insert_environment_rows(rows[index:index + batch_size])
        log.debug("Environment sync progress: %d/%d", min(index + batch_size, total), total)
        if pause_s and index + batch_size < total:
            await asyncio.sleep(pause_s)


async def wait_for_pending_inserts() -> None:
    if _insert_tasks:
        await asyncio.gather(*list(_insert_tasks), return_exceptions=True)


def iter_child_yaml_files(env_root: Path):
    if not env_root.is_dir():
        raise ValueError(f"env root {env_root} is not a directory")
    for subdir in sorted(env_root.iterdir()):
        if not subdir.is_dir() or subdir.name.startswith("__"):
            continue
        for path in sorted(subdir.iterdir()):
            if path.is_file() and path.suffix.lower() in (".yaml", ".yml"):
                yield path


def _resolve_env_config_path(
    *,
    env_config: Union[str, Path],
    env_root: Union[str, Path] = "env",
) -> Path:
    root = Path(env_root)
    path = Path(env_config)
    if not path.is_absolute() and not path.exists() and (root / path).exists():
        path = root / path
    if not path.is_file():
        raise ValueError(f"env_config must be an existing yaml file, got: {path}")
    if path.suffix.lower() not in (".yaml", ".yml"):
        raise ValueError(f"env_config must be a .yaml/.yml file, got: {path}")
    return path


def all_env_yaml_load(
    env_root: Union[str, Path] = "env",
    *,
    env_config: Union[str, Path, None] = None,
) -> List[Dict]:
    yaml_config_list: List[Dict] = []
    root = Path(env_root)
    if env_config:
        yaml_path = _resolve_env_config_path(env_config=env_config, env_root=root)
        yaml_config_list.extend(load_yaml_configs(str(yaml_path)) or [])
        return yaml_config_list
    for yaml_path in iter_child_yaml_files(root):
        try:
            yaml_config_list.extend(load_yaml_configs(str(yaml_path)) or [])
        except Exception as exc:
            log.warning("[SKIP] Failed to parse yaml file: %s -> %s", yaml_path, exc)
    return yaml_config_list


async def sync_configs_to_db(
    data_manager: Any,
    yaml_configs: List[Dict],
    storage_type: str,
    startup_submit_count: int = 100,
    followup_submit_batch: int = 100,
    *,
    rebuild_table: bool = False,
    resume: bool = False,
) -> None:
    """Synchronize configs without leaking a connection or backend client."""
    if storage_type not in {"sqlite", "cloud"}:
        raise ValueError(f"Unknown storage type: {storage_type}")
    if rebuild_table and resume:
        raise ValueError("--rebuild-table and --resume cannot be used together")

    await data_manager.init()
    job_id = data_manager.job_id
    set_job_db_processing_done(job_id, False)
    try:
        existing = await data_manager.list_environment_rows(job_id=job_id)
        if existing and resume:
            unfinished_ids = [
                str(row.get("env_id") or "")
                for row in existing
                if not bool(row.get("finished", False)) and row.get("env_id")
            ]
            if unfinished_ids:
                await data_manager.delete_session_step_rows(
                    job_id=job_id,
                    session_ids=unfinished_ids,
                )
            set_job_db_processing_done(job_id, True)
            log.info("Resuming existing job_id=%s; finished environments will be skipped", job_id)
            return
        if existing and not rebuild_table:
            raise RuntimeError(
                f"job_id={job_id!r} already exists; use --resume to continue it "
                "or --rebuild-table to start it over"
            )
        if existing:
            await data_manager.delete_job_rows(job_id)

        rows = _expand_environment_rows(job_id, yaml_configs)
        startup_count = max(0, int(startup_submit_count))
        followup_batch = max(1, int(followup_submit_batch))
        first_batch = rows[:startup_count]
        if first_batch:
            await data_manager.insert_environment_rows(first_batch)
        remaining = rows[startup_count:]
        if remaining:
            _schedule_insert_task(
                job_id,
                _do_bulk_insert(data_manager, remaining, followup_batch),
                task_name=f"{storage_type}-env-sync",
            )
        else:
            set_job_db_processing_done(job_id, True)
        log.debug(
            "Environment sync scheduled: initial=%d remaining=%d job_id=%s",
            len(first_batch),
            len(remaining),
            job_id,
        )
    except Exception:
        set_job_db_processing_done(job_id, True)
        raise


def _expand_environment_rows(job_id: str, yaml_configs: List[Dict]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for config in yaml_configs:
        env_name = str(config["env_name"]).strip()
        env_params = config.get("env_params") or {}
        image = config.get("env_image") or ""
        env_num = config.get("env_num", 1)
        task_idx = config.get("task_idx", 1)
        if not isinstance(env_num, int) or env_num < 1:
            raise ValueError(
                f"env_num must be a positive integer, got {env_num!r} for env '{env_name}'"
            )
        group_id = str(uuid.uuid5(uuid.NAMESPACE_OID, f"{env_name}:{task_idx}"))
        for _ in range(env_num):
            rows.append({
                "job_id": job_id,
                "env_id": str(uuid.uuid4()),
                "env_name": env_name,
                "env_params": dict(env_params),
                "image": str(image),
                "group_id": group_id,
            })
    return rows
