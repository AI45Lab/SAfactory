from pathlib import Path
import asyncio
import sqlite3
import json
import os
import time
import uuid
import logging
from collections import defaultdict
from typing import List, Dict, Any, Optional, Union, Set

from tortoise.transactions import in_transaction

from .load_yaml import load_yaml_configs
from core.data_manager.models import JobEnvironment, SessionStep

log = logging.getLogger("yaml_aggregator")

# Module-level set to keep background insert tasks alive until they complete
_insert_tasks: Set[asyncio.Task] = set()
_job_db_processing_done: Dict[str, bool] = {}


def set_job_db_processing_done(job_id: str, done: bool) -> None:
    """Record whether a job will append any more environment rows."""
    normalized_job_id = str(job_id or "").strip()
    if not normalized_job_id:
        return
    _job_db_processing_done[normalized_job_id] = bool(done)


def is_job_db_processing_done(job_id: str) -> bool:
    """Return True once a job's env-config producer reaches terminal state."""
    normalized_job_id = str(job_id or "").strip()
    if not normalized_job_id:
        return False
    return bool(_job_db_processing_done.get(normalized_job_id, False))


def _schedule_insert_task(job_id: str, coro: Any, *, task_name: str) -> asyncio.Task:
    """Track a background insert task and flip the job terminal flag on completion."""

    async def _runner() -> None:
        try:
            await coro
        finally:
            set_job_db_processing_done(job_id, True)

    task = asyncio.create_task(_runner(), name=f"{task_name}:{job_id}")
    _insert_tasks.add(task)
    task.add_done_callback(_insert_tasks.discard)
    return task


async def _do_bulk_insert(pending_records: list, batch_size: int = 500) -> None:
    """Background coroutine: bulk-insert pending JobEnvironment records into SQLite.

    Each batch is committed in its own transaction so records become visible to
    AgentPoolManager incrementally, rather than only after the entire insert completes.
    """
    total = len(pending_records)
    batch_size_raw = os.environ.get("AIEVOBOX_SQLITE_BULK_INSERT_BATCH_SIZE")
    pause_raw = os.environ.get("AIEVOBOX_SQLITE_BULK_INSERT_PAUSE_S")

    try:
        batch_size = max(1, int(batch_size_raw or batch_size))
    except (TypeError, ValueError):
        log.warning(
            "Invalid AIEVOBOX_SQLITE_BULK_INSERT_BATCH_SIZE=%r; using %d",
            batch_size_raw,
            batch_size,
        )
        batch_size = max(1, int(batch_size))

    try:
        pause_s = max(0.0, float(pause_raw or 0.0))
    except (TypeError, ValueError):
        log.warning("Invalid AIEVOBOX_SQLITE_BULK_INSERT_PAUSE_S=%r; using 0.0", pause_raw)
        pause_s = 0.0

    try:
        log.debug(
            "Bulk insert start: total=%d batch_size=%d pause_s=%.3f",
            total,
            batch_size,
            pause_s,
        )
        for i in range(0, total, batch_size):
            async with in_transaction():
                await JobEnvironment.bulk_create(pending_records[i:i + batch_size])
            log.debug("Bulk insert progress: %d/%d", min(i + batch_size, total), total)
            if pause_s > 0.0 and i + batch_size < total:
                await asyncio.sleep(pause_s)
        log.debug("Bulk insert done: %d env records", total)
    except Exception:
        log.exception("Background bulk insert failed for %d records", total)


async def _do_bulk_cloud_insert(env_manager, pending_configs: list, batch_size: int = 500) -> None:
    """Background coroutine: bulk-insert pending env config dicts into cloud storage.

    Saves configs in batches so records become visible incrementally.
    """
    total = len(pending_configs)
    try:
        for i in range(0, total, batch_size):
            batch = pending_configs[i:i + batch_size]
            await asyncio.to_thread(env_manager.save_config, batch)
            log.debug("Cloud bulk insert progress: %d/%d", min(i + batch_size, total), total)
        log.debug("Cloud bulk insert done: %d env configs", total)
    except Exception:
        log.exception("Background cloud bulk insert failed for %d configs", total)


async def wait_for_pending_inserts() -> None:
    """Wait for all background env-config insert tasks to complete."""
    if _insert_tasks:
        log.debug("Waiting for %d pending insert task(s)...", len(_insert_tasks))
        await asyncio.gather(*_insert_tasks, return_exceptions=True)
        log.debug("All pending insert tasks completed.")


def iter_child_yaml_files(env_root: Path):
    """Iterate all child yaml files under the given env root."""
    if not env_root.is_dir():
        raise ValueError(f"env root {env_root} is not a directory")

    for subdir in sorted(env_root.iterdir()):
        if not subdir.is_dir():
            continue
        if subdir.name.startswith("__"):
            continue
        for p in sorted(subdir.iterdir()):
            if p.is_file() and p.suffix.lower() in (".yaml", ".yml"):
                yield p


def _resolve_env_config_path(
    *,
    env_config: Union[str, Path],
    env_root: Union[str, Path] = "env",
) -> Path:
    """Resolve env_config to an existing yaml/yml file path.

    - If env_config is an absolute path, use it directly.
    - If it's a relative path:
        1) try as-is (relative to current working dir)
        2) if not found, try joined with env_root
    """
    root = Path(env_root)
    p = Path(env_config)

    if not p.is_absolute() and not p.exists():
        p2 = root / p
        if p2.exists():
            p = p2

    if not p.is_file():
        raise ValueError(f"env_config must be an existing yaml file, got: {p}")

    if p.suffix.lower() not in (".yaml", ".yml"):
        raise ValueError(f"env_config must be a .yaml/.yml file, got: {p}")

    return p


def all_env_yaml_load(
    env_root: Union[str, Path] = "env",
    *,
    env_config: Union[str, Path, None] = None,
) -> List[Dict]:
    """Load env yaml configs.

    - If env_config is provided: only load that yaml file.
    - Else: load all yaml files under env_root.
    """
    yaml_config_list = []
    env_root = Path(env_root)

    if env_config:
        yaml_path = _resolve_env_config_path(env_config=env_config, env_root=env_root)
        log.debug("Loading env config: %s", yaml_path)
        yaml_config_list.extend(load_yaml_configs(str(yaml_path)) or [])
        return yaml_config_list

    for yaml_path in iter_child_yaml_files(env_root):
        log.debug("Loading env config: %s", yaml_path)
        try:
            yaml_configs = load_yaml_configs(str(yaml_path))
        except Exception as e:
            log.warning("[SKIP] Failed to parse yaml file: %s -> %s", yaml_path, e)
            continue
        yaml_config_list.extend(yaml_configs)

    return yaml_config_list


async def sync_configs_to_db(
    data_manager,
    yaml_configs: List[Dict],
    storage_type: str,
    startup_submit_count: int = 100,
    followup_submit_batch: int = 100,
    *,
    rebuild_table: bool = False,
    resume: bool = False,
) -> Any:
    """
    Sync YAML configurations to the database.

    Existing jobs must explicitly choose either rebuild or resume. Rebuild
    deletes only the current job; resume keeps existing configs and skips sync.

    Returns:
        SQLite: sqlite3.Connection for manager usage
        Cloud: env_manager instance
    """
    await data_manager.init()
    job_id = data_manager.job_id
    set_job_db_processing_done(job_id, False)

    try:
        if rebuild_table and resume:
            raise ValueError("--rebuild-table and --resume cannot be used together")

        existing_cloud_configs: List[Dict] = []
        if storage_type == "sqlite":
            job_exists = await JobEnvironment.filter(job_id=job_id).exists()
        elif storage_type == "cloud":
            existing_cloud_configs = await _get_cloud_job_configs(data_manager)
            job_exists = bool(existing_cloud_configs)
        else:
            raise ValueError(f"Unknown storage type: {storage_type}")

        if job_exists and resume:
            await _delete_unfinished_session_steps(
                data_manager,
                storage_type,
                existing_cloud_configs,
            )
            if storage_type == "cloud":
                _restore_cloud_env_cache(data_manager, existing_cloud_configs)
            set_job_db_processing_done(job_id, True)
            log.info("Resuming existing job_id=%s; finished environments will be skipped", job_id)
            return data_manager.get_sync_connection() if storage_type == "sqlite" else data_manager.strategy.env_manager
        if job_exists and not rebuild_table:
            raise RuntimeError(
                f"job_id={job_id!r} already exists; use --resume to continue it "
                "or --rebuild-table to start it over"
            )

        if job_exists:
            await _delete_job_data(data_manager, storage_type, existing_cloud_configs)
            log.info("Deleted existing data for job_id=%s before rebuild", job_id)

        if storage_type == "sqlite":
            return await _sync_sqlite(
                data_manager,
                yaml_configs,
                startup_submit_count,
                followup_submit_batch,
            )
        if storage_type == "cloud":
            return await _sync_cloud(
                data_manager,
                yaml_configs,
                startup_submit_count,
                followup_submit_batch,
            )
    except Exception:
        set_job_db_processing_done(job_id, True)
        raise


def _escape_sql_literal(value: str) -> str:
    return str(value).replace("'", "''")


async def _get_cloud_job_configs(data_manager) -> List[Dict]:
    env_manager = data_manager.strategy.env_manager
    query = f"job_id = '{_escape_sql_literal(data_manager.job_id)}'"
    rows: List[Dict] = []
    offset = 0
    page_size = 1000
    while True:
        page = await asyncio.to_thread(
            env_manager.get_env_configs,
            limit=page_size,
            offset=offset,
            filter_query=query,
        )
        if not page:
            break
        rows.extend(dict(row) for row in page)
        if len(page) < page_size:
            break
        offset += len(page)
    return rows


def _restore_cloud_env_cache(data_manager, configs: List[Dict]) -> None:
    for config in configs:
        row = data_manager.strategy._normalize_env_config(config)
        env_id = str(row.get("env_id") or "")
        if env_id:
            data_manager.strategy._env_configs[env_id] = row


async def _delete_unfinished_session_steps(
    data_manager,
    storage_type: str,
    cloud_configs: List[Dict],
) -> None:
    job_id = data_manager.job_id
    if storage_type == "sqlite":
        unfinished_env_ids = await JobEnvironment.filter(
            job_id=job_id,
            finished=False,
        ).values_list("env_id", flat=True)
        if not unfinished_env_ids:
            return
        session_ids = await SessionStep.filter(
            job_id=job_id,
            session_id__in=unfinished_env_ids,
        ).distinct().values_list("session_id", flat=True)
        if session_ids:
            await SessionStep.filter(
                job_id=job_id,
                session_id__in=session_ids,
            ).delete()
        return

    client = data_manager.strategy.client
    session_ids = []
    for config in cloud_configs:
        if config.get("finished", False):
            continue
        env_id = str(config.get("env_id") or "")
        if not env_id:
            continue
        query = (
            f"job_id = '{_escape_sql_literal(job_id)}' AND "
            f"session_id = '{_escape_sql_literal(env_id)}'"
        )
        rows = await asyncio.to_thread(
            client.query_data,
            filter_query=query,
            limit=1,
            columns=["session_id"],
            partition=job_id,
            checkout_latest=True,
        )
        if rows:
            session_ids.append(str(rows[0].get("session_id") or env_id))

    for session_id in session_ids:
        query = (
            f"job_id = '{_escape_sql_literal(job_id)}' AND "
            f"session_id = '{_escape_sql_literal(session_id)}'"
        )
        await asyncio.to_thread(client.delete_landing, query)


async def _delete_job_data(data_manager, storage_type: str, cloud_configs: List[Dict]) -> None:
    job_id = data_manager.job_id
    if storage_type == "sqlite":
        async with in_transaction() as connection:
            await SessionStep.filter(job_id=job_id).using_db(connection).delete()
            await JobEnvironment.filter(job_id=job_id).using_db(connection).delete()
        return

    strategy = data_manager.strategy
    query = f"job_id = '{_escape_sql_literal(job_id)}'"
    await asyncio.to_thread(strategy.client.delete_landing, query)
    for config in cloud_configs:
        env_id = str(config.get("env_id") or "")
        if env_id and not await asyncio.to_thread(strategy.env_manager.delete_config, env_id):
            raise RuntimeError(f"failed to delete cloud env config env_id={env_id}")
        strategy._env_configs.pop(env_id, None)


async def _sync_sqlite(
    data_manager,
    yaml_configs: List[Dict],
    startup_submit_count: int,
    followup_submit_batch: int,
) -> sqlite3.Connection:
    """Sync configs to SQLite.

    - Reuses existing env records where the config matches (env_name + env_params).
    - Soft-deletes (is_deleted=True) any active env no longer present in the YAML.
    - Creates new records for newly added envs.
    """
    job_id = data_manager.job_id

    def _params_key(env_params) -> str:
        return json.dumps(env_params or {}, sort_keys=True)

    # Load existing active envs for this job only
    existing_envs = await JobEnvironment.filter(
        job_id=job_id, is_deleted=False
    ).order_by("id")

    # Group existing records by (env_name, params_key) for efficient matching
    existing_groups: Dict[str, List[JobEnvironment]] = defaultdict(list)
    for env in existing_envs:
        key = f"{env.env_name}:{_params_key(env.env_params)}"
        existing_groups[key].append(env)

    added = updated = soft_deleted = 0
    matched_env_ids: set = set()
    pending_records: list = []

    for cfg in yaml_configs:
        env_name = cfg["env_name"].strip()
        env_params = cfg.get("env_params") or {}
        image = cfg.get("env_image") or ""
        env_num = cfg.get("env_num", 1)
        task_idx = cfg.get("task_idx", 1)

        if not isinstance(env_num, int) or env_num < 1:
            raise ValueError(
                f"env_num must be a positive integer, got {env_num!r} for env '{env_name}'"
            )

        group_id = str(uuid.uuid5(uuid.NAMESPACE_OID, f"{env_name}:{task_idx}"))
        group_key = f"{env_name}:{_params_key(env_params)}"
        existing_list = existing_groups.get(group_key, [])

        for i in range(env_num):
            if i < len(existing_list):
                # Reuse the existing record; update metadata if needed
                env = existing_list[i]
                matched_env_ids.add(env.env_id)
                changed = False
                if env.group_id != group_id:
                    env.group_id = group_id
                    changed = True
                if (env.image or "") != image:
                    env.image = image
                    changed = True
                if changed:
                    await env.save()
                    updated += 1
            else:
                # Collect for bulk insert
                new_env_id = str(uuid.uuid4())
                pending_records.append(JobEnvironment(
                    job_id=job_id,
                    env_id=new_env_id,
                    env_name=env_name,
                    env_params=env_params,
                    image=image,
                    group_id=group_id,
                ))
                matched_env_ids.add(new_env_id)
                added += 1

    startup_submit_count = max(0, int(startup_submit_count))
    followup_submit_batch = max(1, int(followup_submit_batch))

    # Commit the first batch synchronously so AgentPoolManager's initial DB query
    # (build_binding_plan) always finds enough rows to warm the pool.
    if pending_records:
        async with in_transaction():
            await JobEnvironment.bulk_create(pending_records[:startup_submit_count])
        log.debug(
            "Initial sync insert: %d/%d env records committed",
            min(startup_submit_count, len(pending_records)),
            len(pending_records),
        )
        remaining = pending_records[startup_submit_count:]
        if remaining:
            _schedule_insert_task(
                job_id,
                _do_bulk_insert(remaining, batch_size=followup_submit_batch),
                task_name="sqlite-env-sync",
            )
            log.debug("Scheduled background bulk insert: %d remaining env records", len(remaining))

    # Soft-delete any active envs that are no longer in the YAML
    for env in existing_envs:
        if env.env_id not in matched_env_ids:
            env.is_deleted = True
            await env.save()
            soft_deleted += 1

    log.debug(
        "Sync complete: added=%d updated=%d soft_deleted=%d kept=%d",
        added, updated, soft_deleted, len(matched_env_ids) - added,
    )

    if not pending_records or len(pending_records) <= startup_submit_count:
        set_job_db_processing_done(job_id, True)

    return data_manager.get_sync_connection()


async def _sync_cloud(
    data_manager,
    yaml_configs: List[Dict],
    startup_submit_count: int,
    followup_submit_batch: int,
) -> Any:
    """Sync configs to cloud storage (S3) using append-only batched inserts.

    Commits the first batch synchronously so downstream consumers find data
    immediately, then schedules remaining records as a background task so the
    main training loop is not blocked.
    """
    env_manager = data_manager.strategy.env_manager

    job_id = data_manager.job_id
    pending_configs: list = []
    startup_submit_count = max(0, int(startup_submit_count))
    followup_submit_batch = max(1, int(followup_submit_batch))

    for cfg in yaml_configs:
        env_name = cfg["env_name"].strip()
        env_params = cfg.get("env_params") or {}
        image = cfg.get("env_image") or ""
        env_num = cfg.get("env_num", 1)
        task_idx = cfg.get("task_idx", 1)

        if not isinstance(env_num, int) or env_num < 1:
            raise ValueError(
                f"env_num must be a positive integer, got {env_num!r} for env '{env_name}'"
            )

        group_id = str(uuid.uuid5(uuid.NAMESPACE_OID, f"{env_name}:{task_idx}"))

        for _ in range(env_num):
            env_id = str(uuid.uuid4())
            config_dict = {
                "job_id": job_id,
                "env_id": env_id,
                "env_name": env_name,
                "env_params": env_params,
                "image": image,
                "group_id": group_id,
                "created_at": int(time.time()),
            }
            pending_configs.append(config_dict)
            # Update in-memory cache immediately so get_all_environments is consistent
            data_manager.strategy._env_configs[env_id] = config_dict

    if pending_configs:
        first_batch = pending_configs[:startup_submit_count]
        await asyncio.to_thread(env_manager.save_config, first_batch)
        log.debug(
            "Initial cloud sync insert: %d/%d env configs committed",
            len(first_batch),
            len(pending_configs),
        )
        remaining = pending_configs[startup_submit_count:]
        if remaining:
            _schedule_insert_task(
                job_id,
                _do_bulk_cloud_insert(
                    env_manager,
                    remaining,
                    batch_size=followup_submit_batch,
                ),
                task_name="cloud-env-sync",
            )
            log.debug(
                "Scheduled background cloud bulk insert: %d remaining env configs",
                len(remaining),
            )

    if not pending_configs or len(pending_configs) <= startup_submit_count:
        set_job_db_processing_done(job_id, True)

    return env_manager
