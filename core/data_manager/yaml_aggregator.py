from pathlib import Path
import asyncio
import sqlite3
import json
import uuid
import logging
from collections import defaultdict
from typing import List, Dict, Any, Optional, Union, Set

from tortoise.transactions import in_transaction

from .load_yaml import load_yaml_configs
from core.data_manager.models import JobEnvironment

log = logging.getLogger("yaml_aggregator")

# Module-level set to keep background insert tasks alive until they complete
_insert_tasks: Set[asyncio.Task] = set()


async def _do_bulk_insert(pending_records: list, batch_size: int = 500) -> None:
    """Background coroutine: bulk-insert pending JobEnvironment records into SQLite.

    Each batch is committed in its own transaction so records become visible to
    EnvPoolManager incrementally, rather than only after the entire insert completes.
    """
    total = len(pending_records)
    try:
        for i in range(0, total, batch_size):
            async with in_transaction():
                await JobEnvironment.bulk_create(pending_records[i:i + batch_size])
            log.info("Bulk insert progress: %d/%d", min(i + batch_size, total), total)
        log.info("Bulk insert done: %d env records", total)
    except Exception:
        log.exception("Background bulk insert failed for %d records", total)


async def wait_for_pending_inserts() -> None:
    """Wait for all background env-config insert tasks to complete."""
    if _insert_tasks:
        log.info("Waiting for %d pending insert task(s)...", len(_insert_tasks))
        await asyncio.gather(*_insert_tasks, return_exceptions=True)
        log.info("All pending insert tasks completed.")


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
        log.info("Loading env config: %s", yaml_path)
        yaml_config_list.extend(load_yaml_configs(str(yaml_path)) or [])
        return yaml_config_list

    for yaml_path in iter_child_yaml_files(env_root):
        log.info("Loading env config: %s", yaml_path)
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
) -> Any:
    """
    Sync YAML configurations to the database.

    For SQLite: Uses the JobEnvironment table; removed envs are soft-deleted.
    For Cloud: Uploads to S3 via EnvConfigManager (full replace).

    Returns:
        SQLite: sqlite3.Connection for manager usage
        Cloud: env_manager instance
    """
    await data_manager.init()

    if storage_type == "sqlite":
        return await _sync_sqlite(data_manager, yaml_configs)
    elif storage_type == "cloud":
        return await _sync_cloud(data_manager, yaml_configs)
    else:
        raise ValueError(f"Unknown storage type: {storage_type}")


async def _sync_sqlite(data_manager, yaml_configs: List[Dict]) -> sqlite3.Connection:
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

    # Commit the first batch synchronously so EnvPoolManager's initial DB query
    # (build_binding_plan) always finds data. The rest goes to a background task.
    if pending_records:
        BATCH = 500
        async with in_transaction():
            await JobEnvironment.bulk_create(pending_records[:BATCH])
        log.info(
            "Initial sync insert: %d/%d env records committed",
            min(BATCH, len(pending_records)), len(pending_records),
        )
        remaining = pending_records[BATCH:]
        if remaining:
            task = asyncio.create_task(_do_bulk_insert(remaining, batch_size=BATCH))
            _insert_tasks.add(task)
            task.add_done_callback(_insert_tasks.discard)
            log.info("Scheduled background bulk insert: %d remaining env records", len(remaining))

    # Soft-delete any active envs that are no longer in the YAML
    for env in existing_envs:
        if env.env_id not in matched_env_ids:
            env.is_deleted = True
            await env.save()
            soft_deleted += 1

    log.info(
        "Sync complete: added=%d updated=%d soft_deleted=%d kept=%d",
        added, updated, soft_deleted, len(matched_env_ids) - added,
    )

    return data_manager.get_sync_connection()


async def _sync_cloud(data_manager, yaml_configs: List[Dict]) -> Any:
    """Sync configs to cloud storage (S3) — full replace strategy."""

    # Remove all existing configs before re-uploading
    data_manager.strategy.env_manager.clean_all_configs()

    job_id = data_manager.job_id

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
            await data_manager.add_environment(
                env_name=env_name,
                env_params=env_params,
                image=image,
                group_id=group_id,
                job_id=job_id,
            )

    return data_manager.strategy.env_manager
