import asyncio
import copy
import json
import logging
import os
import subprocess
import sys
import threading
import time
import urllib.request
import uuid
import numpy as np
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, List, Optional, Set

# Add rl directory to path for utils import
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from utils import get_env

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

# Add AIEvoBox to path
AIEVOBOX_ROOT = get_env("AIEVOBOX_ROOT")
if AIEVOBOX_ROOT not in sys.path:
    sys.path.insert(0, AIEVOBOX_ROOT)

from core.data_manager.manager import DataManager

# Setup logging
LOG_DIR = os.environ.get("AIEVOBOX_RUN_DIR") or os.path.join(AIEVOBOX_ROOT, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "buffer_server.log")

logger = logging.getLogger("buffer_server")
logger.setLevel(logging.DEBUG)

# File handler with rotation
file_handler = RotatingFileHandler(
    LOG_FILE, maxBytes=50*1024*1024, backupCount=5, encoding="utf-8"
)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
))
logger.addHandler(file_handler)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter(
    "%(asctime)s [%(levelname)s] %(message)s"
))
logger.addHandler(console_handler)

logger.info(f"Buffer Server logging to: {LOG_FILE}")

app = FastAPI(title="Rollout Buffer Server", debug=True)

# Track subprocesses
aievobox_process: Optional[subprocess.Popen] = None

# DataManager for querying the database
data_manager: Optional[DataManager] = None

# LLM Proxy URL (constructed from host and port)
_llm_proxy_host = get_env("LLM_PROXY_HOST")
_llm_proxy_port = get_env("LLM_PROXY_PORT")
llm_proxy_base_url: str = f"http://{_llm_proxy_host}:{_llm_proxy_port}"
llm_proxy_url: str = f"http://{_llm_proxy_host}:{_llm_proxy_port}/v1"

# Track last served step ID for cursor-based pagination
last_served_id: int = 0

# Pending items by instance_id (for grouping)
pending_items_by_instance: Dict[str, List[Dict[str, Any]]] = {}

# Completed session ids by instance_id. The DB may contain every environment
# step, but this server only reads rows marked is_trainable by the rollout side.
completed_sessions_by_instance: Dict[str, Set[str]] = {}

# Dropped incomplete groups are ignored if late rows arrive later.
dropped_instance_ids: Set[str] = set()

# Group size (set by /start_rollout)
group_size: int = 1


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("Invalid %s=%r, fallback to %s", name, raw, default)
        return default


INCOMPLETE_GROUP_TTL_SECONDS = _env_float("AIEVOBOX_BUFFER_INCOMPLETE_GROUP_TTL_SECONDS", 1800.0)


@app.middleware("http")
async def set_body_size(request: Request, call_next):
    request._body_size_limit = 1_073_741_824  # 1GB
    response = await call_next(request)
    return response


class BufferResponse(BaseModel):
    success: bool
    message: str = ""
    data: Optional[Dict[str, Any]] = None


def _parse_timestamp(ts: Optional[str]) -> Optional[float]:
    """Parse timestamp string to float."""
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts).timestamp()
    except Exception:
        try:
            return float(ts)
        except Exception:
            return None


def _build_item_from_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a database row to the expected item format."""
    # Parse stored prompt (JSON serialized messages list)
    prompt_str = row.get("prompt", "")
    if isinstance(prompt_str, str):
        base_messages = json.loads(prompt_str) if prompt_str else []
    else:
        base_messages = prompt_str
    messages = base_messages + [{"role": "assistant", "content": row.get("response", "")}]

    session_id = row.get("session_id", "")
    env_id = row.get("env_id", "")
    group_id = row.get("group_id", "")

    # 从 env_state 中解析 weight_version
    weight_version = 0
    if env_state_raw := row.get("env_state"):
        weight_version = int(json.loads(env_state_raw)["weight_version"])

    extra_info = {
        "timestamp": _parse_timestamp(row.get("session_end_time")) or _parse_timestamp(row.get("timestamp")) or time.time(),
        "steps": row.get("step_id", 0),
        # 注意：finish_reason 与 truncated 不完全等价，finish_reason 仅用于训练侧标记截断状态
        "finish_reason": "length" if row.get("truncated", False) else "stop",
        "session_id": session_id,
        "env_id": env_id,
        "group_id": group_id,
        "weight_version": weight_version,
        "truncated": row.get("truncated", False),
        "step_pk": row.get("step_pk"),
        "is_session_completed": bool(row.get("is_session_completed", False)),
    }

    return {
        "uid": str(uuid.uuid4()),
        "instance_id": str(group_id),
        "messages": messages,
        "reward": float(row.get("reward", 0.0)),
        "extra_info": extra_info,
    }


def _propagate_terminal_rewards(group: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    terminal_rewards_by_session: Dict[str, float] = {}
    for item in group:
        extra = item.get("extra_info") or {}
        session_id = str(extra.get("session_id") or "")
        if session_id and bool(extra.get("is_session_completed", False)):
            terminal_rewards_by_session[session_id] = float(item.get("reward", 0.0))

    for item in group:
        extra = item.setdefault("extra_info", {})
        session_id = str(extra.get("session_id") or "")
        if session_id not in terminal_rewards_by_session:
            continue
        original_reward = float(item.get("reward", 0.0))
        extra["step_reward"] = original_reward
        extra["terminal_reward"] = terminal_rewards_by_session[session_id]
        item["reward"] = terminal_rewards_by_session[session_id]

    return group


def _group_session_ids(bucket: List[Dict[str, Any]]) -> Set[str]:
    session_ids = set()
    for item in bucket:
        extra = item.get("extra_info") or {}
        session_id = str(extra.get("session_id") or "")
        if session_id:
            session_ids.add(session_id)
    return session_ids


def _group_latest_timestamp(bucket: List[Dict[str, Any]]) -> float:
    timestamps = []
    for item in bucket:
        extra = item.get("extra_info") or {}
        try:
            timestamps.append(float(extra.get("timestamp") or 0.0))
        except (TypeError, ValueError):
            pass
    return max(timestamps) if timestamps else 0.0


def _notify_llm_proxy_clear_sessions(session_ids: Set[str], reason: str, group_ids: List[str]) -> None:
    if not session_ids:
        return

    # Best-effort cleanup: rollout serving should continue even if the proxy is unavailable.
    payload = json.dumps(
        {
            "session_ids": sorted(session_ids),
            "reason": reason,
            "group_ids": group_ids,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{llm_proxy_base_url}/admin/clear_sessions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            body = response.read().decode("utf-8", errors="replace")
        logger.info("llm_proxy clear_sessions response: %s", body)
    except Exception as exc:
        logger.warning(
            "Failed to notify llm_proxy to clear %d sessions for %s: %s",
            len(session_ids),
            reason,
            exc,
        )


def _drop_pending_groups(group_ids: List[str], reason: str) -> None:
    global pending_items_by_instance, completed_sessions_by_instance, dropped_instance_ids

    now = time.time()
    session_ids: Set[str] = set()
    dropped_summary = {}
    for group_id in group_ids:
        bucket = pending_items_by_instance.pop(group_id, [])
        completed_sessions = completed_sessions_by_instance.pop(group_id, set())
        dropped_instance_ids.add(group_id)
        group_session_ids = _group_session_ids(bucket)
        latest_ts = _group_latest_timestamp(bucket)
        session_ids.update(group_session_ids)
        dropped_summary[group_id] = {
            "items": len(bucket),
            "sessions": len(group_session_ids),
            "completed_sessions": len(completed_sessions),
            "age_seconds": round(now - latest_ts, 1) if latest_ts else None,
        }

    if not dropped_summary:
        return

    logger.warning(
        "Dropped incomplete rollout groups: reason=%s groups=%s",
        reason,
        dropped_summary,
    )
    _notify_llm_proxy_clear_sessions(session_ids, reason, group_ids)


def _filter_dropped_group_items(new_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not dropped_instance_ids:
        return new_items

    # Late DB rows from dropped groups must not recreate stale pending state.
    kept = []
    skipped_session_ids: Set[str] = set()
    skipped_group_ids: Set[str] = set()
    for item in new_items:
        group_id = str(item.get("instance_id", ""))
        if group_id not in dropped_instance_ids:
            kept.append(item)
            continue

        skipped_group_ids.add(group_id)
        extra = item.get("extra_info") or {}
        session_id = str(extra.get("session_id") or "")
        if session_id:
            skipped_session_ids.add(session_id)

    if skipped_group_ids:
        logger.warning(
            "Skipped late rows from dropped rollout groups: groups=%s sessions=%d",
            sorted(skipped_group_ids),
            len(skipped_session_ids),
        )
        _notify_llm_proxy_clear_sessions(
            skipped_session_ids,
            "late_rows_from_dropped_groups",
            sorted(skipped_group_ids),
        )

    return kept


def cleanup_incomplete_pending_groups() -> None:
    if not pending_items_by_instance:
        return

    # Drop groups that never reached the configured repeat count within the TTL.
    now = time.time()
    ttl_drop_group_ids = []
    if INCOMPLETE_GROUP_TTL_SECONDS <= 0:
        return

    for group_id, bucket in pending_items_by_instance.items():
        if len(completed_sessions_by_instance.get(group_id, set())) >= group_size:
            continue
        latest_ts = _group_latest_timestamp(bucket)
        if latest_ts and now - latest_ts >= INCOMPLETE_GROUP_TTL_SECONDS:
            ttl_drop_group_ids.append(group_id)

    if ttl_drop_group_ids:
        _drop_pending_groups(ttl_drop_group_ids, "incomplete_group_ttl")


async def fetch_new_items_from_db(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Fetch new trainable rows from the database using cursor-based pagination."""
    global data_manager, last_served_id

    if data_manager is None:
        return []

    items = []
    try:
        rows = await data_manager.fetch_done_steps_with_context(
            after_id=last_served_id,
            limit=limit or 100
        )
    except Exception as e:
        logger.error(f"fetch_done_steps_with_context error: {e}")
        return []

    for row in rows:
        step_pk = row.get("step_pk")
        try:
            item = _build_item_from_row(row)
            items.append(item)
            # Update cursor to the latest processed id
            if not last_served_id or step_pk > last_served_id:
                last_served_id = step_pk
        except Exception as e:
            logger.error(f"Error building item from row: {e}")
            continue

    return items


def accumulate_and_pop_ready_groups(new_items: List[Dict[str, Any]], max_groups: Optional[int] = None) -> tuple:
    """Accumulate items and return ready groups."""
    global pending_items_by_instance, completed_sessions_by_instance, group_size

    ready_groups = []
    finished_instance_ids = []

    # Add new items to pending
    for item in new_items:
        instance_id = str(item.get("instance_id", ""))
        if not instance_id:
            continue
        pending_items_by_instance.setdefault(instance_id, []).append(item)
        extra = item.get("extra_info") or {}
        session_id = str(extra.get("session_id") or "")
        if session_id and bool(extra.get("is_session_completed", False)):
            completed_sessions_by_instance.setdefault(instance_id, set()).add(session_id)

    # Check for complete groups. A complete group means all K trajectories for
    # this prompt group have emitted their final row.
    to_delete = []
    for instance_id, bucket in pending_items_by_instance.items():
        if max_groups is not None and len(ready_groups) >= max_groups:
            break
        completed_sessions = completed_sessions_by_instance.get(instance_id, set())
        if len(completed_sessions) >= group_size:
            if len(completed_sessions) > group_size:
                logger.warning(
                    "Group %s has %d completed sessions, expected group_size=%d",
                    instance_id,
                    len(completed_sessions),
                    group_size,
                )
            group = sorted(
                bucket,
                key=lambda item: (
                    str((item.get("extra_info") or {}).get("session_id") or ""),
                    int((item.get("extra_info") or {}).get("steps") or 0),
                    int((item.get("extra_info") or {}).get("step_pk") or 0),
                ),
            )
            group = _propagate_terminal_rewards(group)
            ready_groups.append((instance_id, group))
            finished_instance_ids.append(instance_id)
            to_delete.append(instance_id)

    for k in to_delete:
        pending_items_by_instance.pop(k, None)
        completed_sessions_by_instance.pop(k, None)

    return ready_groups, finished_instance_ids


@app.post("/get_rollout_data", response_model=BufferResponse)
async def get_rollout_data(request: Request):
    global pending_items_by_instance, completed_sessions_by_instance

    try:
        payload = await request.json()
    except Exception:
        payload = {}
    if not isinstance(payload, dict):
        payload = {}
    max_groups = payload.get("max_groups")
    try:
        max_groups = int(max_groups) if max_groups is not None else None
    except (TypeError, ValueError):
        max_groups = None
    if max_groups is not None and max_groups <= 0:
        max_groups = None

    # Fetch new items from database and accumulate groups
    new_items = await fetch_new_items_from_db(limit=None)
    new_items = _filter_dropped_group_items(new_items)
    ready_groups, finished_ids = accumulate_and_pop_ready_groups(new_items, max_groups=max_groups)
    cleanup_incomplete_pending_groups()

    # Log pending status
    pending_counts = {
        k: {
            "items": len(v),
            "completed_sessions": len(completed_sessions_by_instance.get(k, set())),
        }
        for k, v in pending_items_by_instance.items()
    }
    logger.info(f"new_items={len(new_items)}, ready_groups={len(ready_groups)}, pending={pending_counts}")

    # Flatten groups to items
    ready_items = [item for _, group in ready_groups for item in group]
    rewards = [float(item.get("reward", 0.0)) for item in ready_items]

    total_samples = len(ready_items)
    avg_reward = sum(rewards) / len(rewards) if rewards else 0.0

    # 统计权重版本信息，用于后续在 Slime 侧计算数据 age
    weight_versions: List[int] = []
    for item in ready_items:
        extra = item.get("extra_info") or {}
        wv = extra.get("weight_version", 0)
        try:
            weight_versions.append(int(wv))
        except Exception:
            weight_versions.append(0)

    if weight_versions:
        max_wv = max(weight_versions)
        mean_wv = sum(weight_versions) / len(weight_versions)
    else:
        max_wv = 0.0
        mean_wv = 0.0
    finished_groups = list(sorted(set(finished_ids)))

    meta_info = {
        "total_samples": total_samples,
        "avg_reward": avg_reward,
        "finished_groups": finished_groups,
        "avg_weight_version": mean_wv,
        "max_weight_version": max_wv,
    }

    if total_samples == 0:
        return BufferResponse(
            success=False,
            message="No data available to read",
            data={"data": [], "meta_info": meta_info},
        )

    logger.info(f"Returning {total_samples} items")

    return BufferResponse(
        success=True,
        message=f"Successfully read {total_samples} items",
        data={"data": ready_items, "meta_info": meta_info},
    )


async def init_data_manager(job_session: str, storage_type: str, db_url: str, restart_training: bool = False):
    """Initialize the DataManager for querying the database."""
    global data_manager, last_served_id
    data_manager = DataManager(job_id=job_session, storage_type=storage_type, db_url=db_url)
    await data_manager.init()
    logger.info(f"DataManager initialized with {storage_type} DB: {db_url}, job_session: {job_session}")

    # Initialize cursor based on restart_training flag
    if restart_training:
        last_served_id = await data_manager.get_max_step_id()
        logger.info(f"restart_training=True, initialized last_served_id={last_served_id}")


def start_aievobox_process(data: dict):
    """Start AIEvoBox launcher.py as a subprocess.

    NOTE: LLM Proxy is now hosted in-process by slime_generator.
    It must already be running before this function is called.
    """
    global aievobox_process, group_size, last_served_id, pending_items_by_instance, completed_sessions_by_instance, dropped_instance_ids, data_manager

    # Set group size (num_repeat_per_sample)
    group_size = max(1, int(data.get("num_repeat_per_sample", 16)))

    # Clear state for new rollout
    restart_training = data.get("restart_training", False)
    if restart_training:
        pending_items_by_instance.clear()
        completed_sessions_by_instance.clear()
        dropped_instance_ids.clear()
        logger.info("restart_training=True, cleared pending items")

    # Keep a single job_session for both reader and writer process.
    job_session = str(data.get("job_session") or uuid.uuid4().hex)
    
    # Mode
    mode = os.environ.get("AIEVOBOX_MODE", "local")

    # Database path
    storage_type = os.environ.get("STORAGE_TYPE", "sqlite")
    db_url = os.environ.get("AIEVOBOX_DB_URL", f"sqlite:///{AIEVOBOX_ROOT}/rl/rl.db")

    # Run async init in a new event loop (since we're in a thread)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(
            init_data_manager(job_session=job_session, storage_type=storage_type, db_url=db_url, restart_training=restart_training)
        )
    finally:
        loop.close()

    # Build launcher.py command line arguments
    aievobox_root = os.environ.get("AIEVOBOX_ROOT", "/root/AIEvoBox")
    launcher_script = os.path.join(aievobox_root, "launcher.py")
    env_root = get_env("AIEVOBOX_ENV_ROOT")
    env_config = os.environ.get("AIEVOBOX_ENV_CONFIG")
    max_steps = int(get_env("AIEVOBOX_MAX_STEPS") or 10)
    message_cut = int(get_env("AIEVOBOX_MESSAGE_CUT") or 0)
    llm_model = get_env("RL_MODEL") or "default"
    llm_temperature = float(get_env("LLM_TEMPERATURE") or 1.0)
    pool_size = int(get_env("AIEVOBOX_POOL_SIZE") or 16)
    rl_epoch = int(get_env("RL_EPOCH") or 1)
    env_transport = os.environ.get("AIEVOBOX_ENV_TRANSPORT", "http")
    multiplier = os.environ.get("AIEVOBOX_MULTIPLIER", 1.2)

    cmd = [
        "python3", launcher_script,
        "--mode", mode,
        "--db-path", db_url,
        "--storage-type", storage_type,
        *(["--env-config", env_config] if env_config else ["--env-root", env_root]),
        "--llm-base-url", llm_proxy_url,
        "--llm-model", llm_model,
        "--llm-temperature", str(llm_temperature),
        "--max-steps", str(max_steps),
        "--message-cut", str(message_cut),
        "--pool-size", str(pool_size),
        "--multiplier", str(multiplier),
        "--job-id", job_session,
        "--no-rebuild-table",
        "--rl-use-session-suffix-url",
        "--rl-group-size", str(group_size),
        "--rl-epoch", str(rl_epoch),
        "--env-transport", env_transport,
        "--env-http-timeout-s", "600",
    ]

    logger.info(f"Starting launcher.py: {' '.join(cmd)}")
    logger.info(f"Config: group_size={group_size}, db_url={db_url}")
    logger.info(f"LLM Proxy URL: {llm_proxy_url}")

    try:
        aievobox_process = subprocess.Popen(
            cmd,
            cwd=aievobox_root,
            stdout=None,  # Inherit stdout
            stderr=None,  # Inherit stderr
        )
        logger.info(f"launcher.py started with PID: {aievobox_process.pid}")
    except Exception as e:
        logger.error(f"Failed to start launcher.py: {e}")
        raise


@app.post("/start_rollout")
async def start_rollout(request: Request):
    global aievobox_process

    payload = await request.json()

    # Check if AIEvoBox is already running
    if aievobox_process is not None and aievobox_process.poll() is None:
        return {"message": "AIEvoBox is already running", "pid": aievobox_process.pid}

    # Start AIEvoBox in a background thread
    thread = threading.Thread(target=start_aievobox_process, args=(payload,), daemon=True)
    thread.start()

    return {"message": "Rollout started"}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "aievobox_running": aievobox_process is not None and aievobox_process.poll() is None,
        "llm_proxy_running": llm_proxy_process is not None and llm_proxy_process.poll() is None,
        "data_manager_initialized": data_manager is not None,
    }


if __name__ == "__main__":
    port = int(get_env("BUFFER_SERVER_PORT"))
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        limit_concurrency=1000,  # Connection concurrency limit
        # limit_max_requests=1000000,  # Maximum request limit
        timeout_keep_alive=5,  # Keep-alive timeout,
    )
