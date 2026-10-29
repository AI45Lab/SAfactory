import asyncio
import atexit
import copy
import json
import logging
import os
import re
import subprocess
import sys
import threading
import time
import uuid
import numpy as np
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, List, Optional
from rollout_progress import RolloutProgress, checkpoint_path, database_identity
from rollout_groups import accumulate_group_batch

# Add rl directory to path for utils import
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from utils import get_env
import gateway_autostart

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

# Add AIEvoBox to path
AIEVOBOX_ROOT = get_env("AIEVOBOX_ROOT")
if AIEVOBOX_ROOT not in sys.path:
    sys.path.insert(0, AIEVOBOX_ROOT)

from core.data_manager.manager import DataManager

# Keep scan progress scoped to one training run.
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

# Ensure the auto-started gateway is torn down when buffer_server exits.
atexit.register(gateway_autostart.stop)

# DataManager for querying the database
data_manager: Optional[DataManager] = None

progress: Optional[RolloutProgress] = None
inflight_batch: Optional["BufferResponse"] = None
inflight_env_ids: set[str] = set()
buffer_lock = asyncio.Lock()

# Group size (set by /start_rollout)
group_size: int = 1


@app.middleware("http")
async def set_body_size(request: Request, call_next):
    request._body_size_limit = 1_073_741_824  # 1GB
    response = await call_next(request)
    return response


class BufferResponse(BaseModel):
    success: bool
    message: str = ""
    data: Optional[Dict[str, Any]] = None


class BatchAck(BaseModel):
    batch_id: str


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


def _required_gateway_base_url() -> str:
    gateway_base_url = str(os.environ.get("AIEVOBOX_GATEWAY_BASE_URL") or "").strip().rstrip("/")
    if not gateway_base_url:
        raise RuntimeError(
            "AIEVOBOX_GATEWAY_BASE_URL is required; the llm_proxy cannot be used as a gateway"
        )
    return gateway_base_url


def _assistant_message_from_stored_response(response: Any) -> Optional[Dict[str, Any]]:
    """Rebuild this step's assistant turn from ``session_steps.response``.

    What the column holds depends on the storage backend: the whole chat
    completion envelope on sqlite, the extracted message on cloud, and plain
    assistant text on legacy rows. All three resolve to the same turn here.

    The message is returned verbatim. The training-side mask builder compares
    every non-``content`` key for exact equality against the turn it generated,
    so neither dropping fields nor synthesizing them is safe -- an unexpected
    shape must fail loudly instead of matching by luck.
    """
    if not response:
        return None

    payload: Any = response
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError:
            # Legacy rows store the assistant text directly.
            return {"role": "assistant", "content": response}

    if isinstance(payload, dict):
        if payload.get("role"):
            return payload
        # The sqlite backend persists the whole envelope. Reading the message
        # back out is what keeps those trajectories trainable at all.
        choices = payload.get("choices")
        if isinstance(choices, list):
            for choice in choices:
                message = choice.get("message") if isinstance(choice, dict) else None
                if isinstance(message, dict) and message.get("role"):
                    return message
        return None
    if isinstance(payload, list):
        return next(
            (item for item in payload if isinstance(item, dict) and item.get("role")),
            None,
        )
    return None


def _build_item_from_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a database row to the expected item format."""
    # Parse stored prompt (JSON serialized messages list)
    prompt_str = row.get("prompt") or row.get("messages", "")
    if isinstance(prompt_str, str):
        base_messages = json.loads(prompt_str) if prompt_str else []
    else:
        base_messages = prompt_str
    messages = list(base_messages or [])
    # ``prompt`` holds the request-side history; this step's own assistant turn
    # lives in ``response``. Relying on it reappearing in a later step's request
    # would lose the final turn of every trajectory.
    if assistant := _assistant_message_from_stored_response(row.get("response", "")):
        messages.append(assistant)

    session_id = row.get("session_id", "")
    env_id = row.get("env_id") or session_id
    group_id = row.get("group_id", "")

    # 从 env_state 中解析 weight_version
    weight_version = 0
    if env_state_raw := row.get("env_state") or row.get("meta_json"):
        try:
            env_state = json.loads(env_state_raw) if isinstance(env_state_raw, str) else env_state_raw
            raw_weight_version = env_state.get("weight_version") if isinstance(env_state, dict) else None
            if raw_weight_version is not None:
                weight_version = int(raw_weight_version)
        except (TypeError, ValueError, json.JSONDecodeError):
            weight_version = 0

    truncated = row.get("truncated", row.get("is_truncated", False))
    extra_info = {
        "timestamp": _parse_timestamp(row.get("session_end_time") or row.get("created_at")) or _parse_timestamp(row.get("timestamp")) or time.time(),
        "steps": row.get("step_id", 0),
        # 注意：finish_reason 与 truncated 不完全等价，finish_reason 仅用于训练侧标记截断状态
        "finish_reason": "length" if truncated else "stop",
        "session_id": session_id,
        "env_id": env_id,
        "group_id": group_id,
        "weight_version": weight_version,
        "truncated": truncated,
    }

    return {
        "uid": str(uuid.uuid4()),
        "instance_id": str(group_id),
        "messages": messages,
        "reward": float(row.get("reward", 0.0)),
        "extra_info": extra_info,
    }


async def fetch_new_items_from_db(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Scan new envs and retry completed envs until Slime acknowledges them."""
    if data_manager is None or progress is None:
        return []

    try:
        rows = await progress.terminal_rows(data_manager, page_size=limit or 500)
    except Exception as e:
        logger.error(f"fetch finished env rows error: {e}")
        return []

    items = []
    for row in rows:
        try:
            item = _build_item_from_row(row)
            items.append(item)
        except Exception as e:
            logger.error(f"Error building item from row: {e}")
            continue

    return items


@app.post("/get_rollout_data", response_model=BufferResponse)
async def get_rollout_data(request: Request):
    global inflight_batch, inflight_env_ids
    async with buffer_lock:
        if inflight_batch is not None:
            return inflight_batch
        return await _create_rollout_batch()


async def _create_rollout_batch() -> BufferResponse:
    global inflight_batch, inflight_env_ids
    global_batch = int(get_env("RL_GLOBAL_BATCH_SIZE") or 0)
    if global_batch <= 0 or group_size <= 0 or global_batch % group_size:
        raise ValueError("RL_GLOBAL_BATCH_SIZE must be a positive multiple of group_size")
    new_items = await fetch_new_items_from_db(limit=None)
    ready_groups, finished_ids = accumulate_group_batch(
        {}, new_items, group_size=group_size,
        required_groups=global_batch // group_size,
    )

    # Log pending status
    logger.info(f"new_items={len(new_items)}, ready_groups={len(ready_groups)}, pending_envs={len(progress.pending) if progress else 0}")

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
    batch_id = uuid.uuid4().hex if ready_items else None

    meta_info = {
        "total_samples": total_samples,
        "avg_reward": avg_reward,
        "finished_groups": finished_groups,
        "avg_weight_version": mean_wv,
        "max_weight_version": max_wv,
        "batch_id": batch_id,
    }

    if total_samples == 0:
        return BufferResponse(
            success=False,
            message="No data available to read",
            data={"data": [], "meta_info": meta_info},
        )

    logger.info(f"Returning {total_samples} items")

    response = BufferResponse(
        success=True,
        message=f"Successfully read {total_samples} items",
        data={"data": ready_items, "meta_info": meta_info},
    )
    inflight_env_ids = {str(item["extra_info"]["env_id"]) for item in ready_items}
    inflight_batch = response
    return response


@app.post("/ack_rollout_data")
async def ack_rollout_data(ack: BatchAck):
    global inflight_batch, inflight_env_ids
    async with buffer_lock:
        if progress is not None and progress.last_acked_batch == ack.batch_id:
            return {"message": "Rollout batch already acknowledged"}
        current = (inflight_batch.data or {}).get("meta_info", {}) if inflight_batch else {}
        if current.get("batch_id") != ack.batch_id:
            raise HTTPException(status_code=409, detail="Unknown rollout batch")
        if progress is None:
            raise HTTPException(status_code=503, detail="Rollout progress is not initialized")
        progress.acknowledge(ack.batch_id, inflight_env_ids)
        inflight_batch = None
        inflight_env_ids = set()
    return {"message": "Rollout batch acknowledged"}


async def init_data_manager(job_session: str, storage_type: str, db_url: str, restart_training: bool = False):
    """Initialize the DataManager for querying the database."""
    global data_manager, progress, inflight_batch, inflight_env_ids
    data_manager = DataManager(job_id=job_session, storage_type=storage_type, db_url=db_url)
    await data_manager.init()
    progress = RolloutProgress.load(
        checkpoint_path(LOG_DIR, db_url), db_url=db_url, job_session=job_session
    )
    if restart_training and not progress.loaded:
        await progress.skip_existing_finished(data_manager)
    progress.save()
    inflight_batch = None
    inflight_env_ids = set()
    logger.info(f"DataManager initialized with {storage_type} DB: {db_url}, job_session: {job_session}")


def start_aievobox_process(data: dict):
    """Start AIEvoBox launcher.py as a subprocess.

    NOTE: LLM Proxy is now hosted in-process by slime_generator.
    It must already be running before this function is called.
    """
    global aievobox_process, group_size, data_manager

    # Set group size (num_repeat_per_sample)
    group_size = int(data.get("num_repeat_per_sample", 16))

    restart_training = data.get("restart_training", False)

    # Database path
    storage_type = os.environ.get("STORAGE_TYPE", "sqlite")
    db_url = os.environ.get("AIEVOBOX_DB_URL", f"sqlite:///{AIEVOBOX_ROOT}/rl/rl.db")
    state_path = checkpoint_path(LOG_DIR, db_url)
    saved_session = None
    if state_path.exists():
        saved = json.loads(state_path.read_text(encoding="utf-8"))
        if saved.get("db_url") == db_url and saved.get("db_identity") == database_identity(db_url):
            saved_session = saved.get("job_session")
    job_session = str(data.get("job_session") or saved_session or uuid.uuid4().hex)
    gateway_base_url = _required_gateway_base_url()

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

    # Bring up the gateway before the launcher: it must front the llm_proxy and
    # the launcher validates gateway /readyz at startup. Idempotent across rollouts.
    gateway_autostart.ensure_started(aievobox_root=aievobox_root, config_dir=LOG_DIR)

    launcher_script = os.path.join(aievobox_root, "launcher.py")
    agent_root = get_env("AIEVOBOX_AGENT_ROOT") or "env"
    agent_config = os.environ.get("AIEVOBOX_AGENT_CONFIG")
    agent_start_config = os.environ.get("AIEVOBOX_AGENT_START_CONFIG")
    # v2 docker/rjob runs need the container startup definition (env_types). When
    # not set explicitly, derive it from the agent config path:
    # env/<name>/<name>_config.yaml -> env/<name>/<name>_start.yaml.
    if not agent_start_config and agent_config:
        derived = re.sub(r"_config\.ya?ml$", "_start.yaml", agent_config)
        if derived != agent_config and os.path.exists(derived):
            agent_start_config = derived
    mode = get_env("AIEVOBOX_MODE") or "docker"
    max_steps = int(get_env("AIEVOBOX_MAX_STEPS") or 10)
    llm_model = get_env("RL_MODEL") or "default"
    llm_temperature = float(get_env("LLM_TEMPERATURE") or 1.0)
    pool_size = int(get_env("AIEVOBOX_POOL_SIZE") or 16)
    rl_epoch = int(get_env("RL_EPOCH") or 1)
    evaluation_flag = str(os.environ.get("AIEVOBOX_ENABLE_EVALUATION") or "").strip().lower()
    evaluation_enabled = evaluation_flag in {"1", "true", "yes", "on"}

    cmd = [
        "python3", launcher_script,
        "--mode", mode,
        "--db-path", db_url,
        "--storage-type", storage_type,
        *(["--agent-config", agent_config] if agent_config else ["--agent-root", agent_root]),
        *(["--agent-start-config", agent_start_config] if agent_start_config else []),
        *(["--enable-evaluation"] if evaluation_enabled else []),
        "--gateway-base-url", gateway_base_url,
        "--llm-model", llm_model,
        "--llm-temperature", str(llm_temperature),
        "--max-steps", str(max_steps),
        "--pool-size", str(pool_size),
        "--job-id", job_session,
        "--no-rebuild-table",
        "--rl-group-size", str(group_size),
        "--rl-epoch", str(rl_epoch),
    ]

    logger.info(f"Starting launcher.py: {' '.join(cmd)}")
    logger.info(f"Config: group_size={group_size}, db_url={db_url}")
    logger.info(f"Gateway base URL: {gateway_base_url}")

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

    try:
        _required_gateway_base_url()
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

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
