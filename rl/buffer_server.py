import asyncio
import copy
import json
import logging
import os
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

# Add AIEvoBox to path
AIEVOBOX_ROOT = os.environ.get("AIEVOBOX_ROOT", "/root/AIEvoBox")
if AIEVOBOX_ROOT not in sys.path:
    sys.path.insert(0, AIEVOBOX_ROOT)

from core.data_manager.manager import DataManager

# Setup logging
LOG_DIR = os.path.join(AIEVOBOX_ROOT, "logs")
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
llm_proxy_process: Optional[subprocess.Popen] = None

# DataManager for querying the database
data_manager: Optional[DataManager] = None

# LLM Proxy URL
llm_proxy_url: str = os.environ.get("LLM_PROXY_URL", "http://127.0.0.1:8890")

# Track last served step ID for cursor-based pagination
last_served_id: int = 0

# Pending items by instance_id (for grouping)
pending_items_by_instance: Dict[str, List[Dict[str, Any]]] = {}

# Group size (set by /start_rollout)
group_size: int = 1


def default_is_valid_group(group_data, min_valid_group_size, task_type):
    instance_id, samples = group_data
    return len(samples) >= min_valid_group_size


def default_get_group_data_meta_info(temp_data: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """
    Default implementation for getting meta information about the temporary data
    collected between get_batch calls.
    """
    if not temp_data:
        return {
            "total_samples": 0,
            "num_groups": 0,
            "avg_group_size": 0,
            "avg_reward": 0,
        }

    meta_info = {"total_samples": 0, "num_groups": len(temp_data)}

    all_rewards = []
    # Calculate per-group statistics
    for instance_id, samples in temp_data.items():
        group_size = len(samples)
        group_rewards = [s["reward"] for s in samples]  # Calculate group reward standard deviation
        meta_info["total_samples"] += group_size
        all_rewards.extend(group_rewards)
    # Calculate global statistics
    meta_info["avg_group_size"] = meta_info["total_samples"] / meta_info["num_groups"]

    if all_rewards:
        meta_info["avg_reward"] = sum(all_rewards) / len(all_rewards)
    else:
        meta_info["avg_reward"] = 0
    return meta_info


@app.middleware("http")
async def set_body_size(request: Request, call_next):
    request._body_size_limit = 1_073_741_824  # 1GB
    response = await call_next(request)
    return response


class BufferResponse(BaseModel):
    success: bool
    message: str = ""
    data: Optional[Dict[str, Any]] = None


class BufferQueue:
    def __init__(
        self,
        group_size,
        task_type="math",
        transform_group_func=None,
        is_valid_group_func=None,
        get_group_data_meta_info_func=None,
    ):
        self.data = {}
        self.temp_data = {}
        self.group_timestamps = {}
        self.group_size = group_size
        self.task_type = task_type

        # Set up function handlers with defaults
        self.is_valid_group_func = is_valid_group_func or default_is_valid_group
        self.get_group_data_meta_info_func = get_group_data_meta_info_func or default_get_group_data_meta_info
        self.transform_group_func = transform_group_func or (lambda group, task_type: group)

    def append(self, item):
        instance_id = item["instance_id"]
        current_time = time.time()

        # Update timestamp for this group
        self.group_timestamps[instance_id] = current_time

        if instance_id not in self.temp_data:
            self.temp_data[instance_id] = [copy.deepcopy(item)]
        else:
            self.temp_data[instance_id].append(copy.deepcopy(item))

        if instance_id not in self.data:
            self.data[instance_id] = [item]
        else:
            self.data[instance_id].append(item)

    def _get_valid_groups_with_timeout(self, del_data=False):
        """Get valid groups including timeout-based groups"""
        valid_groups = {}
        timed_out_groups = {}
        finished_groups = []

        for instance_id, group_data in self.data.items():
            if self.is_valid_group_func((instance_id, group_data), self.group_size, self.task_type):
                valid_groups[instance_id] = group_data

        # Remove finished groups and timed out groups with insufficient data
        if del_data:
            for instance_id in finished_groups:
                self.data.pop(instance_id, None)
                self.group_timestamps.pop(instance_id, None)
                # logger.debug(f"Removed finished group {instance_id}")

        # Combine normal valid groups and timeout groups
        all_valid_groups = {**valid_groups, **timed_out_groups}

        return all_valid_groups, finished_groups

    def get(self):
        output = {"data": [], "meta_info": {}}

        # Get meta information about temp data before processing
        meta_info = self.get_group_data_meta_info_func(self.temp_data)
        output["meta_info"] = meta_info

        valid_groups, finished_groups = self._get_valid_groups_with_timeout(del_data=True)
        output["meta_info"]["finished_groups"] = finished_groups

        # logger.debug(f"meta info: {json.dumps(meta_info, indent=2)}")

        valid_groups = list(valid_groups.items())

        for instance_id, group in valid_groups:
            # First filter individual items
            transformed_group = self.transform_group_func((instance_id, group), self.task_type)
            output["data"].extend(transformed_group[1])

            if instance_id in self.data:
                self.data.pop(instance_id)

        return output

    def __len__(self):
        valid_groups, _ = self._get_valid_groups_with_timeout()
        num = sum([len(v) for v in valid_groups.values()])
        num_of_all_groups = sum([len(v) for v in self.data.values()])
        # logger.debug(f"valid_groups: {len(valid_groups)}, num: {num}, num_of_all_groups: {num_of_all_groups}")
        return num


class RolloutBuffer:
    def __init__(
        self,
        group_size=16,
        task_type="math",
        transform_group_func=None,
        is_valid_group_func=None,
        get_group_data_meta_info_func=None,
    ):
        self.buffer = BufferQueue(
            group_size=group_size,
            task_type=task_type,
            transform_group_func=transform_group_func,
            is_valid_group_func=is_valid_group_func,
            get_group_data_meta_info_func=get_group_data_meta_info_func,
        )
        self.lock = threading.RLock()
        self.not_empty = threading.Condition(self.lock)
        self.total_written = 0
        self.total_read = 0
        self.task_type = task_type

    def write(self, data):
        with self.lock:
            self.buffer.append(data)
            self.total_written += 1
            self.not_empty.notify_all()
        return data

    def read(self):
        with self.not_empty:
            if len(self.buffer) == 0:
                return {"data": [], "meta_info": {}}

            # Don't clear temp_data for regular read operations
            result = self.buffer.get()
            self.total_read += len(result["data"])
            return result


buffer = RolloutBuffer()


@app.post("/buffer/write", response_model=BufferResponse)
async def write_to_buffer(request: Request):
    try:
        data = await request.json()
        item = buffer.write(data)
        return BufferResponse(
            success=True,
            message="Data has been successfully written to buffer",
            data={"data": [item], "meta_info": "write to buffer"},
        )
    except Exception as e:
        logger.error(f"Write failed: {str(e)}")
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Write failed: {str(e)}")


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
    # Parse stored prompt (may be JSON serialized messages)
    base_messages: List[Dict[str, Any]] = []
    try:
        prompt_str = row.get("prompt", "")
        if prompt_str:
            loaded = json.loads(prompt_str)
            if isinstance(loaded, list):
                for msg in loaded:
                    if isinstance(msg, dict) and "role" in msg:
                        base_messages.append(msg)
    except Exception:
        base_messages = [{"role": "user", "content": str(row.get("prompt", ""))}]

    messages = list(base_messages)
    messages.append({"role": "assistant", "content": row.get("response", "")})

    session_id = row.get("session_id", "")
    env_id = row.get("env_id", "")

    # 从 env_state 中解析 weight_version（若存在）
    weight_version = 0
    env_state_raw = row.get("env_state")
    if env_state_raw:
        try:
            state = json.loads(env_state_raw)
            wv = state.get("weight_version")
            if wv is None:
                weight_version = 0
            elif isinstance(wv, str) and wv == "default":
                weight_version = 0
            else:
                try:
                    weight_version = int(wv)
                except Exception:
                    weight_version = 0
        except Exception:
            weight_version = 0

    extra_info = {
        "timestamp": _parse_timestamp(row.get("session_end_time")) or _parse_timestamp(row.get("timestamp")) or time.time(),
        "steps": row.get("step_id", 0),
        "finish_reason": "stop",
        "session_id": session_id,
        "env_id": env_id,
        "weight_version": weight_version,
    }

    return {
        "uid": str(uuid.uuid4()),
        "instance_id": str(env_id) if env_id else str(session_id),
        "messages": messages,
        "reward": float(row.get("reward", 0.0)),
        "extra_info": extra_info,
    }


async def fetch_new_items_from_db(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Fetch new completed steps from the database using cursor-based pagination."""
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
            if step_pk > last_served_id:
                last_served_id = step_pk
        except Exception as e:
            logger.error(f"Error building item from row: {e}")
            continue

    return items


def accumulate_and_pop_ready_groups(new_items: List[Dict[str, Any]]) -> tuple:
    """Accumulate items and return ready groups."""
    global pending_items_by_instance, group_size

    ready_groups = []
    finished_instance_ids = []

    # Add new items to pending
    for item in new_items:
        instance_id = str(item.get("instance_id", ""))
        if not instance_id:
            continue
        pending_items_by_instance.setdefault(instance_id, []).append(item)

    # Check for complete groups
    to_delete = []
    for instance_id, bucket in pending_items_by_instance.items():
        while len(bucket) >= group_size:
            group = bucket[:group_size]
            del bucket[:group_size]
            ready_groups.append((instance_id, list(group)))
            finished_instance_ids.append(instance_id)
        if not bucket:
            to_delete.append(instance_id)

    for k in to_delete:
        pending_items_by_instance.pop(k, None)

    return ready_groups, finished_instance_ids


def normalize_group_rewards(groups: List[tuple], eps: float = 1e-8) -> List[Dict[str, Any]]:
    """Normalize rewards within each group (GRPO style)."""
    ready_items = []
    raw_rewards_for_meta = []

    for instance_id, group in groups:
        raw_rewards = [float(item.get("reward", 0.0)) for item in group]
        raw_rewards_for_meta.extend(raw_rewards)

        if not raw_rewards:
            continue

        mean_r = sum(raw_rewards) / len(raw_rewards)
        var_r = sum((r - mean_r) ** 2 for r in raw_rewards) / len(raw_rewards)
        std_r = var_r ** 0.5

        if std_r < eps:
            normalized = [0.0 for _ in raw_rewards]
        else:
            normalized = [(r - mean_r) / (std_r + eps) for r in raw_rewards]

        for item, r_raw, r_norm in zip(group, raw_rewards, normalized):
            item["raw_reward"] = r_raw
            item["reward"] = r_norm
            ready_items.append(item)

    return ready_items, raw_rewards_for_meta


@app.post("/get_rollout_data", response_model=BufferResponse)
async def get_rollout_data(request: Request):
    global pending_items_by_instance

    # First check pending groups
    ready_groups, finished_ids = accumulate_and_pop_ready_groups([])

    # Fetch new items from database
    new_items = await fetch_new_items_from_db(limit=None)
    if new_items:
        more_groups, more_finished = accumulate_and_pop_ready_groups(new_items)
        ready_groups.extend(more_groups)
        finished_ids.extend(more_finished)

    # Log pending status
    pending_counts = {k: len(v) for k, v in pending_items_by_instance.items()}
    logger.info(f"new_items={len(new_items)}, ready_groups={len(ready_groups)}, pending={pending_counts}")

    # Normalize rewards
    ready_items, raw_rewards = normalize_group_rewards(ready_groups)

    total_samples = len(ready_items)
    avg_reward = sum(raw_rewards) / len(raw_rewards) if raw_rewards else 0.0

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


async def init_data_manager(db_url: str, restart_training: bool = False):
    """Initialize the DataManager for querying the database."""
    global data_manager, last_served_id
    data_manager = DataManager(db_url=db_url)
    await data_manager.init()
    logger.info(f"DataManager initialized with DB: {db_url}")

    # Initialize cursor based on restart_training flag
    if restart_training:
        last_served_id = await data_manager.get_max_step_id()
        logger.info(f"restart_training=True, initialized last_served_id={last_served_id}")


def start_llm_proxy() -> subprocess.Popen:
    """Start the LLM Proxy as a subprocess."""
    aievobox_root = os.environ.get("AIEVOBOX_ROOT", "/root/AIEvoBox")
    llm_proxy_script = os.path.join(aievobox_root, "rl", "llm_proxy.py")

    env = os.environ.copy()
    env["LLM_PROXY_HOST"] = "0.0.0.0"
    env["LLM_PROXY_PORT"] = "8890"

    logger.info(f"Starting LLM Proxy: {llm_proxy_script}")

    process = subprocess.Popen(
        ["python3", llm_proxy_script],
        env=env,
        stdout=None,
        stderr=None,
    )
    logger.info(f"LLM Proxy started with PID: {process.pid}")
    return process


def init_llm_proxy(tokenizer_path: str, remote_engine_url: str, max_length: int = None, max_retries: int = 10):
    """Initialize the LLM Proxy with tokenizer and remote engine URL."""
    import requests

    init_url = f"{llm_proxy_url}/init"
    payload = {
        "tokenizer_path": tokenizer_path,
        "remote_engine_url": remote_engine_url,
        "max_length": max_length,
    }

    for attempt in range(max_retries):
        try:
            resp = requests.post(init_url, json=payload, timeout=10)
            if resp.status_code == 200:
                logger.info(f"LLM Proxy initialized successfully")
                return True
            else:
                logger.warning(f"LLM Proxy init failed: {resp.status_code}")
        except Exception as e:
            logger.warning(f"LLM Proxy init attempt {attempt+1}/{max_retries} failed: {e}")
        time.sleep(2)

    logger.error(f"Failed to initialize LLM Proxy after {max_retries} attempts")
    return False


def start_aievobox_process(data: dict):
    """Start AIEvoBox as a subprocess."""
    global aievobox_process, llm_proxy_process, group_size, last_served_id, pending_items_by_instance, data_manager

    # Set group size
    group_size = int(data.get("num_repeat_per_sample", 16))

    # Clear state for new rollout
    restart_training = data.get("restart_training", False)
    if restart_training:
        pending_items_by_instance.clear()
        logger.info("restart_training=True, cleared pending items")

    # Initialize DataManager
    db_url = os.environ.get("AIEVOBOX_DB_URL", f"sqlite:///{AIEVOBOX_ROOT}/rl/rl.db")

    # Run async init in a new event loop (since we're in a thread)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(init_data_manager(db_url, restart_training=restart_training))
    finally:
        loop.close()

    # Start LLM Proxy if not running
    if llm_proxy_process is None or llm_proxy_process.poll() is not None:
        llm_proxy_process = start_llm_proxy()
        time.sleep(2)  # Wait for proxy to start

    # Initialize LLM Proxy with tokenizer and remote engine URL
    tokenizer_path = data.get("tokenizer_path", "")
    remote_engine_url = data.get("remote_engine_url", "")
    # 从环境变量读取 LLM_MAX_LENGTH
    max_length_str = os.environ.get("LLM_MAX_LENGTH")
    max_length = int(max_length_str) if max_length_str else None
    if tokenizer_path and remote_engine_url:
        init_llm_proxy(tokenizer_path, remote_engine_url, max_length=max_length)

    # Prepare environment variables for AIEvoBox
    # AIEvoBox should call LLM Proxy instead of remote engine directly
    env = os.environ.copy()
    env["AIEVOBOX_ROLLOUT_CONFIG"] = json.dumps(data)
    env["AIEVOBOX_DB_URL"] = db_url
    env["ROLLOUT_BUFFER_URL"] = data.get("remote_buffer_url", os.environ.get("ROLLOUT_BUFFER_URL", "http://127.0.0.1:8889"))
    env["LLM_PROXY_URL"] = llm_proxy_url  # AIEvoBox uses LLM Proxy
    env["REMOTE_ENGINE_URL"] = remote_engine_url  # Keep original for reference
    env["NUM_REPEAT_PER_SAMPLE"] = str(group_size)
    env["ROLLOUT_MAX_STEPS"] = str(data.get("max_steps", 10))

    # AIEvoBox entry script path
    aievobox_root = os.environ.get("AIEVOBOX_ROOT", "/root/AIEvoBox")
    aievobox_script = os.path.join(aievobox_root, "rl", "aievobox_runner.py")

    logger.info(f"Starting AIEvoBox subprocess: {aievobox_script}")
    logger.info(f"Config: group_size={group_size}, db_url={db_url}")
    logger.info(f"LLM Proxy URL: {llm_proxy_url}")

    try:
        aievobox_process = subprocess.Popen(
            ["python3", aievobox_script],
            env=env,
            stdout=None,  # Inherit stdout
            stderr=None,  # Inherit stderr
        )
        logger.info(f"AIEvoBox started with PID: {aievobox_process.pid}")
    except Exception as e:
        logger.error(f"Failed to start AIEvoBox: {e}")
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
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8889,
        limit_concurrency=1000,  # Connection concurrency limit
        # limit_max_requests=1000000,  # Maximum request limit
        timeout_keep_alive=5,  # Keep-alive timeout,
    )
