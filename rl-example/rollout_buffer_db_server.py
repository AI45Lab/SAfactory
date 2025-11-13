import os
import sys
import uuid
import json
import asyncio
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, HTTPException, Request
import httpx
from openai import OpenAI
from pydantic import BaseModel

"""
DB-backed rollout buffer server for Trading env.

Uses AIEvoBox's DataManager/Interactor to produce data into DB, and Tortoise ORM
models to read freshly completed steps (done=True). Each done step is treated as
one complete trajectory sample (system+user+assistant), without merging steps.
"""

_EVOBOX_ROOT = os.environ.get("AIEVOBOX_ROOT", "/root/AIEvoBox")
if os.path.isdir(_EVOBOX_ROOT) and _EVOBOX_ROOT not in sys.path:
    sys.path.insert(0, _EVOBOX_ROOT)
try:
    from core.data_manager.manager import DataManager
    from core.data_manager.models import InteractionSession, InteractionStep
    from core.agent.base_agent import APIAgent
    from core.interactor import Interactor
    from core.types.base import deserialize_prompt_output
    # Ensure envs are registered into registry
    import importlib
    try:
        importlib.import_module("env.tradinggym.trading_env")
        print("[DB Buffer] Imported env.tradinggym.trading_env for registry")
    except Exception as e:
        print(f"[DB Buffer] Warning: failed to import trading env module: {e}")
except Exception as e:
    raise RuntimeError(f"Failed to import AIEvoBox modules: {e}")


class BufferResponse(BaseModel):
    success: bool
    message: str = ""
    data: Optional[Dict[str, Any]] = None


app = FastAPI(title="Trading DB Rollout Buffer", debug=True)


# In-memory served cache to avoid duplicate delivery
_SERVED_DONE_STEPS: set[int] = set()


def _msg_to_dict(openai_message) -> Dict[str, Any]:
    return {
        "role": openai_message.role,
        "content": [item.root.model_dump() for item in openai_message.content],
    }


def _flatten_openai_message_to_text(openai_message) -> str:
    try:
        parts = []
        for item in openai_message.content:
            root = item.root
            t = getattr(root, "text", None)
            if isinstance(t, str) and t.strip():
                parts.append(t.strip())
        return "\n".join(parts)
    except Exception:
        # Fallback to string dump
        return _truncate_str(str(openai_message), 400)

# 这里需要继承Agent类？
class StringContentAgent:
    def __init__(self, api_key: str, base_url: str, model: str, temperature: float = 1.0) -> None:
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.temperature = temperature
        self._base_url = base_url.rstrip("/")
        self._router_base = self._base_url[:-3] if self._base_url.endswith("/v1") else self._base_url

    def generate(self, prompt_output) -> str:
        # Convert PromptOutput to OpenAI messages with string content
        messages: list[dict[str, str]] = []
        try:
            sys_str = _flatten_openai_message_to_text(prompt_output.system_message)
            if sys_str:
                messages.append({"role": "system", "content": sys_str})
        except Exception:
            pass
        usr_str = _flatten_openai_message_to_text(prompt_output.user_message)
        messages.append({"role": "user", "content": usr_str})

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
        )
        return resp.choices[0].message.content

    def is_healthy(self, timeout: float = 3.0) -> bool:
        try:
            url = f"{self._router_base}/health"
            with httpx.Client(timeout=timeout) as c:
                r = c.get(url)
                return r.status_code == 200
        except Exception:
            return False

    async def is_healthy_async(self, timeout: float = 3.0) -> bool:
        try:
            url = f"{self._router_base}/health"
            async with httpx.AsyncClient(timeout=timeout) as c:
                r = await c.get(url)
                return r.status_code == 200
        except Exception:
            return False


def _first_text_from_msg_dict(msg_dict: Dict[str, Any]) -> str:
    try:
        content = msg_dict.get("content", [])
        if isinstance(content, list) and content:
            item = content[0]
            root = item.get("root", item)
            t = root.get("text")
            return t if isinstance(t, str) else ""
    except Exception:
        pass
    return ""


def _prompt_to_sys_user_str(prompt_json: str) -> Tuple[Optional[str], str]:
    """Extract plain strings for system/user; ignore type wrappers and multimodal parts."""
    try:
        if deserialize_prompt_output is not None:
            prompt_obj = deserialize_prompt_output(prompt_json)
            sys_d = _msg_to_dict(prompt_obj.system_message)
            usr_d = _msg_to_dict(prompt_obj.user_message)
        else:
            d = json.loads(prompt_json)
            sys_d = d.get("system_message", {})
            usr_d = d.get("user_message", {})
        sys_s = _first_text_from_msg_dict(sys_d) if sys_d else None
        usr_s = _first_text_from_msg_dict(usr_d)
        return sys_s, usr_s
    except Exception:
        # Fallback: treat the whole prompt as user text
        return None, str(prompt_json)


def _to_oai_messages_from_prompt_response(prompt_json: str, response_text: str) -> List[Dict[str, Any]]:
    sys_s, usr_s = _prompt_to_sys_user_str(prompt_json)
    messages: List[Dict[str, Any]] = []
    if sys_s:
        messages.append({"role": "system", "content": sys_s})
    messages.append({"role": "user", "content": usr_s})
    messages.append({"role": "assistant", "content": response_text})
    return messages


def _prompt_to_sys_user(prompt_json: str) -> Tuple[Optional[str], str]:
    return _prompt_to_sys_user_str(prompt_json)


def _truncate_str(s: str, max_len: int = 160) -> str:
    if not isinstance(s, str):
        s = str(s)
    if len(s) <= max_len:
        return s
    head = max_len // 2
    tail = max_len - head
    return f"{s[:head]}...{s[-tail:]}"


def _human_preview(obj: Any, *, max_str: int = 160, max_list: int = 3, max_dict_keys: int = 10, _depth: int = 0) -> Any:
    try:
        # Primitives
        if obj is None or isinstance(obj, (int, float, bool)):
            return obj
        if isinstance(obj, str):
            return _truncate_str(obj, max_len=max_str)
        # Lists / tuples
        if isinstance(obj, (list, tuple)):
            shown = []
            for i, v in enumerate(obj):
                if i >= max_list:
                    shown.append(f"... ({len(obj) - i} more)")
                    break
                shown.append(_human_preview(v, max_str=max_str, max_list=max_list, max_dict_keys=max_dict_keys, _depth=_depth + 1))
            return shown
        # Dicts
        if isinstance(obj, dict):
            shown = {}
            for i, (k, v) in enumerate(obj.items()):
                if i >= max_dict_keys:
                    shown["..."] = f"({len(obj) - i} more keys)"
                    break
                shown[k] = _human_preview(v, max_str=max_str, max_list=max_list, max_dict_keys=max_dict_keys, _depth=_depth + 1)
            return shown
        # Fallback to string
        return _truncate_str(str(obj), max_len=max_str)
    except Exception:
        return "<unserializable>"


async def _build_item_from_done_step(step: InteractionStep) -> Dict[str, Any]:
    # Build messages from this single step only
    sys_s, usr_s = _prompt_to_sys_user(step.prompt)  # plain strings
    messages: List[Dict[str, Any]] = []
    if sys_s:
        messages.append({"role": "system", "content": sys_s})
    messages.append({"role": "user", "content": usr_s})
    messages.append({"role": "assistant", "content": step.response})

    # Fetch related session info for reward and timestamp
    total_reward = None
    end_ts = None
    try:
        await step.fetch_related("session")
        sess = step.session
        total_reward = getattr(sess, "total_reward", None)
        end_ts = getattr(sess, "end_time", None)
    except Exception:
        pass

    extra_info = {
        "timestamp": (end_ts.timestamp() if end_ts else (step.timestamp.timestamp() if step.timestamp else None)),
        "steps": step.step_id,
        "finish_reason": "stop",
        "round_number": 1,  # exactly one assistant turn in this packaged sample
    }
    return {
        "uid": str(uuid.uuid4()),
        "instance_id": str(uuid.uuid4()),
        "messages": messages,
        "reward": float(total_reward if total_reward is not None else (step.reward or 0.0)),
        "extra_info": extra_info,
    }


async def _iter_new_items_from_db(dm: DataManager, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    await dm.init()
    done_steps = await InteractionStep.filter(done=True).order_by("timestamp").limit(limit or 10000)
    items: List[Dict[str, Any]] = []
    for st in done_steps:
        rid = getattr(st, "id", None)
        if rid in _SERVED_DONE_STEPS:
            continue
        try:
            item = await _build_item_from_done_step(st)
            items.append(item)
            if rid is not None:
                _SERVED_DONE_STEPS.add(rid)
        except Exception as e:
            print(f"[DB Buffer] Skip done step due to error: {e}")
            continue
    return items


# Global roll state
class RollState:
    def __init__(self):
        self.db_url = os.environ.get("AIEVOBOX_DB_URL", f"sqlite:////root/AIEvoBox/trading_multi_envs.db")
        self.started = False
        self.env_params = {}


STATE = RollState()


@app.post("/start_rollout")
async def start_rollout(request: Request):
    payload = await request.json()
    remote_engine_url = payload.get("remote_engine_url")
    sampling_params = payload.get("sampling_params", {})
    # We do not directly use sampling_params here
    try:
        print(
            "[DB Buffer][start_rollout] payload:",
            json.dumps(
                {
                    "remote_engine_url": remote_engine_url,
                    "sampling_params": {k: sampling_params.get(k) for k in ["max_tokens", "temperature", "top_p"] if k in sampling_params},
                },
                ensure_ascii=False,
            ),
        )
    except Exception as _e:
        print(f"[DB Buffer][start_rollout] preview error: {_e}")

    # Normalize remote_engine_url to OpenAI-compatible path
    if remote_engine_url and not remote_engine_url.rstrip("/").endswith("/v1"):
        remote_engine_url = remote_engine_url.rstrip("/") + "/v1"
        print("[DB Buffer][start_rollout] normalized remote_engine_url to:", remote_engine_url)

    # Allow TRADING_* envs to flow through
    env_params = {
        "data_dir": os.environ.get("TRADING_DATA_DIR"),
        "price_filename": os.environ.get("TRADING_PRICE_FILE"),
        "tweet_filename": (os.environ.get("TRADING_TWEET_FILE") or None),
        "window_size": int(os.environ.get("TRADING_WINDOW", "7")),
    }
    STATE.env_params = env_params
    if not remote_engine_url:
        raise HTTPException(status_code=400, detail="remote_engine_url is required")

    if STATE.started:
        return {"message": "Rollout already started"}

    async def _run_interactor_job():
        dm = DataManager(db_url=os.environ.get("AIEVOBOX_DB_URL", STATE.db_url))
        await dm.init()
        envs = await dm.get_all_environments()
        print(f"[DB Buffer][interactor] existing env count={len(envs)}")
        if not envs:
            await dm.add_environment_config(
                env_name="trading_gym",
                **{k: v for k, v in env_params.items() if v is not None},
            )
            print("[DB Buffer][interactor] created default env 'trading_gym'")
        else:
            # Update existing trading_gym env with required params if missing
            for cfg in envs:
                if getattr(cfg, "env_name", None) != "trading_gym":
                    continue
                params = getattr(cfg, "env_params", {}) or {}
                required = ("data_dir", "price_filename")
                needs = [k for k in required if not params.get(k)]
                if needs:
                    merged = dict(params)
                    for k in ("data_dir", "price_filename", "tweet_filename", "window_size"):
                        if env_params.get(k) is not None:
                            merged[k] = env_params.get(k)
                    try:
                        await dm.update_environment_config(
                            env_name="trading_gym",
                            env_id=getattr(cfg, "env_id"),
                            **merged,
                        )
                        print(f"[DB Buffer][interactor] updated env params for {cfg.env_id}")
                    except Exception as e:
                        print(f"[DB Buffer][interactor] failed to update env params: {e}")

        # Create agent and interactor (use string-content agent to avoid 400 on list content)
        api_key = os.environ.get("OPENAI_API_KEY", "test")
        model = os.environ.get("AIEVOBOX_MODEL", "custom")
        agent = StringContentAgent(api_key=api_key, base_url=remote_engine_url, model=model, temperature=1.0)
        visual_dir = os.environ.get("AIEVOBOX_VIS_DIR", "/tmp/aievobox_vis")
        max_workers = int(os.environ.get("AIEVOBOX_MAX_WORKERS", "1"))
        interactor = Interactor(
            agent=agent,
            data_manager=dm,
            max_workers=max_workers,
            max_steps=int(os.environ.get("TRADING_MAX_STEPS", 64)),
            visual_save_path=visual_dir,
        )
        while True:
            envs = await dm.get_all_environments()
            # 测试：Only run trading_gym envs
            envs = [cfg for cfg in envs if getattr(cfg, "env_name", None) == "trading_gym"]
            if not envs:
                await asyncio.sleep(1.0)
                continue
            print(f"[DB Buffer][interactor] will run {len(envs)} env(s) concurrently (max_workers={max_workers})")
            # Use Interactor's built-in concurrency to run each environment once (one prompt per env)
            try:
                # 这里有一个定义问题，一个env不应该视作一条samples，但是现在就是等价的。
                # 另外一个问题：现在在模型同步权重的时候是不是需要考虑agent如果unhealthy怎么处理。
                # grpo还需要相同的 instance_id，但是考虑到我们现在的测试只重复跑一个env，所以先不加上了。
                await interactor.run_all_environments()
            except Exception as e:
                print(f"[DB Buffer][interactor] run_all_environments error: {e}")
                await asyncio.sleep(1.0)
                continue

    asyncio.create_task(_run_interactor_job())
    STATE.started = True
    return {"message": "Rollout started (interactor running in background)"}


@app.post("/get_rollout_data", response_model=BufferResponse)
async def get_rollout_data(request: Request):
    try:
        dm = DataManager(db_url=os.environ.get("AIEVOBOX_DB_URL", STATE.db_url))
        # Ensure we return a slightly oversized chunk to avoid exact-equal fetch in client
        # 这个可能是slime的实现的问题：如果使用rollout_num=1，那么有可能会导致一个判断数据的地方出问题。
        try:
            min_return = int(os.environ.get("ROLLBUF_MIN_RETURN", "17"))
            wait_ms = int(os.environ.get("ROLLBUF_WAIT_MS", "2000"))
        except Exception:
            min_return, wait_ms = 17, 2000

        # Try to collect at least `min_return` items within wait budget
        deadline = asyncio.get_event_loop().time() + (wait_ms / 1000.0)
        items = await _iter_new_items_from_db(dm, limit=None)
        while len(items) < min_return and asyncio.get_event_loop().time() < deadline:
            await asyncio.sleep(0.2)
            items = await _iter_new_items_from_db(dm, limit=None)

        # Ensure minimum return; if more are available, return all (minimum semantics)
        total_samples = len(items)
        avg_reward = sum((it.get("reward") or 0.0) for it in items) / total_samples if total_samples > 0 else 0.0
        meta_info = {"total_samples": total_samples, "avg_reward": avg_reward, "finished_groups": []}
        # Human-readable preview to stdout
        try:
            preview_items = _human_preview(items, max_str=200, max_list=3, max_dict_keys=10)
            # Only print a small slice for readability
            if isinstance(preview_items, list) and len(preview_items) > 3:
                preview_items = preview_items[:3] + [f"... ({len(items) - 3} more items)"]
            print("[DB Buffer][get_rollout_data] meta=", json.dumps(meta_info, ensure_ascii=False))
            print("[DB Buffer][get_rollout_data] sample_preview=", json.dumps(preview_items, ensure_ascii=False, indent=2))
        except Exception as _e:
            print(f"[DB Buffer][get_rollout_data] preview error: {_e}")
        if total_samples == 0:
            return BufferResponse(success=False, message="No data available to read", data={"data": [], "meta_info": meta_info})
        return BufferResponse(success=True, message=f"Successfully read {total_samples} items", data={"data": items, "meta_info": meta_info})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch rollout data from DB: {e}")


# Optional: write endpoint for compatibility (no-op)
@app.post("/buffer/write", response_model=BufferResponse)
async def write_to_buffer(request: Request):
    return BufferResponse(success=True, message="DB-backed buffer does not accept writes", data={"data": [], "meta_info": {}})


def main():
    import uvicorn

    host = os.environ.get("ROLLBUF_HOST", "0.0.0.0")
    port = int(os.environ.get("ROLLBUF_PORT", 8889))
    uvicorn.run(app, host=host, port=port, limit_concurrency=1000, timeout_keep_alive=5)


if __name__ == "__main__":
    main()
