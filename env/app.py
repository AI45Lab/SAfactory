import os
import sys

current_file_path = os.path.abspath(__file__)
examples_dir = os.path.dirname(current_file_path)
project_root = os.path.dirname(examples_dir)
sys.path.append(project_root)

import logging
import threading
import json
import ray

from typing import Any, Dict, Optional, Tuple, List, Callable, Type
from openai.types.chat import ChatCompletionMessageParam
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field

from core.types.base import ResetOutput, RenderOutput, StepOutput, dumps_json_bytes

from env.registry import (_import_os_env, _import_search_env, _import_emb_env, _import_gym_env,
                          _import_android_gym, _import_trading_env, _import_mc_env, _import_mc_gpu_env,
                          _import_geo3k_vl_test_env, _import_qa_gym,
                          _import_geo3k_vl_test_env, _import_qa_gym,
                          _import_geo3k_vl_test_env,
                          _import_oepnclaw_env,
                          _import_oepnclaw_env, _import_dab_env, _import_dw_env)

ENV_CLASS_REGISTRY: Dict[str, Callable[[], Type]] = {
    "android_gym": _import_android_gym,
    "search": _import_search_env,
    "trading_gym": _import_trading_env,
    "mc": _import_mc_env,
    "emb": _import_emb_env,
    "git_gym": _import_gym_env,
    "os_gym": _import_os_env,
    "mc_gpu": _import_mc_gpu_env,
    "geo3k_vl_test": _import_geo3k_vl_test_env,
    "qa_gym": _import_qa_gym,
    "openclawgym": _import_oepnclaw_env
    "openclawgym": _import_oepnclaw_env,
    "dabstepgym": _import_dab_env,
    "discoveryworld": _import_dw_env
}


logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("env-service")

# -------------------------------------------------------------------
# Ray actor: one actor hosts one env instance
# -------------------------------------------------------------------
@ray.remote(max_concurrency=2)
class EnvActor:
    """
    A single actor hosts one env instance.
    """

    def __init__(self, envname: str, id_: str, create_kwargs: Optional[Dict[str, Any]] = None):
        self.envname = envname
        self.id = str(id_)

        logger.info(f"Actor Init: {envname}/{id_}")

        try:
            env_import_func = ENV_CLASS_REGISTRY[envname]
            EnvCls = env_import_func()
        except KeyError:
            logger.error(f"Unknown envname: {envname}")
            raise ValueError(f"Unknown envname: {envname} (available: {list(ENV_CLASS_REGISTRY.keys())})")
        except ImportError as e:
            logger.error(f"Import failed for {envname}: {e}")
            raise ImportError(
                f"Failed to import env '{envname}': {e}\n"
                f"Please install dependencies for {envname} environment first."
            )

        self.env = EnvCls(**(create_kwargs or {}))

    def reset(self, seed: Optional[int] = None) -> bytes:
        out: ResetOutput = self.env.reset(seed=seed)
        return dumps_json_bytes(out)

    def step(self, action: str) -> bytes:
        out: StepOutput = self.env.step(action)
        return dumps_json_bytes(out)

    def render(self) -> bytes:
        out: RenderOutput = self.env.render()
        return dumps_json_bytes(out)

    def get_task_prompt(self) -> bytes:
        out: List[ChatCompletionMessageParam] = self.env.get_task_prompt()
        return dumps_json_bytes(out)

    def close(self) -> dict:
        logger.info(f"Actor Close: {self.envname}/{self.id}")
        return self.env.close()

    def is_done(self) -> bool:
        return self.env.is_done()

    def health(self) -> bool:
        return self.env.health()

    def describe(self) -> dict:
        return {
            "env": self.envname,
            "id": self.id,
            "class": self.env.__class__.__name__,
            "done": self.env.is_done(),
        }


# -------------------------------------------------------------------
# In-process map: (envname, id) -> EnvActor handle
# -------------------------------------------------------------------
_ENV_ACTORS: Dict[Tuple[str, str], ray.actor.ActorHandle] = {}
_ENV_LOCK = threading.RLock()


def _key(envname: str, env_id: str) -> Tuple[str, str]:
    return envname, str(env_id)


def _init_ray_if_needed() -> None:
    """Init Ray once; connect to a cluster if RAY_ADDRESS is set."""
    if ray.is_initialized():
        return
    address = os.getenv("RAY_ADDRESS")
    if address:
        logger.info(f"Connecting to Ray cluster at {address}")
        ray.init(address=address)
    else:
        logger.info("Starting local Ray")
        ray.init()


# -------------------------------------------------------------------
# FastAPI models
# -------------------------------------------------------------------
class ResetRequest(BaseModel):
    env_param: Any = Field(
        ...,
        description="Env creation parameters, either a JSON object or a JSON string stored in DB (env_param)."
    )
    seed: Optional[int] = Field(
        default=None,
        description="Optional seed passed into env.reset(seed=...).",
    )


class StepRequest(BaseModel):
    action: str = Field(..., description="Action string passed into env.step(action).")


# -------------------------------------------------------------------
# FastAPI app
# -------------------------------------------------------------------
app = FastAPI(
    title="Env HTTP Service (Ray + TradingGym)",
    version="0.2.0",
)


@app.on_event("startup")
def on_startup() -> None:
    _init_ray_if_needed()


@app.get("/envs")
async def list_envs() -> Dict[str, Any]:
    """List all currently tracked env actors."""
    with _ENV_LOCK:
        envs = [{"envname": k[0], "id": k[1]} for k in _ENV_ACTORS.keys()]
    return {"envs": envs}


@app.post("/{envname}/{env_id}/reset")
async def reset_env(envname: str, env_id: str, req: ResetRequest) -> Response:
    """
    Reset env identified by (envname, env_id).
    """
    _init_ray_if_needed()

    if envname not in ENV_CLASS_REGISTRY:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown envname '{envname}'. "
                   f"Available: {list(ENV_CLASS_REGISTRY.keys())}",
        )

    raw_param = req.env_param
    if isinstance(raw_param, str):
        try:
            create_kwargs = json.loads(raw_param)
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode failed in reset: {e}")
            raise HTTPException(status_code=400, detail=f"Invalid env_param JSON string: {e}")
    elif isinstance(raw_param, dict):
        create_kwargs = dict(raw_param)
    else:
        raise HTTPException(status_code=400, detail="env_param must be a JSON object or JSON string.")

    key = _key(envname, env_id)

    # 1. Fast Path: Optimistic read without lock (Python dict get is atomic)
    actor = _ENV_ACTORS.get(key)

    # 2. Slow Path: Create only if missing
    if actor is None:
        with _ENV_LOCK:
            # 3. Double-Check: Verify again inside lock to handle race conditions
            actor = _ENV_ACTORS.get(key)

            if actor is None:
                try:
                    logger.info(f"Reset: {envname}/{env_id} (Creating new actor)")
                    actor = EnvActor.remote(envname, env_id, create_kwargs)
                    _ENV_ACTORS[key] = actor
                except Exception as e:
                    # Clean up if creation fails
                    _ENV_ACTORS.pop(key, None)
                    if actor is not None:
                        try:
                            ray.kill(actor,no_restart=True)
                        except Exception as e:
                            pass
                    logger.error(f"Actor creation failed: {e}")
                    raise HTTPException(status_code=500, detail=f"Actor creation failed: {e}")
            else:
                logger.info(f"Reset: {envname}/{env_id} (Reusing actor created by another thread)")
    else:
        logger.info(f"Reset: {envname}/{env_id} (Reusing existing actor)")

    # 4. Execute Reset
    try:
        result_bytes: bytes = await actor.reset.remote(req.seed)
        return Response(content=result_bytes, media_type="application/json")

    except Exception as e:
        logger.error(f"Reset execution failed for {envname}/{env_id}: {e}")

        with _ENV_LOCK:
            if _ENV_ACTORS.get(key) == actor:
                _ENV_ACTORS.pop(key, None)
                try:
                    actor.close().remote()
                except Exception as e:
                    pass
                try:
                    ray.kill(actor, no_restart=True)
                except Exception:
                    pass

        raise HTTPException(status_code=500, detail=f"Reset execution failed: {e}")


@app.post("/{envname}/{env_id}/step")
async def step_env(envname: str, env_id: str, req: StepRequest) -> Response:
    """Forward step(action) to the existing actor."""
    key = _key(envname, env_id)

    actor = _ENV_ACTORS.get(key)

    if actor is None:
        logger.warning(f"Step on missing actor: {envname}/{env_id}")
        raise HTTPException(status_code=404, detail=f"Env actor not found: {envname}:{env_id}")
    result_bytes: bytes = await actor.step.remote(req.action)
    return Response(content=result_bytes, media_type="application/json")


@app.get("/{envname}/{env_id}/render")
async def render_env(envname: str, env_id: str) -> Response:
    """Forward render() to the existing actor."""
    key = _key(envname, env_id)
    actor = _ENV_ACTORS.get(key)

    if actor is None:
        raise HTTPException(status_code=404, detail=f"Env actor not found: {envname}:{env_id}")
    result_bytes: bytes = await actor.render.remote()
    return Response(content=result_bytes, media_type="application/json")


@app.get("/{envname}/{env_id}/get_task_prompt")
async def get_task_prompt(envname: str, env_id: str) -> Response:
    """Forward get_task_prompt() to the existing actor."""
    key = _key(envname, env_id)
    actor = _ENV_ACTORS.get(key)
    if actor is None:
        raise HTTPException(status_code=404, detail=f"Env actor not found: {envname}:{env_id}")
    result_bytes: bytes = await actor.get_task_prompt.remote()
    return Response(content=result_bytes, media_type="application/json")


@app.get("/{envname}/{env_id}/is_done")
async def is_done(envname: str, env_id: str) -> Dict[str, Any]:
    """Check if the env is done."""
    key = _key(envname, env_id)
    with _ENV_LOCK:
        actor = _ENV_ACTORS.get(key)
    if actor is None:
        raise HTTPException(status_code=404, detail=f"Env actor not found: {envname}:{env_id}")
    value: bool = await actor.is_done.remote()
    return {"envname": envname, "id": env_id, "done": value}


@app.get("/{envname}/{env_id}/health")
async def health(envname: str, env_id: str) -> Dict[str, Any]:
    """Health check for this env actor."""
    key = _key(envname, env_id)
    with _ENV_LOCK:
        actor = _ENV_ACTORS.get(key)
    if actor is None:
        raise HTTPException(status_code=404, detail=f"Env actor not found: {envname}:{env_id}")
    value: bool = await actor.health.remote()
    return {"envname": envname, "id": env_id, "healthy": value}


@app.get("/{envname}/{env_id}/describe")
async def describe(envname: str, env_id: str) -> Dict[str, Any]:
    """Expose EnvActor.describe()."""
    key = _key(envname, env_id)
    with _ENV_LOCK:
        actor = _ENV_ACTORS.get(key)
    if actor is None:
        raise HTTPException(status_code=404, detail=f"Env actor not found: {envname}:{env_id}")
    info: dict = await actor.describe.remote()
    return info


@app.delete("/{envname}/{env_id}")
async def close_env(envname: str, env_id: str) -> Dict[str, Any]:
    """Close and remove the env actor for (envname, env_id)."""
    key = _key(envname, env_id)
    with _ENV_LOCK:
        actor = _ENV_ACTORS.pop(key, None)
    if actor is not None:
        try:
            await actor.close.remote()
            logger.info(f"Closed: {envname}/{env_id}")
        except Exception:
            pass
        ray.kill(actor, no_restart=True)
    return {"status": "ok", "envname": envname, "id": env_id}


if __name__ == "__main__":
    import uvicorn

    _init_ray_if_needed()
    host = "0.0.0.0"
    port = 36663
    logger.info(f"Starting server at {host}:{port}")
    uvicorn.run(app, host=host, port=port)