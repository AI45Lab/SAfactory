import os
import sys
current_file_path = os.path.abspath(__file__)
examples_dir = os.path.dirname(current_file_path)
project_root = os.path.dirname(examples_dir)
sys.path.append(project_root)

import threading
import json
import dataclasses
import numpy as np
import ray

from typing import Any, Dict, Optional, Tuple, List
from openai.types.chat import ChatCompletionMessageParam
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field

# from env.tradinggym.trading_env import TradingGym
# from env.mc.mc_env import MCGym
# from env.gitgym.git_env import GitGym
# from env.dabstep.dabstep_env import DABStepEnv
# from env.dwgym.dw_env import DiscoveryWorldEnv
# from env.embodiedgym.embodied_env import EmbodiedAlfredGym
# from env.androidgym.android_env import AndroidGym
# from env.search.search_env import SearchEnv
from env.mc_gpu_gym.mc_gpu_env import MCGPUGym

from core.types.base import ResetOutput, RenderOutput, StepOutput, dumps_json_bytes

# -------------------------------------------------------------------
# Env registry: map envname -> concrete env class
# Add more envs here if you have multiple implementations.
# -------------------------------------------------------------------
ENV_CLASS_REGISTRY: Dict[str, type] = {
    # "android_gym": AndroidGym
    # "mc_gym": MCGym,
    "mc_gpu_gym": MCGPUGym,
    # "TradingGym": TradingGym,    # convenience alias
    # "git_gym": GitGym,
    # "dab_gym": DABStepEnv,
    # "dab": DABStepEnv,
    # "dwgym": DiscoveryWorldEnv,
    # "emb": EmbodiedAlfredGym
    # "search": SearchEnv,
}



# -------------------------------------------------------------------
# Ray actor: one actor hosts one env instance
# -------------------------------------------------------------------
@ray.remote(max_concurrency=2)
class EnvActor:
    """
    A single actor hosts one env instance.

    The env class is selected by 'envname' using ENV_CLASS_REGISTRY.
    The env must implement:
        reset(seed: Optional[int]) -> ResetOutput
        step(action: str) -> StepOutput
        render() -> RenderOutput
        get_task_prompt() -> PromptOutput
        close() -> dict
        is_done() -> bool
        health() -> bool
    """

    def __init__(self, envname: str, id_: str, create_kwargs: Optional[Dict[str, Any]] = None):
        self.envname = envname
        self.id = str(id_)

        try:
            EnvCls = ENV_CLASS_REGISTRY[envname]
        except KeyError:
            raise ValueError(f"Unknown envname: {envname} (available: {list(ENV_CLASS_REGISTRY.keys())})")

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
# This lives only inside the HTTP process; it routes HTTP calls to actors.
# -------------------------------------------------------------------
_ENV_ACTORS: Dict[Tuple[str,str], ray.actor.ActorHandle] = {}
_ENV_LOCK = threading.RLock()


def _key(envname: str, env_id: str) -> Tuple[str, str]:
    return envname, str(env_id)


def _init_ray_if_needed() -> None:
    """Init Ray once; connect to a cluster if RAY_ADDRESS is set."""
    if ray.is_initialized():
        return
    address = os.getenv("RAY_ADDRESS")
    if address:
        ray.init(address=address)
    else:
        ray.init()


# -------------------------------------------------------------------
# FastAPI models
# -------------------------------------------------------------------
class ResetRequest(BaseModel):
    """
    Reset will always create a NEW actor for (envname, env_id) and call env.reset().

    - env_param: env creation parameters, recommended to be a JSON object.
      For backward compatibility, a JSON string is also accepted
      (e.g. directly from DB), and will be json.loads() into a dict.
    """
    env_param: Any = Field(
        ...,
        description=(
            "Env creation parameters, either a JSON object or a JSON string "
            "stored in DB (env_param)."
        ),
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

    - Always creates a new EnvActor.
    - If an actor already exists, it is closed and replaced.
    - Returns JSON bytes from EnvActor.reset().
    """

    _init_ray_if_needed()

    # Validate envname early
    if envname not in ENV_CLASS_REGISTRY:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown envname '{envname}'. "
                   f"Available: {list(ENV_CLASS_REGISTRY.keys())}",
        )

    # ---- decode env_param (can be JSON object or JSON string) ----
    raw_param = req.env_param

    if isinstance(raw_param, str):
        # case 1: directly pass the JSON string from DB (env_param)
        try:
            create_kwargs: Dict[str, Any] = json.loads(raw_param)
        except json.JSONDecodeError as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid env_param JSON string: {e}",
            )
    elif isinstance(raw_param, dict):
        # case 2 (recommended): send JSON object directly
        create_kwargs = dict(raw_param)
    else:
        raise HTTPException(
            status_code=400,
            detail="env_param must be a JSON object or JSON string.",
        )


    key = _key(envname, env_id)

    # Create new actor and replace old one atomically
    with _ENV_LOCK:
        old_actor = _ENV_ACTORS.get(key)
        actor = EnvActor.remote(envname, env_id, create_kwargs)
        _ENV_ACTORS[key] = actor

    # Best-effort cleanup of old actor (if any)
    if old_actor is not None:
        try:
            await old_actor.close.remote()
        except Exception:
            pass
        ray.kill(old_actor, no_restart=True)
    # Call reset on the new actor, return JSON bytes directly
    result_bytes: bytes = await actor.reset.remote(req.seed)
    return Response(content=result_bytes, media_type="application/json")


@app.post("/{envname}/{env_id}/step")
async def step_env(envname: str, env_id: str, req: StepRequest) -> Response:
    """Forward step(action) to the existing actor."""
    key = _key(envname, env_id)
    with _ENV_LOCK:
        actor = _ENV_ACTORS.get(key)
    if actor is None:
        raise HTTPException(status_code=404, detail=f"Env actor not found: {envname}:{env_id}")
    result_bytes: bytes = await actor.step.remote(req.action)
    return Response(content=result_bytes, media_type="application/json")


@app.get("/{envname}/{env_id}/render")
async def render_env(envname: str, env_id:str) -> Response:
    """Forward render() to the existing actor."""
    key = _key(envname, env_id)
    with _ENV_LOCK:
        actor = _ENV_ACTORS.get(key)
    if actor is None:
        raise HTTPException(status_code=404, detail=f"Env actor not found: {envname}:{env_id}")
    result_bytes: bytes = await actor.render.remote()
    return Response(content=result_bytes, media_type="application/json")


@app.get("/{envname}/{env_id}/get_task_prompt")
async def get_task_prompt(envname: str, env_id: str) -> Response:
    """Forward get_task_prompt() to the existing actor."""
    key = _key(envname, env_id)
    with _ENV_LOCK:
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


@app.get("{envname}/{env_id}/describe")
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
        except Exception:
            pass
        ray.kill(actor, no_restart=True)

    return {"status": "ok", "envname": envname, "id": env_id}


# -------------------------------------------------------------------
# Local entry point: run this file directly to start the HTTP service
# -------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    _init_ray_if_needed()

    host = "0.0.0.0"
    port = 36663

    uvicorn.run(app, host=host, port=port)
