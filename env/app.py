import os
import sys
import asyncio

current_file_path = os.path.abspath(__file__)
examples_dir = os.path.dirname(current_file_path)
project_root = os.path.dirname(examples_dir)
sys.path.append(project_root)

import logging
import threading
import json
import time
from dataclasses import dataclass
import ray

from typing import Any, Dict, Optional, Tuple, List, Callable, Type
from openai.types.chat import ChatCompletionMessageParam
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field

from core.types.base import ResetOutput, RenderOutput, StepOutput, dumps_json_bytes

from env.registry import (_import_os_env, _import_search_env, _import_emb_env, _import_gym_env,
                          _import_android_gym, _import_trading_env, _import_mc_env, _import_mc_gpu_env,
                          _import_geo3k_vl_test_env, _import_qa_gym, _import_deepeyes_env,
                          _import_dab_env, _import_dw_env)

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
    "deepeyes_env": _import_deepeyes_env,
    "dabstepgym": _import_dab_env,
    "discoveryworld": _import_dw_env,
}


logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("env-service")


def _elapsed_ms(start_time: float) -> int:
    return int((time.perf_counter() - start_time) * 1000)

# -------------------------------------------------------------------
# Ray actor: one actor hosts one env instance
# -------------------------------------------------------------------
@ray.remote(max_concurrency=1)
class EnvActor:
    """
    A single actor hosts one env instance.
    """

    def __init__(self, envname: str, id_: str, create_kwargs: Optional[Dict[str, Any]] = None):
        init_start = time.perf_counter()
        pid = os.getpid()
        self.envname = envname
        self.id = str(id_)

        logger.info("Actor Init Start: %s/%s pid=%s", envname, id_, pid)

        import_done_ms: Optional[int] = None
        try:
            env_import_func = ENV_CLASS_REGISTRY[envname]
            EnvCls = env_import_func()
            import_done_ms = _elapsed_ms(init_start)
            logger.info(
                "Actor Import Done: %s/%s pid=%s env_class=%s import_ms=%s",
                envname,
                id_,
                pid,
                getattr(EnvCls, "__name__", str(EnvCls)),
                import_done_ms,
            )
        except KeyError:
            logger.error(f"Unknown envname: {envname}")
            raise ValueError(f"Unknown envname: {envname} (available: {list(ENV_CLASS_REGISTRY.keys())})")
        except ImportError as e:
            logger.error(f"Import failed for {envname}: {e}")
            raise ImportError(
                f"Failed to import env '{envname}': {e}\n"
                f"Please install dependencies for {envname} environment first."
            )

        try:
            self.env = EnvCls(**(create_kwargs or {}))
        except Exception:
            logger.exception(
                "Actor Init Failed: %s/%s pid=%s import_ms=%s total_ms=%s",
                envname,
                id_,
                pid,
                import_done_ms,
                _elapsed_ms(init_start),
            )
            raise

        logger.info(
            "Actor Init Done: %s/%s pid=%s env_class=%s total_ms=%s",
            envname,
            id_,
            pid,
            self.env.__class__.__name__,
            _elapsed_ms(init_start),
        )

    def reset(self, seed: Optional[int] = None) -> bytes:
        reset_start = time.perf_counter()
        pid = os.getpid()
        logger.info("Actor Reset Start: %s/%s pid=%s seed=%s", self.envname, self.id, pid, seed)
        try:
            out: ResetOutput = self.env.reset(seed=seed)
        except Exception:
            logger.exception(
                "Actor Reset Failed: %s/%s pid=%s elapsed_ms=%s",
                self.envname,
                self.id,
                pid,
                _elapsed_ms(reset_start),
            )
            raise
        logger.info(
            "Actor Reset Done: %s/%s pid=%s elapsed_ms=%s",
            self.envname,
            self.id,
            pid,
            _elapsed_ms(reset_start),
        )
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
# In-process maps:
#   - ready actors are safe to reuse for step/render/reset
#   - pending actors are still in their initial reset and must not be
#     treated as reusable yet
# -------------------------------------------------------------------
@dataclass
class PendingReset:
    actor: ray.actor.ActorHandle
    future: asyncio.Future[bytes]


_ENV_ACTORS: Dict[Tuple[str, str], ray.actor.ActorHandle] = {}
_ENV_PENDING_RESETS: Dict[Tuple[str, str], PendingReset] = {}
_ENV_LOCK = threading.RLock()


def _key(envname: str, env_id: str) -> Tuple[str, str]:
    return envname, str(env_id)


def _kill_actor(actor: Optional[ray.actor.ActorHandle]) -> None:
    if actor is None:
        return
    try:
        actor.close().remote()
    except Exception:
        pass
    try:
        ray.kill(actor, no_restart=True)
    except Exception:
        pass


def _fail_pending_reset(pending: Optional[PendingReset], exc: Exception) -> None:
    if pending is None or pending.future.done():
        return
    pending.future.set_exception(exc)
    pending.future.add_done_callback(lambda fut: fut.exception())


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
        envs = [{"envname": k[0], "id": k[1], "state": "ready"} for k in _ENV_ACTORS.keys()]
        envs.extend({"envname": k[0], "id": k[1], "state": "pending"} for k in _ENV_PENDING_RESETS.keys())
    return {"envs": envs}


@app.post("/{envname}/{env_id}/reset")
async def reset_env(envname: str, env_id: str, req: ResetRequest) -> Response:
    """
    Reset env identified by (envname, env_id).
    """
    _init_ray_if_needed()
    request_start = time.perf_counter()

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
    pending: Optional[PendingReset] = None
    had_cached_actor = actor is not None
    created_actor = False
    awaited_pending = False
    logger.info(
        "HTTP Reset Request Start: %s/%s cached_actor=%s seed=%s",
        envname,
        env_id,
        had_cached_actor,
        req.seed,
    )

    # 2. Slow Path: Create only if missing
    if actor is None:
        with _ENV_LOCK:
            # 3. Double-Check: Verify again inside lock to handle race conditions
            actor = _ENV_ACTORS.get(key)

            if actor is None:
                pending = _ENV_PENDING_RESETS.get(key)
                if pending is not None:
                    actor = pending.actor
                    awaited_pending = True
                    logger.info(f"Reset: {envname}/{env_id} (Awaiting pending initial reset)")
                else:
                    try:
                        logger.info(f"Reset: {envname}/{env_id} (Creating new actor)")
                        actor = EnvActor.remote(envname, env_id, create_kwargs)
                        pending = PendingReset(
                            actor=actor,
                            future=asyncio.get_running_loop().create_future(),
                        )
                        _ENV_PENDING_RESETS[key] = pending
                        created_actor = True
                    except Exception as e:
                        _ENV_PENDING_RESETS.pop(key, None)
                        _kill_actor(actor)
                        logger.error(f"Actor creation failed: {e}")
                        raise HTTPException(status_code=500, detail=f"Actor creation failed: {e}")
            else:
                logger.info(f"Reset: {envname}/{env_id} (Reusing actor created by another thread)")
    else:
        logger.info(f"Reset: {envname}/{env_id} (Reusing existing actor)")

    # 4a. Another request is already doing the initial reset for this actor.
    if awaited_pending:
        assert pending is not None
        pending_wait_start = time.perf_counter()
        try:
            result_bytes = await asyncio.shield(pending.future)
            logger.info(
                "HTTP Reset Request Done: %s/%s cached_actor=%s created_actor=%s awaited_pending=%s wait_ms=%s total_ms=%s",
                envname,
                env_id,
                had_cached_actor,
                created_actor,
                awaited_pending,
                _elapsed_ms(pending_wait_start),
                _elapsed_ms(request_start),
            )
            return Response(content=result_bytes, media_type="application/json")
        except asyncio.CancelledError:
            logger.warning(
                "HTTP Reset Request Cancelled While Awaiting Pending Reset: %s/%s cached_actor=%s awaited_pending=%s wait_ms=%s total_ms=%s",
                envname,
                env_id,
                had_cached_actor,
                awaited_pending,
                _elapsed_ms(pending_wait_start),
                _elapsed_ms(request_start),
            )
            raise
        except Exception as e:
            logger.error(
                "Pending initial reset failed for %s/%s after wait_ms=%s total_ms=%s: %s",
                envname,
                env_id,
                _elapsed_ms(pending_wait_start),
                _elapsed_ms(request_start),
                e,
            )
            raise HTTPException(status_code=500, detail=f"Reset execution failed: {e}")

    # 4b. Normal reset path: either a ready actor, or the creator request driving
    # the initial reset for a pending actor.
    reset_wait_start = time.perf_counter()
    try:
        result_bytes: bytes = await actor.reset.remote(req.seed)

        if created_actor:
            assert pending is not None
            with _ENV_LOCK:
                current_pending = _ENV_PENDING_RESETS.get(key)
                if current_pending is pending:
                    _ENV_PENDING_RESETS.pop(key, None)
                    _ENV_ACTORS[key] = actor
            if not pending.future.done():
                pending.future.set_result(result_bytes)

        logger.info(
            "HTTP Reset Request Done: %s/%s cached_actor=%s created_actor=%s awaited_pending=%s wait_ms=%s total_ms=%s",
            envname,
            env_id,
            had_cached_actor,
            created_actor,
            awaited_pending,
            _elapsed_ms(reset_wait_start),
            _elapsed_ms(request_start),
        )
        return Response(content=result_bytes, media_type="application/json")

    except asyncio.CancelledError:
        if created_actor:
            assert pending is not None
            with _ENV_LOCK:
                current_pending = _ENV_PENDING_RESETS.get(key)
                if current_pending is pending:
                    _ENV_PENDING_RESETS.pop(key, None)
            _fail_pending_reset(pending, RuntimeError("Initial reset request was cancelled"))
            _kill_actor(actor)

        logger.warning(
            "HTTP Reset Request Cancelled: %s/%s cached_actor=%s created_actor=%s awaited_pending=%s wait_ms=%s total_ms=%s",
            envname,
            env_id,
            had_cached_actor,
            created_actor,
            awaited_pending,
            _elapsed_ms(reset_wait_start),
            _elapsed_ms(request_start),
        )
        raise

    except Exception as e:
        if created_actor:
            assert pending is not None
            with _ENV_LOCK:
                current_pending = _ENV_PENDING_RESETS.get(key)
                if current_pending is pending:
                    _ENV_PENDING_RESETS.pop(key, None)
            _fail_pending_reset(pending, e)
            _kill_actor(actor)
        else:
            with _ENV_LOCK:
                if _ENV_ACTORS.get(key) == actor:
                    _ENV_ACTORS.pop(key, None)
            _kill_actor(actor)

        logger.error(
            "Reset execution failed for %s/%s after wait_ms=%s total_ms=%s: %s",
            envname,
            env_id,
            _elapsed_ms(reset_wait_start),
            _elapsed_ms(request_start),
            e,
        )
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
        pending = _ENV_PENDING_RESETS.pop(key, None)

    if pending is not None:
        _fail_pending_reset(pending, RuntimeError("Actor deleted during initial reset"))
        _kill_actor(pending.actor)
        logger.info(f"Closed pending actor: {envname}/{env_id}")

    if actor is not None:
        try:
            await actor.close.remote()
            logger.info(f"Closed: {envname}/{env_id}")
        except Exception:
            pass
        _kill_actor(actor)
    return {"status": "ok", "envname": envname, "id": env_id}


if __name__ == "__main__":
    import uvicorn

    _init_ray_if_needed()
    host = "0.0.0.0"
    port = 36663
    logger.info(f"Starting server at {host}:{port}")
    uvicorn.run(app, host=host, port=port)
