import asyncio
import logging
import os
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import ray
from fastapi import FastAPI, HTTPException, Response
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel, Field

current_file_path = os.path.abspath(__file__)
examples_dir = os.path.dirname(current_file_path)
project_root = os.path.dirname(examples_dir)
sys.path.append(project_root)

from core.types.base import RenderOutput, ResetOutput, StepOutput, dumps_json_bytes
from env.env_factory import is_supported_env_name, list_supported_env_names, normalize_create_kwargs, resolve_env_class

EnvKey = Tuple[str, str]
JsonDict = Dict[str, Any]


logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("env-service")


def _elapsed_ms(start_time: float) -> int:
    return int((time.perf_counter() - start_time) * 1000)


def _now_monotonic() -> float:
    return time.perf_counter()


@ray.remote(max_concurrency=1)
class EnvActor:
    """A single Ray actor hosting one environment instance."""

    def __init__(self, envname: str, id_: str, create_kwargs: Optional[JsonDict] = None):
        init_start = time.perf_counter()
        pid = os.getpid()
        self.envname = envname
        self.id = str(id_)

        logger.debug("Actor Init Start: %s/%s pid=%s", envname, id_, pid)

        import_done_ms: Optional[int] = None
        try:
            env_cls = resolve_env_class(envname)
            import_done_ms = _elapsed_ms(init_start)
            logger.debug(
                "Actor Import Done: %s/%s pid=%s env_class=%s import_ms=%s",
                envname,
                id_,
                pid,
                getattr(env_cls, "__name__", str(env_cls)),
                import_done_ms,
            )
        except (ValueError, ImportError) as exc:
            logger.error("Import failed for %s: %s", envname, exc)
            raise ImportError(
                f"Failed to import env '{envname}': {exc}\n"
                f"Please install dependencies for {envname} environment first."
            )

        try:
            self.env = env_cls(**(create_kwargs or {}))
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

        logger.debug(
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
        logger.debug("Actor Reset Start: %s/%s pid=%s seed=%s", self.envname, self.id, pid, seed)
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
        logger.debug(
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

    def close(self) -> Any:
        logger.debug("Actor Close: %s/%s", self.envname, self.id)
        return self.env.close()

    def is_done(self) -> bool:
        return self.env.isDone()

    def health(self) -> bool:
        return self.env.isHealth()

    def describe(self) -> JsonDict:
        return {
            "class": self.env.__class__.__name__,
            "done": self.env.isDone(),
        }


@dataclass
class ActorRecord:
    actor: ray.actor.ActorHandle
    envname: str
    env_id: str
    created_at_monotonic: float
    last_used_at_monotonic: float
    last_reset_at_monotonic: float
    reset_count: int = 0


@dataclass
class PendingReset:
    actor: ray.actor.ActorHandle
    future: asyncio.Future[bytes]
    envname: str
    env_id: str
    created_at_monotonic: float


_ENV_ACTORS: Dict[EnvKey, ActorRecord] = {}
_ENV_PENDING_RESETS: Dict[EnvKey, PendingReset] = {}
_ENV_LOCK = threading.RLock()


def _key(envname: str, env_id: str) -> EnvKey:
    return envname, str(env_id)


def _parse_create_kwargs(raw_param: Any) -> JsonDict:
    try:
        return normalize_create_kwargs(raw_param)
    except (TypeError, ValueError) as exc:
        logger.warning("Invalid env_param in reset: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _touch_actor_record(record: ActorRecord, *, is_reset: bool = False) -> None:
    now = _now_monotonic()
    record.last_used_at_monotonic = now
    if is_reset:
        record.last_reset_at_monotonic = now
        record.reset_count += 1


def _create_actor_record(envname: str, env_id: str, create_kwargs: JsonDict) -> ActorRecord:
    now = _now_monotonic()
    return ActorRecord(
        actor=EnvActor.remote(envname, env_id, create_kwargs),
        envname=envname,
        env_id=str(env_id),
        created_at_monotonic=now,
        last_used_at_monotonic=now,
        last_reset_at_monotonic=0.0,
    )


def _create_pending_reset(record: ActorRecord) -> PendingReset:
    return PendingReset(
        actor=record.actor,
        future=asyncio.get_running_loop().create_future(),
        envname=record.envname,
        env_id=record.env_id,
        created_at_monotonic=record.created_at_monotonic,
    )


def _fail_pending_reset(pending: Optional[PendingReset], exc: Exception) -> None:
    if pending is None or pending.future.done():
        return
    pending.future.set_exception(exc)
    pending.future.add_done_callback(lambda fut: fut.exception())


def _pop_actor_state_locked(key: EnvKey) -> Tuple[Optional[ActorRecord], Optional[PendingReset]]:
    return _ENV_ACTORS.pop(key, None), _ENV_PENDING_RESETS.pop(key, None)


async def _close_actor_gracefully(actor: Optional[ray.actor.ActorHandle]) -> None:
    if actor is None:
        return
    try:
        await actor.close.remote()
    except Exception:
        logger.debug("Graceful actor close failed; forcing kill", exc_info=True)


def _force_kill_actor(actor: Optional[ray.actor.ActorHandle]) -> None:
    if actor is None:
        return
    try:
        ray.kill(actor, no_restart=True)
    except Exception:
        logger.debug("Force kill failed", exc_info=True)


async def _dispose_actor(actor: Optional[ray.actor.ActorHandle], *, close_first: bool = True) -> None:
    if actor is None:
        return
    if close_first:
        await _close_actor_gracefully(actor)
    _force_kill_actor(actor)


def _build_actor_description(record: ActorRecord, payload: JsonDict) -> JsonDict:
    return {
        "env": record.envname,
        "id": record.env_id,
        "class": payload.get("class"),
        "done": bool(payload.get("done", False)),
    }


def _get_ready_actor_record_or_404(key: EnvKey) -> ActorRecord:
    with _ENV_LOCK:
        record = _ENV_ACTORS.get(key)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Env actor not found: {key[0]}:{key[1]}")
    return record


async def _await_actor_call(record: ActorRecord, remote_call: Any) -> Any:
    result = await remote_call
    _touch_actor_record(record)
    return result


def _prepare_actor_for_reset(
    envname: str,
    env_id: str,
    raw_param: Any,
) -> Tuple[Optional[ActorRecord], Optional[PendingReset], bool]:
    key = _key(envname, env_id)

    record = _ENV_ACTORS.get(key)
    if record is not None:
        logger.debug("Reset: %s/%s (Reusing existing actor)", envname, env_id)
        return record, None, False

    with _ENV_LOCK:
        record = _ENV_ACTORS.get(key)
        if record is not None:
            logger.debug("Reset: %s/%s (Reusing actor created by another thread)", envname, env_id)
            return record, None, False

        pending = _ENV_PENDING_RESETS.get(key)
        if pending is not None:
            logger.debug("Reset: %s/%s (Awaiting pending initial reset)", envname, env_id)
            return None, pending, False

        create_kwargs = _parse_create_kwargs(raw_param)
        record = _create_actor_record(envname, env_id, create_kwargs)
        pending = _create_pending_reset(record)
        _ENV_PENDING_RESETS[key] = pending
        logger.debug("Reset: %s/%s (Creating new actor)", envname, env_id)
        return record, pending, True


def _commit_pending_reset_success(
    key: EnvKey,
    record: ActorRecord,
    pending: PendingReset,
    result_bytes: bytes,
) -> None:
    with _ENV_LOCK:
        current_pending = _ENV_PENDING_RESETS.get(key)
        if current_pending is pending:
            _ENV_PENDING_RESETS.pop(key, None)
            _ENV_ACTORS[key] = record
    _touch_actor_record(record, is_reset=True)
    if not pending.future.done():
        pending.future.set_result(result_bytes)


def _rollback_pending_reset_failure(key: EnvKey, pending: Optional[PendingReset], exc: Exception) -> None:
    with _ENV_LOCK:
        current_pending = _ENV_PENDING_RESETS.get(key)
        if current_pending is pending:
            _ENV_PENDING_RESETS.pop(key, None)
    _fail_pending_reset(pending, exc)


def _init_ray_if_needed() -> None:
    """Initialize Ray once; connect to a cluster if RAY_ADDRESS is set."""
    if ray.is_initialized():
        return
    address = os.getenv("RAY_ADDRESS")
    if address:
        logger.info("Connecting to Ray cluster at %s", address)
        ray.init(address=address)
    else:
        logger.info("Starting local Ray")
        ray.init()


class ResetRequest(BaseModel):
    env_param: Any = Field(
        ...,
        description="Env creation parameters, either a JSON object or a JSON string stored in DB (env_param).",
    )
    seed: Optional[int] = Field(
        default=None,
        description="Optional seed passed into env.reset(seed=...).",
    )


class StepRequest(BaseModel):
    action: str = Field(..., description="Action string passed into env.step(action).")


app = FastAPI(
    title="Env HTTP Service (Ray + Gym)",
    version="0.2.0",
)


@app.on_event("startup")
def on_startup() -> None:
    _init_ray_if_needed()


@app.on_event("shutdown")
async def on_shutdown() -> None:
    with _ENV_LOCK:
        ready_records = list(_ENV_ACTORS.values())
        pending_resets = list(_ENV_PENDING_RESETS.values())
        _ENV_ACTORS.clear()
        _ENV_PENDING_RESETS.clear()

    for pending in pending_resets:
        _fail_pending_reset(pending, RuntimeError("Env service is shutting down"))

    for record in ready_records:
        await _dispose_actor(record.actor)
    for pending in pending_resets:
        await _dispose_actor(pending.actor)

    if ready_records or pending_resets:
        logger.info(
            "Shutdown cleanup finished: ready=%s pending=%s",
            len(ready_records),
            len(pending_resets),
        )


@app.get("/envs")
async def list_envs() -> Dict[str, Any]:
    """List all currently tracked env actors."""
    with _ENV_LOCK:
        envs = [{"envname": r.envname, "id": r.env_id, "state": "ready"} for r in _ENV_ACTORS.values()]
        envs.extend(
            {"envname": p.envname, "id": p.env_id, "state": "pending"}
            for p in _ENV_PENDING_RESETS.values()
        )
    return {"envs": envs}


@app.post("/{envname}/{env_id}/reset")
async def reset_env(envname: str, env_id: str, req: ResetRequest) -> Response:
    """Reset the environment identified by (envname, env_id)."""
    _init_ray_if_needed()
    request_start = time.perf_counter()

    if not is_supported_env_name(envname):
        raise HTTPException(
            status_code=400,
            detail=f"Unknown envname '{envname}'. Available: {list_supported_env_names()}",
        )

    key = _key(envname, env_id)
    record, pending, created_actor = _prepare_actor_for_reset(envname, env_id, req.env_param)

    if pending is not None and not created_actor:
        pending_wait_start = time.perf_counter()
        try:
            result_bytes = await asyncio.shield(pending.future)
            logger.debug(
                "HTTP Reset Request Done: %s/%s mode=await_pending wait_ms=%s total_ms=%s",
                envname,
                env_id,
                _elapsed_ms(pending_wait_start),
                _elapsed_ms(request_start),
            )
            return Response(content=result_bytes, media_type="application/json")
        except asyncio.CancelledError:
            logger.warning(
                "HTTP Reset Request Cancelled While Awaiting Pending Reset: %s/%s wait_ms=%s total_ms=%s",
                envname,
                env_id,
                _elapsed_ms(pending_wait_start),
                _elapsed_ms(request_start),
            )
            raise
        except Exception as exc:
            logger.error(
                "Pending initial reset failed for %s/%s after wait_ms=%s total_ms=%s: %s",
                envname,
                env_id,
                _elapsed_ms(pending_wait_start),
                _elapsed_ms(request_start),
                exc,
            )
            raise HTTPException(status_code=500, detail=f"Reset execution failed: {exc}")

    assert record is not None

    reset_wait_start = time.perf_counter()
    try:
        result_bytes: bytes = await record.actor.reset.remote(req.seed)

        if created_actor:
            assert pending is not None
            _commit_pending_reset_success(key, record, pending, result_bytes)
        else:
            _touch_actor_record(record, is_reset=True)

        logger.debug(
            "HTTP Reset Request Done: %s/%s mode=%s wait_ms=%s total_ms=%s",
            envname,
            env_id,
            "create" if created_actor else "reuse",
            _elapsed_ms(reset_wait_start),
            _elapsed_ms(request_start),
        )
        return Response(content=result_bytes, media_type="application/json")

    except asyncio.CancelledError:
        if created_actor:
            _rollback_pending_reset_failure(key, pending, RuntimeError("Initial reset request was cancelled"))
            await _dispose_actor(record.actor)

        logger.warning(
            "HTTP Reset Request Cancelled: %s/%s mode=%s wait_ms=%s total_ms=%s",
            envname,
            env_id,
            "create" if created_actor else "reuse",
            _elapsed_ms(reset_wait_start),
            _elapsed_ms(request_start),
        )
        raise

    except Exception as exc:
        if created_actor:
            _rollback_pending_reset_failure(key, pending, exc)
            await _dispose_actor(record.actor)
        else:
            with _ENV_LOCK:
                current_record = _ENV_ACTORS.get(key)
                if current_record is record:
                    _ENV_ACTORS.pop(key, None)
            await _dispose_actor(record.actor)

        logger.error(
            "Reset execution failed for %s/%s after wait_ms=%s total_ms=%s: %s",
            envname,
            env_id,
            _elapsed_ms(reset_wait_start),
            _elapsed_ms(request_start),
            exc,
        )
        raise HTTPException(status_code=500, detail=f"Reset execution failed: {exc}")


@app.post("/{envname}/{env_id}/step")
async def step_env(envname: str, env_id: str, req: StepRequest) -> Response:
    """Forward step(action) to the existing actor."""
    record = _get_ready_actor_record_or_404(_key(envname, env_id))
    result_bytes: bytes = await _await_actor_call(record, record.actor.step.remote(req.action))
    return Response(content=result_bytes, media_type="application/json")


@app.get("/{envname}/{env_id}/render")
async def render_env(envname: str, env_id: str) -> Response:
    """Forward render() to the existing actor."""
    record = _get_ready_actor_record_or_404(_key(envname, env_id))
    result_bytes: bytes = await _await_actor_call(record, record.actor.render.remote())
    return Response(content=result_bytes, media_type="application/json")


@app.get("/{envname}/{env_id}/get_task_prompt")
async def get_task_prompt(envname: str, env_id: str) -> Response:
    """Forward get_task_prompt() to the existing actor."""
    record = _get_ready_actor_record_or_404(_key(envname, env_id))
    result_bytes: bytes = await _await_actor_call(record, record.actor.get_task_prompt.remote())
    return Response(content=result_bytes, media_type="application/json")


@app.get("/{envname}/{env_id}/is_done")
async def is_done(envname: str, env_id: str) -> Dict[str, Any]:
    """Check whether the environment is done."""
    record = _get_ready_actor_record_or_404(_key(envname, env_id))
    value: bool = await _await_actor_call(record, record.actor.is_done.remote())
    return {"envname": envname, "id": env_id, "done": value}


@app.get("/{envname}/{env_id}/health")
async def health(envname: str, env_id: str) -> Dict[str, Any]:
    """Check health for the environment actor."""
    record = _get_ready_actor_record_or_404(_key(envname, env_id))
    value: bool = await _await_actor_call(record, record.actor.health.remote())
    return {"envname": envname, "id": env_id, "healthy": value}


@app.get("/{envname}/{env_id}/describe")
async def describe(envname: str, env_id: str) -> Dict[str, Any]:
    """Expose a compact description of the actor and environment state."""
    record = _get_ready_actor_record_or_404(_key(envname, env_id))
    payload: JsonDict = await _await_actor_call(record, record.actor.describe.remote())
    return _build_actor_description(record, payload)


@app.delete("/{envname}/{env_id}")
async def close_env(envname: str, env_id: str) -> Dict[str, Any]:
    """Close and remove the env actor for (envname, env_id)."""
    key = _key(envname, env_id)
    with _ENV_LOCK:
        record, pending = _pop_actor_state_locked(key)

    if pending is not None:
        _fail_pending_reset(pending, RuntimeError("Actor deleted during initial reset"))
        await _dispose_actor(pending.actor)
        logger.debug("Closed pending actor: %s/%s", envname, env_id)

    if record is not None:
        await _dispose_actor(record.actor)
        logger.debug("Closed actor: %s/%s", envname, env_id)

    return {"status": "ok", "envname": envname, "id": env_id}


if __name__ == "__main__":
    import uvicorn

    _init_ray_if_needed()
    host = "0.0.0.0"
    port = 36663
    logger.info("Starting server at %s:%s", host, port)
    uvicorn.run(app, host=host, port=port)
