import asyncio
from typing import Dict, Tuple, Any, Optional
import uvicorn
import yaml
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
import ray

from db_loader import get_connection
from manager import EnvPoolManager, ActorRec


def load_config(path="config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

CFG = load_config()

# Reasonable defaults for your target load
MAX_INFLIGHT_HTTP = int(CFG.get("server", {}).get("max_inflight_requests", 512))
RAY_CALL_TIMEOUT_S = float(CFG.get("server", {}).get("ray_call_timeout_s", 10.0))
LIST_STATUS_PAR = int(CFG.get("server", {}).get("list_status_parallelism", 128))

# ---------------- App & Manager ----------------

app = FastAPI(title="Ray Env Gateway")

DB = get_connection(CFG)
POOL = EnvPoolManager(CFG, DB)

# Per-actor serialization: ensure we *atomically* create one lock per (env, id)
_LOCKS: Dict[Tuple[str, int], asyncio.Lock] = {}
_LOCKS_MU = asyncio.Lock()  # guards creation of new locks

async def _lock_for(env: str, id_: int) -> asyncio.Lock:
    key = (env, int(id_))
    async with _LOCKS_MU:
        lock = _LOCKS.get(key)
        if lock is None:
            lock = asyncio.Lock()
            _LOCKS[key] = lock
        return lock

# Limit global inflight HTTP to protect the server under bursts
_REQ_SEM = asyncio.Semaphore(MAX_INFLIGHT_HTTP)

@app.middleware("http")
async def limit_concurrency(request: Request, call_next):
    async with _REQ_SEM:
        return await call_next(request)

# ---------------- Helpers ----------------

def _rec_or_404(env: str, id_: int) -> ActorRec:
    rec = POOL.get(env, id_)
    if not rec:
        raise HTTPException(404, "env/id not active")
    if rec.state != "READY":
        raise HTTPException(410, "env closing")
    return rec

async def _await_ref(ref, timeout: Optional[float] = None):
    """
    Await a Ray ObjectRef without blocking the event loop.
    Works with modern Ray (awaitable ObjectRef). Falls back to thread+ray.get otherwise.
    """
    try:
        if timeout is None:
            return await ref  # type: ignore[func-returns-value]
        return await asyncio.wait_for(ref, timeout=timeout)  # type: ignore[arg-type]
    except TypeError:
        # Fallback for Ray versions that don't support 'await ref'
        if timeout is None:
            return await asyncio.to_thread(ray.get, ref)
        try:
            return await asyncio.wait_for(asyncio.to_thread(ray.get, ref), timeout=timeout)
        except asyncio.TimeoutError:
            raise

def _map_ray_error(exc: Exception) -> HTTPException:
    # Map Ray exceptions to clear HTTP statuses; client can retry with jitter.
    import ray as _ray
    if isinstance(exc, asyncio.TimeoutError):
        return HTTPException(status_code=504, detail="ray call timed out")
    if isinstance(exc, _ray.exceptions.RayActorError):
        return HTTPException(status_code=503, detail="actor crashed; manager will refill")
    if isinstance(exc, _ray.exceptions.OutOfMemoryError):
        return HTTPException(status_code=503, detail="actor OOM; manager will refill")
    return HTTPException(status_code=500, detail=f"ray error: {type(exc).__name__}: {exc}")

# ---------------- Models ----------------

class StepBody(BaseModel):
    action: Any | None = None
    timeout_s: float | None = Field(default=None, ge=0.0)

# ---------------- Lifecycle ----------------

@app.on_event("startup")
async def _startup():
    await POOL.start()

@app.on_event("shutdown")
async def _shutdown():
    await POOL.close_all()

# ---------------- Endpoints ----------------

@app.get("/healthz")
def healthz():
    return {"ok": True}

@app.get("/readyz")
def readyz():
    # Ray ready + DB handle present
    return {"ray_initialized": ray.is_initialized(), "db_attached": DB is not None}

@app.get("/pool/status")
async def pool_status():
    actors = await POOL.list_status(parallelism=LIST_STATUS_PAR)
    return {"actors": actors}

@app.post("/{env}/{id}/reset")
async def reset(env: str, id: int):
    rec = _rec_or_404(env, id)
    lock = await _lock_for(env, id)
    async with lock:
        ref = rec.handle.reset.remote()
        try:
            return await _await_ref(ref, timeout=RAY_CALL_TIMEOUT_S)
        except Exception as e:
            # Remove and trigger refill on crash; surface a clean error
            # await POOL.close_and_remove(env, id)
            raise _map_ray_error(e)

@app.post("/{env}/{id}/step")
async def step(env: str, id: int, body: StepBody):
    rec = _rec_or_404(env, id)
    lock = await _lock_for(env, id)
    timeout = body.timeout_s if body.timeout_s is not None else RAY_CALL_TIMEOUT_S
    async with lock:
        ref = rec.handle.step.remote(body.action)
        try:
            return await _await_ref(ref, timeout=timeout)
        except Exception as e:
            # await POOL.close_and_remove(env, id)
            raise _map_ray_error(e)

@app.get("/{env}/{id}/render")
async def render(env: str, id: int):
    rec = _rec_or_404(env, id)
    lock = await _lock_for(env, id)
    async with lock:
        ref = rec.handle.render.remote()
        try:
            return await _await_ref(ref, timeout=RAY_CALL_TIMEOUT_S)
        except Exception as e:
            # await POOL.close_and_remove(env, id)
            raise _map_ray_error(e)

@app.post("/{env}/{id}/close")
async def close_env(env: str, id: int):
    # No need to lock; close_and_remove is internally safe
    await POOL.close_and_remove(env, id)
    return {"accepted": True}

if __name__ == "__main__":
    # IMPORTANT: run a single worker per manager process.
    # Scale horizontally by running more replicas and sharding DB rows (or implement DB row claiming).
    uvicorn.run(
        app,
        host=CFG["server"]["host"],
        port=int(CFG["server"]["port"]),
        log_level="info",
    )
