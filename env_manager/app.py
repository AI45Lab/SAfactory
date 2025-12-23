import os
import sys

# --- make project root importable ---
current_file_path = os.path.abspath(__file__)
examples_dir = os.path.dirname(current_file_path)
project_root = os.path.dirname(examples_dir)
sys.path.append(project_root)

import asyncio
import uvicorn
import yaml
import httpx
from typing import Dict, Tuple, Any, Optional
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from starlette.responses import StreamingResponse

from db_loader import get_connection
from manager import EnvPoolManager
from core.types.base import ResetOutput, StepOutput, RenderOutput


# ---------------- Config ----------------
def load_config(path: str = "env_manager/config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

CFG = load_config()

server_cfg = CFG.get("server", {}) or {}
MAX_INFLIGHT_HTTP = int(server_cfg.get("max_inflight_requests", 1024))
LIST_STATUS_PAR = int(server_cfg.get("list_status_parallelism", 128))

# Upstream (cluster-side HTTP service) settings
cluster_http_cfg = (CFG.get("cluster", {}) or {}).get("http", {}) or {}
CLUSTER_HTTP_PORT: int = int(cluster_http_cfg.get("port", server_cfg.get("port", 8083)))
UPSTREAM_TIMEOUT_S: float = float(cluster_http_cfg.get("timeout_s", 10.0))

# Optional httpx pool tuning (good defaults for high-concurrency gateways)
httpx_pool_max = int(server_cfg.get("httpx_max_connections", 2048))
httpx_keepalive_max = int(server_cfg.get("httpx_max_keepalive", 256))


# ---------------- App----------------
# app = FastAPI(title="Ray Env Router")
origins =[
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


DB = get_connection(CFG)
POOL = EnvPoolManager(CFG, DB)

# Global inflight limiter to protect this router under bursts
_REQ_SEM = asyncio.Semaphore(MAX_INFLIGHT_HTTP)

# Shared httpx client (keep-alive pooling)
HTTP_CLIENT: Optional[httpx.AsyncClient] = None


@app.middleware("http")
async def limit_concurrency(request: Request, call_next):
    async with _REQ_SEM:
        return await call_next(request)

# ---------------- Helpers ----------------
def _binding_or_404(env: str):
    """
    Resolve which Ray cluster (job/head IP) is responsible for this env_name.
    """
    binding = POOL.get_cluster_for_env(env)
    if not binding:
        raise HTTPException(status_code=404, detail=f"env '{env}' not bound to any Ray cluster")
    if not binding.head_ip:
        raise HTTPException(
            status_code=503,
            detail=f"Ray cluster for env '{env}' is not ready (head IP missing)",
        )
    return binding


def _route_for_actor(env: str, env_id: int) -> Tuple[str, int]:
    """
    Prefer per-actor routing (manager tracks which cluster served the actor).
    Fallback to env-level binding if actor-specific route not found.
    """
    route = POOL.get_actor_route(env, env_id)
    if route:
        return route[0], int(route[1])
    binding = _binding_or_404(env)
    return binding.head_ip, CLUSTER_HTTP_PORT


def _strip_hop_by_hop(headers: Dict[str, str]) -> Dict[str, str]:
    """
    Remove hop-by-hop headers per RFC 7230 §6.1 to avoid proxy issues.
    """
    hop = {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
        # also remove request-only headers that should be set by httpx/uvicorn
        "host",
        "content-length",
    }
    return {k: v for k, v in headers.items() if k.lower() not in hop}


async def _proxy_stream(
    env: str,
    env_id: int,
    method: str,
    path_suffix: str,
    request: Request,
    timeout_s: Optional[float] = None,
):
    """
    Zero-transformation proxy:
      - Streams request body to the upstream cluster.
      - Streams response body back to the caller.
      - Preserves status code and content-type.
    """
    if HTTP_CLIENT is None:
        raise HTTPException(status_code=500, detail="upstream HTTP client not initialized")

    head_ip, port = _route_for_actor(env, env_id)
    base_url = f"http://{head_ip}:{int(port)}"
    url = base_url + path_suffix

    # Stream body to upstream; sanitize headers
    upstream_headers = _strip_hop_by_hop(dict(request.headers))
    try:
        resp = await HTTP_CLIENT.request(
            method=method,
            url=url,
            params=request.query_params,
            headers=upstream_headers,
            # Pass the incoming stream directly for zero-copy-ish forwarding
            content=request.stream(),
            timeout=timeout_s or UPSTREAM_TIMEOUT_S,
        )
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Upstream request failed: {e}")

    # Prepare downstream headers (content-type etc.), strip hop-by-hop
    resp_headers = _strip_hop_by_hop(dict(resp.headers))
    media_type = resp_headers.get("content-type")

    # Stream upstream body back to client
    return StreamingResponse(
        resp.aiter_bytes(),
        status_code=resp.status_code,
        media_type=media_type,
        headers=resp_headers,
    )


# ---------------- Lifecycle ----------------

@app.on_event("startup")
async def _startup():
    """
    - Build clusters via EnvPoolManager (scan DB, create RayJobs, poll head IPs),
      then prewarm remote actors.
    - Create one shared httpx client with a tuned connection pool.
    """
    global HTTP_CLIENT
    HTTP_CLIENT = httpx.AsyncClient(
        timeout=UPSTREAM_TIMEOUT_S,
        limits=httpx.Limits(max_connections=httpx_pool_max, max_keepalive_connections=httpx_keepalive_max),
    )
    await POOL.start()

#TODO: when shutdown, we should clean all the rayClusters
@app.on_event("shutdown")
async def _shutdown():
    """
    - Ask EnvPoolManager to clean up its resources.
    - Close the shared HTTP client.
    """
    await POOL.close_all()
    global HTTP_CLIENT
    client, HTTP_CLIENT = HTTP_CLIENT, None
    if client is not None:
        await client.aclose()


# ---------------- Health / Status ----------------

@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.get("/readyz")
def readyz():
    """
    Router is 'ready' when:
    - DB connection is present
    - EnvPoolManager has at least one env->cluster binding
    """
    return {
        "db_attached": DB is not None,
        "clusters_initialized": bool(POOL.env_cluster_map),
    }


@app.get("/pool/status")
async def pool_status():
    """
    Return all the env
    """
    actors = await POOL.list_pool_actors()
    return {"env_actors" :actors}


# ---------------- Env API (pure forwarding) ----------------
@app.post("/{env}/{id}/step")
async def step(env: str, id: str, request: Request):
    timeout_q = request.query_params.get("timeout_s")
    timeout = float(timeout_q) if timeout_q is not None else UPSTREAM_TIMEOUT_S
    return await _proxy_stream(env, id, "POST", f"/{env}/{id}/step", request, timeout_s=timeout)


@app.get("/{env}/{id}/get_task_prompt")
async def get_task_prompt(env: str, id: str, request: Request):
    return await _proxy_stream(env, id, "GET", f"/{env}/{id}/get_task_prompt", request)



@app.get("/{env}/{id}/render")
async def render(env: str, id: str, request: Request):
    return await _proxy_stream(env, id, "GET", f"/{env}/{id}/render", request)


@app.post("/{env}/{id}/close")
async def close_env(env: str, id: str):
    """
    Manager-controlled close:
      - manager will POST /{env}/{id}/close to the owning cluster (best-effort),
      - remove (env,id) from its local pool & routing,
      - reserve next DB row and POST /{env}/{new_id}/reset to lazily create
        a new actor on the appropriate cluster.
    """
    await POOL.close_and_refill(env, id)
    return {"accepted": True}


# ---------------- Entrypoint ----------------

if __name__ == "__main__":
    uvicorn.run(
        app,
        host=CFG["server"]["host"],
        port=int(CFG["server"]["port"]),
        log_level="info",
    )