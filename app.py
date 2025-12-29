from __future__ import annotations

import argparse
import asyncio
import logging
import os
import traceback
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional, Tuple, Union

import httpx
import uvicorn
import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse, StreamingResponse


from manager.db_loader import get_connection
from manager import EnvPoolManager


log = logging.getLogger("env-router")


# ---------------- Config helpers ----------------

def _load_config(path: str) -> dict:
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid YAML config (expected dict at root): {path}")
    return cfg


def _strip_hop_by_hop(headers: Dict[str, str]) -> Dict[str, str]:
    """
    Remove hop-by-hop headers per RFC 7230 §6.1 to avoid proxy issues.
    Also remove headers that uvicorn/httpx should manage.
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
        "host",
        "content-length",
    }
    return {k: v for k, v in headers.items() if k.lower() not in hop}


def _parse_timeout_s(request: Request, default_s: float) -> float:
    q = request.query_params.get("timeout_s")
    if q is None:
        return default_s
    try:
        v = float(q)
        if v <= 0:
            raise ValueError("timeout_s must be > 0")
        return v
    except Exception:
        raise HTTPException(status_code=400, detail=f"Invalid timeout_s='{q}'")


# ---------------- Routing helpers ----------------

def _get_pool(request: Request) -> Any:
    pool = getattr(request.app.state, "pool", None)
    if pool is None:
        raise HTTPException(status_code=503, detail="EnvPoolManager not initialized")
    return pool


def _get_http_client(request: Request) -> httpx.AsyncClient:
    client = getattr(request.app.state, "http_client", None)
    if client is None:
        raise HTTPException(status_code=503, detail="Upstream HTTP client not initialized")
    return client


def _resolve_upstream_for_actor(
    pool: Any,
    env: str,
    env_id: str,
    default_port: int,
) -> Tuple[str, int]:
    """
    Prefer per-actor routing from the manager.
    Fallback to env-level binding.
    Works for both:
      - remote clusters (head_ip is cluster head)
      - local mode (head_ip is 127.0.0.1 / localhost)
    """
    route = pool.get_actor_route(env, env_id)

    if route:
        # Your old code returns Tuple[str, str] (head_ip, port):contentReference[oaicite:8]{index=8}
        if isinstance(route, (tuple, list)) and len(route) == 2:
            host, port = route
            return str(host), int(port)

        # If your new code returns a dataclass-like object
        host = getattr(route, "head_ip", None) or getattr(route, "host", None)
        port = getattr(route, "port", None)
        if host and port:
            return str(host), int(port)

    binding = pool.get_cluster_for_env(env)
    if not binding:
        raise HTTPException(status_code=404, detail=f"env '{env}' not bound to any upstream")
    if not getattr(binding, "head_ip", None):
        raise HTTPException(status_code=503, detail=f"env '{env}' upstream not ready (missing head_ip)")
    return str(binding.head_ip), int(default_port)


async def _proxy_stream(
    request: Request,
    env: str,
    env_id: str,
    method: str,
    path_suffix: str,
    timeout_s: float,
) -> StreamingResponse:
    """
    True streaming proxy:
      - stream request body to upstream
      - stream upstream response body back to caller
      - ensure upstream response is closed even if client disconnects
    """
    pool = _get_pool(request)
    client = _get_http_client(request)

    upstream_port: int = int(request.app.state.upstream_port)
    host, port = _resolve_upstream_for_actor(pool, env, env_id, upstream_port)

    url = f"http://{host}:{port}{path_suffix}"

    # request id for better UX/debugging
    request_id = getattr(request.state, "request_id", None) or uuid.uuid4().hex
    request.state.request_id = request_id

    upstream_headers = _strip_hop_by_hop(dict(request.headers))
    upstream_headers["x-request-id"] = request_id

    req = client.build_request(
        method=method,
        url=url,
        params=request.query_params,
        headers=upstream_headers,
        content=request.stream(),
    )

    try:
        # IMPORTANT: stream=True enables true streaming (no full buffering)
        resp = await client.send(req, stream=True, timeout=timeout_s)
    except httpx.TimeoutException as e:
        raise HTTPException(status_code=504, detail=f"Upstream timeout: {e}")
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Upstream request failed: {e}")

    resp_headers = _strip_hop_by_hop(dict(resp.headers))
    media_type = resp_headers.get("content-type")

    async def body_iter():
        try:
            async for chunk in resp.aiter_bytes():
                yield chunk
        finally:
            await resp.aclose()

    return StreamingResponse(
        body_iter(),
        status_code=resp.status_code,
        media_type=media_type,
        headers=resp_headers,
    )


# ---------------- App factory ----------------

def create_app(config_path: Optional[str] = None) -> FastAPI:
    cfg = _load_config(config_path)

    server_cfg = cfg.get("server", {}) or {}
    cluster_cfg = cfg.get("cluster", {}) or {}
    cluster_http_cfg = (cluster_cfg.get("http", {}) or {})

    # Router listen
    host = str(server_cfg.get("host", "0.0.0.0"))
    port = int(server_cfg.get("port", 36008))

    # Upstream (cluster-side) http service
    upstream_port = int(cluster_http_cfg.get("port", 36663))
    upstream_timeout_s = float(cluster_http_cfg.get("timeout_s", 10.0))

    # Concurrency limit on router (protect from bursts)
    max_inflight = int(server_cfg.get("max_inflight_requests", 1024))
    req_sem = asyncio.Semaphore(max_inflight)

    # httpx pool tuning
    httpx_pool_max = int(server_cfg.get("httpx_max_connections", 2048))
    httpx_keepalive_max = int(server_cfg.get("httpx_max_keepalive", 256))

    # Optional: allow server to start even if POOL.start() fails
    fail_fast = bool(server_cfg.get("fail_fast_startup", True))

    # Optional: CORS
    cors_origins = server_cfg.get("cors_origins") or [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.cfg_path = cfg_path
        app.state.cfg = cfg
        app.state.upstream_port = upstream_port
        app.state.upstream_timeout_s = upstream_timeout_s
        app.state.startup_error = None

        # Init resources
        db = None
        pool = None
        client = None

        try:
            db = get_connection(cfg)
            pool = EnvPoolManager(cfg, db)

            client = httpx.AsyncClient(
                timeout=upstream_timeout_s,
                limits=httpx.Limits(
                    max_connections=httpx_pool_max,
                    max_keepalive_connections=httpx_keepalive_max,
                ),
            )

            app.state.db = db
            app.state.pool = pool
            app.state.http_client = client

            # Start the pool manager (creates clusters or binds local upstream)
            try:
                await pool.start()
                log.info("EnvPoolManager started successfully")
            except Exception:
                app.state.startup_error = traceback.format_exc()
                log.error("EnvPoolManager startup failed:\n%s", app.state.startup_error)
                if fail_fast:
                    raise

            yield

        finally:
            # Shutdown in best-effort order
            try:
                if pool is not None:
                    await pool.close_all()
            except Exception:
                log.exception("POOL.close_all() failed (ignored)")

            try:
                if client is not None:
                    await client.aclose()
            except Exception:
                log.exception("http client close failed (ignored)")

            try:
                if db is not None:
                    db.close()
            except Exception:
                log.exception("db close failed (ignored)")

    app = FastAPI(
        title="RL Env Router",
        description="Routes env requests to the correct Ray cluster/local upstream and manages a warm actor pool.",
        version="2.0",
        lifespan=lifespan,
    )

    # CORS (optional)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins if cors_origins != ["*"] else ["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # One combined middleware for (request-id + concurrency limiting)
    @app.middleware("http")
    async def _request_id_and_limit(request: Request, call_next):
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
        request.state.request_id = request_id

        async with req_sem:
            resp = await call_next(request)

        resp.headers["x-request-id"] = request_id
        return resp

    # Better last-resort error JSON for UX
    @app.exception_handler(Exception)
    async def _unhandled_exception(request: Request, exc: Exception):
        rid = getattr(request.state, "request_id", None)
        log.exception("Unhandled error (request_id=%s): %s", rid, exc)
        return JSONResponse(
            status_code=500,
            content={
                "detail": "Internal server error",
                "request_id": rid,
            },
        )

    # ---------------- UX endpoints ----------------

    @app.get("/", include_in_schema=False)
    async def root():
        # Nice UX: go to docs
        return RedirectResponse(url="/docs")

    @app.get("/healthz")
    async def healthz():
        return {"ok": True}

    @app.get("/readyz")
    async def readyz(request: Request):
        pool = getattr(request.app.state, "pool", None)
        db = getattr(request.app.state, "db", None)
        startup_error = getattr(request.app.state, "startup_error", None)

        clusters_initialized = bool(getattr(pool, "env_cluster_map", None)) if pool else False
        return {
            "db_attached": db is not None,
            "clusters_initialized": clusters_initialized,
            "startup_ok": startup_error is None,
            "startup_error": None if startup_error is None else "see /startup/error",
        }

    @app.get("/startup/error")
    async def startup_error(request: Request):
        err = getattr(request.app.state, "startup_error", None)
        return {"startup_error": err}

    @app.get("/envs")
    async def list_envs(request: Request):
        pool = _get_pool(request)
        env_map = getattr(pool, "env_cluster_map", {}) or {}
        return {"envs": sorted(list(env_map.keys()))}

    @app.get("/clusters")
    async def clusters(request: Request):
        pool = _get_pool(request)
        # Your old manager has list_status() returning env bindings:contentReference[oaicite:9]{index=9}
        return {"bindings": await pool.list_status()}

    @app.get("/pool/actors")
    async def pool_actors(request: Request):
        pool = _get_pool(request)
        return {"env_actors": await pool.list_pool_actors()}

    # ---------------- Env API ----------------
    # These are “pure forwarding” except /close which is manager-controlled.

    @app.post("/{env}/{env_id}/reset")
    async def reset(env: str, env_id: str, request: Request):
        timeout_s = _parse_timeout_s(request, float(request.app.state.upstream_timeout_s))
        return await _proxy_stream(request, env, env_id, "POST", f"/{env}/{env_id}/reset", timeout_s)

    @app.post("/{env}/{env_id}/step")
    async def step(env: str, env_id: str, request: Request):
        timeout_s = _parse_timeout_s(request, float(request.app.state.upstream_timeout_s))
        return await _proxy_stream(request, env, env_id, "POST", f"/{env}/{env_id}/step", timeout_s)

    @app.get("/{env}/{env_id}/get_task_prompt")
    async def get_task_prompt(env: str, env_id: str, request: Request):
        timeout_s = _parse_timeout_s(request, float(request.app.state.upstream_timeout_s))
        return await _proxy_stream(request, env, env_id, "GET", f"/{env}/{env_id}/get_task_prompt", timeout_s)

    @app.get("/{env}/{env_id}/render")
    async def render(env: str, env_id: str, request: Request):
        timeout_s = _parse_timeout_s(request, float(request.app.state.upstream_timeout_s))
        return await _proxy_stream(request, env, env_id, "GET", f"/{env}/{env_id}/render", timeout_s)

    @app.post("/{env}/{env_id}/close")
    async def close_env(env: str, env_id: str, request: Request):
        """
        Manager-controlled close:
          - POST /{env}/{id}/close to upstream (best-effort) is handled inside the manager,
          - remove (env,id) from local pool,
          - reserve next DB row and lazily create a new actor via reset().
        """
        pool = _get_pool(request)
        await pool.close_and_refill(env, env_id)
        return JSONResponse(status_code=202, content={"accepted": True})

    app.state.listen_host = host
    app.state.listen_port = port
    return app

app = create_app()


def main():
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    args = parser.parse_args()
    cfg = _load_config(args.config)
    host = str((cfg.get("server") or {}).get("host", "0.0.0.0"))
    port = int((cfg.get("server") or {}).get("port", 36008))

    uvicorn.run(
        create_app(args.config),
        host=host,
        port=port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
