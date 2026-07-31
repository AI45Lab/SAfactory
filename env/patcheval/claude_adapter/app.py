"""Standalone Claude Code protocol adapter in front of SAfactory Gateway."""
from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager
from typing import Any

import aiohttp
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

from .conversion import anthropic_sse, anthropic_to_openai, openai_to_anthropic


log = logging.getLogger(__name__)


def create_app() -> FastAPI:
    gateway_session_base_url = _required_env("CLAUDE_ADAPTER_GATEWAY_SESSION_BASE_URL").rstrip("/")
    route_model = _required_env("CLAUDE_ADAPTER_ROUTE_MODEL")
    request_timeout_s = _positive_float(os.environ.get("CLAUDE_ADAPTER_REQUEST_TIMEOUT_S"), 2700.0)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        timeout = aiohttp.ClientTimeout(total=request_timeout_s, connect=30)
        app.state.http = aiohttp.ClientSession(timeout=timeout)
        try:
            yield
        finally:
            await app.state.http.close()

    app = FastAPI(title="SAfactory Claude Code Adapter", lifespan=lifespan)

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz() -> dict[str, str]:
        return {"status": "ready"}

    @app.post("/v1/sessions/{session_id}/v1/messages/count_tokens")
    async def count_tokens(session_id: str, request: Request) -> JSONResponse:
        del session_id
        payload = await _json_body(request)
        token_count = max(1, len(json.dumps(payload, ensure_ascii=False, default=str)) // 4)
        return JSONResponse({"input_tokens": token_count})

    @app.post("/v1/sessions/{session_id}/v1/messages")
    async def messages(session_id: str, request: Request) -> Response:
        payload = await _json_body(request)
        openai_payload = anthropic_to_openai(payload, route_model)
        gateway_url = f"{gateway_session_base_url}/{session_id}/chat/completions"
        try:
            async with request.app.state.http.post(
                gateway_url,
                json=openai_payload,
                headers={"Authorization": "Bearer safactory"},
            ) as upstream:
                raw_body = await upstream.read()
                if upstream.status >= 400:
                    detail = raw_body.decode("utf-8", errors="replace")
                    log.warning(
                        "Claude adapter Gateway request failed: session_id=%s status=%d body=%s",
                        session_id,
                        upstream.status,
                        detail[-2000:],
                    )
                    return JSONResponse(
                        {
                            "type": "error",
                            "error": {
                                "type": "api_error",
                                "message": f"SAfactory Gateway returned HTTP {upstream.status}: {detail[-2000:]}",
                            },
                        },
                        status_code=upstream.status,
                    )
        except (aiohttp.ClientError, TimeoutError) as exc:
            log.warning("Claude adapter could not reach Gateway: session_id=%s error=%s", session_id, exc)
            return JSONResponse(
                {
                    "type": "error",
                    "error": {"type": "api_error", "message": str(exc)},
                },
                status_code=503,
            )

        try:
            openai_response = json.loads(raw_body)
            if not isinstance(openai_response, dict):
                raise ValueError("Gateway response must be a JSON object")
            anthropic_response = openai_to_anthropic(openai_response, payload)
        except (json.JSONDecodeError, ValueError) as exc:
            return JSONResponse(
                {
                    "type": "error",
                    "error": {"type": "api_error", "message": str(exc)},
                },
                status_code=502,
            )

        if payload.get("stream"):
            return Response(
                anthropic_sse(anthropic_response),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache"},
            )
        return JSONResponse(anthropic_response)

    return app


async def _json_body(request: Request) -> dict[str, Any]:
    payload = await request.json()
    if not isinstance(payload, dict):
        raise ValueError("request body must be a JSON object")
    return payload


def _required_env(name: str) -> str:
    value = str(os.environ.get(name) or "").strip()
    if not value:
        raise RuntimeError(f"{name} is required")
    return value


def _positive_float(value: Any, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default
