#!/usr/bin/env python3
"""
LLM Proxy Server

This server:
1. Proxies LLM requests to the real engine
2. Records trajectory masks for training loss computation
3. Provides /get_trajectory_mask endpoint for the training client
"""

import asyncio
import json
import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, List, Optional

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from transformers import AutoTokenizer

# Add AIEvoBox to path
AIEVOBOX_ROOT = os.environ.get("AIEVOBOX_ROOT", "/root/AIEvoBox")
if AIEVOBOX_ROOT not in sys.path:
    sys.path.insert(0, AIEVOBOX_ROOT)

# Setup logging
LOG_DIR = os.path.join(AIEVOBOX_ROOT, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "llm_proxy.log")

logger = logging.getLogger("llm_proxy")
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

logger.info(f"LLM Proxy logging to: {LOG_FILE}")

# Add mask directory to path
MASK_DIR = os.path.join(AIEVOBOX_ROOT, "rl", "mask")
if MASK_DIR not in sys.path:
    sys.path.insert(0, MASK_DIR)

from trajectory_mask_builder import TrajectoryMaskBuilder

app = FastAPI(title="LLM Proxy Server", debug=True)

# Global state
tokenizer: Optional[AutoTokenizer] = None
trajectory_mask_builder: Optional[TrajectoryMaskBuilder] = None
remote_engine_url: Optional[str] = None
http_client: Optional[httpx.AsyncClient] = None


class ProxyState:
    """Global state for the proxy server."""

    def __init__(self):
        self.tokenizer: Optional[AutoTokenizer] = None
        self.trajectory_mask_builder: Optional[TrajectoryMaskBuilder] = None
        self.remote_engine_url: Optional[str] = None
        self._http_client: Optional[httpx.AsyncClient] = None

    def get_http_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client with connection pooling."""
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(
                limits=httpx.Limits(
                    max_connections=2048,
                    max_keepalive_connections=512
                ),
                timeout=httpx.Timeout(
                    connect=300.0,
                    read=None,
                    write=None,
                    pool=None,
                ),
            )
        return self._http_client

    async def close(self):
        """Close the HTTP client."""
        if self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None


STATE = ProxyState()


class InitRequest(BaseModel):
    tokenizer_path: str
    remote_engine_url: str


class MaskRequest(BaseModel):
    session_id: str
    messages_str: str


@app.post("/init")
async def init_proxy(request: InitRequest):
    """Initialize the proxy with tokenizer and remote engine URL."""
    global STATE

    try:
        STATE.tokenizer = AutoTokenizer.from_pretrained(
            request.tokenizer_path,
            trust_remote_code=True
        )
        STATE.trajectory_mask_builder = TrajectoryMaskBuilder(tokenizer=STATE.tokenizer)

        # Normalize remote engine URL
        engine_url = request.remote_engine_url.rstrip("/")
        if "/v1" not in engine_url:
            engine_url = engine_url + "/v1"
        STATE.remote_engine_url = engine_url

        logger.info(f"Initialized with tokenizer: {request.tokenizer_path}")
        logger.info(f"Remote engine URL: {STATE.remote_engine_url}")

        return {"success": True, "message": "Proxy initialized"}
    except Exception as e:
        logger.error(f"Init failed: {e}")
        raise HTTPException(status_code=500, detail=f"Init failed: {e}")


@app.post("/v1/{session_id}/chat/completions")
async def proxy_chat_completions(session_id: str, request: Request):
    """
    Proxy chat completions to the remote engine.
    Records the trajectory for mask computation.
    """
    if STATE.remote_engine_url is None:
        raise HTTPException(status_code=503, detail="Proxy not initialized. Call /init first.")

    try:
        payload = await request.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON body: {e}")

    messages = payload.get("messages", [])

    # Forward to remote engine
    http_client = STATE.get_http_client()
    url = f"{STATE.remote_engine_url}/chat/completions"

    try:
        logger.debug(f"Forwarding to {url}")
        resp = await http_client.post(
            url,
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        resp.raise_for_status()
        resp_json = resp.json()
    except Exception as e:
        import traceback
        logger.error(f"Forward failed: {traceback.format_exc()}")
        raise HTTPException(status_code=502, detail=f"Failed to call remote engine: {traceback.format_exc()}")

    # Extract assistant response
    try:
        assistant_text = resp_json["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        assistant_text = ""

    # Save trajectory for mask building
    if STATE.trajectory_mask_builder is not None:
        try:
            await asyncio.to_thread(
                STATE.trajectory_mask_builder.save,
                session_id,
                messages,
                assistant_text
            )
        except Exception as e:
            logger.warning(f"Failed to save trajectory: {e}")

    return resp_json


@app.post("/get_trajectory_mask")
async def get_trajectory_mask(request: MaskRequest):
    """
    Get the trajectory mask for a session.
    Returns mask_ranges as list of (start, end) tuples where mask=1.
    """
    if STATE.trajectory_mask_builder is None:
        raise HTTPException(status_code=503, detail="Proxy not initialized")

    session_id = request.session_id
    messages_str = request.messages_str

    try:
        mask = STATE.trajectory_mask_builder.query(session_id, messages_str)
    except Exception as e:
        logger.error(f"Failed to query mask: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to query mask: {e}")

    # Convert [0,1] list to ranges of 1s: [(start, end), ...]
    mask_ranges = []
    start = None
    for i in range(len(mask)):
        if mask[i] == 1 and start is None:
            start = i
        elif mask[i] == 0 and start is not None:
            mask_ranges.append((start, i))
            start = None
    if start is not None:
        mask_ranges.append((start, len(mask)))

    return {"mask_ranges": mask_ranges}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "initialized": STATE.tokenizer is not None,
        "remote_engine_url": STATE.remote_engine_url,
    }


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    await STATE.close()


def main():
    host = os.environ.get("LLM_PROXY_HOST", "0.0.0.0")
    port = int(os.environ.get("LLM_PROXY_PORT", "8890"))

    logger.info(f"Starting on {host}:{port}")

    uvicorn.run(
        app,
        host=host,
        port=port,
        timeout_keep_alive=5,
    )


if __name__ == "__main__":
    main()
