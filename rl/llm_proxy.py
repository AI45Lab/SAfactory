#!/usr/bin/env python3
"""
LLM Proxy Server

This server:
1. Proxies LLM requests to the real engine via /generate API
2. Records trajectory (tokens, mask, logprobs) for training
3. Provides /get_tokens, /get_logprobs endpoints for the training client
"""

import asyncio
import json
import logging
import os
import sys
import time
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, List, Optional

# Add rl directory to path for utils import
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from utils import get_env

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from transformers import AutoTokenizer

# Add AIEvoBox to path
AIEVOBOX_ROOT = get_env("AIEVOBOX_ROOT")
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


class ProxyState:
    """Global state for the proxy server."""

    def __init__(self):
        self.tokenizer: Optional[AutoTokenizer] = None
        self.trajectory_mask_builder: Optional[TrajectoryMaskBuilder] = None
        self.remote_engine_url: Optional[str] = None  # Base URL without /v1
        self._http_client: Optional[httpx.AsyncClient] = None
        self.max_length: Optional[int] = None
        # Sampling params
        self.temperature: float = 1.0
        self.top_p: float = 1.0

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
    max_length: Optional[int] = None
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0


class MaskRequest(BaseModel):
    session_id: str
    messages_str: str


class TokensRequest(BaseModel):
    session_id: str
    messages_str: str


class LogprobsRequest(BaseModel):
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

        # Normalize remote engine URL (remove trailing slash and /v1 if present)
        engine_url = request.remote_engine_url.rstrip("/")
        if engine_url.endswith("/v1"):
            engine_url = engine_url[:-3]
        STATE.remote_engine_url = engine_url
        STATE.max_length = request.max_length
        STATE.temperature = request.temperature or 1.0
        STATE.top_p = request.top_p or 1.0

        logger.info(f"Initialized with tokenizer: {request.tokenizer_path}")
        logger.info(f"Remote engine URL: {STATE.remote_engine_url}")
        logger.info(f"Max length: {STATE.max_length}")
        logger.info(f"Temperature: {STATE.temperature}, Top-p: {STATE.top_p}")

        return {"success": True, "message": "Proxy initialized"}
    except Exception as e:
        logger.error(f"Init failed: {e}")
        raise HTTPException(status_code=500, detail=f"Init failed: {e}")


@app.post("/v1/{session_id}/chat/completions")
async def proxy_chat_completions(session_id: str, request: Request):
    """
    Proxy chat completions to the remote engine via /generate API.
    Records the trajectory (tokens, mask, logprobs) for training.
    """
    if STATE.remote_engine_url is None or STATE.tokenizer is None:
        raise HTTPException(status_code=503, detail="Proxy not initialized. Call /init first.")

    try:
        payload = await request.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON body: {e}")

    messages = payload.get("messages", [])

    # Get sampling params from payload or use defaults
    temperature = payload.get("temperature", STATE.temperature)
    top_p = payload.get("top_p", STATE.top_p)
    max_tokens = payload.get("max_tokens")

    # Prepare input_ids using matching logic
    # This reuses historical tokens and only tokenizes new context
    input_ids, matched_record, matched_prefix_len, matched_tokens_count = STATE.trajectory_mask_builder.prepare_input_ids(
        session_id, messages
    )

    # Calculate max_new_tokens
    if STATE.max_length is not None:
        remaining = STATE.max_length - len(input_ids)
        if max_tokens is not None:
            max_new_tokens = min(max_tokens, remaining)
        else:
            max_new_tokens = remaining

        if max_new_tokens <= 0:
            logger.warning(f"Token budget exhausted: input_ids={len(input_ids)}, max_length={STATE.max_length}")
            # Return empty response
            return {
                "id": f"chatcmpl-{session_id}-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": "proxy",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": ""},
                    "finish_reason": "length"
                }],
                "usage": {
                    "prompt_tokens": len(input_ids),
                    "completion_tokens": 0,
                    "total_tokens": len(input_ids)
                }
            }
    else:
        max_new_tokens = max_tokens or 256

    # Build /generate request
    generate_payload = {
        "input_ids": input_ids,
        "sampling_params": {
            "temperature": temperature,
            "top_p": top_p,
            "max_new_tokens": max_new_tokens,
        },
        "return_logprob": True,
        "stream": False,
    }

    # Call /generate endpoint
    http_client = STATE.get_http_client()
    url = f"{STATE.remote_engine_url}/generate"

    try:
        logger.debug(f"Calling /generate: input_ids length={len(input_ids)}, max_new_tokens={max_new_tokens}")
        resp = await http_client.post(
            url,
            json=generate_payload,
            headers={"Content-Type": "application/json"}
        )
        resp.raise_for_status()
        resp_json = resp.json()
        logger.debug(f"Generate response: text length={len(resp_json.get('text', ''))}")
    except Exception as e:
        import traceback
        logger.error(f"Generate failed: {traceback.format_exc()}")
        raise HTTPException(status_code=502, detail=f"Failed to call /generate: {traceback.format_exc()}")

    # Extract response data
    output_ids = resp_json.get("output_ids", [])
    output_logprobs = resp_json.get("meta_info", {}).get("output_token_logprobs", [])
    finish_reason_info = resp_json.get("meta_info", {}).get("finish_reason", {})

    # Determine finish_reason
    if isinstance(finish_reason_info, dict):
        finish_reason = finish_reason_info.get("type", "stop")
    else:
        finish_reason = "stop"

    # Get assistant_text from generate API response (already decoded)
    assistant_text = resp_json.get("text", "")

    # Save trajectory
    if STATE.trajectory_mask_builder is not None:
        try:
            await asyncio.to_thread(
                STATE.trajectory_mask_builder.save,
                session_id,
                messages,
                input_ids,
                output_ids,
                output_logprobs,
                finish_reason,
                matched_record,
                matched_tokens_count,
                assistant_text,  # 传递 generate API 返回的 text
            )
        except Exception as e:
            import traceback
            logger.warning(f"Failed to save trajectory: {traceback.format_exc()}")

    # Build OpenAI-compatible response
    response = {
        "id": f"chatcmpl-{session_id}-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "proxy",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": assistant_text
            },
            "finish_reason": finish_reason
        }],
        "usage": {
            "prompt_tokens": len(input_ids),
            "completion_tokens": len(output_ids),
            "total_tokens": len(input_ids) + len(output_ids)
        },
        "metadata": {
            "weight_version": resp_json.get("meta_info", {}).get("weight_version")
        }
    }

    return response


@app.post("/get_trajectory_mask")
async def get_trajectory_mask(request: MaskRequest):
    """
    Get the trajectory mask for a session (legacy interface).
    Returns mask_ranges as list of (start, end) tuples where mask=1.
    """
    if STATE.trajectory_mask_builder is None:
        raise HTTPException(status_code=503, detail="Proxy not initialized")

    session_id = request.session_id
    messages_str = request.messages_str

    try:
        # Use the new token-level query
        tokens, response_mask = STATE.trajectory_mask_builder.query_tokens(session_id, messages_str)

        # 按 max_length 截断
        if STATE.max_length is not None and len(tokens) > STATE.max_length:
            tokens = tokens[:STATE.max_length]
            response_mask = response_mask[:STATE.max_length]

        # Convert response_mask to ranges
        mask_ranges = []
        start = None
        for i in range(len(response_mask)):
            if response_mask[i] == 1 and start is None:
                start = i
            elif response_mask[i] == 0 and start is not None:
                mask_ranges.append((start, i))
                start = None
        if start is not None:
            mask_ranges.append((start, len(response_mask)))

        return {"mask_ranges": mask_ranges, "tokens": tokens, "response_mask": response_mask}
    except Exception as e:
        logger.error(f"Failed to query mask: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to query mask: {e}")


@app.post("/get_tokens")
async def get_tokens(request: TokensRequest):
    """
    Get the tokens and response_mask for a session.
    """
    if STATE.trajectory_mask_builder is None:
        raise HTTPException(status_code=503, detail="Proxy not initialized")

    session_id = request.session_id
    messages_str = request.messages_str

    try:
        tokens, response_mask = STATE.trajectory_mask_builder.query_tokens(session_id, messages_str)

        # 按 max_length 截断
        if STATE.max_length is not None and len(tokens) > STATE.max_length:
            tokens = tokens[:STATE.max_length]
            response_mask = response_mask[:STATE.max_length]

        return {"tokens": tokens, "response_mask": response_mask}
    except Exception as e:
        logger.error(f"Failed to query tokens: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to query tokens: {e}")


@app.post("/get_logprobs")
async def get_logprobs(request: LogprobsRequest):
    """
    Get the logprobs for a session.
    """
    if STATE.trajectory_mask_builder is None:
        raise HTTPException(status_code=503, detail="Proxy not initialized")

    session_id = request.session_id
    messages_str = request.messages_str

    try:
        logprobs = STATE.trajectory_mask_builder.query_logprobs(session_id, messages_str)
        return {"logprobs": logprobs}
    except Exception as e:
        logger.error(f"Failed to query logprobs: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to query logprobs: {e}")


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
    port = int(get_env("LLM_PROXY_PORT"))

    logger.info(f"Starting on 0.0.0.0:{port}")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        timeout_keep_alive=5,
    )


def start() -> "subprocess.Popen":
    """Start the LLM Proxy as a subprocess."""
    import subprocess

    llm_proxy_script = os.path.join(os.path.dirname(__file__), "llm_proxy.py")
    logger.info(f"Starting LLM Proxy: {llm_proxy_script}")

    process = subprocess.Popen(
        ["python3", llm_proxy_script],
        stdout=None,
        stderr=None,
    )
    logger.info(f"LLM Proxy started with PID: {process.pid}")
    return process


def init(tokenizer_path: str, remote_engine_url: str, max_length: int = None, max_retries: int = 10) -> bool:
    """Initialize the LLM Proxy with tokenizer and remote engine URL."""
    import requests

    host = get_env("LLM_PROXY_HOST")
    port = get_env("LLM_PROXY_PORT")
    init_url = f"http://{host}:{port}/init"

    payload = {
        "tokenizer_path": tokenizer_path,
        "remote_engine_url": remote_engine_url,
        "max_length": max_length,
    }

    for attempt in range(max_retries):
        try:
            resp = requests.post(init_url, json=payload, timeout=10)
            if resp.status_code == 200:
                logger.info("LLM Proxy initialized successfully")
                return True
            else:
                logger.warning(f"LLM Proxy init failed: {resp.status_code}")
        except Exception as e:
            logger.warning(f"LLM Proxy init attempt {attempt+1}/{max_retries} failed: {e}")
        time.sleep(2)

    logger.error(f"Failed to initialize LLM Proxy after {max_retries} attempts")
    return False


if __name__ == "__main__":
    main()
