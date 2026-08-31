#!/usr/bin/env python3
"""
LLM Proxy Server

Embedded in the slime_generator process.  Provides a single HTTP endpoint
fronted by the Safactory gateway (docker -> gateway -> llm_proxy -> sglang):

    POST /v1/chat/completions   (session id via X-Safactory-Session-Id header)

Training data (tokens, masks, mm_train_inputs) is read directly from the
shared TrajectoryMaskBuilder in memory — no HTTP round-trip needed.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
import copy
import json
import logging
import os
import re
import sys
import time
from logging.handlers import RotatingFileHandler
from typing import Any, Optional

# Add rl directory to path for utils import
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from utils import get_env

import httpx
from fastapi import FastAPI, HTTPException, Request

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
logger.propagate = False

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
if os.getenv("LLM_PROXY_ENABLE_CONSOLE_LOG") == "1":
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

from trajectory_mask_builder import PreparedPrompt, TrajectoryMaskBuilder

app = FastAPI(title="LLM Proxy Server", debug=True)


# Qwen3.5/3.8 emit tool calls as *text* using the chat-template's
# `<function=NAME>...<parameter=KEY>VALUE</parameter>...</function>` format,
# wrapped in special tool-call delimiter tokens (e.g. `<|tool_call_begin|>`/
# `<|tool_call_end|>`). OpenHands, however, goes through litellm's `openai/`
# provider and only executes a tool when the response carries a structured
# OpenAI `tool_calls` field. Without conversion the agent emits one tool
# call as plain text, OpenHands ignores it, and the agent stops after a
# single turn (no patch, reward 0).
#
# This parser extracts `<function=...>` blocks from the raw model text and
# converts them into OpenAI `tool_calls` so OpenHands can execute them. The
# raw `assistant_text` is still what gets recorded into the training
# trajectory (see record_generation call below), so RL training data is
# unaffected — this conversion only shapes the response handed back to the
# agent.
_FUNCTION_BLOCK_RE = re.compile(r"<function=(\w+)>(.*?)</function>", re.DOTALL)
_PARAM_BLOCK_RE = re.compile(r"<parameter=(\w+)>(.*?)</parameter>", re.DOTALL)
# Trailing tool-call delimiter token (a `<...>` tag) right before the first
# `<function=` — stripped from the reasoning content we return.
_TRAILING_TAG_RE = re.compile(r"\s*<[^>]*>\s*$")


def _parse_qwen_tool_calls(assistant_text: str):
    """Convert Qwen text-format tool calls in `assistant_text` into OpenAI
    `tool_calls`. Returns (content, tool_calls, finish_reason):
      - content: reasoning text with tool-call blocks removed (None if empty)
      - tool_calls: list of OpenAI tool_call dicts, or None if none found
      - finish_reason: "tool_calls" if any, else unchanged (caller decides)
    """
    blocks = list(_FUNCTION_BLOCK_RE.finditer(assistant_text))
    if not blocks:
        return assistant_text, None, None

    tool_calls = []
    for idx, blk in enumerate(blocks):
        name = blk.group(1)
        args = {}
        for p in _PARAM_BLOCK_RE.finditer(blk.group(2)):
            args[p.group(1)] = p.group(2).strip("\n")
        tool_calls.append({
            "id": f"call_{idx}",
            "type": "function",
            "function": {
                "name": name,
                "arguments": json.dumps(args, ensure_ascii=False),
            },
        })

    # Content = text before the first tool-call block, with the trailing
    # tool-call delimiter tag stripped. Text between/after blocks is just
    # delimiter noise, discard it.
    content = assistant_text[: blocks[0].start()]
    content = _TRAILING_TAG_RE.sub("", content).strip()
    return (content or None), tool_calls, "tool_calls"


def _normalize_messages_for_qwen_template(messages):
    """Normalize OpenAI-format messages so the Qwen3.5/3.8 chat template can
    render them. The template iterates tool_call.arguments via the `items`
    filter, so each tool_call's arguments must be a dict; litellm/openai send
    arguments as a JSON string, which makes `items` raise "Can only get item
    pairs from a mapping". Parse it back to a dict. Also coerces non-string
    `content` to a string (the template calls content.startswith/endswith).
    Mutates messages in place and returns them.
    """
    for msg in messages or []:
        if not isinstance(msg, dict):
            continue
        tool_calls = msg.get("tool_calls")
        if isinstance(tool_calls, list):
            for tc in tool_calls:
                if not isinstance(tc, dict):
                    continue
                fn = tc.get("function")
                if not isinstance(fn, dict):
                    continue
                args = fn.get("arguments")
                if isinstance(args, str):
                    try:
                        fn["arguments"] = json.loads(args) if args else {}
                    except (json.JSONDecodeError, ValueError):
                        fn["arguments"] = {"_raw": args}
        content = msg.get("content")
        if content is None:
            # OpenAI allows assistant messages with tool_calls to omit content
            # (or set it to null). The Qwen chat template and qwen_vl_utils'
            # extract_vision_info both access message["content"] directly
            # (not .get), so a missing key raises KeyError. Default to "".
            msg["content"] = ""
        elif not isinstance(content, str):
            try:
                msg["content"] = json.dumps(content, ensure_ascii=False)
            except Exception:
                msg["content"] = str(content)
    return messages


def _resolve_proxy_workers() -> int:
    default_workers = min(32, max(8, os.cpu_count() or 8))
    raw = os.getenv("AIEVOBOX_LLM_PROXY_WORKERS")
    if not raw:
        return default_workers
    try:
        return max(1, int(raw))
    except ValueError:
        logger.warning(
            "Invalid AIEVOBOX_LLM_PROXY_WORKERS=%r, fallback to default=%d",
            raw,
            default_workers,
        )
        return default_workers


_PROXY_WORKERS = _resolve_proxy_workers()


@app.middleware("http")
async def set_body_size(request: Request, call_next):
    # VLM requests can include base64 images in the prompt.
    request._body_size_limit = 1_073_741_824  # 1GB
    response = await call_next(request)
    return response


class ProxyState:
    """Global state for the proxy server."""

    def __init__(self):
        self.tokenizer = None
        self.processor: Optional[Any] = None
        self.trajectory_mask_builder: Optional[TrajectoryMaskBuilder] = None
        self.remote_engine_url: Optional[str] = None  # Base URL without /v1
        self._http_client: Optional[httpx.AsyncClient] = None
        self._builder_executor: Optional[ThreadPoolExecutor] = None
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

    def get_builder_executor(self) -> ThreadPoolExecutor:
        """Get or create the dedicated builder executor."""
        if self._builder_executor is None:
            self._builder_executor = ThreadPoolExecutor(
                max_workers=_PROXY_WORKERS,
                thread_name_prefix="llm-proxy",
            )
            logger.info(
                "Initialized llm_proxy executor: workers=%d",
                _PROXY_WORKERS,
            )
        return self._builder_executor

    async def close(self):
        """Close the HTTP client."""
        if self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None
        if self._builder_executor is not None:
            self._builder_executor.shutdown(wait=False, cancel_futures=True)
            self._builder_executor = None


STATE = ProxyState()


@app.post("/v1/chat/completions")
async def proxy_chat_completions(request: Request):
    """
    Proxy chat completions to the remote engine via /generate API.
    Records the trajectory (tokens, mask, logprobs) for training.

    The Safactory gateway always fronts this proxy and passes the session id
    via the X-Safactory-Session-Id header; that id keys the training trajectory.
    """
    session_id = request.headers.get("x-safactory-session-id")
    if not session_id:
        raise HTTPException(
            status_code=400,
            detail="missing X-Safactory-Session-Id header (gateway must front the llm_proxy)",
        )
    if STATE.remote_engine_url is None or STATE.tokenizer is None:
        raise HTTPException(status_code=503, detail="Proxy not initialized.")

    try:
        payload = await request.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON body: {e}")

    messages = payload.get("messages", [])
    # OpenHands sends OpenAI-style `tools` so the model sees the real tool
    # definitions (terminal/file_editor/...). Without passing them through,
    # the chat template never renders the <tools> system block and the model
    # hallucinates tool names (e.g. `bash` instead of `terminal`), so OpenHands
    # rejects every call ("Tool 'Bash' not found") and the agent never produces
    # a patch. Forward them to the mask builder so the prompt (and thus the
    # recorded trajectory) includes the tools system block.
    tools = payload.get("tools")
    # Normalize OpenAI tool_calls (arguments as JSON string) and non-string
    # content so the Qwen chat template can render them; otherwise the
    # template's `items` filter raises "Can only get item pairs from a mapping".
    messages = _normalize_messages_for_qwen_template(messages)

    # Get sampling params from payload or use defaults
    temperature = payload.get("temperature", STATE.temperature)
    top_p = payload.get("top_p", STATE.top_p)
    max_tokens = payload.get("max_tokens")

    # Prepare input tokens (with multimodal state like cumulative image_data).
    # Run in thread pool to avoid blocking the event loop (CPU-bound tokenization)
    loop = asyncio.get_running_loop()
    builder_executor = STATE.get_builder_executor()
    prep: PreparedPrompt = await loop.run_in_executor(
        builder_executor,
        STATE.trajectory_mask_builder.prepare_generate_input,
        session_id,
        messages,
        tools,
    )
    input_ids = prep.input_ids
    image_data = prep.image_data

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
    if image_data:
        generate_payload["image_data"] = image_data

    # Call /generate endpoint
    http_client = STATE.get_http_client()
    url = f"{STATE.remote_engine_url}/generate"

    # Session affinity: when the SGLang router runs the `consistent_hashing`
    # policy, it pins all turns of a session (keyed by this header) to one
    # worker so the worker's RadixAttention prefix tree keeps reusing the
    # session's growing history. Without it the default cache_aware policy
    # scatters turns across workers and ~78% of prefills recompute the full
    # prompt from scratch. Harmless when the router uses a non-hashing policy
    # (unknown header is ignored).
    gen_headers: dict[str, str] = {"Content-Type": "application/json"}
    if session_id:
        gen_headers["X-SMG-Routing-Key"] = session_id

    try:
        logger.debug(f"Calling /generate: input_ids length={len(input_ids)}, max_new_tokens={max_new_tokens}")
        resp = await http_client.post(
            url,
            json=generate_payload,
            headers=gen_headers
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

    # Convert Qwen text-format tool calls into OpenAI `tool_calls` so OpenHands
    # executes them. Without this the agent stops after one turn (see
    # _parse_qwen_tool_calls docstring).
    msg_content, tool_calls, tool_finish = _parse_qwen_tool_calls(assistant_text)
    if tool_calls:
        message_obj = {"role": "assistant", "content": msg_content, "tool_calls": tool_calls}
        resp_finish_reason = tool_finish
    else:
        message_obj = {"role": "assistant", "content": assistant_text}
        resp_finish_reason = finish_reason

    # Save trajectory. Pass a NORMALIZED copy of message_obj (tool_calls
    # arguments as dict, content as string) so the trie stores the same
    # message format as the DB after _normalize_messages_for_qwen_template.
    # The raw assistant_text is still used for token/mask computation.
    # We normalize a COPY so the response sent to OpenHands keeps string
    # arguments (OpenAI spec requires JSON string, not dict).
    if STATE.trajectory_mask_builder is not None:
        try:
            trie_msg = _normalize_messages_for_qwen_template(
                [copy.deepcopy(message_obj)]
            )[0]
            await loop.run_in_executor(
                builder_executor,
                STATE.trajectory_mask_builder.record_generation,
                prep,
                output_ids,
                output_logprobs,
                assistant_text,
                finish_reason,
                trie_msg,
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
            "message": message_obj,
            "finish_reason": resp_finish_reason
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
