#!/usr/bin/env python3
"""Queued judge proxy for multi-agent rollout environments."""

import asyncio
import json
import logging
import os
import sys
import time
import uuid
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from utils import get_env

AIEVOBOX_ROOT = get_env("AIEVOBOX_ROOT")
if AIEVOBOX_ROOT not in sys.path:
    sys.path.insert(0, AIEVOBOX_ROOT)

from env.multi_qagym.simple_llm_judge import SimpleLLMJudge

LOG_DIR = os.path.join(AIEVOBOX_ROOT, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "judge_proxy.log")

logger = logging.getLogger("judge_proxy")
logger.setLevel(logging.DEBUG)
logger.propagate = False

file_handler = RotatingFileHandler(
    LOG_FILE,
    maxBytes=50 * 1024 * 1024,
    backupCount=5,
    encoding="utf-8",
)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
logger.addHandler(file_handler)

if os.getenv("JUDGE_PROXY_ENABLE_CONSOLE_LOG", "1") == "1":
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(console_handler)

app = FastAPI(title="Judge Proxy Server", debug=True)


class JudgeRequest(BaseModel):
    query: str
    response: str
    target_model_holder: Optional[str] = None


class JudgeResponse(BaseModel):
    score: float
    reason: str
    elapsed_s: float


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        return max(1, int(raw))
    except ValueError:
        logger.warning("Invalid %s=%r, fallback to %d", name, raw, default)
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("Invalid %s=%r, fallback to %.1f", name, raw, default)
        return default


class JudgeProxyState:
    def __init__(self):
        self.max_concurrency = _env_int("JUDGE_PROXY_MAX_CONCURRENCY", 1)
        self._semaphore = asyncio.Semaphore(self.max_concurrency)
        self._judge: Optional[SimpleLLMJudge] = None
        self.dump_inputs = os.getenv("JUDGE_PROXY_DUMP_INPUTS", "0") == "1"
        self.dump_dir = os.getenv("JUDGE_PROXY_DUMP_DIR") or os.path.join(LOG_DIR, "judge_proxy_inputs")
        if self.dump_inputs:
            os.makedirs(self.dump_dir, exist_ok=True)

    def get_judge(self, target_model_holder: Optional[str] = None) -> SimpleLLMJudge:
        if self._judge is None:
            base_url = os.getenv("JUDGE_BASE_URL") or os.getenv("OPENAI_BASE_URL")
            model_name = os.getenv("JUDGE_MODEL_NAME") or os.getenv("JUDGE_MODEL") or "gpt-4o-mini"
            api_key = os.getenv("JUDGE_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
            if not base_url:
                raise RuntimeError("JUDGE_BASE_URL or OPENAI_BASE_URL is required")
            self._judge = SimpleLLMJudge(
                api_key=api_key,
                base_url=base_url,
                model_name=model_name,
                temperature=_env_float("JUDGE_TEMPERATURE", 0.1),
                success_threshold=_env_int("JUDGE_SUCCESS_THRESHOLD", 5),
                verbose=os.getenv("JUDGE_VERBOSE", "0") == "1",
                target_model_holder=target_model_holder or os.getenv("JUDGE_TARGET_MODEL_HOLDER", "OpenAI"),
                timeout_s=_env_float("JUDGE_TIMEOUT_S", 180.0),
            )
            logger.info(
                "Initialized judge: base_url=%s model=%s max_concurrency=%d timeout_s=%.1f",
                base_url,
                model_name,
                self.max_concurrency,
                self._judge.timeout_s,
            )
        return self._judge


STATE = JudgeProxyState()


def _write_judge_dump(dump_path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(dump_path), exist_ok=True)
    tmp_path = f"{dump_path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, dump_path)


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "max_concurrency": STATE.max_concurrency,
        "judge_initialized": STATE._judge is not None,
    }


@app.post("/judge", response_model=JudgeResponse)
async def judge(request: JudgeRequest):
    t0 = time.perf_counter()
    async with STATE._semaphore:
        wait_s = time.perf_counter() - t0
        logger.info("judge request dequeued after %.3fs response_chars=%d", wait_s, len(request.response or ""))
        judge_model = STATE.get_judge(request.target_model_holder)
        request_id = uuid.uuid4().hex
        dump_path = None
        dump_payload = None
        if STATE.dump_inputs:
            judge_prompt = judge_model._build_judge_prompt(request.query, request.response)
            dump_path = os.path.join(STATE.dump_dir, f"{int(time.time())}-{request_id}.json")
            dump_payload = {
                "request_id": request_id,
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "status": "started",
                "query": request.query,
                "response": request.response,
                "target_model_holder": request.target_model_holder,
                "judge_prompt": judge_prompt,
                "query_chars": len(request.query or ""),
                "response_chars": len(request.response or ""),
                "judge_prompt_chars": len(judge_prompt),
            }
            _write_judge_dump(dump_path, dump_payload)
            logger.info("judge input dumped request_id=%s path=%s", request_id, dump_path)

        try:
            score, reason = await asyncio.to_thread(
                judge_model.evaluate_response,
                request.query,
                request.response,
            )
        except Exception as exc:
            if dump_path and dump_payload is not None:
                dump_payload.update(
                    {
                        "status": "error",
                        "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "elapsed_s": time.perf_counter() - t0,
                        "error": repr(exc),
                    }
                )
                try:
                    _write_judge_dump(dump_path, dump_payload)
                except Exception:
                    logger.exception("failed to write judge error dump request_id=%s", request_id)
            logger.exception("judge request failed: %s", exc)
            raise HTTPException(status_code=500, detail=str(exc))

        elapsed_s = time.perf_counter() - t0
        logger.info("judge response score=%.1f elapsed=%.3fs reason=%r", score, elapsed_s, reason[:200])
        diagnostics = getattr(judge_model, "last_diagnostics", None)
        if dump_path and dump_payload is not None:
            dump_payload.update(
                {
                    "status": "ok",
                    "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "elapsed_s": elapsed_s,
                    "judge_output": {
                        "score": float(score),
                        "reason": str(reason),
                        "diagnostics": diagnostics,
                    },
                }
            )
            try:
                _write_judge_dump(dump_path, dump_payload)
                logger.info("judge output dumped request_id=%s path=%s", request_id, dump_path)
            except Exception:
                logger.exception("failed to write judge output dump request_id=%s", request_id)
        if diagnostics and "Failed to evaluate" in str(reason):
            logger.error("judge fallback diagnostics=%s", json.dumps(diagnostics, ensure_ascii=False)[:4000])
        return JudgeResponse(score=float(score), reason=str(reason), elapsed_s=elapsed_s)


if __name__ == "__main__":
    port = int(os.getenv("JUDGE_PROXY_PORT", "18892"))
    host = os.getenv("JUDGE_PROXY_HOST", "0.0.0.0")
    logger.info("Judge Proxy logging to: %s", LOG_FILE)
    logger.info("Starting judge proxy on %s:%d", host, port)
    uvicorn.run(app, host=host, port=port, timeout_keep_alive=5)
