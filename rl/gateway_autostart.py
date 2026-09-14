"""Auto-start a Safactory gateway that fronts the RL llm_proxy.

In the two-layer RL topology (docker -> gateway -> llm_proxy -> sglang) the
gateway must run on the same machine as the launcher and its DB, because for
SQLite the gateway telemetry and the launcher trajectory rows share one file.
buffer_server owns that machine, so it starts the gateway here:

- generate a gateway config from the RL env vars: one route (keyed by RL_MODEL)
  whose base_url points at the in-process llm_proxy, storage sharing the RL DB
  (AIEVOBOX_DB_URL, so it matches launcher --db-path), and max_steps=-1 so the
  gateway never injects synthetic stops that would truncate RL rollouts;
- launch `python -m gateway --config <generated>` and wait for /readyz.

Disable with AIEVOBOX_GATEWAY_AUTOSTART=0 to use an external/manual gateway.
"""
from __future__ import annotations

import logging
import os
import subprocess
import time
import urllib.request
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger("buffer_server")

_gateway_process: Optional[subprocess.Popen] = None


def autostart_enabled() -> bool:
    val = os.environ.get("AIEVOBOX_GATEWAY_AUTOSTART", "1")
    return str(val).strip().lower() not in ("0", "false", "no")


def build_gateway_config(*, aievobox_root: str) -> Dict[str, Any]:
    """Build a gateway config dict routing to the in-process RL llm_proxy."""
    db_url = os.environ.get("AIEVOBOX_DB_URL", f"sqlite:///{aievobox_root}/rl/rl.db")
    storage_type = os.environ.get("STORAGE_TYPE", "sqlite")
    port = int(os.environ.get("AIEVOBOX_GATEWAY_PORT", "8000"))
    route = os.environ.get("RL_MODEL", "default")
    proxy_host = os.environ.get("LLM_PROXY_HOST", "127.0.0.1")
    proxy_port = os.environ.get("LLM_PROXY_PORT", "18890")
    max_concurrency = int(
        os.environ.get("AIEVOBOX_LLM_MAX_CONCURRENCY")
        or os.environ.get("AIEVOBOX_POOL_SIZE")
        or 256
    )
    # Per-session LLM step budget enforced by the gateway. -1 = unlimited
    # (default: trust the agent / llm_proxy to bound rollout length). Set
    # AIEVOBOX_GATEWAY_MAX_STEPS to a non-negative integer to hard-cap runaway
    # rollouts; the gateway will inject a synthetic `max_steps_reached` stop
    # once a session reaches that many LLM calls.
    max_steps = int(os.environ.get("AIEVOBOX_GATEWAY_MAX_STEPS", "-1"))
    if max_steps < -1:
        raise ValueError("AIEVOBOX_GATEWAY_MAX_STEPS must be -1 or a non-negative integer")
    return {
        "listen_host": "0.0.0.0",
        "listen_port": port,
        "base_session_path": "/v1/sessions",
        # -1: never enforce a per-session step budget / inject synthetic stops,
        # so gateway does not truncate RL generations. llm_proxy owns rollout length.
        # Set AIEVOBOX_GATEWAY_MAX_STEPS>=0 to hard-cap rollout steps as a fallback
        # when the agent / llm_proxy fails to terminate on its own.
        "max_steps": max_steps,
        "storage_type": storage_type,
        # Must match launcher --db-path (= AIEVOBOX_DB_URL) or launcher /readyz fails.
        "storage_config": {"db_url": db_url},
        "llm_routes": {
            route: {
                # llm_proxy exposes POST /v1/chat/completions; gateway appends
                # /chat/completions and forwards the session id via header.
                "base_url": f"http://{proxy_host}:{proxy_port}/v1",
                "api_key": None,
                "supports_stream": False,
                "max_concurrency": max_concurrency,
            }
        },
    }


def _readyz_url() -> str:
    host = os.environ.get("AIEVOBOX_GATEWAY_HOST", "127.0.0.1")
    port = os.environ.get("AIEVOBOX_GATEWAY_PORT", "8000")
    return f"http://{host}:{port}/readyz"


def _wait_ready(url: str, timeout_s: float) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=3) as resp:
                if 200 <= resp.status < 300:
                    return True
        except Exception:
            pass
        time.sleep(1.0)
    return False


def ensure_started(*, aievobox_root: str, config_dir: str) -> Optional[subprocess.Popen]:
    """Start the gateway once (idempotent). Returns the process, or None if disabled."""
    global _gateway_process
    if not autostart_enabled():
        logger.info(
            "gateway autostart disabled (AIEVOBOX_GATEWAY_AUTOSTART=0); "
            "assuming an external gateway at %s",
            os.environ.get("AIEVOBOX_GATEWAY_BASE_URL", "<unset>"),
        )
        return None
    if _gateway_process is not None and _gateway_process.poll() is None:
        return _gateway_process

    cfg = build_gateway_config(aievobox_root=aievobox_root)
    os.makedirs(config_dir, exist_ok=True)
    cfg_path = os.path.join(config_dir, "gateway.rl.generated.yaml")
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

    cmd = ["python3", "-m", "gateway", "--config", cfg_path]
    logger.info("Starting gateway: %s (config=%s)", " ".join(cmd), cfg_path)
    _gateway_process = subprocess.Popen(cmd, cwd=aievobox_root)

    ready_url = _readyz_url()
    timeout_s = float(os.environ.get("AIEVOBOX_GATEWAY_READY_TIMEOUT_S", "60"))
    if _wait_ready(ready_url, timeout_s=timeout_s):
        logger.info("gateway ready at %s (pid=%s)", ready_url, _gateway_process.pid)
    else:
        logger.error(
            "gateway did not become ready at %s within %.0fs; check logs/gateway.log",
            ready_url,
            timeout_s,
        )
    return _gateway_process


def stop() -> None:
    global _gateway_process
    if _gateway_process is not None and _gateway_process.poll() is None:
        logger.info("Stopping gateway (pid=%s)", _gateway_process.pid)
        _gateway_process.terminate()
        try:
            _gateway_process.wait(timeout=10)
        except Exception:
            _gateway_process.kill()
    _gateway_process = None
