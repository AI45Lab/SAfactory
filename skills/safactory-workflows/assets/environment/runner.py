#!/usr/bin/env python3
"""Fixed SAfactory protocol shell.  Standard part: do not edit per benchmark.

This file is copied unchanged into every environment (see ``env/prmeval``).
It owns only the SAfactory runtime contract: reading the
SimulationStartRequest, resolving the Gateway session URL, invoking the
environment hook, and emitting exactly one result JSON on stdout plus the
optional ``SAFACTORY_RESULT_PATH`` artifact.  Benchmark-specific behavior
lives in ``adapter.py:run_case``.  When the protocol evolves, replace this
whole file mechanically instead of editing around benchmark logic.
"""

from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path
import sys
from typing import Any


def read_request() -> dict[str, Any]:
    raw = sys.stdin.read().strip() or os.environ.get("SAFACTORY_START_REQUEST_JSON", "")
    if not raw:
        raise ValueError("missing SimulationStartRequest JSON")
    request = json.loads(raw)
    if not isinstance(request, dict):
        raise ValueError("SimulationStartRequest must be a JSON object")
    return request


def run_episode(request: dict[str, Any]) -> dict[str, Any]:
    session_id = _required_text(request.get("session_id"), "session_id")
    env_params = request.get("env_params")
    if not isinstance(env_params, dict):
        raise TypeError("env_params must be a JSON object")
    task = env_params.get("dataset")
    if not isinstance(task, dict):
        raise TypeError("env_params.dataset must be a JSON object")

    session_url = _first_text(
        os.environ.get("SAFACTORY_GATEWAY_SESSION_URL_CONTAINER"),
        _gateway_session_url(request, session_id),
        os.environ.get("OPENROUTER_BASE_URL"),
        os.environ.get("OPENAI_BASE_URL"),
    )
    if not session_url:
        raise ValueError("cannot resolve Gateway session URL")

    # Import the environment hook lazily and with stdout guarded: native
    # libraries may print, and stdout must contain exactly one result JSON.
    # Protocol checks stay runnable without the adapter's native dependencies
    # by swapping in a fixture adapter (contract_smoke --adapter).
    with contextlib.redirect_stdout(sys.stderr):
        from adapter import run_case

        metrics, step_count = run_case(request, task, session_url)
    if not isinstance(metrics, dict):
        raise TypeError("adapter.run_case metrics must be a JSON object")
    if type(step_count) is not int or step_count < 0:
        raise ValueError("adapter.run_case step_count must be a nonnegative integer")
    json.dumps(metrics, ensure_ascii=False, allow_nan=False)
    return {
        "session_id": session_id,
        "status": "succeeded",
        "total_reward": 0.0,
        "step_count": step_count,
        "terminated": True,
        "truncated": False,
        "error_text": None,
        "metrics": metrics,
    }


def main() -> int:
    session_id = os.environ.get("SAFACTORY_SESSION_ID", "")
    try:
        request = read_request()
        session_id = str(request.get("session_id") or session_id)
        result = run_episode(request)
    except Exception as exc:  # Runtime failures are represented in the result JSON.
        result = {
            "session_id": session_id,
            "status": "failed",
            "total_reward": 0.0,
            "step_count": 0,
            "terminated": True,
            "truncated": False,
            "error_text": str(exc),
            "metrics": {},
        }
    _write_result(result)
    return 0


def _write_result(result: dict[str, Any]) -> None:
    artifact = str(os.environ.get("SAFACTORY_RESULT_PATH") or "").strip()
    if artifact:
        try:
            path = Path(artifact)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n",
                encoding="utf-8",
            )
        except Exception as exc:
            # Keep the stdout result intact; the framework parses stdout first.
            print(f"SAFACTORY_RUNNER_DIAGNOSTIC result_artifact_write_failed: {exc}", file=sys.stderr)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)


def _gateway_session_url(request: dict[str, Any], session_id: str) -> str:
    base = str(request.get("gateway_base_url") or "").rstrip("/")
    if not base:
        return ""
    # request_env normally performs this rewrite.  Keep the fallback useful
    # when the runner is invoked by hand or by a local contract test.
    from urllib.parse import urlsplit, urlunsplit

    parts = urlsplit(base)
    if parts.hostname in {"127.0.0.1", "localhost", "::1"}:
        netloc = "host.docker.internal"
        if parts.port is not None:
            netloc = f"{netloc}:{parts.port}"
        base = urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment)).rstrip("/")
    return f"{base}/{session_id}"


def _first_text(*values: Any) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _required_text(value: Any, name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"SimulationStartRequest missing {name}")
    return text


if __name__ == "__main__":
    raise SystemExit(main())
