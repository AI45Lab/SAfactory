#!/usr/bin/env python3
"""Fixed SAfactory protocol shell, based on docs/guides/custom-environment.md.

Put environment-specific behavior in adapter.py; keep this shell unchanged.
"""
from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path
import sys
from urllib.parse import urlsplit, urlunsplit


def read_request():
    raw = sys.stdin.read().strip() or os.environ.get("SAFACTORY_START_REQUEST_JSON", "")
    if not raw:
        raise ValueError("missing SimulationStartRequest JSON")
    request = json.loads(raw)
    if not isinstance(request, dict):
        raise ValueError("SimulationStartRequest must be a JSON object")
    return request


def main():
    result = {
        "session_id": os.environ.get("SAFACTORY_SESSION_ID", ""),
        "status": "failed", "total_reward": 0.0, "step_count": 0,
        "terminated": True, "truncated": False, "error_text": None, "metrics": {},
    }
    try:
        request = read_request()
        result["session_id"] = str(request["session_id"])
        session_url = os.environ.get("SAFACTORY_GATEWAY_SESSION_URL_CONTAINER") or _gateway_session_url(
            request, result["session_id"]
        )
        if not session_url:
            raise ValueError("cannot resolve Gateway session URL")
        task = (request.get("env_params") or {}).get("dataset") or {}
        with contextlib.redirect_stdout(sys.stderr):
            from adapter import run_case

            metrics, step_count = run_case(request, task, session_url)
        if not isinstance(metrics, dict):
            raise ValueError("run_case metrics must be a JSON object")
        if type(step_count) is not int or step_count < 0:
            raise ValueError("run_case step_count must be a nonnegative integer")
        json.dumps(metrics, allow_nan=False)
        result.update(status="succeeded", metrics=metrics, step_count=step_count)
    except Exception as exc:
        result["error_text"] = str(exc)

    artifact = os.environ.get("SAFACTORY_RESULT_PATH")
    if artifact:
        try:
            path = Path(artifact)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(result, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        except Exception as exc:
            result.update(status="failed", error_text=f"cannot write result artifact: {exc}")
    print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)
    return 0  # Controlled failures are result JSON, not process failures.


def _gateway_session_url(request, session_id):
    base = str(request.get("gateway_base_url") or "").rstrip("/")
    if not base:
        return ""
    parts = urlsplit(base)
    if parts.hostname in {"127.0.0.1", "localhost", "::1"}:
        netloc = "host.docker.internal"
        if parts.port is not None:
            netloc = f"{netloc}:{parts.port}"
        base = urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment)).rstrip("/")
    return f"{base}/{session_id}"


if __name__ == "__main__":
    raise SystemExit(main())
