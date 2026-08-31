"""Structured timing log for offline analysis.

Writes one JSON line per event to a single append-only file so the buffer
server (rollout side) and the slime generator (training side) can both record
into the same log even when they run as separate processes.

Default path is ``${LOG_ROOT}/timing.jsonl`` (shared by both sides via the
env.sh), overridable with ``SAFACTORY_TIMING_LOG``. Each line carries an
``event`` field and a monotonic ``ts`` (epoch seconds) plus whatever fields
the caller passes. Lines are flushed immediately so a crash never loses
already-recorded events.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Optional

_DEFAULT_LOG_NAME = "timing.jsonl"
_file_handle = None
_file_path: Optional[str] = None


def _resolve_path() -> str:
    override = os.environ.get("SAFACTORY_TIMING_LOG", "").strip()
    if override:
        return override
    log_root = os.environ.get("LOG_ROOT", "").strip() or "/tmp"
    return os.path.join(log_root, _DEFAULT_LOG_NAME)


def _ensure_handle():
    global _file_handle, _file_path
    path = _resolve_path()
    if _file_handle is not None and _file_path == path:
        return _file_handle
    if _file_handle is not None:
        try:
            _file_handle.flush()
            _file_handle.close()
        except Exception:
            pass
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    # Line-buffered append: each write is a full line, flushed right away.
    _file_handle = open(path, "a", buffering=1, encoding="utf-8")
    _file_path = path
    return _file_handle


def emit(event: str, **fields: Any) -> None:
    """Append one timing event as a JSON line.

    Never raises: logging must not affect the training/rollout process.
    """
    try:
        record: Dict[str, Any] = {"event": event, "ts": time.time()}
        record.update(fields)
        line = json.dumps(record, ensure_ascii=False, default=str)
        handle = _ensure_handle()
        handle.write(line + "\n")
        handle.flush()
    except Exception:
        # Best-effort: drop on the floor rather than killing the run.
        pass


def now_s() -> float:
    """Monotonic seconds, for callers that measure spans themselves."""
    return time.perf_counter()
