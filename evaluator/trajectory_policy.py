"""Evaluator-owned rules for classifying persisted trajectory rows."""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, Optional


NON_TRAJECTORY_EVENT_TYPES = {
    "gateway_session_close",
    "episode_summary",
    "evaluation_summary",
}


def metadata_from_row(row: Dict[str, Any]) -> Dict[str, Any]:
    value = row.get("meta_json")
    if isinstance(value, dict):
        return dict(value)
    try:
        parsed = json.loads(value) if value else {}
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def is_session_sealing_event(row: Dict[str, Any]) -> bool:
    metadata = metadata_from_row(row)
    return bool(
        row.get("is_session_completed")
        or row.get("is_terminal")
        or metadata.get("is_session_completed")
        or metadata.get("event_type") in NON_TRAJECTORY_EVENT_TYPES
    )


def is_trajectory_step(row: Dict[str, Any]) -> bool:
    metadata = metadata_from_row(row)
    event_type = metadata.get("event_type")
    if event_type in NON_TRAJECTORY_EVENT_TYPES or metadata.get("synthetic_stop"):
        return False
    if event_type == "gateway_inference":
        try:
            return int(metadata.get("status_code") or 200) < 400
        except (TypeError, ValueError):
            return True
    return bool(_has_messages(row.get("messages")) or row.get("response"))


def select_reward_target(rows: Iterable[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    candidates = [row for row in rows if is_trajectory_step(row)]
    if not candidates:
        return None
    return max(candidates, key=lambda row: (
        int(row.get("step_id") or 0),
        str(row.get("created_at") or ""),
        str(row.get("record_id") or row.get("id") or ""),
    ))


def _has_messages(value: Any) -> bool:
    try:
        parsed = json.loads(value) if isinstance(value, str) else value
    except Exception:
        return bool(value)
    return bool(parsed)
