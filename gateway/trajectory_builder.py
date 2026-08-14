"""Gateway-owned construction and lifecycle selection for persisted trajectory rows."""

from __future__ import annotations

import uuid
from typing import Any, Dict, Iterable, Optional

from gateway.models import GatewaySessionBinding, GatewayTelemetryRecord


NON_TRAJECTORY_EVENT_TYPES = {
    "gateway_session_close",
    "episode_summary",
    "evaluation_summary",
}


def build_gateway_step_row(
    *,
    binding: GatewaySessionBinding,
    record: GatewayTelemetryRecord,
    messages: Any,
    response: Any,
    metadata: Dict[str, Any],
    dataset: Any = None,
    provider_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    meta_json = dict(metadata)
    if dataset is not None:
        meta_json["dataset"] = dataset
    if provider_meta:
        meta_json.update(provider_meta)
    record_id = str(uuid.uuid5(
        uuid.NAMESPACE_URL,
        ":".join((
            str(binding.job_id or "gateway"),
            record.session_id,
            str(record.seq_id),
            record.requested_model,
            record.event_type,
        )),
    ))
    return {
        "record_id": record_id,
        "session_id": record.session_id,
        "env_id": record.session_id,
        "step_id": record.seq_id,
        "env_name": binding.env_name or "gateway",
        "llm_model": record.requested_model,
        "group_id": binding.group_id or "",
        "job_id": binding.job_id or "gateway",
        "messages": messages,
        "request": record.request,
        "response": response,
        "step_reward": 0.0,
        "reward": None,
        "meta_json": meta_json,
        "is_terminal": bool(record.is_truncated),
        "is_truncated": bool(record.is_truncated),
        "is_session_completed": False,
        "is_trainable": False,
    }


def select_latest_trajectory_record_ids(
    rows: Iterable[Dict[str, Any]],
    *,
    models: Iterable[str] = (),
) -> list[str]:
    requested_models = {str(model) for model in models if model}
    latest: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        meta_json = row.get("meta_json")
        if not isinstance(meta_json, dict):
            meta_json = {}
        if meta_json.get("event_type") in NON_TRAJECTORY_EVENT_TYPES:
            continue
        model = str(row.get("llm_model") or "")
        if requested_models and model not in requested_models:
            continue
        previous = latest.get(model)
        current_key = (
            int(row.get("step_id") or 0),
            str(row.get("created_at") or ""),
            str(row.get("record_id") or row.get("id") or ""),
        )
        previous_key = (
            int(previous.get("step_id") or 0),
            str(previous.get("created_at") or ""),
            str(previous.get("record_id") or previous.get("id") or ""),
        ) if previous else None
        if previous_key is None or current_key > previous_key:
            latest[model] = row
    return [
        str(row.get("record_id") or row.get("id"))
        for row in latest.values()
        if row.get("record_id") or row.get("id")
    ]
