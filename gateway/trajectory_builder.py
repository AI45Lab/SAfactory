"""Gateway-owned construction and lifecycle selection for persisted trajectory rows."""

from __future__ import annotations

import uuid
from typing import Any, Dict, Optional

from gateway.models import GatewaySessionBinding, GatewayTelemetryRecord


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
        "is_terminal": False,
        "is_truncated": bool(record.is_truncated),
        "is_session_completed": False,
        "is_trainable": False,
    }
