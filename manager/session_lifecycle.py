"""Manager-owned fallback lifecycle operations for unevaluated runs."""

from __future__ import annotations

from typing import Any, Dict


NON_TRAJECTORY_EVENT_TYPES = {
    "gateway_session_close",
    "episode_summary",
    "evaluation_summary",
}


async def complete_latest_session_step(
    data_manager: Any,
    *,
    session_id: str,
    job_id: str,
    llm_model: str | None = None,
) -> int:
    rows = await data_manager.list_session_steps(
        session_id,
        job_id=job_id,
        checkout_latest=True,
    )
    candidates = [
        row for row in rows
        if (not llm_model or str(row.get("llm_model") or "") == llm_model)
        and _is_trajectory_row(row)
    ]
    if not candidates:
        return 0
    latest = max(candidates, key=lambda row: (
        int(row.get("step_id") or 0),
        str(row.get("created_at") or ""),
        str(row.get("record_id") or row.get("id") or ""),
    ))
    return await data_manager.update_session_step_rows(
        job_id=job_id,
        record_id=str(latest.get("record_id") or latest.get("id")),
        updates={"is_session_completed": True, "is_terminal": True},
    )


def _is_trajectory_row(row: Dict[str, Any]) -> bool:
    meta_json = row.get("meta_json")
    if not isinstance(meta_json, dict):
        meta_json = {}
    return meta_json.get("event_type") not in NON_TRAJECTORY_EVENT_TYPES
