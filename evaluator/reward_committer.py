from __future__ import annotations

import json
import logging
import uuid
from typing import Any

from core.data_manager.manager import DataManager
from core.perf_trace import PerfTrace
from evaluator.eval_types import EvalResult, EvalStatus, to_jsonable
from evaluator.trajectory_policy import metadata_from_row, select_reward_target

log = logging.getLogger("evaluator.reward_committer")


class RewardCommitter:
    def __init__(
        self,
        *,
        db_url: str,
        storage_type: str = "sqlite",
        data_manager: Any | None = None,
    ) -> None:
        self.storage_type = str(storage_type or "sqlite").strip().lower()
        if self.storage_type not in {"sqlite", "cloud"}:
            raise ValueError(f"RewardCommitter does not support storage type {storage_type!r}")
        self._owns_data_manager = data_manager is None
        if data_manager is None:
            if self.storage_type == "cloud":
                raise ValueError("RewardCommitter cloud mode requires a data manager")
            data_manager = DataManager(job_id="", storage_type="sqlite", db_url=db_url)
        self.data_manager = data_manager

    async def commit(
        self,
        *,
        session_id: str,
        eval_result: EvalResult,
    ) -> None:
        if eval_result.status not in {
            EvalStatus.SUCCEEDED,
            EvalStatus.SUCCEEDED.value,
            EvalStatus.TRUNCATED,
            EvalStatus.TRUNCATED.value,
        }:
            raise ValueError(
                f"cannot commit reward for evaluation status {eval_result.status!r}"
            )
        trace = PerfTrace(
            "evaluator.reward_commit",
            logger=log,
            context={
                "session_id": session_id,
                "score": eval_result.normalized_score_10,
                "status": eval_result.status,
                "storage_type": self.storage_type,
            },
        )
        log.info(
            "EVAL REWARD commit start: session=%s score=%.4f status=%s storage=%s db=%s",
            session_id,
            eval_result.normalized_score_10,
            eval_result.status,
            self.storage_type,
            None,
        )
        try:
            init = getattr(self.data_manager, "init", None)
            if callable(init):
                await init()
            with trace.span("data_manager_commit"):
                await self._commit_data_manager(
                    session_id=session_id,
                    eval_result=eval_result,
                )
            log.info(
                "EVAL REWARD commit complete: session=%s score=%.4f",
                session_id,
                eval_result.normalized_score_10,
            )
            trace.emit_summary(status="success")
        except Exception as exc:
            trace.emit_summary(status="failed", error_type=type(exc).__name__, error=str(exc))
            raise
        finally:
            if self._owns_data_manager:
                await self.data_manager.close()

    async def _commit_data_manager(
        self,
        *,
        session_id: str,
        eval_result: EvalResult,
    ) -> None:
        truncated = eval_result.status in {
            EvalStatus.TRUNCATED,
            EvalStatus.TRUNCATED.value,
        }
        rows = await self.data_manager.list_session_steps(
            session_id,
            checkout_latest=True,
        )
        terminal = select_reward_target(rows)
        log.info(
            "EVAL REWARD rows: session=%s total_rows=%d terminal_found=%s",
            session_id,
            len(rows),
            terminal is not None,
        )
        if terminal is None:
            metadata = self._build_reward_metadata(
                session_id=session_id,
                eval_result=eval_result,
            )
            summary_metadata = _as_eval_summary_metadata(metadata)
            summary = _existing_eval_summary_row(rows, session_id)
            if summary is None:
                reference = rows[-1] if rows else {}
                insert_rows = getattr(self.data_manager, "insert_session_step_rows", None)
                if callable(insert_rows):
                    summary_row = {
                        "record_id": str(uuid.uuid4()),
                        "session_id": session_id,
                        "env_id": session_id,
                        "step_id": _next_step_id(rows),
                        "env_name": str(reference.get("env_name") or "gateway"),
                        "llm_model": str(reference.get("llm_model") or ""),
                        "group_id": str(reference.get("group_id") or ""),
                        "job_id": str(reference.get("job_id") or self.data_manager.job_id or ""),
                        "messages": [],
                        "request": None,
                        "response": "",
                        "step_reward": eval_result.normalized_score_10,
                        "reward": eval_result.normalized_score_10,
                        "meta_json": _load_meta_json(summary_metadata),
                        "is_terminal": True,
                        "is_truncated": truncated,
                        "is_session_completed": True,
                        "is_trainable": False,
                    }
                    if eval_result.ground_truth_answer is not None:
                        summary_row["ground_truth_answer"] = eval_result.ground_truth_answer
                    record_ids = await insert_rows([summary_row])
                    recorded = len(record_ids)
                else:
                    # Compatibility for injected legacy test doubles.
                    recorded = await self.data_manager.record_evaluation_summary(
                        session_id=session_id,
                        step_id=_next_step_id(rows),
                        reward=eval_result.normalized_score_10,
                        env_state=summary_metadata,
                        truncated=truncated,
                    )
            else:
                summary_updates = {
                    "step_reward": eval_result.normalized_score_10,
                    "reward": eval_result.normalized_score_10,
                    "meta_json": _merge_meta_json(summary.get("meta_json"), summary_metadata),
                    "is_terminal": True,
                    **({"is_truncated": True} if truncated else {}),
                    "is_session_completed": True,
                }
                if eval_result.ground_truth_answer is not None:
                    summary_updates["ground_truth_answer"] = eval_result.ground_truth_answer
                recorded = await _update_persisted_row(
                    self.data_manager,
                    summary,
                    summary_updates,
                )
            if recorded <= 0:
                raise RuntimeError(
                    "Cannot commit evaluation reward: evaluation summary "
                    f"was not persisted for {session_id}"
                )
            log.info(
                "EVAL REWARD summary persisted: session=%s step_id=%d",
                session_id,
                _next_step_id(rows) if summary is None else int(summary.get("step_id") or 0),
            )
            return

        metadata = self._build_reward_metadata(
            session_id=session_id,
            eval_result=eval_result,
        )
        meta_json = _merge_meta_json(terminal.get("meta_json"), metadata)
        terminal_updates = {
            "step_reward": eval_result.normalized_score_10,
            "reward": eval_result.normalized_score_10,
            "meta_json": meta_json,
            "is_terminal": True,
            **({"is_truncated": True} if truncated else {}),
            "is_session_completed": True,
        }
        if eval_result.ground_truth_answer is not None:
            terminal_updates["ground_truth_answer"] = eval_result.ground_truth_answer
        updated = await _update_persisted_row(
            self.data_manager,
            terminal,
            terminal_updates,
        )
        if updated <= 0:
            raise RuntimeError(
                f"Cannot commit evaluation reward: session row was not updated for {session_id}"
            )

    def _build_reward_metadata(self, *, session_id: str, eval_result: EvalResult) -> str:
        return json.dumps(
            {
                "eval": {
                    "session_id": session_id,
                    "status": eval_result.status,
                    "normalized_score_10": eval_result.normalized_score_10,
                    "reason": eval_result.reason,
                    "result": to_jsonable(eval_result),
                    "context": to_jsonable(eval_result.evaluation_context),
                }
            },
            ensure_ascii=False,
        )


def _merge_meta_json(existing: Any, new_metadata: str) -> str:
    if isinstance(existing, dict):
        existing_obj = dict(existing)
    else:
        try:
            existing_obj = json.loads(existing) if existing else {}
        except Exception:
            existing_obj = {"legacy_meta_json": existing}
    try:
        new_obj = json.loads(new_metadata)
    except Exception:
        new_obj = {"eval": {"raw": new_metadata}}
    return json.dumps(_deep_merge(existing_obj, new_obj), ensure_ascii=False)


def _deep_merge(existing: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    merged = dict(existing)
    for key, value in updates.items():
        current = merged.get(key)
        merged[key] = _deep_merge(current, value) if isinstance(current, dict) and isinstance(value, dict) else value
    return merged


def _existing_eval_summary_row(rows: list[dict[str, Any]], session_id: str) -> dict[str, Any] | None:
    for row in reversed(rows):
        meta_json = metadata_from_row(row)
        if meta_json.get("event_type") != "evaluation_summary":
            continue
        eval_metadata = meta_json.get("eval")
        if isinstance(eval_metadata, dict) and eval_metadata.get("session_id") == session_id:
            return row
    return None


def _next_step_id(rows: list[dict[str, Any]]) -> int:
    if not rows:
        return 0
    return max(int(row["step_id"] or 0) for row in rows) + 1


def _as_eval_summary_metadata(metadata: str) -> str:
    obj = _load_meta_json(metadata)
    obj["event_type"] = "evaluation_summary"
    return json.dumps(obj, ensure_ascii=False)


def _load_meta_json(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    try:
        parsed = json.loads(value) if value else {}
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


async def _update_persisted_row(
    data_manager: Any,
    row: dict[str, Any],
    updates: dict[str, Any],
) -> int:
    record_id = str(row.get("record_id") or row.get("id") or "")
    update_rows = getattr(data_manager, "update_session_step_rows", None)
    if record_id and callable(update_rows):
        return await update_rows(
            job_id=str(row.get("job_id") or "") or None,
            record_id=record_id,
            updates=updates,
        )
    return await data_manager.update_session_step(
        str(row.get("session_id") or ""),
        int(row.get("step_id") or 0),
        updates,
    )
