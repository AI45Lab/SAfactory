from __future__ import annotations

import json
import logging
from typing import Any

from core.data_manager.manager import DataManager
from core.perf_trace import PerfTrace
from evaluator.eval_types import EvalResult, EvalStatus, to_jsonable

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
        terminal = next(
            (row for row in reversed(rows) if _is_trainable_step(row)),
            None,
        )
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
                recorded = await self.data_manager.record_evaluation_summary(
                    session_id=session_id,
                    step_id=_next_step_id(rows),
                    reward=eval_result.normalized_score_10,
                    env_state=summary_metadata,
                    truncated=truncated,
                )
            else:
                recorded = await self.data_manager.update_session_step(
                    session_id,
                    int(summary.get("step_id") or 0),
                    {
                        "step_reward": eval_result.normalized_score_10,
                        "reward": eval_result.normalized_score_10,
                        "env_state": _merge_env_state(
                            summary.get("env_state"),
                            summary_metadata,
                        ),
                        "is_terminal": True,
                        **({"is_truncated": True} if truncated else {}),
                        "is_session_completed": True,
                    },
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
        env_state = _merge_env_state(terminal.get("env_state"), metadata)
        updated = await self.data_manager.update_session_step(
            session_id,
            int(terminal.get("step_id") or 0),
            {
                "step_reward": eval_result.normalized_score_10,
                "reward": eval_result.normalized_score_10,
                "env_state": env_state,
                "is_terminal": True,
                **({"is_truncated": True} if truncated else {}),
                "is_session_completed": True,
            },
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
                }
            },
            ensure_ascii=False,
        )


def _merge_env_state(existing: Any, new_metadata: str) -> str:
    if isinstance(existing, dict):
        existing_obj = dict(existing)
    else:
        try:
            existing_obj = json.loads(existing) if existing else {}
        except Exception:
            existing_obj = {"previous_env_state": existing}
    try:
        new_obj = json.loads(new_metadata)
    except Exception:
        new_obj = {"eval": {"raw": new_metadata}}
    existing_obj.update(new_obj)
    return json.dumps(existing_obj, ensure_ascii=False)


_NON_TRAINABLE_EVENT_TYPES = {
    "gateway_session_close",
    "episode_summary",
    "evaluation_summary",
}


def _is_trainable_step(row: dict[str, Any]) -> bool:
    env_state = _load_env_state(row["env_state"])
    event_type = env_state.get("event_type")
    if event_type in _NON_TRAINABLE_EVENT_TYPES:
        return False
    if env_state.get("synthetic_stop"):
        return False
    if event_type == "gateway_inference":
        try:
            return int(env_state.get("status_code") or 200) < 400
        except (TypeError, ValueError):
            return True
    return bool(_has_messages(row["messages"]) or row["response"])


def _last_trainable_row(rows: list[dict[str, Any]], trainable_ids: list[int]) -> dict[str, Any] | None:
    trainable = set(trainable_ids)
    for row in reversed(rows):
        if int(row["id"]) in trainable:
            return row
    return None


def _existing_eval_summary_row(rows: list[dict[str, Any]], session_id: str) -> dict[str, Any] | None:
    for row in reversed(rows):
        env_state = _load_env_state(row["env_state"])
        if env_state.get("event_type") != "evaluation_summary":
            continue
        eval_metadata = env_state.get("eval")
        if isinstance(eval_metadata, dict) and eval_metadata.get("session_id") == session_id:
            return row
    return None


def _next_step_id(rows: list[dict[str, Any]]) -> int:
    if not rows:
        return 0
    return max(int(row["step_id"] or 0) for row in rows) + 1


def _as_eval_summary_metadata(metadata: str) -> str:
    obj = _load_env_state(metadata)
    obj["event_type"] = "evaluation_summary"
    return json.dumps(obj, ensure_ascii=False)


def _load_env_state(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    try:
        parsed = json.loads(value) if value else {}
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _has_messages(value: Any) -> bool:
    try:
        parsed = json.loads(value) if isinstance(value, str) else value
    except Exception:
        return bool(value)
    return bool(parsed)
