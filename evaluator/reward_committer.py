from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
from typing import Any

from core.perf_trace import PerfTrace
from evaluator.eval_types import EvalResult, to_jsonable
from evaluator.trajectory_reader import _sqlite_path

log = logging.getLogger("evaluator.reward_committer")


class RewardCommitter:
    def __init__(self, *, db_url: str) -> None:
        self.db_path = _sqlite_path(db_url)

    async def commit(
        self,
        *,
        session_id: str,
        eval_result: EvalResult,
    ) -> None:
        trace = PerfTrace(
            "evaluator.reward_commit",
            logger=log,
            context={
                "session_id": session_id,
                "score": eval_result.normalized_score_10,
                "status": eval_result.status,
                "db_path": self.db_path,
            },
        )
        log.info(
            "EVAL REWARD commit start: session=%s score=%.4f status=%s db=%s",
            session_id,
            eval_result.normalized_score_10,
            eval_result.status,
            self.db_path,
        )
        try:
            with trace.span("sqlite_commit"):
                await asyncio.to_thread(
                    self._commit_sqlite,
                    session_id=session_id,
                    eval_result=eval_result,
                )
            log.info(
                "EVAL REWARD commit complete: session=%s score=%.4f",
                session_id,
                eval_result.normalized_score_10,
            )
            trace.emit_summary(status="success")
        except asyncio.CancelledError:
            trace.emit_summary(status="cancelled", error_type="CancelledError")
            raise
        except Exception as exc:
            trace.emit_summary(status="failed", error_type=type(exc).__name__, error=str(exc))
            raise

    def _commit_sqlite(
        self,
        *,
        session_id: str,
        eval_result: EvalResult,
    ) -> None:
        metadata = self._build_reward_metadata(session_id=session_id, eval_result=eval_result)
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT id, step_id, env_name, messages, response, env_state
                FROM session_steps
                WHERE session_id = ?
                ORDER BY step_id ASC, id ASC
                """,
                (session_id,),
            ).fetchall()
            trainable_ids = [int(row["id"]) for row in rows if _is_trainable_step(row)]
            terminal = _last_trainable_row(rows, trainable_ids)
            log.info(
                "EVAL REWARD commit rows: session=%s total_rows=%d trainable_rows=%d terminal_found=%s",
                session_id,
                len(rows),
                len(trainable_ids),
                terminal is not None,
            )
            if terminal is None:
                summary = _existing_eval_summary_row(rows, session_id)
                summary_metadata = _as_eval_summary_metadata(metadata)
                if summary is None:
                    conn.execute(
                        """
                        INSERT INTO session_steps
                        (session_id, step_id, env_name, llm_model, group_id, job_id, messages,
                         response, step_reward, reward, env_state, is_terminal,
                         is_truncated, is_session_completed, is_trainable)
                        VALUES (?, ?, 'gateway', '', '', '', '[]', '', ?, ?, ?, 1, 0, 1, 0)
                        """,
                        (
                            session_id,
                            _next_step_id(rows),
                            eval_result.normalized_score_10,
                            eval_result.normalized_score_10,
                            summary_metadata,
                        ),
                    )
                else:
                    env_state = _merge_env_state(summary["env_state"], summary_metadata)
                    conn.execute(
                        """
                        UPDATE session_steps
                        SET step_reward = ?, reward = ?, env_state = ?,
                            is_terminal = 1, is_session_completed = 1, is_trainable = 0
                        WHERE id = ?
                        """,
                        (eval_result.normalized_score_10, eval_result.normalized_score_10, env_state, summary["id"]),
                    )
                conn.commit()
                return

            env_state = _merge_env_state(terminal["env_state"], metadata)
            conn.execute(
                """
                UPDATE session_steps
                SET step_reward = ?, reward = ?, env_state = ?,
                    is_terminal = 1, is_session_completed = 1, is_trainable = 1
                WHERE id = ?
                """,
                (eval_result.normalized_score_10, eval_result.normalized_score_10, env_state, terminal["id"]),
            )
            _mark_trainable(conn, trainable_ids)
            conn.commit()

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


def _is_trainable_step(row: sqlite3.Row) -> bool:
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


def _last_trainable_row(rows: list[sqlite3.Row], trainable_ids: list[int]) -> sqlite3.Row | None:
    trainable = set(trainable_ids)
    for row in reversed(rows):
        if int(row["id"]) in trainable:
            return row
    return None


def _existing_eval_summary_row(rows: list[sqlite3.Row], session_id: str) -> sqlite3.Row | None:
    for row in reversed(rows):
        env_state = _load_env_state(row["env_state"])
        if env_state.get("event_type") != "evaluation_summary":
            continue
        eval_metadata = env_state.get("eval")
        if isinstance(eval_metadata, dict) and eval_metadata.get("session_id") == session_id:
            return row
    return None


def _next_step_id(rows: list[sqlite3.Row]) -> int:
    if not rows:
        return 0
    return max(int(row["step_id"] or 0) for row in rows) + 1


def _as_eval_summary_metadata(metadata: str) -> str:
    obj = _load_env_state(metadata)
    obj["event_type"] = "evaluation_summary"
    return json.dumps(obj, ensure_ascii=False)


def _load_env_state(value: Any) -> dict[str, Any]:
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


def _mark_trainable(conn: sqlite3.Connection, ids: list[int]) -> None:
    for offset in range(0, len(ids), 500):
        chunk = ids[offset : offset + 500]
        if not chunk:
            continue
        placeholders = ",".join("?" for _ in chunk)
        conn.execute(
            f"UPDATE session_steps SET is_trainable = 1 WHERE id IN ({placeholders})",
            tuple(chunk),
        )
