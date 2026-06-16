from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from evaluator.eval_types import EvalRequest, EvalResult, to_jsonable


class RunStatus(str, Enum):
    QUEUED = "queued"
    CONTAINER_READY = "container_ready"
    RUNNING = "running"
    ROLLOUT_SUCCEEDED = "rollout_succeeded"
    ROLLOUT_TRUNCATED = "rollout_truncated"
    ROLLOUT_FAILED = "rollout_failed"
    AWAITING_EVAL = "awaiting_eval"
    EVALUATING = "evaluating"
    EVAL_SUCCEEDED = "eval_succeeded"
    EVAL_FAILED = "eval_failed"
    REWARD_COMMITTED = "reward_committed"
    RELEASING_CONTAINER = "releasing_container"
    DONE = "done"
    FAILED = "failed"


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class RunRecord:
    session_id: str
    job_id: str = ""
    status: RunStatus = RunStatus.QUEUED
    created_at: datetime = field(default_factory=_now)
    updated_at: datetime = field(default_factory=_now)
    rollout_result: dict[str, Any] | None = None
    eval_request: EvalRequest | None = None
    eval_result: EvalResult | None = None
    error_text: str | None = None
    reward: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class InMemoryRunRegistry:
    def __init__(self) -> None:
        self._records: dict[str, RunRecord] = {}
        self._lock = asyncio.Lock()

    async def create_run(
        self,
        *,
        job_id: str = "",
        session_id: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> RunRecord:
        if not session_id:
            raise ValueError("session_id is required")
        async with self._lock:
            record = RunRecord(
                session_id=session_id,
                job_id=job_id,
                status=RunStatus.CONTAINER_READY,
                metadata=dict(metadata or {}),
            )
            self._records[record.session_id] = record
            return record

    async def mark_running(self, session_id: str) -> RunRecord:
        return await self._set(session_id, RunStatus.RUNNING)

    async def mark_rollout_finished(self, session_id: str, result: Any) -> RunRecord:
        if getattr(result, "truncated", False) or getattr(result, "status", "") == "truncated":
            status = RunStatus.ROLLOUT_TRUNCATED
        elif getattr(result, "status", "") == "succeeded":
            status = RunStatus.ROLLOUT_SUCCEEDED
        else:
            status = RunStatus.ROLLOUT_FAILED
        return await self._set(session_id, status, rollout_result=to_jsonable(result))

    async def mark_awaiting_eval(self, session_id: str, eval_request: EvalRequest) -> RunRecord:
        return await self._set(session_id, RunStatus.AWAITING_EVAL, eval_request=eval_request)

    async def mark_evaluating(self, session_id: str) -> RunRecord:
        return await self._set(session_id, RunStatus.EVALUATING)

    async def mark_eval_finished(self, session_id: str, result: EvalResult) -> RunRecord:
        status = RunStatus.EVAL_SUCCEEDED if result.status == "succeeded" else RunStatus.EVAL_FAILED
        return await self._set(session_id, status, eval_result=result)

    async def mark_reward_committed(self, session_id: str, reward: float) -> RunRecord:
        return await self._set(session_id, RunStatus.REWARD_COMMITTED, reward=reward)

    async def mark_releasing_container(self, session_id: str) -> RunRecord:
        return await self._set(session_id, RunStatus.RELEASING_CONTAINER)

    async def mark_failed(self, session_id: str, error_text: str) -> RunRecord:
        return await self._set(session_id, RunStatus.FAILED, error_text=error_text)

    async def mark_done(self, session_id: str) -> RunRecord:
        return await self._set(session_id, RunStatus.DONE)

    async def get(self, session_id: str) -> RunRecord | None:
        async with self._lock:
            return self._records.get(session_id)

    async def get_by_session(self, session_id: str) -> RunRecord | None:
        async with self._lock:
            return self._records.get(session_id)

    async def list_by_status(self, *statuses: RunStatus) -> list[RunRecord]:
        wanted = set(statuses)
        async with self._lock:
            return [record for record in self._records.values() if record.status in wanted]

    async def snapshot(self) -> dict[str, Any]:
        async with self._lock:
            return {
                session_id: to_jsonable(record)
                for session_id, record in self._records.items()
            }

    async def remove(self, session_id: str) -> None:
        async with self._lock:
            self._records.pop(session_id, None)

    async def _set(self, session_id: str, status: RunStatus, **updates: Any) -> RunRecord:
        if not session_id:
            raise ValueError("session_id is required")
        async with self._lock:
            record = self._records.get(session_id)
            if record is None:
                record = RunRecord(session_id=session_id)
                self._records[session_id] = record
            record.status = status
            record.updated_at = _now()
            for key, value in updates.items():
                setattr(record, key, value)
            return record
