from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List


log = logging.getLogger("core.data_manager.upsert_batcher")

UpsertWriter = Callable[[List[Dict[str, Any]]], Awaitable[List[str]]]
_STOP = object()


@dataclass(slots=True)
class _UpsertRequest:
    rows: List[Dict[str, Any]]
    future: asyncio.Future[List[str]]


class SessionStepUpsertBatcher:
    """Serialize complete-row upserts and acknowledge them after persistence."""

    def __init__(
        self,
        writer: UpsertWriter,
        *,
        flush_interval: float = 10.0,
        batch_size: int = 100,
        queue_size: int = 1000,
    ) -> None:
        self._writer = writer
        self._flush_interval = max(0.0, float(flush_interval))
        self._batch_size = max(1, int(batch_size))
        self._queue: asyncio.Queue[_UpsertRequest | object] = asyncio.Queue(
            maxsize=max(1, int(queue_size)),
        )
        self._state_lock = asyncio.Lock()
        self._task: asyncio.Task[None] | None = None
        self._closing = False

    async def submit(self, rows: List[Dict[str, Any]]) -> List[str]:
        if not rows:
            return []

        future: asyncio.Future[List[str]] = asyncio.get_running_loop().create_future()
        request = _UpsertRequest(rows=rows, future=future)
        async with self._state_lock:
            if self._closing:
                raise RuntimeError("session-step upsert batcher is closed")
            if self._task is None:
                self._task = asyncio.create_task(
                    self._run(),
                    name="session-step-upsert-batcher",
                )
            await self._queue.put(request)
        return await future

    async def close(self) -> None:
        async with self._state_lock:
            if not self._closing:
                self._closing = True
                if self._task is not None:
                    await self._queue.put(_STOP)
            task = self._task
        if task is not None:
            await task

    async def _run(self) -> None:
        while True:
            item = await self._queue.get()
            if item is _STOP:
                self._queue.task_done()
                return

            requests = [item]
            row_count = len(item.rows)
            stop_after_flush = False
            deadline = asyncio.get_running_loop().time() + self._flush_interval

            while row_count < self._batch_size:
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    break
                try:
                    item = await asyncio.wait_for(self._queue.get(), timeout=remaining)
                except asyncio.TimeoutError:
                    break
                if item is _STOP:
                    self._queue.task_done()
                    stop_after_flush = True
                    break
                requests.append(item)
                row_count += len(item.rows)

            try:
                await self._flush(requests)
            finally:
                for _ in requests:
                    self._queue.task_done()

            if stop_after_flush:
                return

    async def _flush(self, requests: List[_UpsertRequest]) -> None:
        # Complete rows use last-enqueued-wins semantics for duplicate keys.
        rows_by_key: Dict[tuple[str, str], Dict[str, Any]] = {}
        for request in requests:
            for row in request.rows:
                key = (str(row["job_id"]), str(row["record_id"]))
                rows_by_key[key] = row

        try:
            await self._writer(list(rows_by_key.values()))
        except Exception as exc:
            log.exception(
                "Session-step batch upsert failed: requests=%d rows=%d",
                len(requests),
                len(rows_by_key),
            )
            for request in requests:
                if not request.future.done():
                    request.future.set_exception(exc)
            return

        for request in requests:
            if not request.future.done():
                request.future.set_result([
                    str(row["record_id"])
                    for row in request.rows
                ])
