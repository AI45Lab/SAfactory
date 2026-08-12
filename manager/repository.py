from __future__ import annotations

import asyncio
import logging
import sqlite3
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

from core.perf_trace import PerfTrace

from .db_loader import (
    get_active_data,
    get_active_data_after_id,
    get_all_image,
    get_env_image_map,
)

log = logging.getLogger("manager.repository")

DB_FETCH_WARN_SECONDS = 1.0


class AgentDataRepository:
    """
    Thin repository around db_loader helpers.

    The repository owns row-reservation state so callers can reserve buffered rows
    without holding the actor-pool state lock across database reads.
    """

    def __init__(
        self,
        conn: Any,
        *,
        job_id: str = "",
        db_processing_done_checker: Optional[Callable[[], bool]] = None,
    ) -> None:
        self._conn = conn
        self._job_id = str(job_id or "").strip() or None
        self._db_processing_done_checker = db_processing_done_checker

        self._cursor_reads_enabled = isinstance(conn, sqlite3.Connection) or callable(
            getattr(conn, "get_env_configs", None)
        )
        self._last_seen_id: int = 0
        self._fallback_offset: int = 0
        self._row_buffer: Deque[Dict[str, Any]] = deque()
        self._fetch_lock = asyncio.Lock()
        self._db_processing_done_cached: bool = False
        self._stop_db_reads: bool = False
        self._fetch_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="safactory-agent-db-fetch",
        )
        self._pending_fetch: Optional[asyncio.Future[Tuple[List[Dict[str, Any]], int, int]]] = None
        self._pending_fetch_args: Optional[Tuple[int, int, int]] = None

    def reset_cursor(self) -> None:
        self._last_seen_id = 0
        self._fallback_offset = 0
        self._row_buffer.clear()
        self._db_processing_done_cached = False
        self._stop_db_reads = False
        self._pending_fetch = None
        self._pending_fetch_args = None

    def close(self) -> None:
        self._pending_fetch = None
        self._pending_fetch_args = None
        self._fetch_executor.shutdown(wait=False, cancel_futures=True)

    def get_env_image_map(self) -> Dict[str, str]:
        trace = PerfTrace(
            "manager.repository.get_env_image_map",
            logger=log,
            context={"operation": "db_read", "table": "job_environments", "job_id": self._job_id},
        )
        try:
            with trace.span("db_read.env_image_map"):
                m = get_env_image_map(self._conn, job_id=self._job_id) or {}
            trace.emit_summary(status="success", row_count=len(m))
        except Exception as exc:
            trace.emit_summary(status="failed", error_type=type(exc).__name__, error=str(exc))
            raise
        out: Dict[str, str] = {}
        for k, v in m.items():
            out[str(k)] = "" if v is None else str(v)
        return out

    def get_image_to_env_map(self) -> Dict[str, str]:
        trace = PerfTrace(
            "manager.repository.get_image_to_env_map",
            logger=log,
            context={"operation": "db_read", "table": "job_environments", "job_id": self._job_id},
        )
        try:
            with trace.span("db_read.image_to_env_map"):
                m = get_all_image(self._conn, job_id=self._job_id) or {}
            trace.emit_summary(status="success", row_count=len(m))
        except Exception as exc:
            trace.emit_summary(status="failed", error_type=type(exc).__name__, error=str(exc))
            raise
        out: Dict[str, str] = {}
        for k, v in m.items():
            out[str(k)] = str(v)
        return out

    async def prime(
        self,
        startup_batch_size: int,
        *,
        fetch_timeout_s: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Reserve the initial warm-pool rows in one batched pass."""
        return await self.reserve_rows(
            startup_batch_size,
            fetch_batch_size=max(1, int(startup_batch_size or 1)),
            fetch_timeout_s=fetch_timeout_s,
        )

    async def reserve_rows(
        self,
        limit: int,
        *,
        fetch_batch_size: int,
        wait_for_rows: bool = False,
        poll_interval_s: float = 1.0,
        max_wait_s: Optional[float] = None,
        fetch_timeout_s: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        requested = max(0, int(limit))
        if requested <= 0:
            return []

        batch_size = max(1, int(fetch_batch_size))
        poll_interval = max(0.1, float(poll_interval_s or 1.0))

        loop = asyncio.get_running_loop()
        deadline = None
        if max_wait_s is not None:
            deadline = loop.time() + max(0.0, float(max_wait_s))

        while True:
            async with self._fetch_lock:
                reserved = self._drain_buffer_locked(requested)

                while len(reserved) < requested:
                    if self._stop_db_reads:
                        break

                    if not self._row_buffer:
                        fetch_limit = max(batch_size, requested - len(reserved))
                        started_at = time.perf_counter()
                        fetch_rows: List[Dict[str, Any]] = []
                        try:
                            fetch_rows = await self._fetch_rows_async(
                                fetch_limit,
                                timeout_s=fetch_timeout_s,
                            )
                        except TimeoutError:
                            log.warning(
                                "agent DB fetch timed out; will retry scheduling new rows "
                                "limit=%d last_seen_id=%d job_id=%s",
                                int(fetch_limit),
                                int(self._last_seen_id),
                                self._job_id or "<all>",
                            )
                            break
                        elapsed = time.perf_counter() - started_at
                        if elapsed >= DB_FETCH_WARN_SECONDS:
                            log.warning(
                                "agent DB fetch took %.2fs limit=%d last_seen_id=%d buffered=%d job_id=%s",
                                elapsed,
                                int(fetch_limit),
                                int(self._last_seen_id),
                                len(self._row_buffer),
                                self._job_id or "<all>",
                            )
                        if fetch_rows:
                            self._row_buffer.extend(fetch_rows)
                        if not self._row_buffer:
                            break

                    reserved.extend(self._drain_buffer_locked(requested - len(reserved)))

                self._update_stop_state_locked()
                if reserved or not wait_for_rows or self._stop_db_reads:
                    return reserved
                if deadline is not None and loop.time() >= deadline:
                    log.warning(
                        "agent DB row wait timed out after %.1fs job_id=%s last_seen_id=%d",
                        max(0.0, float(max_wait_s or 0.0)),
                        self._job_id or "<all>",
                        int(self._last_seen_id),
                    )
                    return reserved

                log.debug(
                    "agent DB has no reservable rows yet; waiting for producer job_id=%s last_seen_id=%d",
                    self._job_id or "<all>",
                    int(self._last_seen_id),
                )

            sleep_s = poll_interval
            if deadline is not None:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    return reserved
                sleep_s = min(sleep_s, remaining)
            await asyncio.sleep(sleep_s)

    async def reserve_one(
        self,
        *,
        fetch_batch_size: int,
        wait_for_rows: bool = False,
        poll_interval_s: float = 1.0,
        max_wait_s: Optional[float] = None,
        fetch_timeout_s: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        rows = await self.reserve_rows(
            1,
            fetch_batch_size=fetch_batch_size,
            wait_for_rows=wait_for_rows,
            poll_interval_s=poll_interval_s,
            max_wait_s=max_wait_s,
            fetch_timeout_s=fetch_timeout_s,
        )
        if not rows:
            return None
        return rows[0]

    async def is_exhausted(self) -> bool:
        async with self._fetch_lock:
            self._update_stop_state_locked()
            return bool(self._stop_db_reads)

    def _drain_buffer_locked(self, limit: int) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        while self._row_buffer and len(rows) < limit:
            rows.append(self._row_buffer.popleft())
        return rows

    async def _fetch_rows_async(
        self,
        limit: int,
        *,
        timeout_s: Optional[float],
    ) -> List[Dict[str, Any]]:
        fetch_args = (int(limit), int(self._last_seen_id), int(self._fallback_offset))
        task = self._get_or_start_fetch_task(fetch_args)

        try:
            if timeout_s is None or float(timeout_s) <= 0.0:
                rows, last_seen_id, fallback_offset = await task
            else:
                rows, last_seen_id, fallback_offset = await asyncio.wait_for(
                    asyncio.shield(task),
                    timeout=float(timeout_s),
                )
        except asyncio.TimeoutError as exc:
            raise TimeoutError from exc
        finally:
            if task.done():
                self._pending_fetch = None
                self._pending_fetch_args = None

        if rows:
            self._last_seen_id = last_seen_id
            self._fallback_offset = fallback_offset
            return rows

        if self._db_processing_done_checker is None or self._db_processing_done_cached:
            self._update_stop_state_locked()
            return []

        try:
            self._db_processing_done_cached = bool(self._db_processing_done_checker())
        except Exception:
            log.warning("db_processing_done_checker failed; assuming producer is still active", exc_info=True)

        self._update_stop_state_locked()
        return []

    def _get_or_start_fetch_task(
        self,
        fetch_args: Tuple[int, int, int],
    ) -> asyncio.Future[Tuple[List[Dict[str, Any]], int, int]]:
        if (
            self._pending_fetch is not None
            and self._pending_fetch_args == fetch_args
        ):
            return self._pending_fetch

        loop = asyncio.get_running_loop()
        task = loop.run_in_executor(
            self._fetch_executor,
            self._fetch_rows_snapshot,
            fetch_args[0],
            fetch_args[1],
            fetch_args[2],
        )
        self._pending_fetch = task
        self._pending_fetch_args = fetch_args
        return task

    def _fetch_rows_snapshot(
        self,
        limit: int,
        last_seen_id: int,
        fallback_offset: int,
    ) -> Tuple[List[Dict[str, Any]], int, int]:
        trace = PerfTrace(
            "manager.repository.fetch_rows",
            logger=log,
            context={
                "operation": "db_read",
                "table": "job_environments",
                "job_id": self._job_id,
                "limit": int(limit),
                "last_seen_id": int(last_seen_id),
                "fallback_offset": int(fallback_offset),
                "cursor_reads_enabled": self._cursor_reads_enabled,
            },
        )
        if self._cursor_reads_enabled:
            try:
                with trace.span("db_read.active_data_after_id"):
                    rows = get_active_data_after_id(
                        self._conn,
                        int(limit),
                        int(last_seen_id),
                        job_id=self._job_id,
                    ) or []
                next_last_seen_id = int(last_seen_id)
                if rows:
                    next_last_seen_id = int(rows[-1].get("id") or next_last_seen_id)
                trace.emit_summary(
                    status="success",
                    row_count=len(rows),
                    next_last_seen_id=next_last_seen_id,
                    next_fallback_offset=int(fallback_offset),
                )
                return rows, next_last_seen_id, int(fallback_offset)
            except Exception as exc:
                trace.emit_summary(status="failed", error_type=type(exc).__name__, error=str(exc))
                raise

        try:
            with trace.span("db_read.active_data_page"):
                rows = get_active_data(
                    self._conn,
                    int(limit),
                    int(fallback_offset),
                    job_id=self._job_id,
                ) or []
            next_fallback_offset = int(fallback_offset)
            if rows:
                next_fallback_offset += len(rows)
            trace.emit_summary(
                status="success",
                row_count=len(rows),
                next_last_seen_id=int(last_seen_id),
                next_fallback_offset=next_fallback_offset,
            )
            return rows, int(last_seen_id), next_fallback_offset
        except Exception as exc:
            trace.emit_summary(status="failed", error_type=type(exc).__name__, error=str(exc))
            raise

    def _update_stop_state_locked(self) -> None:
        if self._stop_db_reads:
            return
        self._stop_db_reads = self._db_processing_done_cached and not self._row_buffer
