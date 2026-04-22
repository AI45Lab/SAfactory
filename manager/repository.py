from __future__ import annotations

import asyncio
import logging
import sqlite3
import time
from collections import deque
from typing import Any, Callable, Deque, Dict, List, Optional

from .db_loader import (
    get_active_data,
    get_active_data_after_id,
    get_all_image,
    get_env_image_map,
)

log = logging.getLogger("manager.repository")

DB_FETCH_WARN_SECONDS = 1.0


class EnvDataRepository:
    """
    Thin repository around db_loader helpers.

    The repository owns row-reservation state so callers can reserve buffered rows
    without holding the actor-pool state lock across database reads.
    """

    def __init__(
        self,
        conn: Optional[sqlite3.Connection],
        *,
        job_id: str = "",
        db_processing_done_checker: Optional[Callable[[], bool]] = None,
    ) -> None:
        self._conn = conn
        self._job_id = str(job_id or "").strip() or None
        self._db_processing_done_checker = db_processing_done_checker

        self._cursor_reads_enabled = isinstance(conn, sqlite3.Connection)
        self._last_seen_id: int = 0
        self._fallback_offset: int = 0
        self._row_buffer: Deque[Dict[str, Any]] = deque()
        self._fetch_lock = asyncio.Lock()
        self._db_processing_done_cached: bool = False
        self._stop_db_reads: bool = False

    def reset_cursor(self) -> None:
        self._last_seen_id = 0
        self._fallback_offset = 0
        self._row_buffer.clear()
        self._db_processing_done_cached = False
        self._stop_db_reads = False

    def get_env_image_map(self) -> Dict[str, str]:
        m = get_env_image_map(self._conn, job_id=self._job_id) or {}
        out: Dict[str, str] = {}
        for k, v in m.items():
            out[str(k)] = "" if v is None else str(v)
        return out

    def get_image_to_env_map(self) -> Dict[str, str]:
        m = get_all_image(self._conn, job_id=self._job_id) or {}
        out: Dict[str, str] = {}
        for k, v in m.items():
            out[str(k)] = str(v)
        return out

    async def prime(self, startup_batch_size: int) -> List[Dict[str, Any]]:
        """Reserve the initial warm-pool rows in one batched pass."""
        return await self.reserve_rows(
            startup_batch_size,
            fetch_batch_size=max(1, int(startup_batch_size or 1)),
        )

    async def reserve_rows(self, limit: int, *, fetch_batch_size: int) -> List[Dict[str, Any]]:
        requested = max(0, int(limit))
        if requested <= 0:
            return []

        batch_size = max(1, int(fetch_batch_size))

        async with self._fetch_lock:
            reserved = self._drain_buffer_locked(requested)

            while len(reserved) < requested:
                if self._stop_db_reads:
                    break

                if not self._row_buffer:
                    fetch_limit = max(batch_size, requested - len(reserved))
                    started_at = time.perf_counter()
                    await asyncio.to_thread(self._fill_buffer_locked, fetch_limit)
                    elapsed = time.perf_counter() - started_at
                    if elapsed >= DB_FETCH_WARN_SECONDS:
                        log.warning(
                            "env DB fetch took %.2fs limit=%d last_seen_id=%d buffered=%d job_id=%s",
                            elapsed,
                            int(fetch_limit),
                            int(self._last_seen_id),
                            len(self._row_buffer),
                            self._job_id or "<all>",
                        )
                    if not self._row_buffer:
                        break

                reserved.extend(self._drain_buffer_locked(requested - len(reserved)))

            self._update_stop_state_locked()
            return reserved

    async def reserve_one(self, *, fetch_batch_size: int) -> Optional[Dict[str, Any]]:
        rows = await self.reserve_rows(1, fetch_batch_size=fetch_batch_size)
        if not rows:
            return None
        return rows[0]

    def _drain_buffer_locked(self, limit: int) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        while self._row_buffer and len(rows) < limit:
            rows.append(self._row_buffer.popleft())
        return rows

    def _fill_buffer_locked(self, limit: int) -> None:
        rows = self._fetch_rows_locked(limit)
        if rows:
            self._row_buffer.extend(rows)
            return

        if self._db_processing_done_checker is None or self._db_processing_done_cached:
            self._update_stop_state_locked()
            return

        try:
            self._db_processing_done_cached = bool(self._db_processing_done_checker())
        except Exception:
            log.warning("db_processing_done_checker failed; assuming producer is still active", exc_info=True)

        self._update_stop_state_locked()

    def _fetch_rows_locked(self, limit: int) -> List[Dict[str, Any]]:
        if self._cursor_reads_enabled:
            rows = get_active_data_after_id(
                self._conn,
                int(limit),
                int(self._last_seen_id),
                job_id=self._job_id,
            ) or []
            if rows:
                self._last_seen_id = int(rows[-1].get("id") or self._last_seen_id)
            return rows

        rows = get_active_data(
            self._conn,
            int(limit),
            int(self._fallback_offset),
            job_id=self._job_id,
        ) or []
        if rows:
            self._fallback_offset += len(rows)
        return rows

    def _update_stop_state_locked(self) -> None:
        self._stop_db_reads = self._db_processing_done_cached and not self._row_buffer
