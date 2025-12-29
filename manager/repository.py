from __future__ import annotations

import sqlite3
from typing import Any, Dict, List, Optional

from .db_loader import get_active_data, get_env_image_map, get_all_image


class EnvDataRepository:
    """
    Thin DB repository around db_loader.py helpers.

    Notes:
      - Maintains an offset cursor for sequential row reservation.
      - sqlite3 connections are generally not concurrency-friendly; the manager/pool
        should guard calls with an asyncio.Lock (done in ActorPool).
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        self._offset: int = 0

    @property
    def offset(self) -> int:
        return self._offset

    def reset_cursor(self) -> None:
        self._offset = 0

    def get_env_image_map(self) -> Dict[str, str]:
        m = get_env_image_map(self._conn) or {}
        out: Dict[str, str] = {}
        for k, v in m.items():
            out[str(k)] = "" if v is None else str(v)
        return out

    def get_image_to_env_map(self) -> Dict[str, str]:
        m = get_all_image(self._conn) or {}
        out: Dict[str, str] = {}
        for k, v in m.items():
            out[str(k)] = str(v)
        return out

    def fetch_active_rows(self, limit: int) -> List[Dict[str, Any]]:
        rows = get_active_data(self._conn, int(limit), int(self._offset)) or []
        self._offset += len(rows)
        return rows

    def fetch_one_active_row(self) -> Optional[Dict[str, Any]]:
        rows = get_active_data(self._conn, 1, int(self._offset)) or []
        if not rows:
            return None
        self._offset += 1
        return rows[0]
