from __future__ import annotations

import inspect
import sqlite3
from collections.abc import Mapping
from typing import List, Dict, Any, Optional

REMOTE_FETCH_PAGE_SIZE = 1000


class EnvConfigCacheReader:
    """
    Synchronous scheduler reader backed by an in-memory env-config cache.

    Cloud storage keeps scheduler rows in the strategy cache while the SDK owns
    the remote persistence.  The manager scheduler is intentionally synchronous
    at this boundary, so this adapter exposes the same shape as remote SDK
    readers without requiring an event loop hop inside repository fetches.
    """

    def __init__(self, env_configs: Mapping[str, Any]) -> None:
        self._env_configs = env_configs

    def get_env_configs(
        self,
        *,
        limit: Optional[int] = None,
        offset: int = 0,
        job_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        rows = _rows_from_mapping_cache(self._env_configs, job_id=job_id)
        start = max(0, int(offset or 0))
        if limit is None:
            return rows[start:]
        end = start + max(0, int(limit))
        return rows[start:end]

    def get_all_environments(self, job_id: Optional[str] = None) -> List[Dict[str, Any]]:
        return _rows_from_mapping_cache(self._env_configs, job_id=job_id)


def scheduler_db_reader(storage_type: str, data_manager: Any, conn: Any) -> Any:
    """
    Return the object AgentDataRepository should read from.

    SQLite uses the raw sqlite3 connection.  Cloud mode does not have a raw sync
    DB connection, so prefer a strategy-provided sync reader and fall back to the
    strategy cache used by CloudStrategy.
    """
    if str(storage_type or "").strip().lower() == "sqlite":
        return conn

    strategy = getattr(data_manager, "strategy", None)
    for candidate in (strategy, data_manager, conn):
        if candidate is None:
            continue
        get_env_configs = getattr(candidate, "get_env_configs", None)
        if callable(get_env_configs) and not inspect.iscoroutinefunction(get_env_configs):
            return candidate
        env_configs = getattr(candidate, "_env_configs", None)
        if isinstance(env_configs, Mapping):
            return EnvConfigCacheReader(env_configs)
        env_cache = getattr(candidate, "_env_cache", None)
        if isinstance(env_cache, Mapping):
            return EnvConfigCacheReader(env_cache)

    return conn


def _supports_job_id_kw(fn: Any) -> bool:
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return False

    for param in sig.parameters.values():
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            return True
        if param.name == "job_id":
            return True
    return False


def _supports_kw(fn: Any, kw_name: str) -> bool:
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return False

    for param in sig.parameters.values():
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            return True
        if param.name == kw_name:
            return True
    return False


def _requires_kw(fn: Any, kw_name: str) -> bool:
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return False

    for param in sig.parameters.values():
        if param.name != kw_name:
            continue
        return (
            param.kind in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            )
            and param.default is inspect._empty
        )
    return False


def _invoke_sync_reader(fn: Any, *args: Any, job_id: Optional[str] = None, **kwargs: Any) -> Any:
    if job_id and _supports_job_id_kw(fn):
        kwargs["job_id"] = job_id
    result = fn(*args, **kwargs)
    if inspect.isawaitable(result):
        raise TypeError("db_loader requires synchronous reader methods")
    return result


def _normalize_remote_row(row: Dict[str, Any], index: int) -> Dict[str, Any]:
    normalized = dict(row)
    if "image" not in normalized and "env_image" in normalized:
        normalized["image"] = normalized.get("env_image")
    if normalized.get("id") is None:
        normalized["id"] = index
    return normalized


def _normalize_rows(rows: Any) -> List[Dict[str, Any]]:
    if not rows:
        return []

    normalized: List[Dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        if isinstance(row, dict):
            normalized.append(_normalize_remote_row(row, index))
        else:
            try:
                normalized.append(_normalize_remote_row(dict(row), index))
            except Exception:
                continue
    return normalized


def _rows_from_mapping_cache(env_configs: Mapping[str, Any], job_id: Optional[str] = None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for index, value in enumerate(env_configs.values(), start=1):
        if not isinstance(value, dict):
            try:
                value = dict(value)
            except Exception:
                continue
        row = _normalize_remote_row(value, index)
        if job_id and str(row.get("job_id") or "") != job_id:
            continue
        rows.append(row)
    return rows


def _filter_rows_by_job_id(rows: List[Dict[str, Any]], job_id: Optional[str]) -> List[Dict[str, Any]]:
    if not job_id:
        return rows

    has_job_id = any("job_id" in row for row in rows)
    if not has_job_id:
        return rows

    return [row for row in rows if str(row.get("job_id") or "") == job_id]


def _load_remote_rows(
    conn: Any,
    *,
    limit: Optional[int] = None,
    offset: int = 0,
    job_id: Optional[str] = None,
) -> Optional[List[Dict[str, Any]]]:
    get_env_configs = getattr(conn, "get_env_configs", None)
    if callable(get_env_configs):
        supports_limit = _supports_kw(get_env_configs, "limit")
        supports_offset = _supports_kw(get_env_configs, "offset")
        requires_limit = _requires_kw(get_env_configs, "limit")

        def _fetch_page(page_offset: int, page_limit: Optional[int]) -> List[Dict[str, Any]]:
            kwargs: Dict[str, Any] = {}
            if supports_offset:
                kwargs["offset"] = page_offset
            if supports_limit and page_limit is not None:
                kwargs["limit"] = page_limit
            rows = _invoke_sync_reader(get_env_configs, job_id=job_id, **kwargs)
            return _filter_rows_by_job_id(_normalize_rows(rows), job_id)

        if limit is not None:
            if offset and supports_limit and not supports_offset:
                rows = _fetch_page(0, offset + limit)
                return rows[offset: offset + limit]
            return _fetch_page(offset, limit)

        if supports_limit or requires_limit:
            page_size = REMOTE_FETCH_PAGE_SIZE
            all_rows: List[Dict[str, Any]] = []
            page_offset = offset
            while True:
                page_rows = _fetch_page(page_offset, page_size)
                if not page_rows:
                    break
                all_rows.extend(page_rows)
                if len(page_rows) < page_size:
                    break
                if not supports_offset:
                    break
                page_offset += len(page_rows)
            return all_rows

        return _fetch_page(offset, None)

    get_all_environments = getattr(conn, "get_all_environments", None)
    if callable(get_all_environments):
        rows = _invoke_sync_reader(get_all_environments, job_id=job_id)
        normalized = _filter_rows_by_job_id(_normalize_rows(rows), job_id)
        if limit is None:
            return normalized[offset:]
        return normalized[offset: offset + limit]

    return None


def _build_env_image_map(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for row in rows:
        env_name = row.get("env_name")
        image = row.get("image")
        if env_name is None:
            continue
        env_name = str(env_name)
        if image:
            result[env_name] = image
        elif env_name not in result:
            result[env_name] = None
    return result


def _build_image_to_env_map(rows: List[Dict[str, Any]]) -> Dict[str, str]:
    image_to_env: Dict[str, str] = {}
    for row in rows:
        img = str(row.get("image") or "").strip()
        env = str(row.get("env_name") or "").strip()
        if img and env and img not in image_to_env:
            image_to_env[img] = env
    return image_to_env


def _coerce_row_id(row: Dict[str, Any]) -> Optional[int]:
    value = row.get("id")
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def get_active_data(
    conn: Any,
    limit: int,
    offset: int,
    job_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Return a paginated slice of active agent rows from the legacy table."""
    if isinstance(conn, sqlite3.Connection):
        filters = ["is_deleted = 0"]
        params: List[Any] = []
        if job_id:
            filters.append("job_id = ?")
            params.append(job_id)
        query = """
        SELECT
            id, job_id, env_id, env_name, env_params, image, group_id
        FROM job_environments
        WHERE {where_clause}
        ORDER BY id ASC
        LIMIT ? OFFSET ?;
        """
        cursor = conn.execute(query.format(where_clause=" AND ".join(filters)), tuple(params + [limit, offset]))
        cols = [d[0] for d in cursor.description]
        return [dict(zip(cols, row)) for row in cursor.fetchall()]

    rows = _load_remote_rows(conn, limit=limit, offset=offset, job_id=job_id)
    if rows is not None:
        return rows
    return []


def get_active_data_after_id(
    conn: Any,
    limit: int,
    after_id: int,
    job_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Return active agent rows whose primary key is greater than ``after_id``."""
    if isinstance(conn, sqlite3.Connection):
        filters = ["is_deleted = 0", "id > ?"]
        params: List[Any] = [after_id]
        if job_id:
            filters.append("job_id = ?")
            params.append(job_id)
        query = """
        SELECT
            id, job_id, env_id, env_name, env_params, image, group_id
        FROM job_environments
        WHERE {where_clause}
        ORDER BY id ASC
        LIMIT ?;
        """
        cursor = conn.execute(query.format(where_clause=" AND ".join(filters)), tuple(params + [limit]))
        cols = [d[0] for d in cursor.description]
        return [dict(zip(cols, row)) for row in cursor.fetchall()]

    rows = _load_remote_rows(conn, job_id=job_id)
    if rows is None:
        return []

    filtered_rows: List[Dict[str, Any]] = []
    for row in rows:
        row_id = _coerce_row_id(row)
        if row_id is None or row_id <= after_id:
            continue
        normalized_row = dict(row)
        normalized_row["id"] = row_id
        filtered_rows.append(normalized_row)

    filtered_rows.sort(key=lambda row: int(row["id"]))
    return filtered_rows[:limit]


def get_env_image_map(conn: Any, job_id: Optional[str] = None) -> Dict[str, Any]:
    """Return a mapping of legacy env_name -> image for all active agents."""
    if isinstance(conn, sqlite3.Connection):
        filters = ["is_deleted = 0"]
        params: List[Any] = []
        if job_id:
            filters.append("job_id = ?")
            params.append(job_id)
        query = """
        SELECT env_name, image
        FROM job_environments
        WHERE {where_clause}
        ORDER BY id ASC;
        """
        cursor = conn.execute(query.format(where_clause=" AND ".join(filters)), tuple(params))
        result: Dict[str, Any] = {}
        for env_name, image in cursor.fetchall():
            if env_name is None:
                continue
            if image:
                result[env_name] = image
            elif env_name not in result:
                result[env_name] = None
        return result

    rows = _load_remote_rows(conn, job_id=job_id)
    if rows is not None:
        return _build_env_image_map(rows)

    get_map = getattr(conn, "get_env_image_map", None)
    if callable(get_map):
        return _invoke_sync_reader(get_map, job_id=job_id) or {}
    return {}


def get_all_image(conn: Any, job_id: Optional[str] = None) -> Dict[str, str]:
    """Return a mapping of image -> legacy env_name for all active agents."""
    if isinstance(conn, sqlite3.Connection):
        filters = [
            "is_deleted = 0",
            "image IS NOT NULL AND TRIM(image) != ''",
            "env_name IS NOT NULL",
        ]
        params: List[Any] = []
        if job_id:
            filters.append("job_id = ?")
            params.append(job_id)
        query = """
        SELECT image, env_name
        FROM job_environments
        WHERE {where_clause};
        """
        cursor = conn.execute(query.format(where_clause=" AND ".join(filters)), tuple(params))
        image_to_env: Dict[str, str] = {}
        for image, env_name in cursor.fetchall():
            img = (image or "").strip()
            env = (env_name or "").strip()
            if img and env and img not in image_to_env:
                image_to_env[img] = env
        return image_to_env

    rows = _load_remote_rows(conn, job_id=job_id)
    if rows is not None:
        return _build_image_to_env_map(rows)

    get_map = getattr(conn, "get_all_image", None)
    if callable(get_map):
        return _invoke_sync_reader(get_map, job_id=job_id) or {}
    return {}
