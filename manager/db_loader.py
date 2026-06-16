from __future__ import annotations

import inspect
import sqlite3
from typing import List, Dict, Any, Optional

REMOTE_FETCH_PAGE_SIZE = 1000


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


def _normalize_rows(rows: Any) -> List[Dict[str, Any]]:
    if not rows:
        return []

    normalized: List[Dict[str, Any]] = []
    for row in rows:
        if isinstance(row, dict):
            normalized.append(dict(row))
        else:
            try:
                normalized.append(dict(row))
            except Exception:
                continue
    return normalized


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
    conn: Optional[sqlite3.Connection],
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
    conn: Optional[sqlite3.Connection],
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


def get_env_image_map(conn: Optional[sqlite3.Connection], job_id: Optional[str] = None) -> Dict[str, Any]:
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


def get_all_image(conn: Optional[sqlite3.Connection], job_id: Optional[str] = None) -> Dict[str, str]:
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
