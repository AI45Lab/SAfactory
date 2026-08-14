"""Compatibility query helpers backed exclusively by ``DataManager``.

Runtime code uses :mod:`manager.repository`; these helpers remain for callers
that need the historical names without receiving a raw DB connection.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from core.data_manager.manager import DataManager


class EnvConfigCacheReader:
    def __init__(self, data_manager: DataManager) -> None:
        self._data_manager = data_manager

    async def get_env_configs(
        self,
        limit: Optional[int] = None,
        offset: int = 0,
        job_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        return await self._data_manager.list_environment_rows(
            job_id=job_id,
            offset=offset,
            limit=limit,
            finished=False,
            is_deleted=False,
        )


def scheduler_db_reader(
    storage_type: str,
    data_manager: DataManager,
    conn: Any = None,
) -> DataManager:
    """Return the public data boundary; ``conn`` is ignored for compatibility."""
    if storage_type not in {"sqlite", "cloud"}:
        raise ValueError(f"Unknown storage type: {storage_type}")
    return data_manager


async def get_active_data(
    data_manager: DataManager,
    limit: int,
    offset: int,
    job_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    return await data_manager.list_environment_rows(
        job_id=job_id,
        offset=offset,
        limit=limit,
        finished=False,
        is_deleted=False,
    )


async def get_active_data_after_id(
    data_manager: DataManager,
    limit: int,
    after_id: int,
    job_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    return await data_manager.list_environment_rows(
        job_id=job_id,
        after_id=after_id,
        limit=limit,
        finished=False,
        is_deleted=False,
    )


async def get_env_image_map(
    data_manager: DataManager,
    job_id: Optional[str] = None,
) -> Dict[str, Any]:
    rows = await data_manager.list_environment_rows(
        job_id=job_id,
        finished=False,
        is_deleted=False,
    )
    return {
        str(row.get("env_name") or ""): row.get("image")
        for row in rows
        if row.get("env_name")
    }


async def get_all_image(
    data_manager: DataManager,
    job_id: Optional[str] = None,
) -> Dict[str, str]:
    rows = await data_manager.list_environment_rows(
        job_id=job_id,
        finished=False,
        is_deleted=False,
    )
    return {
        str(row.get("image") or ""): str(row.get("env_name") or "")
        for row in rows
        if row.get("image") and row.get("env_name")
    }
