from __future__ import annotations

import sqlite3
from typing import List, Dict, Any, Optional


def get_active_data(
    conn: Optional[sqlite3.Connection],
    limit: int,
    offset: int,
) -> List[Dict[str, Any]]:
    """Return a paginated slice of active (non-deleted) environment rows."""
    if isinstance(conn, sqlite3.Connection):
        query = """
        SELECT
            id, env_id, env_name, env_params, image, group_id
        FROM job_environments
        WHERE is_deleted = 0
        ORDER BY id ASC
        LIMIT ? OFFSET ?;
        """
        cursor = conn.execute(query, (limit, offset))
        cols = [d[0] for d in cursor.description]
        return [dict(zip(cols, row)) for row in cursor.fetchall()]
    else:
        return conn.get_env_configs(limit=limit, offset=offset)


def get_env_image_map(conn: Optional[sqlite3.Connection]) -> Dict[str, Any]:
    """Return a mapping of env_name -> image for all active environments."""
    if isinstance(conn, sqlite3.Connection):
        query = """
        SELECT env_name, image
        FROM job_environments
        WHERE is_deleted = 0
        ORDER BY id ASC;
        """
        cursor = conn.execute(query)
        result: Dict[str, Any] = {}
        for env_name, image in cursor.fetchall():
            if env_name is None:
                continue
            if image:
                result[env_name] = image
            elif env_name not in result:
                result[env_name] = None
        return result
    else:
        return conn.get_env_image_map()


def get_all_image(conn: Optional[sqlite3.Connection]) -> Dict[str, str]:
    """Return a mapping of image -> env_name for all active environments."""
    if isinstance(conn, sqlite3.Connection):
        query = """
        SELECT image, env_name
        FROM job_environments
        WHERE is_deleted = 0
          AND image IS NOT NULL AND TRIM(image) != ''
          AND env_name IS NOT NULL;
        """
        cursor = conn.execute(query)
        image_to_env: Dict[str, str] = {}
        for image, env_name in cursor.fetchall():
            img = (image or "").strip()
            env = (env_name or "").strip()
            if img and env and img not in image_to_env:
                image_to_env[img] = env
        return image_to_env
    else:
        return conn.get_all_image()
