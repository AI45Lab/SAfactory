from __future__ import annotations

import sqlite3
from typing import List, Dict, Any, Optional


def get_connection(cfg: dict) -> sqlite3.Connection:
    db_cfg = (cfg or {}).get("database", {})
    driver = db_cfg.get("driver", "sqlite")
    if driver != "sqlite":
        raise NotImplementedError(f"Only sqlite is supported right now (got {driver})")

    db_path = db_cfg.get("sqlite_path")
    if db_path:
        conn = sqlite3.connect(db_path)
        print(f"Connected to database: {db_path}")
        return conn
    else:
        raise Exception("No database path specified")


def get_active_data(conn: Optional[sqlite3.Connection], limit: int, offset: int, tableName:str="environment_configs") -> List[Dict[str, Any]]:
    if isinstance(conn, sqlite3.Connection):
        query = f"""
        SELECT
            id, env_name, env_id, env_params, image, group_id
        FROM {tableName}
        ORDER BY id ASC
        LIMIT ? OFFSET ?;
        """
        cursor = conn.execute(query, (limit, offset))
        cols = [d[0] for d in cursor.description]
        return [dict(zip(cols, row)) for row in cursor.fetchall()]
    else:
        configs = conn.get_env_configs(limit=limit, offset=offset)
        return configs

def get_env_image_map(conn: Optional[sqlite3.Connection], tableName:str="environment_configs") -> Dict[str, Any]:
    """
    scan the db to load all the envs required image.
    """
    if isinstance(conn, sqlite3.Connection):
        query = f"""
        SELECT env_name, image
        FROM {tableName}
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
        image_map = conn.get_env_image_map()
        return image_map

def get_all_image(conn: Optional[sqlite3.Connection], tableName:str="environment_configs") -> Dict[str,str]:
    if isinstance(conn, sqlite3.Connection):
        image_to_env : Dict[str, str]={}
        query=f"""
        SELECT image, env_name 
        FROM {tableName}
        WHERE image IS NOT NULL AND TRIM(image) !='' AND env_name IS NOT NULL
        """
        cursor = conn.execute(query)
        for image, env_name in cursor.fetchall():
            img =(image or "").strip()
            env= (env_name or "").strip()
            if not img or not env:
                continue
            if img not in image_to_env:
                image_to_env[img]=env
        return image_to_env
    else:
        image_map = conn.get_all_image()
        return image_map