import sqlite3
from typing import List, Dict, Any


def get_connection(cfg: dict) -> sqlite3.Connection:
    db_cfg = (cfg or {}).get("database", {})
    driver = db_cfg.get("driver", "sqlite")
    if driver != "sqlite":
        raise NotImplementedError(f"Only sqlite is supported right now (got {driver})")

    db_path = db_cfg.get("sqlite_path") or "trading_envs.db"
    conn = sqlite3.connect(db_path)
    print(f"Connected to database: {db_path}")
    return conn


def get_active_data(conn: sqlite3.Connection, limit: int, offset: int) -> List[Dict[str, Any]]:
    query = """
    SELECT
        id, env_name, env_id, data_dir,
        price_filename, tweet_filename,
        visual_save_path, window_size, is_active
    FROM env_configs
    WHERE is_active = 1
    ORDER BY id ASC
    LIMIT ? OFFSET ?;
    """
    cursor = conn.execute(query, (limit, offset))
    cols = [d[0] for d in cursor.description]
    return [dict(zip(cols, row)) for row in cursor.fetchall()]


def compose_create_kwargs(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "data_dir": row["data_dir"],
        "price_filename": row["price_filename"],
        "tweet_filename": row["tweet_filename"],
        "visual_save_path": row["visual_save_path"],
        "window_size": int(row["window_size"]),
        "env_id": row["env_id"],
    }
