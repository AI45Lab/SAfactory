import sqlite3
from typing import List, Tuple


def load_env_lists(db_path: str, table_name: str = "trad") -> Tuple[List[str], List[str]]:
    """
    Read env_name and env_id from a sqlite database table and return two lists:
    (env_name_list, env_id_list)

    Schema expected:
      id INTEGER PK AUTOINCREMENT,
      env_name TEXT NOT NULL,
      env_id TEXT NOT NULL,
      env_param TEXT NULL,
      image TEXT NULL
    """
    # basic hardening: avoid SQL injection via table_name
    if not table_name.replace("_", "").isalnum():
        raise ValueError(f"Invalid table_name: {table_name!r}")

    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute(f'SELECT env_name, env_id FROM "{table_name}" ORDER BY id ASC')
        rows = cur.fetchall()
    finally:
        conn.close()

    env_names = [r[0] for r in rows]
    env_ids = [r[1] for r in rows]
    return env_names, env_ids
