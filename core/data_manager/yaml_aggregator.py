from pathlib import Path
import sqlite3
import json
import uuid
from load_yaml import load_yaml_configs


TABLE_NAME = "v2"
TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
  id        INTEGER PRIMARY KEY AUTOINCREMENT,
  env       TEXT    NOT NULL,
  env_id    TEXT    NOT NULL,
  env_param TEXT    NULL,
  image     TEXT    NULL
);
"""


def iter_child_yaml_files(env_root: Path):
    """iter all the child yaml files under the given env root"""

    if not env_root.is_dir():
        raise ValueError(f"env root {env_root} is not a directory")

    for subdir in sorted(env_root.iterdir()):
        if not subdir.is_dir():
            continue
        if subdir.name.startswith("__"):
            # skip __pycache__ etc.
            continue

        for p in sorted(subdir.iterdir()):
            if p.is_file() and p.suffix.lower() in (".yaml", ".yml"):
                yield p


def _make_base_id() -> str:
    return uuid.uuid4().hex


def insert_from_yaml(conn: sqlite3.Connection, yaml_path: Path) -> None:
    try:
        configs = load_yaml_configs(str(yaml_path))
    except Exception as e:
        print(f"[SKIP] failed to parse the yaml file: {yaml_path} -> {e}")
        return

    if not configs:
        return

    cur = conn.cursor()

    for cfg in configs:
        env = cfg.get("env_name") or cfg.get("env")
        if not env:
            continue

        env_num = int(cfg.get("env_num", 1) or 1)
        env_params = cfg.get("env_params") or {}
        image = cfg.get("env_image")

        env_param_str = json.dumps(env_params, ensure_ascii=False)

        for _ in range(env_num):
            env_id = _make_base_id()

            cur.execute(
                f"""
                INSERT INTO {TABLE_NAME} (env, env_id, env_param, image)
                VALUES (?, ?, ?, ?)
                """,
                (env, env_id, env_param_str, image),
            )

    conn.commit()


def populate_env_table(db_path: str | Path, env_root: str | Path = "env") -> None:
    """iter all the yaml files under the given env root and insert it into the given db path"""

    db_path = Path(db_path)
    env_root = Path(env_root)

    conn = sqlite3.connect(db_path)
    try:
        conn.execute(TABLE_SQL)

        for yaml_path in iter_child_yaml_files(env_root):
            print(f"Populating from: {yaml_path}")
            insert_from_yaml(conn, yaml_path)

    finally:
        conn.close()

