from pathlib import Path
import sqlite3
import json

import yaml  # pip install pyyaml

TABLE_SQL = """
CREATE TABLE IF NOT EXISTS v2 (
  id         INTEGER PRIMARY KEY,
  env_name   TEXT    NOT NULL,
  env_id     TEXT    NOT NULL,
  env_param  TEXT    NULL,
  image      TEXT    NULL,
  entrypoint TEXT
);
"""


def _find_first_yaml_files(env_root: Path) -> dict[str, Path]:

    mapping: dict[str, Path] = {}

    if not env_root.is_dir():
        raise ValueError(f"env root {env_root} is not a directory")

    for subdir in sorted(env_root.iterdir()):
        if not subdir.is_dir():
            continue
        if subdir.name.startswith("__"):
            # skip __pycache__ etc.
            continue

        yaml_files = sorted(
            p
            for p in subdir.iterdir()
            if p.is_file() and p.suffix.lower() in (".yaml", ".yml")
        )
        if yaml_files:
            mapping[subdir.name] = yaml_files[0]

    return mapping


def _insert_from_yaml(conn: sqlite3.Connection, yaml_path: Path) -> None:
    """
    Parse one YAML file and insert rows into table v2.

    YAML format (per item in 'environments'):
      - env_name: trading_gym
        env_image: ...
        env_num: 2
        env_params: { ... }
    """
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    env_list = data.get("environments", [])

    cur = conn.cursor()

    for env_cfg in env_list:
        if not isinstance(env_cfg, dict):
            continue

        env_name = env_cfg.get("env_name")
        image = env_cfg.get("env_image")
        env_num = int(env_cfg.get("env_num", 1))
        params = env_cfg.get("env_params") or {}

        if not env_name:
            # skip invalid entries
            continue

        # Serialize env_params dict to JSON text for env_param column
        env_param_str = json.dumps(params, ensure_ascii=False)

        # Build base env_id using env_name and optional eval_set
        eval_set = params.get("eval_set")
        base_id = env_name if not eval_set else f"{env_name}_{eval_set}"

        # Insert 'env_num' rows with different env_id values
        for i in range(env_num):
            env_id = f"{base_id}_{i + 1}"

            cur.execute(
                """
                INSERT INTO v2 (env_name, env_id, env_param, image, entrypoint)
                VALUES (?, ?, ?, ?, ?)
                """,
                (env_name, env_id, env_param_str, image, None),  # entrypoint currently NULL
            )

    conn.commit()


def populate_env_table(db_path: str | Path, env_root: str | Path = "env") -> None:

    db_path = Path(db_path)
    env_root = Path(env_root)

    conn = sqlite3.connect(db_path)
    try:
        conn.execute(TABLE_SQL)
        yaml_files = _find_first_yaml_files(env_root)
        for subdir_name, yaml_path in yaml_files.items():
            print(f"Populating {subdir_name}")
            _insert_from_yaml(conn, yaml_path)
    finally:
        conn.close()
