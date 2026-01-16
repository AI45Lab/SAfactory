from pathlib import Path
import sqlite3
import json
import uuid
from .load_yaml import load_yaml_configs
from core.data_manager.models import EnvironmentConfig


TABLE_NAME = "v2"
TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
  id        INTEGER PRIMARY KEY AUTOINCREMENT,
  env_name  TEXT    NOT NULL,
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
                INSERT INTO {TABLE_NAME} (env_name, env_id, env_param, image)
                VALUES (?, ?, ?, ?)
                """,
                (env, env_id, env_param_str, image),
            )

    conn.commit()

def _resolve_env_config_path(
    *,
    env_config: str | Path,
    env_root: str | Path = "env",
) -> Path:
    """Resolve env_config to an existing yaml/yml file path.

    - If env_config is an absolute path, use it directly.
    - If it's a relative path:
        1) try as-is (relative to current working dir)
        2) if not exists, try join with env_root
    """
    root = Path(env_root)
    p = Path(env_config)

    if not p.is_absolute() and not p.exists():
        p2 = root / p
        if p2.exists():
            p = p2

    if not p.is_file():
        raise ValueError(f"env_config must be an existing yaml file, got: {p}")

    if p.suffix.lower() not in (".yaml", ".yml"):
        raise ValueError(f"env_config must be a .yaml/.yml file, got: {p}")

    return p



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

def all_env_yaml_load(
    env_root: str | Path = "env",
    *,
    env_config: str | Path | None = None,
):
    """Load env yaml configs.

    - If env_config is provided: only load that yaml file.
    - Else: load all yaml files under env_root.
    """
    yaml_config_list = []
    env_root = Path(env_root)

    if env_config:
        yaml_path = _resolve_env_config_path(env_config=env_config, env_root=env_root)
        print(f"Populating from (env_config): {yaml_path}")
        yaml_config_list.extend(load_yaml_configs(str(yaml_path)) or [])
        return yaml_config_list

    for yaml_path in iter_child_yaml_files(env_root):
        print(f"Populating from: {yaml_path}")
        try:
            yaml_configs = load_yaml_configs(str(yaml_path))
        except Exception as e:
            print(f"[SKIP] failed to parse the yaml file: {yaml_path} -> {e}")
            continue
        yaml_config_list.extend(yaml_configs)

    return yaml_config_list

async def sync_configs_to_db(data_manager, yaml_configs):
    """同步YAML配置到数据库，自动生成env_id"""
    await data_manager.init()

    # 数据库现有配置：通过(env_name + env_params)判断唯一性（避免重复生成env_id）
    db_configs = await EnvironmentConfig.all()
    # 构建唯一标识：env_name + 用户参数的哈希值 + 序号 （区分同配置的不同实例）
    def get_config_key(env_name, env_params, index):
        import json
        params_hash = hash(json.dumps(env_params, sort_keys=True))
        return f"{env_name}_{params_hash}_{index}"
    
    # 现有配置键值映射（包含序号）
    db_config_keys = {}
    for cfg in db_configs:
        # 从现有记录反推序号（同一配置的实例按创建顺序排序）
        same_configs = [c for c in db_configs 
                       if c.env_name == cfg.env_name 
                       and c.env_params == cfg.env_params]
        index = same_configs.index(cfg) + 1  # 序号从1开始
        key = get_config_key(cfg.env_name, cfg.env_params, index)
        db_config_keys[key] = cfg

    added, updated, deleted = 0, 0, 0

    # 处理新增和更新（支持多实例）
    for cfg in yaml_configs:
        env_name = cfg["env_name"].strip()
        env_params = cfg.get("env_params", {})
        image = cfg.get("env_image", "")
        env_num = cfg.get("env_num", 1)  # 默认创建1个实例
        if not isinstance(env_num, int) or env_num < 1:
            raise ValueError(f"env_num必须是正整数，{env_name}配置错误")

        # 为每个序号创建实例
        group_id = str(uuid.uuid4())
        for index in range(1, env_num + 1):
            config_key = get_config_key(env_name, env_params, index)
            
            if config_key not in db_config_keys:
                # 新增实例：自动生成env_id
                await EnvironmentConfig.create(
                    env_name=env_name,
                    env_params=env_params,
                    image=image,
                    group_id=group_id
                )
                added += 1
                print(f"新增环境配置实例：{env_name}（序号：{index}/{env_num}，自动生成env_id）")
            else:
                # 实例已存在，无需操作
                print(f"环境配置实例已存在：{env_name}（序号：{index}/{env_num}，使用现有env_id）")

    # 处理删除：数据库中存在但YAML中不需要的实例
    yaml_config_keys = set()
    for cfg in yaml_configs:
        env_name = cfg["env_name"].strip()
        env_params = cfg.get("env_params", {})
        env_num = cfg.get("env_num", 1)
        for index in range(1, env_num + 1):
            key = get_config_key(env_name, env_params, index)
            yaml_config_keys.add(key)

    # 删除不在YAML配置中的实例
    for config_key, db_cfg in db_config_keys.items():
        if config_key not in yaml_config_keys:
            await db_cfg.delete()
            deleted += 1
            print(f"删除环境配置实例：{db_cfg.env_name}（env_id: {db_cfg.env_id}）")

    print(f"\n配置同步结果：新增{added} | 保留{len(yaml_config_keys)-added} | 删除{deleted}")
    
    conn = data_manager.get_sync_connection()
    
    return conn