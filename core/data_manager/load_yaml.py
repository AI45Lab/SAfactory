import json
import logging
import os
from functools import lru_cache
from typing import Any, Dict, List

import numpy as np
import yaml

from core.runtime_metadata import SAFACTORY_INTERNAL_ENV_KEY

log = logging.getLogger("core.data_manager.load_yaml")

DATASET_REF_KEY = "__dataset_ref__"


@lru_cache(maxsize=16)
def _cached_parquet_file(path: str):
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError("materializing parquet row references requires pyarrow") from exc
    return pq.ParquetFile(path)


@lru_cache(maxsize=4)
def _cached_parquet_row_group(
    path: str,
    row_group: int,
    columns: tuple[str, ...] | None,
) -> List[Dict[str, Any]]:
    parquet_file = _cached_parquet_file(path)
    return parquet_file.read_row_group(
        row_group,
        columns=list(columns) if columns else None,
    ).to_pylist()


def _convert_numpy_types(obj: Any) -> Any:
    """递归转换 numpy 类型为 Python 原生类型"""
    if isinstance(obj, np.ndarray):
        return [_convert_numpy_types(item) for item in obj.tolist()]
    elif isinstance(obj, np.generic):
        # 处理所有 numpy 标量类型
        return obj.item()
    elif isinstance(obj, dict):
        return {k: _convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_convert_numpy_types(item) for item in obj]
    return obj


def _normalize_dataset_columns(value: Any) -> List[str] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)):
        raise ValueError("dataset_columns must be a list of parquet column names")
    columns = [str(item).strip() for item in value if str(item).strip()]
    if not columns:
        raise ValueError("dataset_columns must contain at least one parquet column name")
    return columns


def _build_parquet_row_refs(path: str, columns: List[str] | None = None) -> List[Dict[str, Any]]:
    """
    Build lightweight references to each parquet row instead of eagerly
    materializing the full dataset into memory.
    """
    try:
        import pyarrow.parquet as pq
    except ImportError:
        raise ImportError("parquet_row_ref 模式需要安装 pyarrow: pip install pyarrow")

    parquet_file = pq.ParquetFile(path)
    if columns:
        missing = sorted(set(columns) - set(parquet_file.schema_arrow.names))
        if missing:
            raise ValueError(f"parquet dataset is missing requested columns: {missing}")
    refs: List[Dict[str, Any]] = []
    row_idx = 0

    for row_group in range(parquet_file.num_row_groups):
        group_rows = parquet_file.metadata.row_group(row_group).num_rows
        for row_in_group in range(group_rows):
            dataset_ref: Dict[str, Any] = {
                "kind": "parquet_row",
                "path": os.path.abspath(path),
                "row_group": row_group,
                "row_in_group": row_in_group,
                "row_idx": row_idx,
            }
            if columns:
                dataset_ref["columns"] = list(columns)
            refs.append({DATASET_REF_KEY: dataset_ref})
            row_idx += 1

    return refs


def materialize_dataset_item(item: Any) -> Any:
    """Resolve one lightweight parquet row reference into a JSON-safe dataset row."""
    if not isinstance(item, dict):
        return item
    raw_ref = item.get(DATASET_REF_KEY)
    if not isinstance(raw_ref, dict):
        return item
    if str(raw_ref.get("kind") or "") != "parquet_row":
        raise ValueError(f"unsupported dataset reference kind: {raw_ref.get('kind')!r}")

    path = str(raw_ref.get("path") or "").strip()
    if not path:
        raise ValueError("parquet dataset reference is missing path")
    columns = _normalize_dataset_columns(raw_ref.get("columns"))
    try:
        row_group = int(raw_ref["row_group"])
        row_in_group = int(raw_ref["row_in_group"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("parquet dataset reference has invalid row coordinates") from exc

    parquet_file = _cached_parquet_file(path)
    if row_group < 0 or row_group >= parquet_file.num_row_groups:
        raise IndexError(f"parquet row_group out of range: {row_group}")
    group_rows = parquet_file.metadata.row_group(row_group).num_rows
    if row_in_group < 0 or row_in_group >= group_rows:
        raise IndexError(f"parquet row_in_group out of range: {row_in_group}")

    rows = _cached_parquet_row_group(path, row_group, tuple(columns) if columns else None)
    if row_in_group >= len(rows):
        raise RuntimeError("parquet dataset reference resolved to no row")
    materialized = _convert_numpy_types(rows[row_in_group])
    extras = {key: value for key, value in item.items() if key != DATASET_REF_KEY}
    extras.update(materialized)
    return extras


def materialize_dataset_env_params(env_params: Dict[str, Any]) -> Dict[str, Any]:
    """Materialize env_params.dataset when it contains a lightweight row reference."""
    materialized = dict(env_params or {})
    if "dataset" in materialized:
        materialized["dataset"] = materialize_dataset_item(materialized["dataset"])
    return materialized


def load_dataset_file(
    base_dir: str,
    path: str,
    load_mode: str = "eager",
    columns: List[str] | None = None,
):
    """
    根据后缀加载数据文件
    支持: .json, .jsonl, .yaml/.yml, .parquet
    返回: list[dict] 或 list[any]
    """
    if not os.path.isabs(path):
        path = os.path.join(base_dir, path)

    if not os.path.exists(path):
        raise FileNotFoundError(f"dataset文件不存在：{path}")

    _, ext = os.path.splitext(path)
    ext = ext.lower()

    data_list = []

    try:
        # 1. JSONL (每行为一个JSON对象)
        if ext == ".jsonl":
            with open(path, "r", encoding="utf-8") as f:
                for line_no, line in enumerate(f, 1):
                    line = line.strip()
                    if not line or line.startswith("//"):
                        continue
                    try:
                        data_list.append(json.loads(line))
                    except json.JSONDecodeError as exc:
                        raise ValueError(
                            f"JSONL第{line_no}行不是合法JSON: {exc.msg} "
                            f"(column {exc.colno})"
                        ) from exc

        # 2. JSON (标准列表)
        elif ext == ".json":
            with open(path, "r", encoding="utf-8") as f:
                content = json.load(f)
                if isinstance(content, list):
                    data_list = content
                else:
                    raise ValueError(f"JSON文件内容必须是列表: {path}")

        # 3. YAML (标准列表)
        elif ext in [".yaml", ".yml"]:
            with open(path, "r", encoding="utf-8") as f:
                content = yaml.safe_load(f)
                if isinstance(content, list):
                    data_list = content
                else:
                    raise ValueError(f"YAML文件内容必须是列表: {path}")

        # 4. Parquet (需安装pandas和pyarrow/fastparquet)
        elif ext == ".parquet":
            if load_mode == "parquet_row_ref":
                return _build_parquet_row_refs(path, columns=columns)

            try:
                import pandas as pd
            except ImportError:
                raise ImportError("加载parquet文件需要安装pandas: pip install pandas pyarrow")

            df = pd.read_parquet(path, columns=columns)
            # 将DataFrame转换为字典列表，并转换 numpy 类型
            data_list = [_convert_numpy_types(row) for row in df.to_dict(orient="records")]

        else:
            raise ValueError(f"不支持的文件格式: {ext}，仅支持 json/jsonl/yaml/parquet")

    except Exception as e:
        raise RuntimeError(f"解析dataset文件失败 [{path}]: {str(e)}")

    return data_list


def load_yaml_configs(yaml_path: str) -> List[Dict]:
    """加载YAML配置并验证格式"""
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"环境配置YAML不存在：{yaml_path}")
    yaml_abs_path = os.path.abspath(yaml_path)
    base_dir = os.path.dirname(yaml_abs_path)

    with open(yaml_path, "r", encoding="utf-8") as f:
        config_data = yaml.safe_load(f)
    
    if "environments" not in config_data:
        raise ValueError("YAML配置缺少'environments'根节点")

    configs = []
    
    for idx, env in enumerate(config_data["environments"], 1):
        # 1. 基础校验
        if "env_name" not in env:
            raise ValueError(f"环境配置 #{idx} 缺少 'env_name'")

        env_name = env["env_name"]
        env_num = env.get("env_num", 1)
        # 获取基础 env_params (深拷贝防止引用污染)
        base_params = env.get("env_params", {}).copy()
        
        dataset_path = env.get("dataset")
        dataset_load_mode = str(env.get("dataset_load_mode", "eager")).strip() or "eager"
        dataset_columns = _normalize_dataset_columns(env.get("dataset_columns"))
        dataset_abs_path = ""
        dataset_name = ""
        if dataset_path:
            dataset_abs_path = dataset_path if os.path.isabs(dataset_path) else os.path.join(base_dir, dataset_path)
            dataset_abs_path = os.path.abspath(dataset_abs_path)
            dataset_name = os.path.splitext(os.path.basename(dataset_abs_path))[0]
        
        # 2. 加载 Dataset 数据
        dataset_items = []
        if dataset_path:
            try:
                dataset_items = load_dataset_file(
                    base_dir,
                    dataset_path,
                    load_mode=dataset_load_mode,
                    columns=dataset_columns,
                )
            except Exception as e:
                log.warning("Environment %s dataset load failed; skipping: %s", env_name, e)
                continue
        else:
            # 如果没有dataset，则生成单个配置，仅包含基础params
            dataset_items = [{}]
            
        # 3. 展开生成配置
        # 如果 dataset_items 为空列表（例如空文件），则生成默认num_param
        if dataset_items:
            for i, item in enumerate(dataset_items):
                # 合并参数：dataset中的行数据 覆盖/追加到 env_params
                current_params = base_params.copy()
                current_params['dataset'] = item
                current_params[SAFACTORY_INTERNAL_ENV_KEY] = {
                    "config_path": yaml_abs_path,
                    "config_dir": base_dir,
                    "dataset_path": dataset_abs_path,
                    "dataset_name": dataset_name,
                    "dataset_load_mode": dataset_load_mode,
                    "dataset_columns": dataset_columns,
                }

                # 构造最终配置对象
                config = {
                    "env_name": env_name,
                    "env_num": env_num,
                    "env_params": current_params,
                    "task_idx": i + 1,
                    "env_image": env.get("env_image", "")
                }
                configs.append(config)
        else:
            current_params = base_params.copy()
            current_params['dataset'] = {}
            current_params[SAFACTORY_INTERNAL_ENV_KEY] = {
                "config_path": yaml_abs_path,
                "config_dir": base_dir,
                "dataset_path": dataset_abs_path,
                "dataset_name": dataset_name,
                "dataset_load_mode": dataset_load_mode,
                "dataset_columns": dataset_columns,
            }
            config = {
                "env_name": env_name,
                "env_num": env_num,
                "env_params": current_params,
                "task_idx": 1,
                "env_image": env.get("env_image", "")
            }
            configs.append(config)

    return configs
