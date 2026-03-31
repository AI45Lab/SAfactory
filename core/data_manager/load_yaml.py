import os
import yaml
import json
from typing import Any, Dict, List

import numpy as np


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


def _build_parquet_row_refs(path: str) -> List[Dict[str, Any]]:
    """
    Build lightweight references to each parquet row instead of eagerly
    materializing the full dataset into memory.
    """
    try:
        import pyarrow.parquet as pq
    except ImportError:
        raise ImportError("parquet_row_ref 模式需要安装 pyarrow: pip install pyarrow")

    parquet_file = pq.ParquetFile(path)
    refs: List[Dict[str, Any]] = []
    row_idx = 0

    for row_group in range(parquet_file.num_row_groups):
        group_rows = parquet_file.metadata.row_group(row_group).num_rows
        for row_in_group in range(group_rows):
            refs.append({
                "__dataset_ref__": {
                    "kind": "parquet_row",
                    "path": os.path.abspath(path),
                    "row_group": row_group,
                    "row_in_group": row_in_group,
                    "row_idx": row_idx,
                }
            })
            row_idx += 1

    return refs


def load_dataset_file(base_dir: str, path: str, load_mode: str = "eager"):
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
                for line in f:
                    line = line.strip()
                    if line:
                        data_list.append(json.loads(line))

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
                return _build_parquet_row_refs(path)

            try:
                import pandas as pd
            except ImportError:
                raise ImportError("加载parquet文件需要安装pandas: pip install pandas pyarrow")

            df = pd.read_parquet(path)
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
    base_dir = os.path.dirname(os.path.abspath(yaml_path))

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
        
        # 2. 加载 Dataset 数据
        dataset_items = []
        if dataset_path:
            try:
                dataset_items = load_dataset_file(base_dir, dataset_path, load_mode=dataset_load_mode)
            except Exception as e:
                print(f"环境 [{env_name}] 加载dataset失败: {e} (跳过此环境)")
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
            config = {
                "env_name": env_name,
                "env_num": env_num,
                "env_params": current_params,
                "task_idx": 1,
                "env_image": env.get("env_image", "")
            }
            configs.append(config)

    return configs
