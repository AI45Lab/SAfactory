#!/usr/bin/env python3
"""
generate_alfred_cases.py

将 EmbodiedBench 的 splits.json 转换为 AIEvoBox 框架所需的 JSONL 格式。

使用方法:
    python /mnt/shared-storage-user/evobox-share/gaozhenkun/gzk/AIEvoBox/env/embodiedgym/generate_alfred_cases.py --eval_set base --output /mnt/shared-storage-user/evobox-share/gaozhenkun/gzk/AIEvoBox/env/embodiedgym/alfred_cases.jsonl
    python generate_alfred_cases.py --eval_set all --output alfred_cases_all.jsonl
    python generate_alfred_cases.py --eval_set spatial --splits_json /path/to/splits.json

JSONL 输出格式:
{
    "id": 1,
    "task": "pick_clean_then_place_in_recep-Ladle-None-DiningTable-4/trial_T20190909_164840_568811",
    "repeat_idx": 2,
    "instruction": "Rinse off a ladle and move it to the table.",
    "eval_set": "base"
}
"""

import os
import sys
import json
import argparse
from typing import List, Dict, Any, Optional


def find_splits_json(script_dir: str) -> Optional[str]:
    """
    自动查找 splits.json 文件路径
    
    尝试以下路径（按优先级）：
    1. script_dir/EmbodiedBench-master/embodiedbench/envs/eb_alfred/data/splits/splits.json
    2. 向上查找包含 EmbodiedBench 的目录
    """
    # 标准相对路径
    candidates = [
        os.path.join(script_dir, "EmbodiedBench-master", "embodiedbench", "envs", 
                     "eb_alfred", "data", "splits", "splits.json"),
        os.path.join(script_dir, "EmbodiedBench", "embodiedbench", "envs", 
                     "eb_alfred", "data", "splits", "splits.json"),
    ]
    
    # 检查直接路径
    for path in candidates:
        if os.path.exists(path):
            return path
    
    # 向上查找
    current = script_dir
    for _ in range(5):
        parent = os.path.dirname(current)
        if parent == current:
            break
        
        # 在父目录中查找 EmbodiedBench-master
        for bench_name in ["EmbodiedBench-master", "EmbodiedBench"]:
            test_path = os.path.join(parent, "env", "embodiedgym", bench_name,
                                     "embodiedbench", "envs", "eb_alfred", 
                                     "data", "splits", "splits.json")
            if os.path.exists(test_path):
                return test_path
        current = parent
    
    return None


# 有效的评测集名称
VALID_EVAL_SETS = [
    'base', 'common_sense', 'complex_instruction', 
    'spatial', 'visual_appearance', 'long_horizon'
]


def load_splits(splits_path: str) -> Dict[str, List[Dict[str, Any]]]:
    """加载 splits.json 文件"""
    if not os.path.exists(splits_path):
        raise FileNotFoundError(f"splits.json 文件不存在: {splits_path}")
    
    with open(splits_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 验证加载的数据格式
    if not isinstance(data, dict):
        raise ValueError(f"splits.json 格式错误: 期望 dict, 得到 {type(data).__name__}")
    
    # 检查是否包含已知的评测集
    known_sets = set(data.keys()) & set(VALID_EVAL_SETS)
    if not known_sets:
        raise ValueError(
            f"splits.json 中没有找到有效的评测集。\n"
            f"文件中的键: {list(data.keys())}\n"
            f"期望的评测集: {VALID_EVAL_SETS}\n"
            f"请确认文件路径是否正确: {splits_path}"
        )
    
    return data


def generate_jsonl(
    splits_data: Dict[str, List[Dict[str, Any]]],
    eval_sets: List[str],
    output_path: str,
    down_sample_ratio: float = 1.0
) -> int:
    """
    生成 JSONL 文件
    
    Args:
        splits_data: splits.json 的内容
        eval_sets: 要包含的评测集列表
        output_path: 输出 JSONL 文件路径
        down_sample_ratio: 下采样比例 (0.0-1.0)
    
    Returns:
        写入的任务数量
    """
    task_id = 0
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for eval_set in eval_sets:
            if eval_set not in splits_data:
                print(f"警告: 评测集 '{eval_set}' 在 splits.json 中不存在，跳过")
                print(f"  可用的评测集: {list(splits_data.keys())}")
                continue
            
            tasks = splits_data[eval_set]
            original_count = len(tasks)
            
            # 下采样处理
            if 0 < down_sample_ratio < 1.0:
                step = max(1, round(1.0 / down_sample_ratio))
                tasks = tasks[::step]
            
            print(f"  处理评测集 '{eval_set}': {len(tasks)}/{original_count} 条任务")
            
            for task_info in tasks:
                task_id += 1
                
                # 验证必需字段
                if "task" not in task_info:
                    print(f"  警告: 任务缺少 'task' 字段，跳过: {task_info}")
                    continue
                
                # 构建 JSONL 记录
                record = {
                    "id": task_id,
                    "task": task_info.get("task", ""),
                    "repeat_idx": task_info.get("repeat_idx", 0),
                    "instruction": task_info.get("instruction", ""),
                    "eval_set": eval_set
                }
                
                # 写入一行 JSON
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    return task_id


def main():
    parser = argparse.ArgumentParser(
        description="将 EmbodiedBench splits.json 转换为 AIEvoBox JSONL 格式"
    )
    parser.add_argument(
        "--eval_set",
        type=str,
        default="all",
        help=f"评测集名称，可选: {', '.join(VALID_EVAL_SETS)} 或 'all' (默认: all)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="alfred_cases.jsonl",
        help="输出 JSONL 文件路径 (默认: alfred_cases.jsonl)"
    )
    parser.add_argument(
        "--splits_json",
        type=str,
        default=None,
        help="splits.json 文件路径 (默认: 自动查找)"
    )
    parser.add_argument(
        "--down_sample_ratio",
        type=float,
        default=1.0,
        help="下采样比例 0.0-1.0 (默认: 1.0，不下采样)"
    )
    
    args = parser.parse_args()
    
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"脚本目录: {script_dir}")
    
    # 确定 splits.json 路径
    if args.splits_json:
        splits_path = args.splits_json
    else:
        splits_path = find_splits_json(script_dir)
        if not splits_path:
            print("错误: 无法自动找到 splits.json 文件")
            print("请使用 --splits_json 参数指定路径")
            print(f"期望路径格式: .../EmbodiedBench-master/embodiedbench/envs/eb_alfred/data/splits/splits.json")
            sys.exit(1)
    
    print(f"splits.json 路径: {splits_path}")
    
    # 验证路径
    if not os.path.exists(splits_path):
        print(f"错误: splits.json 文件不存在: {splits_path}")
        sys.exit(1)
    
    # 验证不是 ann_*.json 文件
    if "ann_" in os.path.basename(splits_path):
        print(f"错误: 路径指向 ann_*.json 而不是 splits.json")
        print(f"  当前路径: {splits_path}")
        print(f"  正确路径应该包含: data/splits/splits.json")
        sys.exit(1)
    
    # 确定要处理的评测集
    if args.eval_set.lower() == "all":
        eval_sets = VALID_EVAL_SETS
    else:
        eval_sets = [args.eval_set]
        if args.eval_set not in VALID_EVAL_SETS:
            print(f"警告: '{args.eval_set}' 不是已知的评测集名称")
            print(f"  已知的评测集: {VALID_EVAL_SETS}")
    
    # 加载 splits.json
    print(f"\n加载 splits.json...")
    try:
        splits_data = load_splits(splits_path)
        print(f"  找到评测集: {list(splits_data.keys())}")
        for key in splits_data.keys():
            if key in VALID_EVAL_SETS:
                print(f"    - {key}: {len(splits_data[key])} 条任务")
    except Exception as e:
        print(f"错误: 加载 splits.json 失败: {e}")
        sys.exit(1)
    
    # 确定输出路径
    output_path = args.output
    if not os.path.isabs(output_path):
        output_path = os.path.join(script_dir, output_path)
    
    print(f"\n生成 JSONL 文件...")
    print(f"  输出路径: {output_path}")
    print(f"  评测集: {eval_sets}")
    print(f"  下采样比例: {args.down_sample_ratio}")
    
    count = generate_jsonl(
        splits_data, 
        eval_sets, 
        output_path,
        args.down_sample_ratio
    )
    
    print(f"\n✓ 成功生成 {count} 条任务记录")
    print(f"  输出文件: {output_path}")
    
    # 显示示例
    if count > 0:
        print(f"\n示例记录 (前3条):")
        with open(output_path, 'r') as f:
            for i, line in enumerate(f):
                if i >= 3:
                    break
                record = json.loads(line)
                print(f"  {i+1}. id={record['id']}, task={record['task'][:50]}...")


if __name__ == "__main__":
    main()
