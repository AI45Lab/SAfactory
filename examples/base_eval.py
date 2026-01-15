import os
import sys
current_file_path = os.path.abspath(__file__)
examples_dir = os.path.dirname(current_file_path)
project_root = os.path.dirname(examples_dir)
sys.path.append(project_root)
import argparse
import csv
import asyncio
import json
import yaml
import pandas as pd
from env.tradinggym.trading_env import TradingGym  # 导入环境来注册
from env.androidgym.android_env import AndroidGym
from env.search.search_env import SearchEnv
from env.gitgym.git_env import GitGym
from env.embodiedgym.embodied_env import EmbodiedAlfredGym  # 导入 Alfred 环境来注册
from env.dabstep.dabstep_env import DABStepEnv
from env.dwgym.dw_env import DiscoveryWorldEnv
from env.osgym.os_env import OSGym
from core.llm import StaticBaseURLProvider
from core.interactor import Interactor
from core.data_manager.load_yaml import load_yaml_configs
from core.data_manager.manager import DataManager
from core.data_manager.models import EnvironmentConfig  # 导入模型类
from core.env.env_register import list_registered_envs

DB_PATH = "sqlite://test_envs.db"


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="多环境智能体交互测试工具",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # YAML配置同步参数
    parser.add_argument("--env-config-yaml", type=str, default="env_configs.yaml",
                      help="环境配置YAML文件路径（用于同步到数据库）")
    parser.add_argument("--dry-run", action="store_true",
                      help="仅同步配置不运行交互（用于验证配置）")
    
    # 核心运行参数
    parser.add_argument("--max-workers", type=int, default=4,
                      help="最大并行环境数量")
    parser.add_argument("--max-steps", type=int, default=1000,
                      help="每个环境的最大交互步数")
    parser.add_argument("--visual-save-path", type=str, default="/mnt/shared-storage-user/chenxinquan/ai_sandbox/visualize/test1021",
                      help="步进可视化保存文件夹")
    
    # LLM 配置
    parser.add_argument("--llm-api-key", type=str, default="EMPTY",
                      help="LLM API 密钥")
    parser.add_argument("--llm-base-url", type=str, default="http://localhost:8001/v1",
                      help="LLM API 基础地址")
    parser.add_argument("--llm-model", type=str, default="Qwen3-30B-Instruct",
                      help="LLM 模型名称")
    parser.add_argument("--llm-temperature", type=float, default=0.3,
                      help="LLM 生成响应的温度参数（0-1）")
    parser.add_argument("--llm-proxy", type=str, default="",
                      help="LLM 代理")
    
    return parser.parse_args()

async def sync_configs_to_db(data_manager, yaml_configs, dry_run):
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
        for index in range(1, env_num + 1):
            config_key = get_config_key(env_name, env_params, index)
            
            if config_key not in db_config_keys:
                # 新增实例：自动生成env_id
                if not dry_run:
                    await EnvironmentConfig.create(
                        env_name=env_name,
                        env_params=env_params,  # env_id自动生成
                        image=image
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
            if not dry_run:
                await db_cfg.delete()
            deleted += 1
            print(f"删除环境配置实例：{db_cfg.env_name}（env_id: {db_cfg.env_id}）")

    print(f"\n配置同步结果：新增{added} | 保留{len(yaml_config_keys)-added} | 删除{deleted}")
    await data_manager.close()
    return not dry_run

async def run_interaction(data_manager, args):
    """运行多环境交互逻辑"""

    # 查看已注册环境
    registered_envs = list_registered_envs()
    print(f"已注册的环境类型：{list(registered_envs.keys())}")

    # 获取激活的环境配置
    active_envs = await data_manager.get_all_environments()
    if not active_envs:
        print("未找到激活的环境配置，无法运行交互")
        await data_manager.close()
        return

    print(f"\n将运行以下激活环境（共{len(active_envs)}个）：")
    for env in active_envs:
        # 根据环境类型显示不同的信息
        if env.env_name == 'trading_gym':
            info = f"数据：{env.env_params.get('price_filename', 'N/A')}"
        elif env.env_name == 'embodied_alfred':
            info = f"评测集：{env.env_params.get('eval_set', 'base')}"
        else:
            info = f"参数：{list(env.env_params.keys())}"
        print(f"  - {env.env_name}_{env.env_id}（{info}）")

    # 初始化 base_url_provider
    base_url_provider = StaticBaseURLProvider(base_url=args.llm_base_url)
    print(f"\nLLM 配置：模型={args.llm_model}, base_url={args.llm_base_url}")

    # 运行交互器
    interactor = Interactor(
        base_url_provider=base_url_provider,
        api_key=args.llm_api_key,
        model=args.llm_model,
        data_manager=data_manager,
        temperature=args.llm_temperature,
        llm_proxy=args.llm_proxy,
        max_workers=args.max_workers,
        max_steps=args.max_steps,
        visual_save_path=args.visual_save_path
    )
    
    results = await interactor.run_all_environments()
    print("\n" + "="*50)
    print("所有环境运行结果：")
    for env_key, rewards in results.items():
        avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
        print(f"  {env_key}：平均奖励 = {avg_reward:.2f} ({len(rewards)} episodes)")
    print("="*50)

    # 关闭数据库
    await data_manager.close()

async def main():
    args = parse_args()
    
    try:
        # 1. 加载yaml配置
        yaml_configs = load_yaml_configs(args.env_config_yaml)
        print(f"成功加载YAML配置：{args.env_config_yaml}（共{len(yaml_configs)}条有效配置）")
        
        # 2. 同步配置到数据库
        data_manager = DataManager(storage_type="sqlite", db_url=DB_PATH, enable_buffer=True, buffer_size=100, flush_interval=5.0)
        sync_success = await sync_configs_to_db(data_manager, yaml_configs, args.dry_run)
        
        # 3. 若同步成功且非试运行，运行交互
        if sync_success:
            await run_interaction(data_manager, args)
            
    except Exception as e:
        print(f"程序运行失败：{str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
