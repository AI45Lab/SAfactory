import os
import sys
current_file_path = os.path.abspath(__file__)
examples_dir = os.path.dirname(current_file_path)
project_root = os.path.dirname(examples_dir)
sys.path.append(project_root)
import argparse
import asyncio
import pandas as pd
from core.llm import StaticBaseURLProvider
from core.interactor_server import InteractorServer

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="多环境智能体交互测试工具",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--env-service-url", type=str, default="http://100.100.170.208:36003/trading_gym",
                      help="环境配置YAML文件路径（用于同步到数据库）")
    
    # 核心运行参数
    parser.add_argument("--max-workers", type=int, default=4,
                      help="最大并行环境数量")
    parser.add_argument("--max-steps", type=int, default=1000,
                      help="每个环境的最大交互步数")
    
    # LLM 配置
    parser.add_argument("--llm-api-key", type=str, default="EMPTY",
                      help="LLM API 密钥")
    parser.add_argument("--llm-base-url", type=str, default="http://localhost:8001/v1",
                      help="LLM API 基础地址")
    parser.add_argument("--llm-model", type=str, default="Qwen3-30B-Instruct",
                      help="LLM 模型名称")
    parser.add_argument("--llm-temperature", type=float, default=0.3,
                      help="LLM 生成响应的温度参数（0-1）")
    
    return parser.parse_args()

async def run_interaction(args):
    """运行多环境交互逻辑"""

    # 初始化 base_url_provider
    base_url_provider = StaticBaseURLProvider(base_url=args.llm_base_url)
    print(f"\nLLM 配置：模型={args.llm_model}, base_url={args.llm_base_url}")

    # 运行交互器
    interactor = InteractorServer(
        base_url_provider=base_url_provider,
        api_key=args.llm_api_key,
        model=args.llm_model,
        env_service_url=args.env_service_url,
        temperature=args.llm_temperature,
        max_workers=args.max_workers,
        max_steps=args.max_steps,
    )
    
    results = await interactor.run_all_environments()
    print("\n" + "="*50)
    print("所有环境运行结果：")
    for env_id, total_reward in results.items():
        print(f"  {env_id}：总奖励 = {total_reward:.2f}")
    print("="*50)

async def main():
    args = parse_args()
    
    try:
        await run_interaction(args)
            
    except Exception as e:
        print(f"程序运行失败：{str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())