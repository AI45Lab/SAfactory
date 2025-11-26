import os
import sys
current_file_path = os.path.abspath(__file__)
examples_dir = os.path.dirname(current_file_path)
project_root = os.path.dirname(examples_dir)
sys.path.append(project_root)
import argparse
import asyncio
import pandas as pd
from core.agent.base_agent import APIAgent
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
    
    # Agent配置
    parser.add_argument("--agent-api-key", type=str, default="EMPTY",
                      help="Agent的API密钥")
    parser.add_argument("--agent-base-url", type=str, default="http://localhost:8001/v1",
                      help="Agent的API基础地址")
    parser.add_argument("--agent-model", type=str, default="Qwen3-30B-Instruct",
                      help="Agent使用的模型名称")
    parser.add_argument("--agent-temperature", type=float, default=0.3,
                      help="Agent生成响应的温度参数（0-1）")
    
    return parser.parse_args()

async def run_interaction(args):
    """运行多环境交互逻辑"""

    # 初始化Agent
    agent = APIAgent(
        api_key=args.agent_api_key,
        base_url=args.agent_base_url,
        model=args.agent_model,
        temperature=args.agent_temperature
    )
    print(f"\n智能体初始化完成：模型：{args.agent_model}")

    # 运行交互器
    interactor = InteractorServer(
        agent=agent,
        env_service_url=args.env_service_url,
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