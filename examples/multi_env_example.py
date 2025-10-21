import os
import sys
current_file_path = os.path.abspath(__file__)
examples_dir = os.path.dirname(current_file_path)
project_root = os.path.dirname(examples_dir)
sys.path.append(project_root)
import asyncio
from env.tradinggym.trading_env import TradingGym # 导入环境来注册
from core.agent.base_agent import APIAgent
from core.interactor import Interactor
from core.data_manager.manager import DataManager
from core.env.env_register import list_registered_envs  # 查看已注册环境

async def main():
    # 查看所有已注册的环境（验证注册是否成功）
    registered_envs = list_registered_envs()
    print("已注册的环境:", list(registered_envs.keys()))  # 应输出 ['trading_gym']

    # 1. 初始化数据管理器
    data_manager = DataManager(db_url="sqlite://trading_envs.db")
    await data_manager.init()

    # 2. 配置两个trading_gym环境（使用不同数据集）
    data_dir = "/mnt/shared-storage-user/chenxinquan/ai_sandbox/data/trading"
    visual_save_path = "/mnt/shared-storage-user/chenxinquan/ai_sandbox/visualize/test1020"
    
    # 环境1：AMZN数据集
    await data_manager.add_environment_config(
        env_name="trading_gym",  # 必须与注册的环境名称一致
        env_id=1,
        data_dir=data_dir,
        price_filename="AMZN.csv",
        tweet_filename="amzn_stockmo.csv",
        visual_save_path=visual_save_path,
        window_size=7
    )
    
    # 环境2：AAPL数据集
    await data_manager.add_environment_config(
        env_name="trading_gym",
        env_id=2,
        data_dir=data_dir,
        price_filename="AAPL.csv",
        tweet_filename="aapl_stockmo.csv",
        visual_save_path=visual_save_path,
        window_size=7
    )

    # 3. 初始化Agent
    agent = APIAgent(
        api_key="EMPTY",
        base_url="http://localhost:8001/v1",
        model="Qwen3-30B-Instruct",
        temperature=0.3
    )

    # 4. 运行交互器
    interactor = Interactor(
        agent=agent,
        data_manager=data_manager,
        max_workers=2,
        max_steps=1000
    )
    
    results = await interactor.run_all_environments()
    print("所有环境运行结果:", results)

    # 5. 关闭数据库
    await data_manager.close()

if __name__ == "__main__":
    asyncio.run(main())