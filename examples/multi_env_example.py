import os
import sys
current_file_path = os.path.abspath(__file__)
examples_dir = os.path.dirname(current_file_path)
project_root = os.path.dirname(examples_dir)
sys.path.append(project_root)
import asyncio
from env.tradinggym.trading_env import TradingGym # 导入环境来注册
from env.embodiedgym.embodied_env import EmbodiedAlfredGym # 导入 Alfred 环境来注册
from core.llm import StaticBaseURLProvider
from core.interactor import Interactor
from core.data_manager.manager import DataManager
from core.env.env_register import list_registered_envs  # 查看已注册环境

async def main():
    # 查看所有已注册的环境（验证注册是否成功）
    registered_envs = list_registered_envs()
    print("已注册的环境:", list(registered_envs.keys()))  # 应输出 ['trading_gym', 'embodied_alfred']

    # 1. 初始化数据管理器
    data_manager = DataManager(db_url="sqlite://embodied_alfred_envs.db")
    await data_manager.init()


    # 清理旧配置（避免残留 env_id 等错误参数）
    from core.data_manager.models import EnvironmentConfig
    await EnvironmentConfig.all().delete()
    print("✓ 已清理旧配置")
    
    
    # 2. 配置 embodied_alfred 环境
    # 环境1：Base 评测集
    await data_manager.add_environment_config(
        env_name="embodied_alfred",  # 必须与注册的环境名称一致
        eval_set="base",
        down_sample_ratio=0.1,  # 仅使用 10% 的数据进行快速测试
        resolution=500,
        detection_box=False,
        max_episode_steps=30,
        max_invalid_actions=10,
        exp_name="aievobox_alfred_base_test"
    )
    
    # 环境2：Spatial 评测集（可选）
    await data_manager.add_environment_config(
        env_name="embodied_alfred",
        eval_set="spatial",
        down_sample_ratio=0.1,
        resolution=500,
        detection_box=False,
        max_episode_steps=30,
        max_invalid_actions=10,
        exp_name="aievobox_alfred_spatial_test"
    )

<<<<<<< HEAD
    # 3. 初始化Agent（使用视觉语言模型）
    agent = APIAgent(
        api_key="EMPTY",
        base_url="http://100.99.102.69:8001/v1",
        model="/mnt/shared-storage-user/steai-share/hf-hub/Qwen2.5-VL-7B-Instruct",  # 使用完整路径
        temperature=0.3
    )
=======
    # 3. 初始化 LLM 配置
    base_url_provider = StaticBaseURLProvider(base_url="http://100.99.102.69:8001/v1")
>>>>>>> origin/main

    # 4. 运行交互器
    interactor = Interactor(
        base_url_provider=base_url_provider,
        api_key="EMPTY",
        model="/mnt/shared-storage-user/steai-share/hf-hub/Qwen2.5-VL-7B-Instruct",
        data_manager=data_manager,
<<<<<<< HEAD
=======
        temperature=0.3,
>>>>>>> origin/main
        max_workers=2,  # 并行运行 2 个环境
        max_steps=30,  # Alfred 环境通常 30 步内完成
        visual_save_path="/mnt/shared-storage-user/evobox-share/gaozhenkun/gzk/eval/visualize/test1105"  # 保存可视化结果
    )
    
    results = await interactor.run_all_environments()
    print("所有环境运行结果:", results)

    # 5. 关闭数据库
    await data_manager.close()

if __name__ == "__main__":
    asyncio.run(main())