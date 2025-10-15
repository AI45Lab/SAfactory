import sys
import os
import threading
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)
import argparse
from datetime import datetime
from typing import List, Dict
from controller import APIAgentvllm
from clients import TradingEnvClient
from clients.base_env_client import StepOutput
from db.config import engine
from db import InteractionStep, InteractionSession, SessionLocal, init_db


def run_simulation(env_client: TradingEnvClient, env_name: str, agent: APIAgentvllm, results: Dict[str, List[StepOutput]]):
    """在单个环境上运行模拟并记录日志"""
    db = None
    try:
        print(f"开始在环境 {env_name} 上的模拟")
        step_outputs = []

        db = SessionLocal()
        
        # 创建交互会话记录
        env_id = env_client.env_ids[env_name]
        interaction_session = InteractionSession(
            env_name=env_name,
            env_id=env_id,
            model_name=agent.model,
            start_time=datetime.utcnow(),
            total_reward=0.0,  # 初始化总奖励
            completed=False
        )
        db.add(interaction_session)
        db.commit()
        # 刷新会话以获取自动生成的ID
        db.refresh(interaction_session)
        
        # 重置环境
        step_output = env_client.reset(env_name)
        step_outputs.append(step_output)
        
        step_number = 0
        while not step_output.done:
            step_number += 1
            prompt = step_output.state["text"]["text"]
            response = agent.generate(prompt)

            step_output = env_client.step(response, env_name)
            step_outputs.append(step_output)

            # 记录步骤信息
            interaction_step = InteractionStep(
                session_id=interaction_session.id,
                step_number=step_number,
                env_state=prompt,
                llm_prompt=prompt,
                llm_response=response,
                reward=step_output.reward,
                done=step_output.done
            )
            db.add(interaction_step)
            db.commit()
            print(f"环境 {env_name} - 奖励: {step_output.reward}, 完成状态: {step_output.done}")
            
        results[env_name] = step_outputs
        print(f"环境 {env_name} 模拟完成")
        
    except Exception as e:
        print(f"环境 {env_name} 模拟出错: {str(e)}")
        results[env_name] = []
    finally:
        db.close()

def main(args):
    # 初始化数据库（创建所有表）
    init_db()

    # 初始化API代理
    api_agent = APIAgentvllm(
        api_key=args.api_key,
        base_url=args.base_url,
        model=args.model_name,
        temperature=args.temperature,
    )

    # 初始化交易环境客户端
    env_client = TradingEnvClient(
        env_server_base=args.env_server_base,
        timeout=args.timeout
    )

    # 创建多个环境
    num_envs = args.num_envs
    env_names = [f"trading_env_{i}" for i in range(num_envs)]
    
    for env_name in env_names:
        env_client.create_env(env_name, data_idx=0)  # 可以为不同环境指定不同data_idx

    # 存储每个环境的模拟结果
    simulation_results = {}
    threads = []

    # 在多个环境上并行运行模拟
    for env_name in env_names:
        thread = threading.Thread(
            target=run_simulation,
            args=(env_client, env_name, api_agent, simulation_results)
        )
        threads.append(thread)
        thread.start()

    # 等待所有模拟完成
    for thread in threads:
        thread.join()

    # 处理结果（示例：打印每个环境的总奖励）
    for env_name, steps in simulation_results.items():
        if steps:
            total_reward = sum(step.reward for step in steps)
            print(f"环境 {env_name} 总奖励: {total_reward}")

    # 关闭所有环境
    for env_name in env_names:
        env_client.close_env(env_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="Qwen3-30B-Instruct")
    parser.add_argument('--api_key', type=str, default="EMPTY")
    parser.add_argument('--base_url', type=str, default="http://localhost:8001/v1")
    parser.add_argument('--temperature', type=float, default=0.3)
    parser.add_argument('--timeout', type=int, default=300)
    parser.add_argument('--env_server_base', type=str, default="http://127.0.0.1:36002")
    parser.add_argument('--num_envs', type=int, default=2, help="同时运行的环境数量")
    args = parser.parse_args()
    print(args)
    
    main(args)