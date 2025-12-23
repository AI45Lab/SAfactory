import asyncio
import base64
import os
import requests
import aiohttp
from typing import List, Dict, Tuple, Type
from openai.types.chat import ChatCompletionMessageParam
from core.types.base import PromptOutput, TextContent, OpenAIMessage, MessageContent
from core.data_manager.db_utils import  load_env_lists
from .llm import LLM, BaseURLProvider

class InteractorServer:
    def __init__(
            self,
            base_url_provider: BaseURLProvider,
            api_key: str,
            model: str,
            env_service_url: str,
            temperature: float = 1.0,
            max_workers: int = 5,
            max_steps: int = 1000,
            enable_render: bool = True,
            n_episodes: int = 1,
    ):
        self.base_url_provider = base_url_provider
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        base_url = self.base_url_provider.get_base_url()
        self.llm = LLM(
            api_key=self.api_key,
            base_url=base_url,
            model=self.model,
            temperature=self.temperature,
        )
        self.env_service_url = env_service_url
        self.max_workers = max_workers
        self.max_steps = max_steps
        self.enable_render = enable_render
        self.n_episodes = n_episodes

    async def _post_step(self, env_name: str, env_id: str, action: str) -> Dict:
        """调用环境服务的step接口"""
        step_url = f"{self.env_service_url}/{env_name}/{env_id}/step"
        
        headers = {
            "Content-Type": "application/json",  # 设置 Content-Type 为 application/json
        }
        
        data = {
            "env_id": env_id,
            "action": action
        }
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(step_url, json=data, headers=headers) as response:
                    response.raise_for_status()  # 检查是否有异常的HTTP状态码
                    return await response.json()  # 异步解析JSON响应
            except aiohttp.ClientError as e:
                raise ConnectionError(f"环境{env_id}步骤执行失败: {str(e)}") from e
        
    async def _get_task_prompt(self, env_name: str, env_id: str) -> List[ChatCompletionMessageParam]:
        """获取环境任务提示（适配实际get_task_prompt接口）"""
        prompt_url = f"{self.env_service_url}/{env_name}/{env_id}/get_task_prompt"  # 修正接口路径

        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(prompt_url) as response:
                    response.raise_for_status()  # 检查是否有异常的HTTP状态码
                    result = await response.json()  # 解析JSON响应
                    
                    return result
            
            except Exception as e:
                print(f"Error occurred: {e}")
                raise
        
    async def _run_single_environment(
        self, 
        env_name,
        env_id
    ):
        """在单个远程环境中运行Agent交互循环"""

        total_reward = 0.0
        step_id = 1
        
        try:
            # 1. 环境默认已经reset过
            done = False

            # 2. 交互循环
            while not done and step_id <= self.max_steps:
                # 获取任务提示（使用修改后的接口逻辑）
                prompt = await self._get_task_prompt(env_name, env_id)
                
                # Agent生成响应
                response = await self.llm.generate(prompt)

                # 调用远程环境执行动作
                step_result = await self._post_step(env_name, env_id, response)
                
                # 解析步骤结果
                reward = step_result.get("reward", 0.0)
                terminated = step_result.get("terminated", False)
                truncated = step_result.get("truncated", False)
                done = terminated or truncated
                print("=" * 50)
                print(f"step: {step_id}")
                print(f"reward: {reward}")

                # 更新状态
                total_reward += reward
                step_id += 1


        except Exception as e:
            print(f"环境 {env_name}-{env_id} 出错: {str(e)}")

        return env_name, env_id, total_reward
    
    async def _run_env_episodes(self, env_name: str, env_id: str) -> Tuple[str, List[float]]:
        """
        运行一个环境的所有 episodes（并发）

        同一环境的 n_episodes 同时启动
        """
        env_key = f"{env_name}_{env_id}"

        # 同时启动该环境的所有 episodes
        episode_tasks = [
            self._run_single_environment(env_name, env_id)
            for _ in range(self.n_episodes)
        ]
        results = await asyncio.gather(*episode_tasks, return_exceptions=True)

        # 收集奖励
        rewards = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"环境 {env_key} episode {i} 失败: {result}")
                rewards.append(0.0)
            else:
                _, _, reward = result
                rewards.append(reward)
                print(f"环境 {env_key} episode {i} 完成，奖励: {reward:.4f}")

        return env_key, rewards
    
    async def run_all_environments(self, db_path :str,table_name:Optional[str] = None ) -> Dict[str, float]:
        env_name_list, env_id_list = load_env_lists(db_path, table_name)
        """并行运行所有配置的远程环境"""
        #TODO: below is used for manual test when the env is not aggregated from the yaml configs.
        # env_name_list = ["trading_gym"] * 20
        # env_id_list = [i for i in range(1, 21)]
        semaphore = asyncio.Semaphore(self.max_workers)
        
        async def bounded_task(env_name, env_id):
            async with semaphore:
                return await self._run_single_environment(env_name, env_id)

        tasks = [bounded_task(env_name, env_id) for env_name, env_id in zip(env_name_list, env_id_list)]
        results = {}
        
        for task in asyncio.as_completed(tasks):
            env_name, env_id, total_reward = await task
            results[env_id] = total_reward
            print(f"环境 {env_name}-{env_id} 完成，总奖励: {total_reward}")

        return results