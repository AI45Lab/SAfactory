import asyncio
import base64
import os
import requests
import aiohttp
from typing import List, Dict, Tuple, Type
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
    ):
        self.base_url_provider = base_url_provider
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.env_service_url = env_service_url
        self.max_workers = max_workers
        self.max_steps = max_steps

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
        
    async def _get_task_prompt(self, env_name: str, env_id: str) -> List[Dict]:
        """获取环境任务提示，返回 OpenAI messages 格式"""
        prompt_url = f"{self.env_service_url}/{env_name}/{env_id}/get_task_prompt"

        async with aiohttp.ClientSession() as http_session:
            async with http_session.get(prompt_url) as response:
                response.raise_for_status()
                return await response.json()
        
    async def _run_single_environment(
        self,
        env_name,
        env_id
    ):
        """在单个远程环境中运行 LLM 交互循环"""
        # 创建 LLM 实例
        base_url = self.base_url_provider.get_base_url()
        llm = LLM(
            api_key=self.api_key,
            base_url=base_url,
            model=self.model,
            temperature=self.temperature,
        )

        total_reward = 0.0
        step_id = 1

        try:
            # 1. 环境默认已经 reset 过
            done = False

            # 2. 交互循环
            while not done and step_id <= self.max_steps:
                # 获取任务提示
                prompt = await self._get_task_prompt(env_name, env_id)

                # LLM 生成响应
                response = await llm.generate(prompt)

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
    
    async def run_all_environments(self) -> Dict[str, float]:
        """并行运行所有配置的远程环境"""
        env_name_list = ["trading_gym"] * 7 + ["git_gym"] * 6
        env_id_list = [i for i in range(1, 14)]
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