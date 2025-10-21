import asyncio
import os
from typing import List, Dict, Tuple, Type
from .agent.base_agent import APIAgent
from .data_manager.manager import DataManager
from .data_manager.models import EnvironmentConfig, InteractionSession
from .env.env_register import get_env_class

class Interactor:
    def __init__(
        self,
        agent: APIAgent,
        data_manager: DataManager,
        max_workers: int = 5,
        max_steps: int = 1000
    ):
        self.agent = agent
        self.data_manager = data_manager
        self.max_workers = max_workers  # 最大并行环境数
        self.max_steps = max_steps      # 每个环境最大交互步数

    async def _init_environment(
        self, 
        env_config: EnvironmentConfig
    ) -> object:
        """根据配置初始化环境实例"""
        # 1. 从注册表获取环境类
        env_class: Type[object] = get_env_class(env_config.env_name)
        
        # 2. 解析环境参数
        env_id = env_config.env_id
        env_params = env_config.env_params.copy()
        
        # 3. 动态传入所有环境参数
        try:
            return env_class(env_id=env_id, **env_params)
        except TypeError as e:
            raise ValueError(
                f"初始化环境 {env_config.env_name} 失败：参数不匹配。"
                f"环境所需参数与env_params中的参数不兼容。错误详情：{str(e)}"
            ) from e

    async def _run_single_environment(
        self, 
        env_config: EnvironmentConfig
    ) -> Tuple[InteractionSession, float]:
        """在单个环境中运行Agent交互循环（核心逻辑不变，略作调整）"""
        # 1. 初始化环境（通过注册机制获取的环境类）
        env = await self._init_environment(env_config)
        session = await self.data_manager.create_session(
            env_config=env_config,
            agent_model=self.agent.model
        )

        total_reward = 0.0
        step_id = 1
        
        try:
            # 2. 重置环境获取初始状态（假设所有环境都实现了reset方法）
            obs, info = env.reset()
            done = False

            # 3. 交互循环（假设所有环境都实现了step方法）
            while not done and step_id <= self.max_steps:
                prompt = obs["text"]
                
                # Agent生成响应
                response = await asyncio.to_thread(
                    self.agent.generate, 
                    prompt=prompt
                )

                # 环境执行动作（统一接口假设：step返回(state, reward, done, info)）
                obs, reward, _, done, _ = env.step(response)

                img_filename = f"env_{env_config.env_id}/step_{step_id:04d}.png"
                env.render(img_filename)

                # 记录交互步骤
                await self.data_manager.record_step(
                    session=session,
                    step_id=step_id,
                    prompt=prompt,
                    response=response,
                    reward=reward,
                    done=done
                )

                # 更新状态
                total_reward += reward
                step_id += 1

            # 4. 完成会话记录
            await self.data_manager.update_session(
                session=session,
                total_reward=total_reward,
                is_completed=True
            )

        except Exception as e:
            print(f"环境 {env_config.env_name}_{env_config.env_id} 出错: {str(e)}")
            await self.data_manager.update_session(
                session=session,
                total_reward=total_reward,
                is_completed=False
            )

        return session, total_reward


    async def run_all_environments(self) -> Dict[str, float]:
        """并行运行所有配置的环境（逻辑不变）"""
        env_configs = await self.data_manager.get_all_environments()
        if not env_configs:
            print("没有找到激活的环境配置")
            return {}

        semaphore = asyncio.Semaphore(self.max_workers)
        
        async def bounded_task(env_config):
            async with semaphore:
                return await self._run_single_environment(env_config)

        tasks = [bounded_task(config) for config in env_configs]
        results = {}
        
        for task in asyncio.as_completed(tasks):
            session, total_reward = await task
            env_key = f"{session.env.env_name}_{session.env.env_id}"
            results[env_key] = total_reward
            print(f"环境 {env_key} 完成，总奖励: {total_reward}")

        return results