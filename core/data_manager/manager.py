import uuid
from tortoise import Tortoise
from .models import EnvironmentConfig, InteractionSession, InteractionStep
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from core.types.base import PromptOutput, serialize_prompt_output

class DataManager:
    def __init__(self, db_url: str = "sqlite://trading_envs.db"):
        self.db_url = db_url
        self.initialized = False

    async def init(self):
        """初始化数据库连接"""
        if not self.initialized:
            await Tortoise.init(
                db_url=self.db_url,
                modules={"models": ["core.data_manager.models"]}  # 确保模型路径正确
            )
            await Tortoise.generate_schemas()  # 自动创建数据表
            self.initialized = True

    async def add_environment_config(
        self, 
        env_name: str, 
        **user_params
    ) -> EnvironmentConfig:
        await self.init()
        env_id = str(uuid.uuid4())
        # 若环境配置已存在则获取，不存在则创建
        config, created = await EnvironmentConfig.get_or_create(
            env_name=env_name,
            env_id=env_id,
            defaults={
                "env_params": user_params  # 将用户参数存入JSON字段
            }
        )
        return config

    async def update_environment_config(
        self,
        env_name: str,
        env_id: str,  # 注意：env_id是UUID字符串，不再是int
        **user_params  # 更新用户参数（整体替换或部分更新）
    ) -> Optional[EnvironmentConfig]:
        """更新环境配置的用户参数"""
        await self.init()
        try:
            config = await EnvironmentConfig.get(env_name=env_name, env_id=env_id)
            
            # 若传入的是部分参数，合并到现有参数中；否则整体替换
            if user_params:
                current_params = config.env_params or {}
                current_params.update(user_params)  # 合并更新
                config.env_params = current_params
            
            await config.save()
            return config
        except EnvironmentConfig.DoesNotExist:
            return None

    async def get_all_environments(self) -> List[EnvironmentConfig]:
        """获取所有环境配置"""
        await self.init()
        return await EnvironmentConfig.all()  # 直接返回所有记录

    async def create_session(
        self, 
        env_config: EnvironmentConfig, 
        agent_model: str
    ) -> InteractionSession:
        await self.init()
        return await InteractionSession.create(
            env=env_config,  # 关联EnvironmentConfig实例
            agent_model=agent_model
        )

    async def update_session(
        self, 
        session: InteractionSession, 
        total_reward: float, 
        is_completed: bool = True
    ) -> InteractionSession:
        # 1. 批量查询该会话的所有步骤
        steps = await InteractionStep.filter(session__session_id=session.session_id).order_by("step_id").all()
        # 2. 合并所有步骤的Prompt+Response
        trajectory = ""
        for step in steps:
            trajectory += f"Step {step.step_id}:\nPrompt: {step.prompt}\nResponse: {step.response}\n\n"
        # 3. 更新会话
        session.total_reward = total_reward
        session.trajectory = trajectory
        session.is_completed = is_completed
        session.end_time = datetime.now()
        await session.save()
        return session

    async def record_step(
        self,
        session: InteractionSession,
        step_id: int,
        prompt: PromptOutput,
        response: str,
        reward: float,
        env_state: Optional[str] = None,
        done: bool = False
    ) -> InteractionStep:
        if isinstance(prompt, PromptOutput):
            prompt_str = serialize_prompt_output(prompt)
        else:
            prompt_str = prompt
        
        return await InteractionStep.create(
            session=session,
            step_id=step_id,
            prompt=prompt_str,
            response=response,
            reward=reward,
            env_state=env_state,
            done=done
        )

    async def close(self):
        """关闭数据库连接"""
        if self.initialized:
            await Tortoise.close_connections()
            self.initialized = False