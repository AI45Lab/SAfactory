from core.data_manager.strategy.base_strategy import StorageStrategy
from core.data_manager.models import EnvironmentConfig, InteractionSession, InteractionStep
from core.data_manager.write_buffer import WriteBuffer
from tortoise import Tortoise
from typing import List, Dict, Optional, Tuple
import uuid
import json
from datetime import datetime

class SqliteStrategy(StorageStrategy):
    def __init__(self, db_url, enable_buffer, buffer_size, flush_interval):
        self.db_url = db_url
        self.initialized = False
        self._enable_buffer = enable_buffer
        self._write_buffer: Optional[WriteBuffer] = None
        self._buffer_size = buffer_size
        self._flush_interval = flush_interval

    async def init(self):
        """初始化数据库连接和写入缓冲器"""
        if not self.initialized:
            await Tortoise.init(
                db_url=self.db_url,
                modules={"models": ["core.data_manager.models"]}
            )
            await Tortoise.generate_schemas()
            self.initialized = True

            # 初始化写入缓冲器
            if self._enable_buffer:
                self._write_buffer = WriteBuffer(
                    buffer_size=self._buffer_size,
                    flush_interval=self._flush_interval,
                    auto_start=True,
                    # 按外键依赖顺序 flush：Session 先于 Step
                    flush_order=[InteractionSession, InteractionStep]
                )

    async def add_environment_config(self, env_name: str, **user_params) -> EnvironmentConfig:
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
    
    async def get_all_environments(self) -> List[EnvironmentConfig]:
        """获取所有环境配置"""
        await self.init()
        return await EnvironmentConfig.all()  # 直接返回所有记录

    async def create_session(self, env_config: EnvironmentConfig, llm_model: str) -> InteractionSession:
        """创建 session（走 buffer，update 时会检测并合并到同一对象）"""
        await self.init()
        session = InteractionSession(
            env_id=env_config.env_id,
            llm_model=llm_model
        )
        if self._write_buffer:
            await self._write_buffer.buffer_create(session)
        else:
            await session.save()
        return session

    async def update_session(self, session: InteractionSession, trajectory : str, total_reward: float, is_completed: bool = True) -> InteractionSession:
        """
        更新会话状态

        Args:
            session: 会话实例
            trajectory: 由调用者维护的轨迹字符串
            total_reward: 总奖励
            is_completed: 是否完成
        """
        session.total_reward = total_reward
        session.trajectory = trajectory
        session.is_completed = is_completed
        session.end_time = datetime.now()
        # 使用 buffer 或直接保存
        if self._write_buffer:
            await self._write_buffer.buffer_update(session, {"total_reward", "trajectory", "is_completed", "end_time"})
        else:
            await session.save()
        return session

    async def record_step(self, session: InteractionSession, step_id: int, prompt: list, response: str, reward: float, env_state: Optional[str] = None, done: bool = False):
        """
        记录交互步骤

        使用分离方案：缓冲时只保存 session_id，不依赖 session 对象
        """
        if isinstance(prompt, list):
            # OpenAI messages 格式
            prompt_str = json.dumps(prompt, ensure_ascii=False)
        else:
            prompt_str = str(prompt) if prompt else ""

        # 创建 Step 实例（使用 session_id 而非 session 对象）
        step = InteractionStep(
            session_id=session.session_id,
            step_id=step_id,
            prompt=prompt_str,
            response=response,
            reward=reward,
            env_state=env_state,
            done=done
        )

        # 根据是否启用缓冲决定写入方式
        if self._write_buffer:
            await self._write_buffer.buffer_create(step)
        else:
            await step.save()

        return step

    async def close(self):
        if self._write_buffer:
            await self._write_buffer.stop()
        if self.initialized:
            await Tortoise.close_connections()
    
    @property
    def buffer_stats(self) -> Optional[dict]:
        return self._write_buffer.stats if self._write_buffer else None