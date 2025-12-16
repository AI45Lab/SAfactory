import json
import uuid
from tortoise import Tortoise
from .models import EnvironmentConfig, InteractionSession, InteractionStep
from .write_buffer import WriteBuffer
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from core.types.base import PromptOutput, serialize_prompt_output


class DataManager:
    def __init__(
        self,
        db_url: str = "sqlite://trading_envs.db",
        enable_buffer: bool = True,
        buffer_size: int = 100,
        flush_interval: float = 5.0
    ):
        """
        初始化数据管理器

        Args:
            db_url: 数据库连接 URL
            enable_buffer: 是否启用写入缓冲（高并发场景建议开启）
            buffer_size: 缓冲区大小阈值
            flush_interval: 定时刷新间隔（秒）
        """
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
        llm_model: str
    ) -> InteractionSession:
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

    async def update_session(
        self,
        session: InteractionSession,
        trajectory: str,
        total_reward: float,
        is_completed: bool = True
    ) -> InteractionSession:
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

    async def record_step(
        self,
        session: InteractionSession,
        step_id: int,
        prompt,
        response: str,
        reward: float,
        env_state: Optional[str] = None,
        done: bool = False
    ) -> InteractionStep:
        """
        记录交互步骤

        使用分离方案：缓冲时只保存 session_id，不依赖 session 对象
        """
        # 序列化 prompt
        if isinstance(prompt, PromptOutput):
            prompt_str = serialize_prompt_output(prompt)
        elif isinstance(prompt, list):
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

    async def flush(self) -> Dict[str, int]:
        """手动刷新缓冲区"""
        if self._write_buffer:
            return await self._write_buffer.flush()
        return {"created": 0, "updated": 0}

    async def close(self):
        """关闭数据库连接（先刷新缓冲区确保数据不丢失）"""
        if self.initialized:
            # 先停止缓冲器，确保所有数据写入
            if self._write_buffer:
                await self._write_buffer.stop()
                self._write_buffer = None
            await Tortoise.close_connections()
            self.initialized = False

    @property
    def buffer_stats(self) -> Optional[dict]:
        """获取缓冲区统计信息"""
        if self._write_buffer:
            return self._write_buffer.stats
        return None

    async def fetch_done_steps_with_context(
        self,
        after_id: int = 0,
        limit: int = 100
    ) -> List[Dict]:
        """
        获取已完成的步骤及其上下文信息（游标分页）

        Args:
            after_id: 只返回 id > after_id 的记录
            limit: 最大返回数量

        Returns:
            包含 step、session、env 信息的字典列表
        """
        await self.init()

        steps = await InteractionStep.filter(
            done=True,
            id__gt=after_id
        ).prefetch_related(
            "session", "session__env"
        ).order_by("id").limit(limit)

        results = []
        for step in steps:
            session = step.session
            env = session.env if session else None

            results.append({
                "step_pk": step.id,
                "step_id": step.step_id,
                "prompt": step.prompt,
                "response": step.response,
                "reward": step.reward,
                "timestamp": step.timestamp.isoformat() if step.timestamp else None,
                "done": step.done,
                "session_id": session.session_id if session else None,
                "session_end_time": session.end_time.isoformat() if session and session.end_time else None,
                "env_id": env.env_id if env else None,
                "env_name": env.env_name if env else None,
            })

        return results

    async def get_max_step_id(self) -> int:
        """获取当前最大的 step id，用于初始化游标"""
        await self.init()
        latest = await InteractionStep.all().order_by("-id").first()
        return latest.id if latest else 0