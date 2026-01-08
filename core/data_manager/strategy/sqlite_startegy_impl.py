from core.data_manager.strategy.base_strategy import StorageStrategy
from core.data_manager.models import EnvironmentConfig, InteractionSession, InteractionStep
from core.data_manager.write_buffer import WriteBuffer
from tortoise import Tortoise
from typing import List, Dict, Optional, Tuple
import uuid
import json
import sqlite3
from datetime import datetime

class SqliteStrategy(StorageStrategy):
    def __init__(self, db_url: str, enable_buffer: bool = True, buffer_size: int = 100, flush_interval: float = 5.0):
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

    async def create_session(self, env_id, llm_model: str) -> InteractionSession:
        """创建 session（走 buffer，update 时会检测并合并到同一对象）"""
        await self.init()
        session = InteractionSession(
            env_id=env_id,
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
            
    def get_sync_connection(self) -> sqlite3.Connection:
        """
        解析 db_url 并返回一个原生的 sqlite3 同步连接对象
        """
        # 1. 解析 URL (例如: sqlite:///mnt/.../db.sqlite3?journal_mode=DELETE)
        if not self.db_url.startswith("sqlite://"):
            raise ValueError("只支持 sqlite:// 协议")
        
        # 去掉 sqlite:// 前缀
        url_suffix = self.db_url[9:]
        
        # 处理参数 (去掉 ?journal_mode=...)
        if "?" in url_suffix:
            file_path = url_suffix.split("?")[0]
        else:
            file_path = url_suffix

        # 2. 建立原生连接
        # check_same_thread=False 允许在不同线程使用该连接（根据你的需求可选）
        conn = sqlite3.connect(file_path, check_same_thread=False)
        
        # 推荐：配置 row_factory 以便可以通过列名访问数据
        conn.row_factory = sqlite3.Row
        
        return conn
    
    @property
    def buffer_stats(self) -> Optional[dict]:
        return self._write_buffer.stats if self._write_buffer else None
    
    async def fetch_done_steps_with_context(self, after_id: int = 0, limit: int = 100) -> List[Dict]:
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
                "env_state": step.env_state,
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