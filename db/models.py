from dataclasses import dataclass
from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, Float, Boolean, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from .config import Base  # 从配置文件导入Base

@dataclass
class InteractionSession(Base):
    """交互会话主表，记录整个交互过程的元信息"""
    __tablename__ = "interaction_sessions"
    
    id: int = Column(Integer, primary_key=True, autoincrement=True)
    env_name: str = Column(String(100), nullable=False)  # 环境名称
    env_id: int = Column(Integer, nullable=False)        # 环境ID
    model_name: str = Column(String(100), nullable=False) # LLM模型名称
    start_time: datetime = Column(DateTime, default=datetime.utcnow)
    end_time: datetime = Column(DateTime, nullable=True)
    total_reward: float = Column(Float, default=0.0)
    success: bool = Column(Boolean, default=False)
    completed: bool = Column(Boolean, default=False)
    
    # 关联到步骤日志
    steps = relationship("InteractionStep", back_populates="session", cascade="all, delete-orphan")

@dataclass
class InteractionStep(Base):
    """交互步骤表，记录每一步的详细交互信息"""
    __tablename__ = "interaction_steps"
    
    id: int = Column(Integer, primary_key=True, autoincrement=True)
    session_id: int = Column(Integer, ForeignKey("interaction_sessions.id"), nullable=False)
    step_number: int = Column(Integer, nullable=False)  # 步骤编号
    timestamp: datetime = Column(DateTime, default=datetime.utcnow)
    
    # 环境信息
    env_state: Text = Column(Text, nullable=False)  # 环境状态（observation）
    reward: float = Column(Float, default=0.0)      # 该步骤的奖励
    done: bool = Column(Boolean, default=False)     # 是否完成
    
    # LLM信息
    llm_prompt: Text = Column(Text, nullable=False)   # 输入给LLM的提示
    llm_response: Text = Column(Text, nullable=False) # LLM的响应
    llm_reasoning: Text = Column(Text, nullable=True) # LLM的推理过程（如果有）
    token_count: int = Column(Integer, nullable=True) # 生成的token数量
    
    # 关联到会话
    session = relationship("InteractionSession", back_populates="steps")