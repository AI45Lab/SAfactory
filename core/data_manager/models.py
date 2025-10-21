from tortoise import Model, fields
from datetime import datetime
from typing import Optional
import uuid

class EnvironmentConfig(Model):
    """环境配置模型，关联注册的环境名称"""
    id = fields.IntField(pk=True, autoincrement=True) # 内部自增ID
    env_id = fields.CharField(
        max_length=36, 
        default=lambda: str(uuid.uuid4()),  # 自动生成UUID
        unique=True, 
        description="环境唯一标识UUID"
    )
    env_name = fields.CharField(max_length=100, description="环境名称（需与注册的环境名称一致）")
    env_params = fields.JSONField(description="用户自定义参数")
    created_at = fields.DatetimeField(auto_now_add=True)

    class Meta:
        table = "environment_configs"
        unique_together = ("env_name", "env_id")  # 确保环境名称+ID唯一

class InteractionSession(Model):
    id = fields.IntField(pk=True, autoincrement=True)
    session_id = fields.CharField(
        max_length=36, 
        default=lambda: str(uuid.uuid4()),  # 自动生成UUID
        unique=True, 
        description="会话唯一标识UUID"
    )
    # 通过env_id关联EnvironmentConfig
    env = fields.ForeignKeyField(
        "models.EnvironmentConfig",
        related_name="sessions",
        on_delete=fields.CASCADE,
        to_field="env_id"  # 明确关联EnvironmentConfig的env_id字段（UUID）
    )
    agent_model = fields.CharField(max_length=100)
    start_time = fields.DatetimeField(auto_now_add=True)
    end_time = fields.DatetimeField(null=True)
    total_reward = fields.FloatField(default=0.0)
    trajectory = fields.TextField(default="")
    is_completed = fields.BooleanField(default=False)

    class Meta:
        table = "interaction_sessions"

class InteractionStep(Model):
    id = fields.IntField(pk=True, autoincrement=True)
    # 通过session_id关联InteractionSession（外键关联UUID字段）
    session = fields.ForeignKeyField(
        "models.InteractionSession",
        related_name="steps",
        on_delete=fields.CASCADE,
        to_field="session_id"  # 明确关联InteractionSession的session_id字段（UUID）
    )
    step_id = fields.IntField()
    timestamp = fields.DatetimeField(auto_now_add=True)
    prompt = fields.TextField()
    response = fields.TextField()
    reward = fields.FloatField()
    env_state = fields.TextField(null=True)
    done = fields.BooleanField(default=False)

    class Meta:
        table = "interaction_steps"
        unique_together = ("session", "step_id")