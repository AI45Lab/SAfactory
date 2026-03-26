from tortoise import Model, fields
from typing import Optional, Any, List, Dict
import uuid


class JobEnvironment(Model):
    """
    Table 1: Job-Environment Mapping
    Stores: job_id and env_id as primary identifiers
    Stores: env_name, env_params, image, group_id, created_at
    """
    id = fields.IntField(pk=True, autoincrement=True)
    job_id = fields.CharField(
        max_length=64,
        description="Job session identifier"
    )
    env_id = fields.CharField(
        max_length=36,
        default=lambda: str(uuid.uuid4()),
        unique=True,
        description="Environment unique identifier (UUID)"
    )
    env_name = fields.CharField(max_length=100, description="Environment name")
    env_params = fields.JSONField(default=dict, description="User-defined parameters")
    image = fields.CharField(max_length=100, null=True, description="Environment image")
    group_id = fields.CharField(
        max_length=150,
        null=True,
        description="Group ID for RL GRPO aggregation"
    )
    finished = fields.BooleanField(default=False, description="Has it been completed?")
    is_deleted = fields.BooleanField(default=False, description="Soft-delete flag")
    created_at = fields.DatetimeField(auto_now_add=True)

    class Meta:
        table = "job_environments"
        unique_together = ("job_id", "env_id")
        
        
class SessionStep(Model):
    """
    Table 2: Session-Step Mapping
    Stores full conversation history per step
    Includes: env_name, llm_model, group_id, job_id, messages (JSON), response, step_reward, total_reward, terminated, is_session_completed
    """
    id = fields.IntField(pk=True, autoincrement=True)
    session_id = fields.CharField(
        max_length=36,
        description="Session identifier (equals env_id for compatibility)"
    )
    step_id = fields.IntField(description="Step number within session")
    
    # Environment reference
    env_name = fields.CharField(max_length=100, description="Environment name")

    # LLM info
    llm_model = fields.CharField(max_length=150, description="LLM model name")

    # Session metadata
    group_id = fields.CharField(max_length=150, null=True)
    job_id = fields.CharField(max_length=64, null=True)

    # Conversation history (full messages up to this step)
    messages = fields.TextField(
        description="JSON: Full conversation history including system, user, assistant messages"
    )

    # Current step data
    response = fields.TextField(description="LLM action/response for this step")

    # Rewards
    step_reward = fields.FloatField(default=0.0, description="Reward for this step")
    reward = fields.FloatField(default=0.0, description="Cumulative reward up to this step")

    # State tracking
    env_state = fields.TextField(null=True, description="JSON: Environment state")
    is_terminal = fields.BooleanField(default=False, description="Whether this step is terminal")
    is_truncated = fields.BooleanField(default=False, description="Whether this step is truncated")
    is_session_completed = fields.BooleanField(default=False, description="Whether the session is completed (final record)")
    is_trainable = fields.BooleanField(default=True, description="Whether this step is used for training")

    # Timestamps
    created_at = fields.DatetimeField(auto_now_add=True)

    class Meta:
        table = "session_steps"
        unique_together = ("session_id", "step_id", "created_at")