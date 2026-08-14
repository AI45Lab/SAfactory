"""Backend-neutral contracts used at the DataManager/DAO boundary."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SessionContext:
    """In-memory workflow state; this object is never persisted as a row."""

    session_id: str
    env_id: str
    env_name: str
    llm_model: str
    group_id: str = ""
    job_id: str = ""
    total_reward: float = 0.0
    start_time: float = 0.0
    message_history: List[Dict[str, Any]] = field(default_factory=list)
    is_session_completed: bool = False


@dataclass(frozen=True)
class EnvironmentQuery:
    """Portable environment-row query understood by every storage DAO."""

    job_id: Optional[str] = None
    env_id: Optional[str] = None
    after_id: int = 0
    offset: int = 0
    limit: Optional[int] = None
    finished: Optional[bool] = None
    is_deleted: Optional[bool] = None


@dataclass(frozen=True)
class SessionStepQuery:
    """Portable session-step query understood by every storage DAO."""

    job_id: Optional[str] = None
    session_id: Optional[str] = None
    session_ids: tuple[str, ...] = ()
    record_id: Optional[str] = None
    record_ids: tuple[str, ...] = ()
    step_id: Optional[int] = None
    llm_model: Optional[str] = None
    after_id: int = 0
    limit: Optional[int] = None
    is_terminal: Optional[bool] = None
    is_trainable: Optional[bool] = None
    checkout_latest: bool = False
