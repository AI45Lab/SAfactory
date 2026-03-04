from abc import ABC, abstractmethod
from typing import Any, Optional, List, Dict
from dataclasses import dataclass, field


@dataclass
class SessionContext:
    """
    Runtime session context object.
    This is NOT persisted to database - it's an in-memory object to track session state.
    """
    session_id: str
    env_id: str
    env_name: str
    llm_model: str
    group_id: str = ""
    job_id: str = ""

    # Runtime state (not persisted)
    total_reward: float = 0.0
    start_time: float = 0.0
    message_history: List[Dict] = field(default_factory=list)
    is_session_completed: bool = False


class StorageStrategy(ABC):
    """
    Abstract base class for storage backends.

    Table design:
    - Table 1 (JobEnvironment): job_id + env_id mapping with env config
    - Table 2 (SessionStep): session_id + step_id with full conversation history

    Key design principles:
    - session_id equals env_id for compatibility
    - Each step record contains full conversation history up to that point
    - Last record of a session contains total_reward and is_session_completed=True
    """
    
    @abstractmethod
    async def init(self) -> None:
        """Initialize storage backend (DB connection, schemas, clients)"""
        pass
    
    @abstractmethod
    async def add_environment(
        self,
        job_id: str,
        env_name: str,
        env_params: Dict,
        image: str = "",
        group_id: str = "",
    ) -> str:
        """
        Register a new environment configuration.

        Args:
            job_id: Job session identifier
            env_name: Environment name
            env_params: User-defined parameters
            image: Environment image
            group_id: Group ID for RL GRPO aggregation

        Returns:
            env_id: Generated environment UUID
        """
        pass
    
    @abstractmethod
    async def get_all_environments(self, job_id: Optional[str] = None) -> List[Dict]:
        """
        Retrieve all registered environments, optionally filtered by job_id.

        Returns:
            List of environment configs as dicts
        """
        pass
    
    @abstractmethod
    async def create_session(
        self,
        env_id: str,
        env_name: str,
        llm_model: str,
        group_id: str = "",
        job_id: str = ""
    ) -> SessionContext:
        """
        Create a new session context (in-memory object).
        Note: session_id = env_id by design.

        Returns:
            SessionContext object for tracking session state
        """
        pass

    @abstractmethod
    async def record_step(
        self,
        session: SessionContext,
        step_id: int,
        messages: List[Dict],
        response: str,
        step_reward: float,
        env_state: Optional[str] = None,
        terminated: bool = False,
        truncated: bool = False,
        execution_time: Optional[float] = None
    ) -> None:
        """
        Record a single interaction step with full conversation history.

        The messages parameter should contain the FULL conversation history
        up to and including the current user message (but NOT the assistant response,
        which is stored in the response parameter).

        For SQLite: stores base64 images directly in messages JSON
        For Cloud: uploads binary images to S3, stores URLs in messages JSON

        Args:
            session: Session context object
            step_id: Step number (1-indexed)
            messages: Full conversation history (list of {role, content} dicts)
            response: LLM response/action for this step
            step_reward: Reward for this step
            env_state: Optional JSON string of environment state
            terminated: Whether this is a terminal step
            truncated: Whether episode was truncated
            execution_time: Optional execution time in seconds
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """Clean up resources (DB connections, clients, buffers)"""
        pass

    def get_sync_connection(self) -> Any:
        """
        Get synchronous connection for direct queries (SQLite only).
        Returns None for cloud storage.
        """
        return None

    # Legacy compatibility methods (will be deprecated)
    async def create_session_legacy(self, env_id, llm_model: str, group_id: str = ""):
        """Legacy method for backward compatibility"""
        pass

    async def update_session(self, session, trajectory, total_reward, is_session_completed):
        """Legacy method for backward compatibility"""
        pass