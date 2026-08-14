from abc import ABC, abstractmethod
from typing import Any, Optional, List, Dict

from core.data_manager.contracts import EnvironmentQuery, SessionContext, SessionStepQuery


class StorageStrategy(ABC):
    """
    DAO contract for storage backends.

    Implementations translate these operations to a physical backend. Runtime
    workflow policy belongs to ``DataManager`` and its callers, not here.

    Table design:
    - Table 1 (JobEnvironment): job_id + env_id mapping with env config
    - Table 2 (SessionStep): session_id + step_id with full conversation history

    Callers provide complete row values. Implementations must not infer workflow
    state, classify events, construct evaluation rows, or change trainability.
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
    async def get_environment_by_env_id(self, env_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve one active environment by env_id."""
        pass

    @abstractmethod
    async def list_environment_rows(self, query: EnvironmentQuery) -> List[Dict[str, Any]]:
        """List environment rows using backend-neutral filters."""
        pass

    @abstractmethod
    async def insert_environment_rows(self, rows: List[Dict[str, Any]]) -> List[str]:
        """Insert environment rows and return their env_ids."""
        pass

    @abstractmethod
    async def update_environment_rows(
        self,
        query: EnvironmentQuery,
        updates: Dict[str, Any],
    ) -> int:
        pass

    @abstractmethod
    async def delete_session_step_rows(self, query: SessionStepQuery) -> int:
        pass

    @abstractmethod
    async def delete_job_rows(self, job_id: str) -> None:
        pass

    @abstractmethod
    async def insert_session_step_rows(
        self,
        rows: List[Dict[str, Any]],
    ) -> List[str]:
        """Insert fully constructed session-step rows without applying workflow policy."""
        pass

    @abstractmethod
    async def list_session_step_rows(
        self,
        query: SessionStepQuery,
    ) -> List[Dict[str, Any]]:
        """List raw session-step rows using backend-neutral filters."""
        pass

    @abstractmethod
    async def update_session_step_rows(
        self,
        query: SessionStepQuery,
        updates: Dict[str, Any],
    ) -> int:
        """Update matching session-step rows with caller-decided field values."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Clean up resources (DB connections, clients, buffers)"""
        pass
