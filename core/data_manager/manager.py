import logging
from typing import Optional, List, Dict, Any

from core.data_manager.strategy.base_strategy import StorageStrategy, SessionContext
from core.data_manager.strategy_factory import StorageFactory

log = logging.getLogger("core.data_manager.manager")


class DataManager:
    """
    Unified data manager that delegates to storage strategies.
    """

    def __init__(
        self,
        job_id: str,
        storage_type: str = "sqlite",
        **storage_config
    ):
        self.job_id = job_id
        self.storage_type = storage_type
        self.strategy: Optional[StorageStrategy] = None

        try:
            log.debug("Initializing DataManager with strategy: %r", storage_type)
            self.strategy = StorageFactory.create(job_id, storage_type, **storage_config)
            log.debug("DataManager initialized successfully using %s", self.strategy.__class__.__name__)

        except ValueError as e:
            error_msg = f"Unsupported storage type: '{storage_type}'. Please check registered types."
            log.error("%s Original Error: %s", error_msg, e)
            raise ValueError(error_msg) from e

        except TypeError as e:
            error_msg = f"Invalid configuration for storage type '{storage_type}'."
            log.error("%s Missing or invalid arguments. Original Error: %s", error_msg, e)
            raise ValueError(error_msg) from e

        except Exception as e:
            error_msg = f"Failed to initialize storage strategy '{storage_type}' due to an internal error."
            log.error("%s Original Error: %s", error_msg, e)
            raise RuntimeError(error_msg) from e

    async def init(self) -> None:
        """Initialize the storage strategy"""
        await self.strategy.init()

    async def add_environment(
        self,
        env_name: str,
        env_params: Dict,
        image: str = "",
        group_id: str = "",
        job_id: Optional[str] = None
    ) -> str:
        """
        Register a new environment configuration.

        Args:
            env_name: Environment name
            env_params: User-defined parameters
            image: Environment image
            group_id: Group ID for RL GRPO aggregation
            job_id: Job session identifier (defaults to manager's job_id)

        Returns:
            env_id: Generated environment UUID
        """
        return await self.strategy.add_environment(
            job_id=job_id or self.job_id,
            env_name=env_name,
            env_params=env_params,
            image=image,
            group_id=group_id
        )
    
    async def get_all_environments(self, job_id: Optional[str] = None) -> List[Dict]:
        """Retrieve all registered environments"""
        return await self.strategy.get_all_environments(job_id)

    async def get_environment_by_env_id(self, env_id: str) -> Optional[Dict]:
        """Retrieve one environment config by env_id."""
        return await self.strategy.get_environment_by_env_id(env_id)

    def create_session(
        self,
        env_id: str,
        env_name: str,
        llm_model: str,
        group_id: str = "",
        job_id: Optional[str] = None
    ) -> SessionContext:
        """
        Create a new session context.
        Note: session_id = env_id by design.
        """
        return self.strategy.create_session(
            env_id=env_id,
            env_name=env_name,
            llm_model=llm_model,
            group_id=group_id,
            job_id=job_id or self.job_id
        )

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
        is_trainable: bool = True
    ) -> None:
        """
        Record a single interaction step with full conversation history.

        For SQLite: base64 images stored directly in messages
        For Cloud: images uploaded to S3, URLs stored in messages
        """
        await self.strategy.record_step(
            session=session,
            step_id=step_id,
            messages=messages,
            response=response,
            step_reward=step_reward,
            env_state=env_state,
            terminated=terminated,
            truncated=truncated,
            is_trainable=is_trainable,
        )

    async def record_steps_batch(self, steps: List[Dict[str, Any]]) -> List[Optional[str]]:
        """Persist multiple steps using the backend's native bulk API when available."""
        return await self.strategy.record_steps_batch(steps)

    async def mark_records_completed(self, record_ids: List[str]) -> int:
        """Mark known records completed without a latest-row lookup."""
        return await self.strategy.mark_records_completed(record_ids)

    async def list_session_steps(self, session_id: str) -> List[Dict[str, Any]]:
        """Return persisted rows for one session in trajectory order."""
        return await self.strategy.list_session_steps(session_id)

    async def update_session_step(
        self,
        session_id: str,
        step_id: int,
        updates: Dict[str, Any],
    ) -> int:
        """
        Update one session step by session_id and step_id.

        Returns the number of matched records.
        """
        return await self.strategy.update_session_step(
            session_id=session_id,
            step_id=step_id,
            updates=updates,
        )

    async def patch_session_environment(
        self,
        session_id: str,
        *,
        job_id: str,
        env_name: str,
        group_id: Optional[str] = None,
    ) -> int:
        """Patch persisted session rows after environment metadata is known."""
        return await self.strategy.patch_session_environment(
            session_id=session_id,
            job_id=job_id,
            env_name=env_name,
            group_id=group_id,
        )

    async def mark_latest_session_completed(
        self,
        session_id: str,
        llm_model: Optional[str] = None,
    ) -> int:
        """
        Mark the latest persisted trajectory row for a session as completed.
        When llm_model is provided, only rows for that model are considered.

        Returns the number of updated records.
        """
        return await self.strategy.mark_latest_session_completed(
            session_id=session_id,
            llm_model=llm_model,
        )

    async def close(self) -> None:
        """Close the storage strategy"""
        await self.strategy.close()

    def get_sync_connection(self) -> Any:
        """Get synchronous connection (SQLite only)"""
        return self.strategy.get_sync_connection()
    
    async def fetch_done_steps_with_context(
        self,
        after_id: int = 0,
        limit: int = 100
    ) -> List[Dict]:
        """Fetch completed steps for training data collection"""
        if hasattr(self.strategy, 'fetch_done_steps_with_context'):
            return await self.strategy.fetch_done_steps_with_context(self.job_id, after_id, limit)
        return []

    async def get_max_step_id(self) -> int:
        """Get maximum primary key for pagination"""
        if hasattr(self.strategy, 'get_max_step_id'):
            return await self.strategy.get_max_step_id(self.job_id)
        return 0

    @property
    def buffer_stats(self) -> Optional[dict]:
        """Get buffer statistics (SQLite only)"""
        if hasattr(self.strategy, 'buffer_stats'):
            return self.strategy.buffer_stats
        return None
