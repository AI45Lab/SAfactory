from typing import Optional, List, Dict, Any

from core.data_manager.strategy.base_strategy import StorageStrategy, SessionContext
from core.data_manager.strategy_factory import StorageFactory

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
            print(f"Initializing DataManager with strategy: '{storage_type}'")
            self.strategy = StorageFactory.create(job_id, storage_type, **storage_config)
            print(f"DataManager initialized successfully using {self.strategy.__class__.__name__}")

        except ValueError as e:
            error_msg = f"Unsupported storage type: '{storage_type}'. Please check registered types."
            print(f"{error_msg} Original Error: {e}")
            raise ValueError(error_msg) from e

        except TypeError as e:
            error_msg = f"Invalid configuration for storage type '{storage_type}'."
            print(f"{error_msg} Missing or invalid arguments. Original Error: {e}")
            raise ValueError(error_msg) from e

        except Exception as e:
            error_msg = f"Failed to initialize storage strategy '{storage_type}' due to an internal error."
            print(f"{error_msg} Original Error: {e}")
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