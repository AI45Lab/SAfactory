import logging
import time
from typing import Optional, List, Dict, Any

from core.data_manager.contracts import EnvironmentQuery, SessionContext, SessionStepQuery
from core.data_manager.strategy.base_strategy import StorageStrategy
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
        self._strategy: StorageStrategy

        try:
            log.debug("Initializing DataManager with strategy: %r", storage_type)
            self._strategy = StorageFactory.create(job_id, storage_type, **storage_config)
            log.debug("DataManager initialized successfully using %s", self.backend_name)

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
        await self._strategy.init()

    @property
    def backend_name(self) -> str:
        """Diagnostic backend name without exposing the DAO instance."""
        return self._strategy.__class__.__name__

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
        return await self._strategy.add_environment(
            job_id=job_id or self.job_id,
            env_name=env_name,
            env_params=env_params,
            image=image,
            group_id=group_id
        )
    
    async def get_all_environments(self, job_id: Optional[str] = None) -> List[Dict]:
        """Retrieve all registered environments"""
        return await self._strategy.list_environment_rows(EnvironmentQuery(
            job_id=job_id or self.job_id,
        ))

    async def get_environment_by_env_id(self, env_id: str) -> Optional[Dict]:
        """Retrieve one environment config by env_id."""
        return await self._strategy.get_environment_by_env_id(env_id)

    async def mark_environment_finished(self, env_id: str) -> int:
        """Mark one environment completed for this job."""
        return await self._strategy.mark_environment_finished(env_id)

    async def list_environment_rows(
        self,
        *,
        job_id: Optional[str] = None,
        env_id: Optional[str] = None,
        after_id: int = 0,
        offset: int = 0,
        limit: Optional[int] = None,
        finished: Optional[bool] = None,
        is_deleted: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        """Query environment rows without exposing backend query objects."""
        return await self._strategy.list_environment_rows(EnvironmentQuery(
            job_id=job_id or self.job_id,
            env_id=env_id,
            after_id=max(0, int(after_id)),
            offset=max(0, int(offset)),
            limit=None if limit is None else max(0, int(limit)),
            finished=finished,
            is_deleted=is_deleted,
        ))

    async def insert_environment_rows(self, rows: List[Dict[str, Any]]) -> List[str]:
        """Insert environment rows through the configured DAO."""
        normalized = []
        for row in rows:
            item = dict(row)
            item["job_id"] = str(item.get("job_id") or self.job_id)
            normalized.append(item)
        return await self._strategy.insert_environment_rows(normalized)

    async def update_environment_rows(
        self,
        *,
        env_id: Optional[str] = None,
        job_id: Optional[str] = None,
        finished: Optional[bool] = None,
        is_deleted: Optional[bool] = None,
        updates: Dict[str, Any],
    ) -> int:
        return await self._strategy.update_environment_rows(
            EnvironmentQuery(
                job_id=job_id or self.job_id,
                env_id=env_id,
                finished=finished,
                is_deleted=is_deleted,
            ),
            dict(updates),
        )

    async def delete_session_step_rows(
        self,
        *,
        session_ids: Optional[List[str]] = None,
        job_id: Optional[str] = None,
    ) -> int:
        return await self._strategy.delete_session_step_rows(SessionStepQuery(
            job_id=job_id or self.job_id,
            session_ids=tuple(session_ids or ()),
        ))

    async def delete_job_rows(self, job_id: Optional[str] = None) -> None:
        await self._strategy.delete_job_rows(job_id or self.job_id)

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
        return SessionContext(
            session_id=env_id,
            env_id=env_id,
            env_name=env_name,
            llm_model=llm_model,
            group_id=group_id,
            job_id=job_id or self.job_id,
            start_time=time.perf_counter(),
        )

    async def record_step(
        self,
        session: SessionContext,
        step_id: int,
        messages: List[Dict],
        response: str,
        step_reward: float,
        request: Optional[str] = None,
        env_state: Optional[str] = None,
        terminated: bool = False,
        truncated: bool = False,
        is_trainable: bool = False,
        dataset: Optional[Any] = None,
        reward: Optional[float] = None,
    ) -> None:
        """
        Record a single interaction step with full conversation history.

        For SQLite: base64 images stored directly in messages
        For Cloud: images uploaded to S3, URLs stored in messages
        """
        session.total_reward += float(step_reward or 0.0)
        await self._strategy.record_step(
            session=session,
            step_id=step_id,
            messages=messages,
            response=response,
            step_reward=step_reward,
            reward=reward,
            request=request,
            env_state=env_state,
            dataset=dataset,
            terminated=terminated,
            truncated=truncated,
            is_trainable=is_trainable,
        )

    async def record_steps_batch(self, steps: List[Dict[str, Any]]) -> List[Optional[str]]:
        """Persist multiple steps using the backend's native bulk API when available."""
        for step in steps:
            session = step.get("session")
            if isinstance(session, SessionContext):
                session.total_reward += float(step.get("step_reward") or 0.0)
        return await self._strategy.record_steps_batch(steps)

    async def mark_records_completed(self, record_ids: List[str]) -> int:
        """Mark known records completed without a latest-row lookup."""
        return await self._strategy.mark_records_completed(record_ids)

    async def list_session_steps(
        self,
        session_id: str,
        *,
        checkout_latest: bool = False,
    ) -> List[Dict[str, Any]]:
        """Return persisted rows for one session in trajectory order."""
        return await self._strategy.list_session_steps(
            session_id,
            checkout_latest=checkout_latest,
        )

    async def record_evaluation_summary(
        self,
        session_id: str,
        step_id: int,
        reward: float,
        env_state: str,
        truncated: bool = False,
    ) -> int:
        """Persist a non-trainable evaluation summary row."""
        return await self._strategy.record_evaluation_summary(
            session_id=session_id,
            step_id=step_id,
            reward=reward,
            env_state=env_state,
            truncated=truncated,
        )

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
        return await self._strategy.update_session_step(
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
        return await self._strategy.patch_session_environment(
            session_id=session_id,
            job_id=job_id,
            env_name=env_name,
            group_id=group_id,
        )

    async def mark_latest_session_completed(
        self,
        session_id: str,
        llm_model: Optional[str] = None,
        *,
        is_session_completed: bool = True,
        is_terminal: Optional[bool] = None,
    ) -> int:
        """
        Set the completion state of the latest persisted trajectory row.
        When llm_model is provided, only rows for that model are considered.
        is_terminal can seal a row before evaluator completion.

        Returns the number of updated records.
        """
        return await self._strategy.mark_latest_session_completed(
            session_id=session_id,
            llm_model=llm_model,
            is_session_completed=is_session_completed,
            is_terminal=is_terminal,
        )

    async def close(self) -> None:
        """Close the storage strategy"""
        await self._strategy.close()
    
    async def fetch_done_steps_with_context(
        self,
        after_id: int = 0,
        limit: int = 100
    ) -> List[Dict]:
        """Fetch completed steps for training data collection"""
        if hasattr(self._strategy, 'fetch_done_steps_with_context'):
            return await self._strategy.fetch_done_steps_with_context(self.job_id, after_id, limit)
        return []

    async def get_max_step_id(self) -> int:
        """Get maximum primary key for pagination"""
        if hasattr(self._strategy, 'get_max_step_id'):
            return await self._strategy.get_max_step_id(self.job_id)
        return 0

    @property
    def buffer_stats(self) -> Optional[dict]:
        """Get buffer statistics (SQLite only)"""
        if hasattr(self._strategy, 'buffer_stats'):
            return self._strategy.buffer_stats
        return None
