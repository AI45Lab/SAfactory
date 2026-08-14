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

    Key design principles:
    - session_id equals env_id for compatibility
    - Each step record contains full conversation history up to that point
    - Final reward remains null until the evaluator completes the session
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

    async def get_environment_by_env_id(self, env_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve one environment config by env_id.

        Storage backends can override this with an indexed lookup. The default
        implementation keeps older strategies compatible by scanning
        get_all_environments().
        """
        for env in await self.get_all_environments():
            if not isinstance(env, dict):
                continue
            if str(env.get("env_id") or "") == str(env_id):
                if bool(env.get("is_deleted", False)):
                    continue
                return env
        return None

    async def list_environment_rows(self, query: EnvironmentQuery) -> List[Dict[str, Any]]:
        """List environment rows using backend-neutral filters."""
        rows = await self.get_all_environments(query.job_id)
        filtered = []
        for index, row in enumerate(rows, start=1):
            item = dict(row)
            item.setdefault("id", index)
            if query.env_id and str(item.get("env_id") or "") != query.env_id:
                continue
            if int(item.get("id") or 0) <= query.after_id:
                continue
            if query.finished is not None and bool(item.get("finished", False)) != query.finished:
                continue
            if query.is_deleted is not None and bool(item.get("is_deleted", False)) != query.is_deleted:
                continue
            filtered.append(item)
        filtered.sort(key=lambda item: int(item.get("id") or 0))
        start = max(0, query.offset)
        end = None if query.limit is None else start + max(0, query.limit)
        return filtered[start:end]

    async def insert_environment_rows(self, rows: List[Dict[str, Any]]) -> List[str]:
        """Insert environment rows and return their env_ids."""
        env_ids = []
        for row in rows:
            env_ids.append(await self.add_environment(
                job_id=str(row.get("job_id") or ""),
                env_name=str(row.get("env_name") or ""),
                env_params=dict(row.get("env_params") or {}),
                image=str(row.get("image") or ""),
                group_id=str(row.get("group_id") or ""),
            ))
        return env_ids

    async def update_environment_rows(
        self,
        query: EnvironmentQuery,
        updates: Dict[str, Any],
    ) -> int:
        raise NotImplementedError

    async def delete_session_step_rows(self, query: SessionStepQuery) -> int:
        raise NotImplementedError

    async def delete_job_rows(self, job_id: str) -> None:
        raise NotImplementedError

    @abstractmethod
    async def mark_environment_finished(self, env_id: str) -> int:
        """Mark one environment completed after its full workflow succeeds."""
        pass
    
    @abstractmethod
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
            request: Optional provider-bound request JSON for this step
            env_state: Optional JSON string of environment state
            terminated: Whether this is a terminal step
            truncated: Whether episode was truncated
            is_trainable: Whether this step is eligible for training
            dataset: Optional task dataset stored with the current step
        """
        pass

    async def record_steps_batch(self, steps: List[Dict[str, Any]]) -> List[Optional[str]]:
        """Persist multiple steps.

        Backends may override this to use a native bulk API. The default keeps
        existing strategies compatible and preserves input ordering.
        """
        record_ids: List[Optional[str]] = []
        for step in steps:
            await self.record_step(**step)
            record_ids.append(None)
        return record_ids

    async def mark_records_completed(self, record_ids: List[str]) -> int:
        """Mark known records completed without discovering them by a table scan."""
        return 0

    async def list_session_steps(
        self,
        session_id: str,
        *,
        checkout_latest: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Return persisted rows for one session in trajectory order.

        Storage backends may override this when callers such as evaluators need
        storage-agnostic access to the completed trajectory. Cloud callers can
        request the latest table version when another process performed writes.
        The default keeps older/custom strategies compatible.
        """
        return []

    async def record_evaluation_summary(
        self,
        session_id: str,
        step_id: int,
        reward: float,
        env_state: str,
        truncated: bool = False,
    ) -> int:
        """Persist a non-trainable evaluation result when no trajectory row exists."""
        return 0

    @abstractmethod
    async def update_session_step(
        self,
        session_id: str,
        step_id: int,
        updates: Dict[str, Any],
    ) -> int:
        """
        Update fields for one persisted session step identified by session_id and step_id.

        Returns:
            Number of matched records.
        """
        pass

    async def patch_session_environment(
        self,
        session_id: str,
        *,
        job_id: str,
        env_name: str,
        group_id: Optional[str] = None,
    ) -> int:
        """
        Patch persisted session rows after their environment metadata is known.

        Backends that cannot efficiently or safely rewrite existing rows may
        leave the default no-op behavior.
        """
        return 0

    @abstractmethod
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

        Returns:
            Number of updated records.
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """Clean up resources (DB connections, clients, buffers)"""
        pass
