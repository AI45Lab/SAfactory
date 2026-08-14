import json
import logging
import time
from typing import Optional, List, Dict, Any

from core.data_manager.contracts import EnvironmentQuery, SessionContext, SessionStepQuery
from core.data_manager.image_processing import MessageImageProcessor
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
        self._image_processor: Optional[MessageImageProcessor] = None

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
        if self.storage_type == "cloud" and self._image_processor is None:
            self._image_processor = MessageImageProcessor(
                job_id=self.job_id,
                uploader=getattr(self._strategy, "s3_uploader", None),
            )

    @property
    def backend_name(self) -> str:
        """Diagnostic backend name without exposing the DAO instance."""
        return self._strategy.__class__.__name__

    @property
    def storage_identity(self) -> str:
        """Stable identity used to scope cross-process initialization claims."""
        return ":".join((
            self.storage_type,
            str(getattr(self._strategy, "db_url", "") or ""),
            str(getattr(self._strategy, "env_config_table", "") or ""),
        ))

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
        updated = await self._strategy.update_environment_rows(
            EnvironmentQuery(job_id=self.job_id, env_id=env_id, is_deleted=False),
            {"finished": True},
        )
        if updated != 1:
            raise RuntimeError(
                f"expected one env config for job_id={self.job_id!r} env_id={env_id!r}, "
                f"updated={updated}"
            )
        return updated

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
        record_ids: Optional[List[str]] = None,
        job_id: Optional[str] = None,
    ) -> int:
        return await self._strategy.delete_session_step_rows(SessionStepQuery(
            job_id=job_id or self.job_id,
            session_ids=tuple(session_ids or ()),
            record_ids=tuple(record_ids or ()),
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
        meta_json: Optional[Any] = None,
        env_state: Optional[str] = None,
        terminated: bool = False,
        truncated: bool = False,
        is_trainable: bool = False,
        dataset: Optional[Any] = None,
        reward: Optional[float] = None,
    ) -> Optional[str]:
        """
        Record a single interaction step with full conversation history.

        For SQLite: base64 images stored directly in messages
        For Cloud: images uploaded to S3, URLs stored in messages
        """
        session.total_reward += float(step_reward or 0.0)
        metadata = _metadata_object(meta_json if meta_json is not None else env_state)
        if dataset is not None:
            metadata["dataset"] = dataset
        record_ids = await self.insert_session_step_rows([{
            "session_id": session.session_id,
            "env_id": session.env_id,
            "step_id": step_id,
            "env_name": session.env_name,
            "llm_model": session.llm_model,
            "group_id": session.group_id,
            "job_id": session.job_id,
            "messages": messages,
            "request": request,
            "response": response,
            "step_reward": step_reward,
            "reward": reward,
            "meta_json": metadata,
            "is_terminal": bool(terminated or truncated),
            "is_truncated": bool(truncated),
            "is_session_completed": bool(terminated),
            # Compatibility behavior; new callers should use insert_session_step_rows.
            "is_trainable": False,
        }])
        session.message_history = list(messages)
        return record_ids[0] if record_ids else None

    async def record_steps_batch(self, steps: List[Dict[str, Any]]) -> List[Optional[str]]:
        """Compatibility adapter for the former SessionContext-based write API."""
        rows: List[Dict[str, Any]] = []
        for step in steps:
            session = step.get("session")
            if not isinstance(session, SessionContext):
                rows.append(dict(step))
                continue
            session.total_reward += float(step.get("step_reward") or 0.0)
            metadata = _metadata_object(step.get("meta_json", step.get("env_state")))
            if step.get("dataset") is not None:
                metadata["dataset"] = step["dataset"]
            if isinstance(step.get("provider_meta"), dict):
                metadata.update(step["provider_meta"])
            messages = step.get("messages") or []
            session.message_history = list(messages)
            rows.append({
                "record_id": step.get("record_id"),
                "session_id": session.session_id,
                "env_id": session.env_id,
                "step_id": step.get("step_id", 0),
                "env_name": session.env_name,
                "llm_model": session.llm_model,
                "group_id": session.group_id,
                "job_id": session.job_id,
                "messages": messages,
                "request": step.get("request"),
                "response": step.get("response", ""),
                "step_reward": step.get("step_reward", 0.0),
                "reward": step.get("reward"),
                "meta_json": metadata,
                "is_terminal": bool(step.get("terminated") or step.get("truncated")),
                "is_truncated": bool(step.get("truncated")),
                "is_session_completed": bool(step.get("terminated")),
                "is_trainable": False,
            })
        return list(await self.insert_session_step_rows(rows))

    async def insert_session_step_rows(
        self,
        rows: List[Dict[str, Any]],
    ) -> List[str]:
        """Insert fully constructed logical rows through the configured DAO."""
        if self.storage_type == "cloud" and self._image_processor is None:
            await self.init()
        normalized: List[Dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["job_id"] = str(item.get("job_id") or self.job_id)
            item["meta_json"] = _metadata_object(item.get("meta_json"))
            normalized.append(item)
        if self._image_processor is not None:
            normalized = await self._image_processor.process_rows(normalized)
        return await self._strategy.insert_session_step_rows(normalized)

    async def mark_records_completed(self, record_ids: List[str]) -> int:
        """Compatibility wrapper for exact-ID lifecycle updates."""
        return await self.update_session_step_rows(
            record_ids=record_ids,
            updates={"is_session_completed": True, "is_terminal": True},
        )

    async def list_session_steps(
        self,
        session_id: str,
        *,
        job_id: Optional[str] = None,
        checkout_latest: bool = False,
    ) -> List[Dict[str, Any]]:
        """Return persisted rows for one session in trajectory order."""
        return await self._strategy.list_session_step_rows(SessionStepQuery(
            job_id=job_id or self.job_id or None,
            session_id=session_id,
            checkout_latest=checkout_latest,
        ))

    async def update_session_step_rows(
        self,
        *,
        updates: Dict[str, Any],
        job_id: Optional[str] = None,
        session_id: Optional[str] = None,
        session_ids: Optional[List[str]] = None,
        record_id: Optional[str] = None,
        record_ids: Optional[List[str]] = None,
        step_id: Optional[int] = None,
        llm_model: Optional[str] = None,
    ) -> int:
        """Update rows using explicit caller-owned selection and values."""
        return await self._strategy.update_session_step_rows(
            SessionStepQuery(
                job_id=job_id,
                session_id=session_id,
                session_ids=tuple(session_ids or ()),
                record_id=record_id,
                record_ids=tuple(record_ids or ()),
                step_id=step_id,
                llm_model=llm_model,
            ),
            dict(updates),
        )

    async def record_evaluation_summary(
        self,
        session_id: str,
        step_id: int,
        reward: float,
        env_state: str,
        truncated: bool = False,
    ) -> int:
        """Compatibility wrapper; Evaluator now constructs summary rows directly."""
        record_ids = await self.insert_session_step_rows([{
            "session_id": session_id,
            "env_id": session_id,
            "step_id": step_id,
            "env_name": "gateway",
            "llm_model": "",
            "group_id": "",
            "job_id": self.job_id,
            "messages": [],
            "request": None,
            "response": "",
            "step_reward": reward,
            "reward": reward,
            "meta_json": _metadata_object(env_state),
            "is_terminal": True,
            "is_truncated": truncated,
            "is_session_completed": True,
            "is_trainable": False,
        }])
        return len(record_ids)

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
        return await self._strategy.update_session_step_rows(
            SessionStepQuery(
                job_id=self.job_id or None,
                session_id=session_id,
                step_id=step_id,
            ),
            dict(updates),
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
        updates: Dict[str, Any] = {"job_id": job_id, "env_name": env_name}
        if group_id is not None:
            updates["group_id"] = group_id
        return await self._strategy.update_session_step_rows(
            SessionStepQuery(session_id=session_id),
            updates,
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
        rows = await self.list_session_steps(session_id, checkout_latest=True)
        if llm_model:
            rows = [row for row in rows if row.get("llm_model") == llm_model]
        if not rows:
            return 0
        latest = max(rows, key=lambda row: (
            int(row.get("step_id") or 0),
            str(row.get("created_at") or ""),
            str(row.get("record_id") or row.get("id") or ""),
        ))
        completed = bool(is_session_completed)
        updates: Dict[str, Any] = {
            "is_session_completed": completed,
            "is_terminal": completed if is_terminal is None else bool(is_terminal),
        }
        if not completed:
            updates.update(step_reward=0.0, reward=None)
        return await self.update_session_step_rows(
            job_id=str(latest.get("job_id") or "") or None,
            record_id=str(latest.get("record_id") or latest.get("id")),
            updates=updates,
        )

    async def close(self) -> None:
        """Close the storage strategy"""
        await self._strategy.close()
        self._image_processor = None
    
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


def _metadata_object(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        metadata = dict(value)
    elif not value:
        metadata = {}
    else:
        try:
            parsed = json.loads(value)
        except Exception:
            metadata = {"legacy_metadata": value}
        else:
            metadata = parsed if isinstance(parsed, dict) else {"legacy_metadata": parsed}
    legacy_state = metadata.pop("env_state", None)
    if legacy_state is None:
        return metadata
    legacy_metadata = _metadata_object(legacy_state)
    legacy_metadata.update(metadata)
    return legacy_metadata
