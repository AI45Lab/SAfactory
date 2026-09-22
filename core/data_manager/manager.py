import json
import logging
import time
from typing import Optional, List, Dict, Any

from core.data_manager.contracts import EnvironmentQuery, SessionContext, SessionStepQuery
from core.data_manager.strategy.base_strategy import StorageStrategy
from core.data_manager.strategy_factory import StorageFactory
from core.data_manager.upsert_batcher import SessionStepUpsertBatcher

log = logging.getLogger("core.data_manager.manager")


class DataManager:
    """
    Unified data manager that delegates to storage strategies.
    """

    def __init__(
        self,
        job_id: str,
        storage_type: str = "sqlite",
        *,
        enable_upsert_batching: bool = False,
        upsert_batch_size: int = 100,
        upsert_flush_interval: float = 10.0,
        upsert_queue_size: int = 1000,
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

        self._upsert_batcher = (
            SessionStepUpsertBatcher(
                self._strategy.upsert_session_step_rows,
                batch_size=upsert_batch_size,
                flush_interval=upsert_flush_interval,
                queue_size=upsert_queue_size,
            )
            if enable_upsert_batching
            else None
        )

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

    async def clear_environment_cache(self, env_ids: List[str]) -> int:
        """Clear backend-local environment cache entries."""
        return await self._strategy.clear_environment_cache(env_ids)

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

    async def list_environment_refs(
        self,
        *,
        job_id: Optional[str] = None,
        after_id: int = 0,
        limit: Optional[int] = None,
        finished: Optional[bool] = None,
        is_deleted: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        """Query only the environment identity fields needed by resume cleanup."""
        return await self._strategy.list_environment_refs(EnvironmentQuery(
            job_id=job_id or self.job_id,
            after_id=max(0, int(after_id)),
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
        metadata = _metadata_object(meta_json)
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
            metadata = _metadata_object(step.get("meta_json"))
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
        normalized: List[Dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["job_id"] = str(item.get("job_id") or self.job_id)
            item["meta_json"] = _metadata_object(item.get("meta_json"))
            normalized.append(item)
        return await self._strategy.insert_session_step_rows(normalized)

    async def upsert_session_step_rows(
        self,
        rows: List[Dict[str, Any]],
    ) -> List[str]:
        """Upsert fully constructed logical rows through the configured DAO."""
        normalized: List[Dict[str, Any]] = []
        seen_keys: set[tuple[str, str]] = set()
        for row in rows:
            item = dict(row)
            item["job_id"] = str(item.get("job_id") or self.job_id)
            item["record_id"] = str(item.get("record_id") or "")
            if not item["job_id"] or not item["record_id"]:
                raise ValueError("session-step upsert requires non-empty job_id and record_id")
            key = (item["job_id"], item["record_id"])
            if key in seen_keys:
                raise ValueError(f"duplicate session-step upsert key: {key!r}")
            seen_keys.add(key)
            item["meta_json"] = _metadata_object(item.get("meta_json"))
            normalized.append(item)
        if self._upsert_batcher is not None:
            return await self._upsert_batcher.submit(normalized)
        return await self._strategy.upsert_session_step_rows(normalized)

    async def patch_session_step_rows(
        self,
        rows: List[Dict[str, Any]],
    ) -> List[str]:
        """Patch explicitly supplied reward fields on existing session-step rows."""
        identity_fields = {"job_id", "record_id", "dataset_type"}
        reward_fields = {
            "step_reward", "reward", "meta_json", "is_terminal",
            "is_truncated", "is_session_completed",
        }
        normalized: List[Dict[str, Any]] = []
        seen_keys: set[tuple[str, str]] = set()
        field_shape: set[str] | None = None
        for row in rows:
            item = dict(row)
            unknown = set(item) - identity_fields - reward_fields
            if unknown:
                raise ValueError(f"Unknown session-step patch fields: {sorted(unknown)}")
            supplied = set(item) & reward_fields
            if not supplied:
                raise ValueError("session-step patch requires at least one reward field")
            if field_shape is not None and supplied != field_shape:
                raise ValueError("session-step patches in one batch must use the same fields")
            field_shape = supplied
            item["job_id"] = str(item.get("job_id") or self.job_id)
            item["record_id"] = str(item.get("record_id") or "")
            if not item["job_id"] or not item["record_id"]:
                raise ValueError("session-step patch requires non-empty job_id and record_id")
            key = (item["job_id"], item["record_id"])
            if key in seen_keys:
                raise ValueError(f"duplicate session-step patch key: {key!r}")
            seen_keys.add(key)
            if "meta_json" in item:
                item["meta_json"] = _metadata_object(item["meta_json"])
            normalized.append(item)
        return await self._strategy.patch_session_step_rows(normalized)

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
        record_id: Optional[str] = None,
        record_ids: Optional[List[str]] = None,
        step_id: Optional[int] = None,
        llm_model: Optional[str] = None,
        limit: Optional[int] = None,
        latest_first: bool = False,
        columns: Optional[List[str]] = None,
        checkout_latest: bool = False,
    ) -> List[Dict[str, Any]]:
        """Return persisted rows for one session in the requested order."""
        return await self._strategy.list_session_step_rows(SessionStepQuery(
            job_id=job_id or self.job_id or None,
            session_id=session_id,
            record_id=record_id,
            record_ids=tuple(record_ids or ()),
            step_id=step_id,
            llm_model=llm_model,
            limit=None if limit is None else max(0, int(limit)),
            latest_first=bool(latest_first),
            columns=tuple(columns or ()),
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
        try:
            if self._upsert_batcher is not None:
                await self._upsert_batcher.close()
        finally:
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


def _metadata_object(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if not value:
        return {}
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("meta_json must contain a JSON object")
    return parsed
