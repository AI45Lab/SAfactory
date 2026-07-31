from core.data_manager.strategy.base_strategy import StorageStrategy, SessionContext
from core.data_manager.models import JobEnvironment, SessionStep
from core.data_manager.write_buffer import WriteBuffer
from core.perf_trace import PerfTrace
from tortoise import Tortoise
from typing import List, Dict, Optional, Tuple, Any
import asyncio
import uuid
import json
import sqlite3
import time
import logging
from datetime import datetime

log = logging.getLogger("sqlite_strategy")

RUNTIME_INDEX_SQL = (
    """
    CREATE INDEX IF NOT EXISTS idx_job_environments_job_deleted_id
    ON job_environments(job_id, is_deleted, id)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_session_steps_job_trainable_id
    ON session_steps(job_id, is_trainable, id)
    """,
)

NON_TRAJECTORY_EVENT_TYPES = {
    "gateway_session_close",
    "episode_summary",
    "evaluation_summary",
}


def _json_object(value: Any) -> Dict[str, Any]:
    if not value:
        return {}
    if isinstance(value, dict):
        return dict(value)
    try:
        parsed = json.loads(value)
    except Exception:
        return {"previous_env_state": value}
    return parsed if isinstance(parsed, dict) else {"previous_env_state": parsed}


def _is_trajectory_env_state(value: Any) -> bool:
    event_type = _json_object(value).get("event_type")
    return event_type not in NON_TRAJECTORY_EVENT_TYPES


class SqliteStrategy(StorageStrategy):
    """
    SQLite storage strategy:
    - Table 1 (job_environments): job_id + env_id mapping
    - Table 2 (session_steps): session_id + step_id with full conversation history

    Image handling: Base64 images are stored directly in the messages JSON field.
    This keeps all data self-contained within the SQLite database.
    """
    def __init__(
        self,
        job_id: str,
        db_url: str,
        enable_buffer: bool = True,
        buffer_size: int = 100,
        flush_interval: float = 5.0
    ):
        self.db_url = db_url
        self.job_id = job_id
        self.initialized = False

        self._enable_buffer = enable_buffer
        self._write_buffer: Optional[WriteBuffer] = None
        self._buffer_size = buffer_size
        self._flush_interval = flush_interval

        # Cache for environment configs
        self._env_cache: Dict[str, Dict] = {}

    async def init(self) -> None:
        """Initialize database connection and write buffer"""
        if self.initialized:
            return

        await Tortoise.init(
            db_url=self.db_url,
            modules={"models": ["core.data_manager.models"]}
        )
        await Tortoise.generate_schemas()
        await self._ensure_runtime_schema()
        await self._ensure_runtime_indexes()
        self.initialized = True

        # Initialize write buffer for batched writes
        if self._enable_buffer:
            self._write_buffer = WriteBuffer(
                buffer_size=self._buffer_size,
                flush_interval=self._flush_interval,
                auto_start=True,
                flush_order=[SessionStep]
            )

        log.debug("SQLite strategy initialized: %s", self.db_url)

    async def _ensure_runtime_schema(self) -> None:
        if not self.db_url.startswith("sqlite://"):
            raise ValueError("Only sqlite:// protocol is supported")

        file_path = self.db_url[9:].split("?", 1)[0]

        def add_missing_columns() -> None:
            conn = sqlite3.connect(file_path)
            try:
                columns = {
                    str(row[1])
                    for row in conn.execute("PRAGMA table_info(session_steps)")
                }
                if "request" not in columns:
                    conn.execute("ALTER TABLE session_steps ADD COLUMN request TEXT")
                conn.commit()
            finally:
                conn.close()

        await asyncio.to_thread(add_missing_columns)

    async def _ensure_runtime_indexes(self) -> None:
        if not self.db_url.startswith("sqlite://"):
            raise ValueError("Only sqlite:// protocol is supported")

        file_path = self.db_url[9:].split("?", 1)[0]
        started_at = time.perf_counter()

        def create_indexes() -> None:
            conn = sqlite3.connect(file_path)
            try:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA busy_timeout=5000")
                for sql in RUNTIME_INDEX_SQL:
                    conn.execute(sql)
                conn.commit()
            finally:
                conn.close()

        await asyncio.to_thread(create_indexes)
        elapsed = time.perf_counter() - started_at
        if elapsed >= 1.0:
            log.debug("Ensured SQLite runtime indexes in %.2fs: %s", elapsed, file_path)

    async def add_environment(
        self,
        job_id: str,
        env_name: str,
        env_params: Dict,
        image: str = "",
        group_id: str = ""
    ) -> str:
        """Register a new environment configuration"""
        await self.init()

        env_id = str(uuid.uuid4())

        env_record = JobEnvironment(
            job_id=job_id,
            env_id=env_id,
            env_name=env_name,
            env_params=env_params,
            image=image,
            group_id=group_id
        )
        trace = PerfTrace(
            "sqlite_strategy.add_environment",
            logger=log,
            context={
                "operation": "db_write",
                "table": "job_environments",
                "job_id": job_id,
                "env_id": env_id,
                "env_name": env_name,
            },
        )
        try:
            with trace.span("db_write.job_environment_save", row_count=1):
                await env_record.save()
            trace.emit_summary(status="success", row_count=1)
        except Exception as exc:
            trace.emit_summary(status="failed", error_type=type(exc).__name__, error=str(exc))
            raise

        # Cache the config
        self._env_cache[env_id] = {
            "job_id": job_id,
            "env_id": env_id,
            "env_name": env_name,
            "env_params": env_params,
            "image": image,
            "group_id": group_id
        }

        log.debug("Added environment: %s/%s", env_name, env_id)
        return env_id

    async def get_all_environments(self, job_id: Optional[str] = None) -> List[Dict]:
        """Retrieve all registered environments"""
        await self.init()

        trace = PerfTrace(
            "sqlite_strategy.get_all_environments",
            logger=log,
            context={
                "operation": "db_read",
                "table": "job_environments",
                "job_id": job_id or self.job_id,
                "filter_job_id": bool(job_id),
            },
        )
        try:
            with trace.span("db_read.job_environments_select"):
                if job_id:
                    envs = await JobEnvironment.filter(job_id=job_id)
                else:
                    envs = await JobEnvironment.all()

            rows = [
                {
                    "job_id": e.job_id,
                    "env_id": e.env_id,
                    "env_name": e.env_name,
                    "env_params": e.env_params,
                    "image": e.image,
                    "group_id": e.group_id,
                    "created_at": e.created_at.isoformat() if e.created_at else None
                }
                for e in envs
            ]
            trace.emit_summary(status="success", row_count=len(rows))
            return rows
        except Exception as exc:
            trace.emit_summary(status="failed", error_type=type(exc).__name__, error=str(exc))
            raise

    async def get_environment_by_env_id(self, env_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve one active environment by env_id."""
        await self.init()

        trace = PerfTrace(
            "sqlite_strategy.get_environment_by_env_id",
            logger=log,
            context={
                "operation": "db_read",
                "table": "job_environments",
                "env_id": env_id,
            },
        )
        try:
            with trace.span("db_read.job_environment_select_latest"):
                env = await JobEnvironment.filter(
                    env_id=env_id,
                    is_deleted=False,
                ).order_by("-id").first()
            if env is None:
                trace.emit_summary(status="miss", row_count=0)
                return None

            result = {
                "id": env.id,
                "job_id": env.job_id,
                "env_id": env.env_id,
                "env_name": env.env_name,
                "env_params": env.env_params,
                "image": env.image,
                "group_id": env.group_id,
                "created_at": env.created_at.isoformat() if env.created_at else None,
            }
            trace.emit_summary(status="success", row_count=1, job_id=env.job_id, env_name=env.env_name)
            return result
        except Exception as exc:
            trace.emit_summary(status="failed", error_type=type(exc).__name__, error=str(exc))
            raise

    async def create_session(
        self,
        env_id: str,
        env_name: str,
        llm_model: str,
        group_id: str = "",
        job_id: str = ""
    ) -> SessionContext:
        """Create a new session context (in-memory only)"""
        # session_id = env_id
        session = SessionContext(
            session_id=env_id,
            env_id=env_id,
            env_name=env_name,
            llm_model=llm_model,
            group_id=group_id,
            job_id=job_id or self.job_id,
            total_reward=0.0,
            start_time=time.perf_counter(),
            message_history=[]
        )

        log.debug("Created session: %s for env %s", session.session_id, env_name)
        return session

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
        is_trainable: bool = True
    ) -> None:
        """
        Record a single interaction step.
        Base64 images in messages are stored directly (no extraction).
        """
        await self.init()
        trace = PerfTrace(
            "sqlite_strategy.record_step",
            logger=log,
            context={
                "operation": "db_write",
                "table": "session_steps",
                "session_id": session.session_id,
                "step_id": step_id,
                "model": session.llm_model,
                "job_id": session.job_id,
                "buffered": bool(self._write_buffer),
            },
        )

        try:
            # Update session total reward
            session.total_reward += step_reward

            # Build full message history including current response
            full_messages = list(messages)
            # full_messages.append({"role": "assistant", "content": response})

            # Update session's message history
            session.message_history = full_messages

            # Create step record
            step_record = SessionStep(
                session_id=session.session_id,
                step_id=step_id,
                env_id=session.env_id,
                env_name=session.env_name,
                llm_model=session.llm_model,
                group_id=session.group_id,
                job_id=session.job_id,
                messages=json.dumps(full_messages, ensure_ascii=False),
                request=request,
                response=response,
                step_reward=step_reward,
                reward=session.total_reward,
                env_state=env_state,
                is_terminal=terminated or truncated,
                is_truncated=truncated,
                is_session_completed=terminated or truncated,
                is_trainable=is_trainable,
            )

            # Use buffer or direct save
            if self._write_buffer:
                with trace.span("db_write_buffer.enqueue_create", row_count=1):
                    await self._write_buffer.buffer_create(step_record)
            else:
                with trace.span("db_write.session_step_save", row_count=1):
                    await step_record.save()
            trace.emit_summary(
                status="success",
                row_count=1,
                buffered=bool(self._write_buffer),
                is_terminal=terminated or truncated,
                is_truncated=truncated,
            )
        except Exception as exc:
            trace.emit_summary(status="failed", error_type=type(exc).__name__, error=str(exc))
            raise

        log.debug(
            "Recorded step %d for session %s: reward=%.4f total=%.4f",
            step_id, session.session_id, step_reward, session.total_reward
        )

    async def update_session_step(
        self,
        session_id: str,
        step_id: int,
        updates: Dict[str, Any],
    ) -> int:
        """Update one session_steps row by session_id and step_id."""
        await self.init()

        normalized_updates = self._normalize_session_step_updates(updates)
        if not normalized_updates:
            return 0

        trace = PerfTrace(
            "sqlite_strategy.update_session_step",
            logger=log,
            context={
                "operation": "db_write",
                "table": "session_steps",
                "session_id": session_id,
                "step_id": step_id,
                "field_count": len(normalized_updates),
                "buffered": bool(self._write_buffer),
            },
        )
        try:
            # Make pending buffered creates visible before applying a direct query update.
            if self._write_buffer:
                with trace.span("db_write.flush_pending_creates"):
                    await self._write_buffer.flush_model(SessionStep, operation="create")

            with trace.span("db_write.session_step_update", field_count=len(normalized_updates)):
                updated = await SessionStep.filter(
                    session_id=session_id,
                    step_id=step_id,
                ).update(**normalized_updates)
            trace.emit_summary(status="success", updated_count=updated)
            return updated
        except Exception as exc:
            trace.emit_summary(status="failed", error_type=type(exc).__name__, error=str(exc))
            raise

    async def patch_session_environment(
        self,
        session_id: str,
        *,
        job_id: str,
        env_name: str,
        group_id: Optional[str] = None,
    ) -> int:
        """Patch session rows with their resolved environment metadata."""
        await self.init()

        trace = PerfTrace(
            "sqlite_strategy.patch_session_environment",
            logger=log,
            context={
                "operation": "db_write",
                "table": "session_steps",
                "session_id": session_id,
                "job_id": job_id,
                "env_name": env_name,
                "buffered": bool(self._write_buffer),
            },
        )
        try:
            if self._write_buffer:
                with trace.span("db_write.flush_pending_creates"):
                    await self._write_buffer.flush_model(SessionStep, operation="create")

            updates: Dict[str, Any] = {
                "job_id": job_id,
                "env_name": env_name,
            }
            if group_id is not None:
                updates["group_id"] = group_id

            with trace.span("db_write.patch_session_environment", field_count=len(updates)):
                updated = await SessionStep.filter(session_id=session_id).update(**updates)
            trace.emit_summary(status="success", updated_count=updated)
            return updated
        except Exception as exc:
            trace.emit_summary(status="failed", error_type=type(exc).__name__, error=str(exc))
            raise

    async def mark_latest_session_completed(
        self,
        session_id: str,
        llm_model: Optional[str] = None,
    ) -> int:
        """Mark the latest trajectory row for a session as completed."""
        await self.init()

        trace = PerfTrace(
            "sqlite_strategy.mark_latest_session_completed",
            logger=log,
            context={
                "operation": "db_write",
                "table": "session_steps",
                "session_id": session_id,
                "model": llm_model,
                "buffered": bool(self._write_buffer),
            },
        )
        try:
            # Make pending buffered creates visible before selecting the latest row.
            if self._write_buffer:
                with trace.span("db_write.flush_pending_creates"):
                    await self._write_buffer.flush_model(SessionStep, operation="create")

            query = SessionStep.filter(session_id=session_id)
            if llm_model:
                query = query.filter(llm_model=llm_model)

            with trace.span("db_read.select_latest_session_step", limit=50):
                candidates = await query.order_by("-step_id", "-id").limit(50)
            if not candidates:
                trace.emit_summary(status="miss", candidate_count=0, updated_count=0)
                return 0

            latest = next(
                (step for step in candidates if _is_trajectory_env_state(step.env_state)),
                candidates[0],
            )
            if latest.is_session_completed and latest.is_terminal:
                trace.emit_summary(
                    status="skipped",
                    candidate_count=len(candidates),
                    updated_count=0,
                    step_id=latest.step_id,
                    row_id=latest.id,
                )
                return 0

            with trace.span("db_write.mark_session_completed", row_id=latest.id, step_id=latest.step_id):
                updated = await SessionStep.filter(id=latest.id).update(
                    is_session_completed=True,
                    is_terminal=True,
                )
            trace.emit_summary(
                status="success",
                candidate_count=len(candidates),
                updated_count=updated,
                step_id=latest.step_id,
                row_id=latest.id,
            )
            return updated
        except Exception as exc:
            trace.emit_summary(status="failed", error_type=type(exc).__name__, error=str(exc))
            raise

    def _normalize_session_step_updates(self, updates: Dict[str, Any]) -> Dict[str, Any]:
        if not updates:
            return {}

        valid_fields = set(SessionStep._meta.fields_map.keys())
        blocked_fields = {"id", "created_at"}
        normalized: Dict[str, Any] = {}

        for field, value in updates.items():
            if field not in valid_fields:
                raise ValueError(f"Unknown SessionStep field for update: {field}")
            if field in blocked_fields:
                raise ValueError(f"SessionStep field cannot be updated: {field}")

            if field in {"messages", "request"} and not isinstance(value, str):
                value = json.dumps(value, ensure_ascii=False)
            elif field == "env_state" and isinstance(value, (dict, list)):
                value = json.dumps(value, ensure_ascii=False)

            normalized[field] = value

        return normalized

    async def close(self) -> None:
        """Clean up resources"""
        if self._write_buffer:
            await self._write_buffer.stop()

        if self.initialized:
            await Tortoise.close_connections()
            self.initialized = False

        log.debug("SQLite strategy closed")

    def get_sync_connection(self) -> sqlite3.Connection:
        """Get raw SQLite connection for direct queries"""
        if not self.db_url.startswith("sqlite://"):
            raise ValueError("Only sqlite:// protocol is supported")

        file_path = self.db_url[9:].split("?", 1)[0]
        conn = sqlite3.connect(file_path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.row_factory = sqlite3.Row
        return conn

    @property
    def buffer_stats(self) -> Optional[dict]:
        """Get buffer statistics"""
        return self._write_buffer.stats if self._write_buffer else None

    async def fetch_done_steps_with_context(
        self,
        job_id: str,
        after_id: int = 0,
        limit: int = 100
    ) -> List[Dict]:
        """
        Fetch completed steps for training data collection.
        Uses cursor-based pagination.
        """
        await self.init()

        trace = PerfTrace(
            "sqlite_strategy.fetch_done_steps_with_context",
            logger=log,
            context={
                "operation": "db_read",
                "table": "session_steps",
                "job_id": job_id,
                "after_id": after_id,
                "limit": limit,
            },
        )
        try:
            with trace.span("db_read.fetch_done_steps", limit=limit):
                steps = await SessionStep.filter(
                    job_id=job_id,
                    is_trainable=True,
                    id__gt=after_id
                ).order_by("id").limit(limit)

            rows = [
                {
                    "step_pk": s.id,
                    "step_id": s.step_id,
                    "env_name": s.env_name,
                    "env_id": s.session_id,
                    "env_state": s.env_state,
                    "prompt": s.messages,
                    "request": s.request,
                    "response": s.response,
                    "reward": s.step_reward,
                    "step_reward": s.step_reward,
                    "total_reward": s.reward,
                    "session_id": s.session_id,
                    "session_end_time": s.created_at.isoformat() if s.created_at else None,
                    "group_id": s.group_id,
                    "truncated": s.is_truncated,
                    "is_session_completed": s.is_session_completed,
                }
                for s in steps
            ]
            trace.emit_summary(status="success", row_count=len(rows))
            return rows
        except Exception as exc:
            trace.emit_summary(status="failed", error_type=type(exc).__name__, error=str(exc))
            raise

    async def get_max_step_id(self, job_id: str) -> int:
        """Get maximum primary key for pagination"""
        await self.init()
        trace = PerfTrace(
            "sqlite_strategy.get_max_step_id",
            logger=log,
            context={
                "operation": "db_read",
                "table": "session_steps",
                "job_id": job_id,
            },
        )
        try:
            with trace.span("db_read.max_terminal_step_id"):
                latest = await SessionStep.filter(job_id=job_id, is_terminal=True).order_by("-id").first()
            max_id = latest.id if latest else 0
            trace.emit_summary(status="success", row_count=1 if latest else 0, max_step_id=max_id)
            return max_id
        except Exception as exc:
            trace.emit_summary(status="failed", error_type=type(exc).__name__, error=str(exc))
            raise
