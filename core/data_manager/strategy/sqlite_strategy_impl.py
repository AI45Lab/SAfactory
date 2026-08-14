from core.data_manager.contracts import EnvironmentQuery, SessionContext, SessionStepQuery
from core.data_manager.strategy.base_strategy import StorageStrategy
from core.data_manager.models import JobEnvironment, SessionStep
from core.data_manager.write_buffer import WriteBuffer
from core.perf_trace import PerfTrace
from tortoise import Tortoise
from tortoise.transactions import in_transaction
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

        def ensure_schema() -> None:
            conn = sqlite3.connect(file_path)
            try:
                conn.execute("PRAGMA busy_timeout=30000")
                table_info = list(conn.execute("PRAGMA table_info(session_steps)"))
                columns = {str(row[1]) for row in table_info}
                reward_column = next(
                    (row for row in table_info if str(row[1]) == "reward"),
                    None,
                )
                if reward_column and (bool(reward_column[3]) or reward_column[4] is not None):
                    conn.execute("BEGIN IMMEDIATE")
                    conn.execute("DROP TABLE IF EXISTS session_steps_reward_migration")
                    conn.execute(
                        """
                        CREATE TABLE session_steps_reward_migration (
                            id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
                            session_id VARCHAR(36) NOT NULL,
                            step_id INT NOT NULL,
                            env_name VARCHAR(100) NOT NULL,
                            llm_model VARCHAR(150) NOT NULL,
                            group_id VARCHAR(150),
                            job_id VARCHAR(64),
                            messages TEXT NOT NULL,
                            request TEXT,
                            response TEXT NOT NULL,
                            step_reward REAL NOT NULL DEFAULT 0,
                            reward REAL,
                            env_state TEXT,
                            is_terminal INT NOT NULL DEFAULT 0,
                            is_truncated INT NOT NULL DEFAULT 0,
                            is_session_completed INT NOT NULL DEFAULT 0,
                            is_trainable INT NOT NULL DEFAULT 0,
                            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                            UNIQUE (session_id, step_id, created_at)
                        )
                        """
                    )
                    target_columns = (
                        "id", "session_id", "step_id", "env_name", "llm_model",
                        "group_id", "job_id", "messages", "request", "response",
                        "step_reward", "reward", "env_state", "is_terminal",
                        "is_truncated", "is_session_completed", "is_trainable",
                        "created_at",
                    )
                    missing_defaults = {
                        "id": "NULL",
                        "session_id": "''",
                        "step_id": "0",
                        "env_name": "''",
                        "llm_model": "''",
                        "group_id": "NULL",
                        "job_id": "NULL",
                        "messages": "'[]'",
                        "request": "NULL",
                        "response": "''",
                        "step_reward": "0",
                        "reward": "NULL",
                        "env_state": "NULL",
                        "is_terminal": "0",
                        "is_truncated": "0",
                        "is_session_completed": "0",
                        "is_trainable": "0",
                        "created_at": "CURRENT_TIMESTAMP",
                    }
                    column_list = ", ".join(f'"{column}"' for column in target_columns)
                    nonnull_columns = {
                        "session_id", "step_id", "env_name", "llm_model", "messages",
                        "response", "step_reward", "is_terminal", "is_truncated",
                        "is_session_completed", "is_trainable", "created_at",
                    }
                    select_list = ", ".join(
                        (
                            f'COALESCE("{column}", {missing_defaults[column]})'
                            if column in columns and column in nonnull_columns
                            else f'"{column}"' if column in columns
                            else missing_defaults[column]
                        )
                        for column in target_columns
                    )
                    conn.execute(
                        f"INSERT INTO session_steps_reward_migration ({column_list}) "
                        f"SELECT {select_list} FROM session_steps"
                    )
                    conn.execute("DROP TABLE session_steps")
                    conn.execute(
                        "ALTER TABLE session_steps_reward_migration RENAME TO session_steps"
                    )
                    conn.commit()
                    return
                if "request" not in columns:
                    conn.execute("ALTER TABLE session_steps ADD COLUMN request TEXT")
                conn.commit()
            finally:
                conn.close()

        await asyncio.to_thread(ensure_schema)

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
                    "finished": e.finished,
                    "is_deleted": e.is_deleted,
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
                "finished": env.finished,
                "is_deleted": env.is_deleted,
                "created_at": env.created_at.isoformat() if env.created_at else None,
            }
            trace.emit_summary(status="success", row_count=1, job_id=env.job_id, env_name=env.env_name)
            return result
        except Exception as exc:
            trace.emit_summary(status="failed", error_type=type(exc).__name__, error=str(exc))
            raise

    async def list_environment_rows(self, query: EnvironmentQuery) -> List[Dict[str, Any]]:
        await self.init()
        rows = JobEnvironment.all()
        if query.job_id:
            rows = rows.filter(job_id=query.job_id)
        if query.env_id:
            rows = rows.filter(env_id=query.env_id)
        if query.after_id:
            rows = rows.filter(id__gt=query.after_id)
        if query.finished is not None:
            rows = rows.filter(finished=query.finished)
        if query.is_deleted is not None:
            rows = rows.filter(is_deleted=query.is_deleted)
        rows = rows.order_by("id").offset(query.offset)
        if query.limit is not None:
            rows = rows.limit(query.limit)
        return [self._environment_to_dict(env) for env in await rows]

    async def insert_environment_rows(self, rows: List[Dict[str, Any]]) -> List[str]:
        await self.init()
        if not rows:
            return []
        records: List[JobEnvironment] = []
        env_ids: List[str] = []
        for row in rows:
            env_id = str(row.get("env_id") or uuid.uuid4())
            env_ids.append(env_id)
            records.append(JobEnvironment(
                job_id=str(row.get("job_id") or self.job_id),
                env_id=env_id,
                env_name=str(row.get("env_name") or ""),
                env_params=dict(row.get("env_params") or {}),
                image=str(row.get("image") or ""),
                group_id=str(row.get("group_id") or ""),
                finished=bool(row.get("finished", False)),
                is_deleted=bool(row.get("is_deleted", False)),
            ))
        async with in_transaction():
            await JobEnvironment.bulk_create(records)
        return env_ids

    async def update_environment_rows(
        self,
        query: EnvironmentQuery,
        updates: Dict[str, Any],
    ) -> int:
        await self.init()
        allowed = {"env_name", "env_params", "image", "group_id", "finished", "is_deleted"}
        unknown = set(updates) - allowed
        if unknown:
            raise ValueError(f"Unknown JobEnvironment update fields: {sorted(unknown)}")
        rows = JobEnvironment.all()
        if query.job_id:
            rows = rows.filter(job_id=query.job_id)
        if query.env_id:
            rows = rows.filter(env_id=query.env_id)
        if query.finished is not None:
            rows = rows.filter(finished=query.finished)
        if query.is_deleted is not None:
            rows = rows.filter(is_deleted=query.is_deleted)
        return await rows.update(**updates) if updates else 0

    async def delete_session_step_rows(self, query: SessionStepQuery) -> int:
        await self.init()
        if self._write_buffer:
            await self._write_buffer.flush_model(SessionStep, operation="create")
        rows = SessionStep.all()
        if query.job_id:
            rows = rows.filter(job_id=query.job_id)
        if query.session_id:
            rows = rows.filter(session_id=query.session_id)
        if query.session_ids:
            rows = rows.filter(session_id__in=query.session_ids)
        return await rows.delete()

    async def delete_job_rows(self, job_id: str) -> None:
        await self.init()
        if self._write_buffer:
            await self._write_buffer.flush_model(SessionStep, operation="create")
        async with in_transaction() as connection:
            await SessionStep.filter(job_id=job_id).using_db(connection).delete()
            await JobEnvironment.filter(job_id=job_id).using_db(connection).delete()

    async def mark_environment_finished(self, env_id: str) -> int:
        """Mark one active environment for the current job as finished."""
        await self.init()
        updated = await JobEnvironment.filter(
            job_id=self.job_id,
            env_id=env_id,
            is_deleted=False,
        ).update(finished=True)
        if updated != 1:
            raise RuntimeError(
                f"expected one env config for job_id={self.job_id!r} env_id={env_id!r}, "
                f"updated={updated}"
            )
        return updated

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
        is_trainable: bool = False,
        dataset: Optional[Any] = None,
        reward: Optional[float] = None,
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
            # Build full message history including current response
            full_messages = list(messages)
            # full_messages.append({"role": "assistant", "content": response})

            # Update session's message history
            session.message_history = full_messages

            if dataset is not None:
                state = _json_object(env_state)
                state["dataset"] = dataset
                env_state = json.dumps(state, ensure_ascii=False, default=str)

            # Create step record
            step_record = SessionStep(
                session_id=session.session_id,
                step_id=step_id,
                env_name=session.env_name,
                llm_model=session.llm_model,
                group_id=session.group_id,
                job_id=session.job_id,
                messages=json.dumps(full_messages, ensure_ascii=False),
                request=request,
                response=response,
                step_reward=step_reward,
                reward=reward,
                env_state=env_state,
                is_terminal=terminated or truncated,
                is_truncated=truncated,
                is_session_completed=terminated,
                # Training eligibility is assigned by a later, explicit workflow.
                # Every newly recorded trajectory step starts non-trainable.
                is_trainable=False,
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
            "Recorded step %d for session %s: step_reward=%.4f reward=%s",
            step_id, session.session_id, step_reward, reward,
        )

    async def list_session_steps(
        self,
        session_id: str,
        *,
        checkout_latest: bool = False,
    ) -> List[Dict[str, Any]]:
        await self.init()
        if self._write_buffer:
            await self._write_buffer.flush_model(SessionStep, operation="create")
        rows = await SessionStep.filter(session_id=session_id).order_by("step_id", "id")
        return [self._session_step_to_dict(row) for row in rows]

    async def record_evaluation_summary(
        self,
        session_id: str,
        step_id: int,
        reward: float,
        env_state: str,
        truncated: bool = False,
    ) -> int:
        await self.init()
        row = SessionStep(
            session_id=session_id,
            step_id=step_id,
            env_name="gateway",
            llm_model="",
            group_id="",
            job_id=self.job_id,
            messages="[]",
            response="",
            step_reward=reward,
            reward=reward,
            env_state=env_state,
            is_terminal=True,
            is_truncated=truncated,
            is_session_completed=True,
            is_trainable=False,
        )
        await row.save()
        return 1

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
        *,
        is_session_completed: bool = True,
        is_terminal: Optional[bool] = None,
    ) -> int:
        """Set the completion state of the latest trajectory row for a session."""
        await self.init()
        completed = bool(is_session_completed)
        terminal = completed if is_terminal is None else bool(is_terminal)

        trace = PerfTrace(
            "sqlite_strategy.mark_latest_session_completed",
            logger=log,
            context={
                "operation": "db_write",
                "table": "session_steps",
                "session_id": session_id,
                "model": llm_model,
                "is_session_completed": completed,
                "is_terminal": terminal,
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
            if latest.is_session_completed == completed and latest.is_terminal == terminal:
                trace.emit_summary(
                    status="skipped",
                    candidate_count=len(candidates),
                    updated_count=0,
                    step_id=latest.step_id,
                    row_id=latest.id,
                )
                return 0

            updates: Dict[str, Any] = {
                "is_session_completed": completed,
                "is_terminal": terminal,
            }
            if not completed:
                updates.update(step_reward=0.0, reward=None)
            with trace.span("db_write.mark_session_completed", row_id=latest.id, step_id=latest.step_id):
                updated = await SessionStep.filter(id=latest.id).update(**updates)
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

    @staticmethod
    def _environment_to_dict(env: JobEnvironment) -> Dict[str, Any]:
        return {
            "id": env.id,
            "job_id": env.job_id,
            "env_id": env.env_id,
            "env_name": env.env_name,
            "env_params": env.env_params,
            "image": env.image,
            "group_id": env.group_id,
            "finished": env.finished,
            "is_deleted": env.is_deleted,
            "created_at": env.created_at.isoformat() if env.created_at else None,
        }

    @staticmethod
    def _session_step_to_dict(step: SessionStep) -> Dict[str, Any]:
        return {
            "id": step.id,
            "session_id": step.session_id,
            "step_id": step.step_id,
            "env_name": step.env_name,
            "llm_model": step.llm_model,
            "group_id": step.group_id,
            "job_id": step.job_id,
            "messages": step.messages,
            "request": step.request,
            "response": step.response,
            "step_reward": step.step_reward,
            "reward": step.reward,
            "env_state": step.env_state,
            "is_terminal": step.is_terminal,
            "is_truncated": step.is_truncated,
            "is_session_completed": step.is_session_completed,
            "is_trainable": step.is_trainable,
            "created_at": step.created_at.isoformat() if step.created_at else None,
        }

    async def close(self) -> None:
        """Clean up resources"""
        if self._write_buffer:
            await self._write_buffer.stop()

        if self.initialized:
            await Tortoise.close_connections()
            self.initialized = False

        log.debug("SQLite strategy closed")

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
