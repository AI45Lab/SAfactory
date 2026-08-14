from core.data_manager.contracts import EnvironmentQuery, SessionStepQuery
from core.data_manager.strategy.base_strategy import StorageStrategy
from core.data_manager.models import JobEnvironment, SessionStep
from core.data_manager.write_buffer import WriteBuffer
from core.perf_trace import PerfTrace
from tortoise import Tortoise
from tortoise.transactions import in_transaction
from typing import List, Dict, Optional, Any
import asyncio
import uuid
import json
import sqlite3
import time
import logging

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


def _json_object(value: Any) -> Dict[str, Any]:
    if not value:
        return {}
    if isinstance(value, dict):
        metadata = dict(value)
    else:
        try:
            parsed = json.loads(value)
        except Exception:
            metadata = {"legacy_meta_json": value}
        else:
            metadata = parsed if isinstance(parsed, dict) else {"legacy_meta_json": parsed}
    legacy_state = metadata.pop("env_state", None)
    if legacy_state is None:
        return metadata
    legacy_metadata = _json_object(legacy_state)
    legacy_metadata.update(metadata)
    return legacy_metadata


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
                needs_rebuild = bool(table_info) and (
                    "record_id" not in columns
                    or "meta_json" not in columns
                    or "env_state" in columns
                    or "request" not in columns
                    or reward_column is None
                    or bool(reward_column[3])
                    or reward_column[4] is not None
                )
                if not needs_rebuild:
                    return

                conn.execute("BEGIN IMMEDIATE")
                conn.execute("DROP TABLE IF EXISTS session_steps_schema_migration")
                conn.execute(
                    """
                    CREATE TABLE session_steps_schema_migration (
                        id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
                        record_id VARCHAR(64) NOT NULL UNIQUE,
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
                        meta_json TEXT,
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
                    "id", "record_id", "session_id", "step_id", "env_name",
                    "llm_model", "group_id", "job_id", "messages", "request",
                    "response", "step_reward", "reward", "meta_json", "is_terminal",
                    "is_truncated", "is_session_completed", "is_trainable", "created_at",
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
                    "is_terminal": "0",
                    "is_truncated": "0",
                    "is_session_completed": "0",
                    "is_trainable": "0",
                    "created_at": "CURRENT_TIMESTAMP",
                }
                record_id_expr = (
                    "COALESCE(NULLIF(\"record_id\", ''), 'legacy-' || CAST(\"id\" AS TEXT))"
                    if "record_id" in columns
                    else "'legacy-' || CAST(\"id\" AS TEXT)"
                )
                if "meta_json" in columns and "env_state" in columns:
                    meta_json_expr = "COALESCE(NULLIF(\"meta_json\", ''), \"env_state\")"
                elif "meta_json" in columns:
                    meta_json_expr = '"meta_json"'
                elif "env_state" in columns:
                    meta_json_expr = '"env_state"'
                else:
                    meta_json_expr = "NULL"
                nonnull_columns = {
                    "session_id", "step_id", "env_name", "llm_model", "messages",
                    "response", "step_reward", "is_terminal", "is_truncated",
                    "is_session_completed", "is_trainable", "created_at",
                }
                select_expressions = []
                for column in target_columns:
                    if column == "record_id":
                        expression = record_id_expr
                    elif column == "meta_json":
                        expression = meta_json_expr
                    elif column in columns and column in nonnull_columns:
                        expression = f'COALESCE("{column}", {missing_defaults[column]})'
                    elif column in columns:
                        expression = f'"{column}"'
                    else:
                        expression = missing_defaults[column]
                    select_expressions.append(expression)
                column_list = ", ".join(f'"{column}"' for column in target_columns)
                conn.execute(
                    f"INSERT INTO session_steps_schema_migration ({column_list}) "
                    f"SELECT {', '.join(select_expressions)} FROM session_steps"
                )
                conn.execute("DROP TABLE session_steps")
                conn.execute("ALTER TABLE session_steps_schema_migration RENAME TO session_steps")
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
        if query.record_id:
            rows = rows.filter(record_id=query.record_id)
        if query.record_ids:
            rows = rows.filter(record_id__in=query.record_ids)
        return await rows.delete()

    async def delete_job_rows(self, job_id: str) -> None:
        await self.init()
        if self._write_buffer:
            await self._write_buffer.flush_model(SessionStep, operation="create")
        async with in_transaction() as connection:
            await SessionStep.filter(job_id=job_id).using_db(connection).delete()
            await JobEnvironment.filter(job_id=job_id).using_db(connection).delete()

    async def insert_session_step_rows(
        self,
        rows: List[Dict[str, Any]],
    ) -> List[str]:
        """Persist caller-constructed rows without applying lifecycle policy."""
        await self.init()
        if not rows:
            return []

        records: List[SessionStep] = []
        record_ids: List[str] = []
        for row in rows:
            record_id = str(row.get("record_id") or uuid.uuid4())
            record_ids.append(record_id)
            messages = row.get("messages", [])
            request = row.get("request")
            response = row.get("response", "")
            meta_json = _json_object(row.get("meta_json"))
            records.append(SessionStep(
                record_id=record_id,
                session_id=str(row.get("session_id") or ""),
                step_id=int(row.get("step_id") or 0),
                env_name=str(row.get("env_name") or ""),
                llm_model=str(row.get("llm_model") or ""),
                group_id=str(row.get("group_id") or ""),
                job_id=str(row.get("job_id") or self.job_id),
                messages=(
                    messages
                    if isinstance(messages, str)
                    else json.dumps(messages, ensure_ascii=False, default=str)
                ),
                request=(
                    request
                    if request is None or isinstance(request, str)
                    else json.dumps(request, ensure_ascii=False, default=str)
                ),
                response=(
                    response
                    if isinstance(response, str)
                    else json.dumps(response, ensure_ascii=False, default=str)
                ),
                step_reward=float(row.get("step_reward") or 0.0),
                reward=row.get("reward"),
                meta_json=json.dumps(meta_json, ensure_ascii=False, default=str),
                is_terminal=bool(row.get("is_terminal", False)),
                is_truncated=bool(row.get("is_truncated", False)),
                is_session_completed=bool(row.get("is_session_completed", False)),
                is_trainable=bool(row.get("is_trainable", False)),
            ))

        if self._write_buffer:
            for record in records:
                await self._write_buffer.buffer_create(record)
        else:
            await SessionStep.bulk_create(records)
        return record_ids

    async def list_session_step_rows(
        self,
        query: SessionStepQuery,
    ) -> List[Dict[str, Any]]:
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
        if query.record_id:
            rows = rows.filter(record_id=query.record_id)
        if query.record_ids:
            rows = rows.filter(record_id__in=query.record_ids)
        if query.step_id is not None:
            rows = rows.filter(step_id=query.step_id)
        if query.llm_model:
            rows = rows.filter(llm_model=query.llm_model)
        if query.after_id:
            rows = rows.filter(id__gt=query.after_id)
        if query.is_terminal is not None:
            rows = rows.filter(is_terminal=query.is_terminal)
        if query.is_trainable is not None:
            rows = rows.filter(is_trainable=query.is_trainable)
        rows = rows.order_by("step_id", "id")
        if query.limit is not None:
            rows = rows.limit(query.limit)
        return [self._session_step_to_dict(row) for row in await rows]

    async def update_session_step_rows(
        self,
        query: SessionStepQuery,
        updates: Dict[str, Any],
    ) -> int:
        await self.init()
        normalized = self._normalize_session_step_updates(updates)
        if not normalized:
            return 0
        if self._write_buffer:
            await self._write_buffer.flush_model(SessionStep, operation="create")
        rows = SessionStep.all()
        if query.job_id:
            rows = rows.filter(job_id=query.job_id)
        if query.session_id:
            rows = rows.filter(session_id=query.session_id)
        if query.session_ids:
            rows = rows.filter(session_id__in=query.session_ids)
        if query.record_id:
            rows = rows.filter(record_id=query.record_id)
        if query.record_ids:
            rows = rows.filter(record_id__in=query.record_ids)
        if query.step_id is not None:
            rows = rows.filter(step_id=query.step_id)
        if query.llm_model:
            rows = rows.filter(llm_model=query.llm_model)
        if query.is_terminal is not None:
            rows = rows.filter(is_terminal=query.is_terminal)
        if query.is_trainable is not None:
            rows = rows.filter(is_trainable=query.is_trainable)
        return await rows.update(**normalized)

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

            if field in {"messages", "request", "response"} and not isinstance(value, str):
                value = json.dumps(value, ensure_ascii=False)
            elif field == "meta_json":
                value = json.dumps(_json_object(value), ensure_ascii=False)

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
            "record_id": step.record_id,
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
            "meta_json": _json_object(step.meta_json),
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
                    # Kept as a derived compatibility key because rl/buffer_server.py
                    # intentionally remains unchanged in this refactor.
                    "env_state": s.meta_json,
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
