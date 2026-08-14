import asyncio
from importlib import import_module
import json
import logging
import os
import time
import uuid
import re
import base64
import tempfile
from typing import List, Dict, Optional, Any, Set
from datetime import date

from core.data_manager.contracts import EnvironmentQuery, SessionContext, SessionStepQuery
from core.data_manager.strategy.base_strategy import StorageStrategy
from core.perf_trace import PerfTrace

log = logging.getLogger("cloud_strategy")

WTGatewayClient = None
GatewayConfig = None
EnvConfigManager = None
LandingRecord = None
ChatMessage = None
ContentItem = None
generate_deterministic_id = None
S3Uploader = None
S3Downloader = None

_WT_SDK_IMPORT_ERROR = (
    "Cloud storage requires the optional private dependency "
    "wt-data-platform-sdk, which provides the wt_sdk package. "
    "Install the cloud dependency after configuring repository credentials "
    "or run with --storage-type sqlite to avoid this dependency."
)


def _load_wt_sdk() -> None:
    """Load wt_sdk only when cloud storage is actually used."""
    global WTGatewayClient
    global GatewayConfig
    global EnvConfigManager
    global LandingRecord
    global ChatMessage
    global ContentItem
    global generate_deterministic_id
    global S3Uploader
    global S3Downloader

    if all(
        symbol is not None
        for symbol in (
            WTGatewayClient,
            GatewayConfig,
            EnvConfigManager,
            LandingRecord,
            generate_deterministic_id,
            S3Uploader,
            S3Downloader,
        )
    ):
        return

    if WTGatewayClient is not None and EnvConfigManager is not None:
        _install_mock_wt_sdk_fallbacks()
        if all(
            symbol is not None
            for symbol in (
                WTGatewayClient,
                GatewayConfig,
                EnvConfigManager,
                LandingRecord,
                generate_deterministic_id,
                S3Uploader,
                S3Downloader,
            )
        ):
            return

    try:
        wt_sdk = import_module("wt_sdk")
        wt_sdk_models = import_module("wt_sdk.models")
        wt_sdk_utils = import_module("wt_sdk.utils")
    except ModuleNotFoundError as exc:
        raise RuntimeError(_WT_SDK_IMPORT_ERROR) from exc

    WTGatewayClient = WTGatewayClient or wt_sdk.WTGatewayClient
    GatewayConfig = GatewayConfig or wt_sdk.GatewayConfig
    EnvConfigManager = EnvConfigManager or wt_sdk.EnvConfigManager
    LandingRecord = LandingRecord or wt_sdk_models.LandingRecord
    ChatMessage = ChatMessage or getattr(wt_sdk_models, "ChatMessage", None)
    ContentItem = ContentItem or getattr(wt_sdk_models, "ContentItem", None)
    generate_deterministic_id = generate_deterministic_id or wt_sdk_utils.generate_deterministic_id
    S3Uploader = S3Uploader or wt_sdk_utils.S3Uploader
    S3Downloader = S3Downloader or wt_sdk_utils.S3Downloader

    try:
        LandingRecord(
            dataset_type="RL",
            id="__json_schema_check__",
            created_at=0,
            messages="[]",
            response="{}",
            meta_json="{}",
        )
    except Exception as exc:
        LandingRecord = None
        raise RuntimeError(
            "Installed wt_sdk uses the legacy LandingRecord schema. "
            "Cloud storage requires messages/response/meta_json JSON strings; "
            "reinstall requirements-cloud.txt with --upgrade --force-reinstall. "
            f"Loaded wt_sdk version={getattr(wt_sdk, '__version__', 'unknown')} "
            f"from {getattr(wt_sdk, '__file__', 'unknown')}"
        ) from exc


def _install_mock_wt_sdk_fallbacks() -> None:
    """Provide tiny SDK-like objects for tests that monkeypatch cloud clients."""
    global GatewayConfig
    global LandingRecord
    global ChatMessage
    global ContentItem
    global generate_deterministic_id
    global S3Uploader
    global S3Downloader

    class _Model:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                self.__dict__[key] = value

        def __getattr__(self, _name: str) -> Any:
            return None

        def model_dump(self) -> Dict[str, Any]:
            return dict(self.__dict__)

    class _TableConfig:
        def __init__(
            self,
            db_uri: str = "",
            landing_table: str = "",
            serving_table: str = "",
        ):
            self.db_uri = db_uri
            self.landing_table = landing_table
            self.serving_table = serving_table

    class _S3Config:
        def to_storage_options(self) -> Dict[str, Any]:
            return {}

    class _GatewayConfig:
        def __init__(self, tables: Any = None, **_kwargs):
            self.tables = tables or _TableConfig()
            self.s3 = _S3Config()

    def _deterministic_id(value: Dict[str, Any]) -> str:
        payload = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
        return str(uuid.uuid5(uuid.NAMESPACE_URL, payload))

    class _S3Uploader:
        def upload_file(self, file_path: str, key: str) -> str:
            return f"s3://mock/{key}"

    class _S3Downloader:
        def download_file(self, image_path: str, local_path: str) -> None:
            raise RuntimeError(f"Mock S3Downloader cannot download {image_path!r}")

    GatewayConfig = GatewayConfig or _GatewayConfig
    LandingRecord = LandingRecord or _Model
    ChatMessage = ChatMessage or _Model
    ContentItem = ContentItem or _Model
    generate_deterministic_id = generate_deterministic_id or _deterministic_id
    S3Uploader = S3Uploader or _S3Uploader
    S3Downloader = S3Downloader or _S3Downloader


# Retry configuration
MAX_UPLOAD_RETRIES = 3
RETRY_BACKOFF_BASE = 1.0
NON_TRAJECTORY_EVENT_TYPES = {
    "gateway_session_close",
    "episode_summary",
    "evaluation_summary",
}
CLOUD_DATASET_TYPE = "RL"


def _json_value(value: Any, default: Any) -> Any:
    """Accept SDK JSON strings and already-deserialized Python values."""
    if value is None:
        return default
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return default
    return default


def _escape_sql_literal(value: str) -> str:
    return value.replace("'", "''")


def _cloud_env_state_from_meta(meta_json: Any) -> Dict[str, Any]:
    meta = _json_value(meta_json, {})
    if not isinstance(meta, dict):
        return {}
    env_state = _json_value(meta.get("env_state"), {})
    return env_state if isinstance(env_state, dict) else {}


def _is_trajectory_meta_json(meta_json: Any) -> bool:
    event_type = _cloud_env_state_from_meta(meta_json).get("event_type")
    return event_type not in NON_TRAJECTORY_EVENT_TYPES


def _truthy_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    try:
        if value != value:
            return False
    except Exception:
        pass
    return bool(value)


def _response_text(value: Any) -> str:
    """Extract training text without discarding non-text model output."""
    if value is None:
        return ""
    if isinstance(value, str):
        parsed = _json_value(value, None)
        if parsed is None:
            return value
        value = parsed

    def extract_text(payload: Any) -> str:
        if isinstance(payload, str):
            return payload
        if isinstance(payload, list):
            return "".join(extract_text(item) for item in payload)
        if not isinstance(payload, dict):
            return ""

        content = payload.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            chunks = []
            for item in content:
                if isinstance(item, str):
                    chunks.append(item)
                elif isinstance(item, dict):
                    text = item.get("text")
                    if not isinstance(text, str):
                        text = item.get("content")
                    if isinstance(text, str):
                        chunks.append(text)
            return "".join(chunks)

        output = payload.get("output")
        if isinstance(output, (dict, list)):
            return extract_text(output)
        choices = payload.get("choices")
        if isinstance(choices, list):
            return "".join(
                extract_text(choice.get("message"))
                for choice in choices
                if isinstance(choice, dict)
            )
        for key in ("output_text", "text"):
            text = payload.get(key)
            if isinstance(text, str):
                return text
        return ""

    text = extract_text(value)
    if text:
        return text
    if isinstance(value, dict) and value.get("content") == "":
        has_non_text_output = any(
            item not in (None, "", [], {})
            for key, item in value.items()
            if key not in {"role", "content", "name"}
        )
        if not has_non_text_output:
            return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, default=str)
    return str(value)




class CloudStrategy(StorageStrategy):
    """
    Cloud storage strategy:
    - Table 1 (S3): Environment configs stored via EnvConfigManager
    - Table 2 (LandingTable): Session steps with full conversation history

    Image handling:
    - Extract base64 images from messages
    - Upload binary to S3 with retry logic
    - On failure: store locally as fallback
    - Store S3 URLs (or local paths) in messages JSON
    """

    def __init__(
        self,
        job_id: str,
        db_url: Optional[str] = None,
        enable_buffer: bool = False,
        buffer_size: int = 1,
        flush_interval: float = 1.0,
        landing_table: Optional[str] = None,
        serving_table: Optional[str] = None,
        env_config_table: str = "evaluation_env_config",
        dldb_model: Optional[str] = None,
        enable_dldb_timing_logs: bool = False,
        dldb_metrics_log_path: Optional[str] = None,
    ):
        self.db_url = str(db_url or "").strip()
        self.job_id = job_id
        self.initialized = False
        self.landing_table = str(landing_table or "").strip() or None
        self.serving_table = str(serving_table or "").strip() or None
        self.env_config_table = env_config_table
        self.dldb_model = dldb_model
        self.enable_dldb_timing_logs = enable_dldb_timing_logs
        self.dldb_metrics_log_path = dldb_metrics_log_path

        self.client: Any = None
        self.env_manager: Any = None
        self.s3_uploader: Any = None

        self._enable_buffer = enable_buffer
        self._buffer_size = buffer_size
        self._flush_interval = flush_interval
        self._record_buffer: List[Any] = []
        self._buffer_lock = asyncio.Lock()
        self._flush_lock = asyncio.Lock()
        self._flush_task: Optional[asyncio.Task] = None
        self._running = False
        self._background_tasks: Set[asyncio.Task] = set()
        self._stats = {
            "total_create_buffered": 0,
            "total_create_flushed": 0,
            "flush_count": 0,
        }

        # In-memory caches
        self._env_configs: Dict[str, Dict] = {}
        self._sessions: Dict[str, SessionContext] = {}
        self._record_job_ids: Dict[str, str] = {}

        # Local fallback directory for failed uploads
        self._local_fallback_dir = "saved_images"

    async def init(self):
        """Initialize cloud clients"""
        if self.initialized:
            return

        os.environ.setdefault("AWS_EC2_METADATA_DISABLED", "true")
        _load_wt_sdk()

        # 1. Initialize WTGatewayClient
        config = GatewayConfig(
            dldb_model=self.dldb_model,
            enable_dldb_timing_logs=self.enable_dldb_timing_logs,
            dldb_metrics_log_path=self.dldb_metrics_log_path,
        )
        if self.db_url:
            config.tables.db_uri = self.db_url
        if self.landing_table:
            config.tables.landing_table = self.landing_table
        if self.serving_table:
            config.tables.serving_table = self.serving_table
        self.landing_table = config.tables.landing_table
        self.serving_table = config.tables.serving_table

        try:
            self.client = WTGatewayClient(config)
            log.debug(
                "CloudStrategy initialized with landing_table=%s serving_table=%s db_uri=%s",
                config.tables.landing_table,
                config.tables.serving_table,
                config.tables.db_uri,
            )
        except Exception as e:
            log.error(f"Failed to initialize WTGatewayClient: {e}")
            raise

        # 2. Initialize EnvConfigManager (S3)
        try:
            self.env_manager = EnvConfigManager(
                table_name=self.env_config_table,
                storage_options=config.s3.to_storage_options(),
                dldb_model=self.dldb_model,
                enable_dldb_timing_logs=self.enable_dldb_timing_logs,
                dldb_metrics_log_path=self.dldb_metrics_log_path,
            )
        except Exception as e:
            log.error(f"Failed to initialize EnvConfigManager: {e}")
            raise
        
        # 3. Initialize S3Uploader
        try:
            self.s3_uploader = S3Uploader()
        except Exception as e:
            log.warning(f"Failed to initialize S3Uploader: {e}")
            self.s3_uploader = None

        self.initialized = True

        if self._enable_buffer and not self._running:
            self._running = True
            self._flush_task = asyncio.create_task(self._periodic_flush())
            log.debug(
                "Cloud buffer started: buffer_size=%d flush_interval=%.2fs",
                self._buffer_size,
                self._flush_interval,
            )

    async def _timed_db_call(
        self,
        sdk_operation: str,
        func: Any,
        *args: Any,
        trace_context: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """Run one synchronous DB SDK call in a thread and log its client-side latency."""
        trace = PerfTrace(
            "cloud_strategy.db_sdk_call",
            logger=log,
            context={
                "operation": "db_read" if sdk_operation.startswith("filter") else "db_write",
                "sdk_operation": sdk_operation,
                "table": self.landing_table,
                **(trace_context or {}),
            },
        )
        try:
            with trace.span("db_sdk.call"):
                result = await asyncio.to_thread(func, *args, **kwargs)
            trace.emit_summary(status="success")
            return result
        except asyncio.CancelledError:
            trace.emit_summary(status="cancelled")
            raise
        except Exception as exc:
            trace.emit_summary(status="failed", error_type=type(exc).__name__, error=str(exc))
            raise

    async def add_environment(
        self,
        job_id: str,
        env_name: str,
        env_params: Dict,
        image: str = "",
        group_id: str = ""
    ) -> str:
        """Register environment config to S3"""
        await self.init()
        
        env_id = str(uuid.uuid4())
        
        config_dict = {
            "job_id": job_id,
            "env_id": env_id,
            "env_name": env_name,
            "env_params": env_params,
            "image": image,
            "group_id": group_id,
            "created_at": int(time.time()),
        }
        
        try:
            await asyncio.to_thread(self.env_manager.save_config, config_dict)
            log.debug("Environment config saved to S3: %s", env_id)
        except Exception as e:
            log.error(f"Failed to save env config to S3: {e}")
            
        # Cache locally
        self._env_configs[env_id] = config_dict

        return env_id

    async def get_all_environments(self, job_id: Optional[str] = None) -> List[Dict]:
        """Get all environments from cache"""
        return self._list_env_configs(job_id=job_id)

    async def get_environment_by_env_id(self, env_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve one environment config from cache or cloud EnvConfigManager."""
        await self.init()

        cached = self._env_configs.get(env_id)
        if cached is not None:
            return dict(cached)

        if self.env_manager is None:
            return None

        query = f"env_id = '{_escape_sql_literal(env_id)}'"
        try:
            rows = await asyncio.to_thread(
                self.env_manager.get_env_configs,
                limit=1,
                offset=0,
                filter_query=query,
            )
        except Exception as e:
            log.warning("Failed to fetch cloud env config env_id=%s: %s", env_id, e)
            return None

        if not rows:
            return None

        config = self._normalize_env_config(rows[0])
        if not config.get("env_id"):
            return None

        self._env_configs[str(config["env_id"])] = config
        return dict(config)

    async def list_environment_rows(self, query: EnvironmentQuery) -> List[Dict[str, Any]]:
        """Read environment rows from the authoritative config store."""
        await self.init()
        clauses = []
        if query.job_id:
            clauses.append(f"job_id = '{_escape_sql_literal(query.job_id)}'")
        if query.env_id:
            clauses.append(f"env_id = '{_escape_sql_literal(query.env_id)}'")
        filter_query = " AND ".join(clauses) or None
        page_size = max(100, query.limit or 1000)
        effective_offset = max(0, query.offset, query.after_id)
        normalized: List[Dict[str, Any]] = []
        scanned = 0
        while True:
            page = await asyncio.to_thread(
                self.env_manager.get_env_configs,
                limit=page_size,
                offset=effective_offset + scanned,
                filter_query=filter_query,
            )
            if not page:
                break
            for index, config in enumerate(page, start=effective_offset + scanned + 1):
                row = self._normalize_env_config(config)
                row.setdefault("id", index)
                if query.finished is not None and _truthy_bool(row.get("finished")) != query.finished:
                    continue
                if query.is_deleted is not None and _truthy_bool(row.get("is_deleted")) != query.is_deleted:
                    continue
                env_id = str(row.get("env_id") or "")
                if env_id:
                    self._env_configs[env_id] = row
                normalized.append(row)
                if query.limit is not None and len(normalized) >= query.limit:
                    return normalized
            scanned += len(page)
            if len(page) < page_size:
                break
        return normalized

    async def insert_environment_rows(self, rows: List[Dict[str, Any]]) -> List[str]:
        await self.init()
        if not rows:
            return []
        configs = []
        env_ids = []
        for row in rows:
            env_id = str(row.get("env_id") or uuid.uuid4())
            env_ids.append(env_id)
            config = {
                "job_id": str(row.get("job_id") or self.job_id),
                "env_id": env_id,
                "env_name": str(row.get("env_name") or ""),
                "env_params": dict(row.get("env_params") or {}),
                "image": str(row.get("image") or ""),
                "group_id": str(row.get("group_id") or ""),
                "finished": bool(row.get("finished", False)),
                "is_deleted": bool(row.get("is_deleted", False)),
                "created_at": int(row.get("created_at") or time.time()),
            }
            configs.append(config)
        await asyncio.to_thread(self.env_manager.save_config, configs)
        self._env_configs.update({row["env_id"]: row for row in configs})
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
            raise ValueError(f"Unknown environment update fields: {sorted(unknown)}")
        rows = await self.list_environment_rows(query)
        updated = 0
        for row in rows:
            env_id = str(row.get("env_id") or "")
            if env_id and await asyncio.to_thread(self.env_manager.update_config, env_id, updates):
                cached = self._env_configs.setdefault(env_id, dict(row))
                cached.update(updates)
                updated += 1
        return updated

    async def delete_session_step_rows(self, query: SessionStepQuery) -> int:
        await self.init()
        session_ids = list(query.session_ids)
        if query.session_id:
            session_ids.append(query.session_id)
        if not session_ids:
            clauses = []
            if query.job_id:
                clauses.append(f"job_id = '{_escape_sql_literal(query.job_id)}'")
            if not clauses:
                raise ValueError("job_id or session_ids is required for cloud deletion")
            await asyncio.to_thread(self.client.delete_landing, " AND ".join(clauses))
            return 1
        for session_id in dict.fromkeys(session_ids):
            clauses = []
            if query.job_id:
                clauses.append(f"job_id = '{_escape_sql_literal(query.job_id)}'")
            clauses.append(f"session_id = '{_escape_sql_literal(session_id)}'")
            await asyncio.to_thread(self.client.delete_landing, " AND ".join(clauses))
        return len(set(session_ids))

    async def delete_job_rows(self, job_id: str) -> None:
        await self.init()
        rows = await self.list_environment_rows(EnvironmentQuery(job_id=job_id))
        await asyncio.to_thread(
            self.client.delete_landing,
            f"job_id = '{_escape_sql_literal(job_id)}'",
        )
        for row in rows:
            env_id = str(row.get("env_id") or "")
            if env_id and not await asyncio.to_thread(self.env_manager.delete_config, env_id):
                raise RuntimeError(f"failed to delete cloud env config env_id={env_id}")
            self._env_configs.pop(env_id, None)

    async def mark_environment_finished(self, env_id: str) -> int:
        """Mark one cloud environment for the current job as finished."""
        await self.init()
        config = await self.get_environment_by_env_id(env_id)
        if config is None or str(config.get("job_id") or "") != str(self.job_id):
            raise RuntimeError(
                f"env config does not belong to job_id={self.job_id!r}: env_id={env_id!r}"
            )
        updated = await asyncio.to_thread(
            self.env_manager.update_config,
            env_id,
            {"finished": True},
        )
        if not updated:
            raise RuntimeError(f"failed to mark cloud env config finished: env_id={env_id}")
        self._env_configs[env_id]["finished"] = True
        return 1

    def get_env_configs(
        self,
        limit: Optional[int] = None,
        offset: int = 0,
        job_id: Optional[str] = None,
    ) -> List[Dict]:
        """Synchronous scheduler reader for cached cloud environment configs."""
        configs = [
            row for row in self._list_env_configs(job_id=job_id)
            if not _truthy_bool(row.get("finished", False))
        ]
        start = max(0, int(offset or 0))
        if limit is None:
            return configs[start:]
        end = start + max(0, int(limit))
        return configs[start:end]

    def _list_env_configs(self, job_id: Optional[str] = None) -> List[Dict]:
        rows: List[Dict] = []
        for index, config in enumerate(self._env_configs.values(), start=1):
            row = self._normalize_env_config(config)
            row.setdefault("id", index)
            if job_id and str(row.get("job_id") or "") != str(job_id):
                continue
            rows.append(row)
        return rows

    def _normalize_env_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        row = dict(config)
        if "image" not in row and "env_image" in row:
            row["image"] = row.get("env_image")
        env_params = row.get("env_params")
        if isinstance(env_params, str):
            try:
                parsed = json.loads(env_params)
                row["env_params"] = parsed if isinstance(parsed, dict) else {}
            except Exception:
                row["env_params"] = {}
        return row

    async def create_session(
        self,
        env_id: str,
        env_name: str,
        llm_model: str,
        group_id: str = "",
        job_id: str = ""
    ) -> SessionContext:
        """Create session context (in-memory only)"""
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

        self._sessions[session.session_id] = session
        log.debug("Created cloud session: %s", session.session_id)
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
    ):
        """
        Record step to cloud LandingTable.
        Images are extracted, uploaded to S3 (with retry), and URLs stored.
        """
        await self.init()

        record, record_id = await self._build_step_record(
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

        if self._enable_buffer:
            await self._buffer_record(record)
        else:
            try:
                await self._timed_db_call(
                    "ingest_landing",
                    self.client.ingest_landing,
                    record,
                    trace_context={"session_id": session.session_id, "step_id": step_id},
                )
                log.debug("Step %d recorded to cloud: %s", step_id, record_id)
            except Exception as e:
                log.error("Failed to ingest step %d: %s", step_id, e)
                raise

        if record_id and session.job_id:
            self._record_job_ids[record_id] = str(session.job_id)
        return record_id

    async def record_steps_batch(self, steps: List[Dict[str, Any]]) -> List[Optional[str]]:
        """Build step records in order and persist them with one cloud batch call."""
        await self.init()
        if not steps:
            return []

        records: List[Any] = []
        record_ids: List[Optional[str]] = []
        for step in steps:
            record, record_id = await self._build_step_record(**step)
            records.append(record)
            record_ids.append(record_id)

        if self._enable_buffer:
            await self._buffer_records(records)
        else:
            try:
                await self._timed_db_call(
                    "ingest_landing_batch",
                    self.client.ingest_landing_batch,
                    records,
                    trace_context={"record_count": len(records)},
                )
                log.debug("Recorded %d steps to cloud in one batch", len(records))
            except Exception as e:
                log.error("Failed to ingest %d cloud steps as a batch: %s", len(records), e)
                raise

        for record, record_id in zip(records, record_ids):
            job_id = getattr(record, "job_id", None)
            if record_id and job_id:
                self._record_job_ids[record_id] = str(job_id)
        return record_ids

    async def mark_records_completed(self, record_ids: List[str]) -> int:
        """Mark known landing record IDs completed in their associated HASH buckets."""
        await self.init()
        unique_ids = list(dict.fromkeys(str(record_id) for record_id in record_ids if record_id))
        if not unique_ids:
            return 0
        if self._enable_buffer:
            await self._flush_records()

        ids_by_job: Dict[str, List[str]] = {}
        inferred_ids: List[str] = []
        missing_job_ids: List[str] = []
        for record_id in unique_ids:
            job_id = self._record_job_ids.get(record_id)
            if not job_id and self.job_id:
                job_id = str(self.job_id)
                inferred_ids.append(record_id)
            if not job_id:
                missing_job_ids.append(record_id)
                continue
            ids_by_job.setdefault(job_id, []).append(record_id)

        if inferred_ids:
            log.warning(
                "Record-to-job association unavailable for %d landing records; "
                "falling back to configured job_id=%s",
                len(inferred_ids),
                self.job_id,
            )
        if missing_job_ids:
            log.error(
                "Cannot mark %d landing records completed without job_id; "
                "refusing an all-bucket HASH update",
                len(missing_job_ids),
            )
            raise ValueError(
                "job_id is required to mark landing records completed without "
                "scanning all HASH buckets"
            )

        for job_id, job_record_ids in ids_by_job.items():
            quoted_ids = ", ".join(
                f"'{_escape_sql_literal(record_id)}'"
                for record_id in job_record_ids
            )
            filter_query = (
                f"job_id = '{_escape_sql_literal(job_id)}' "
                f"AND id IN ({quoted_ids})"
            )
            await self._timed_db_call(
                "update_landing",
                self.client.update_landing,
                filter_query,
                {
                    "is_session_completed": True,
                    "is_terminal": True,
                },
                partition=job_id,
                trace_context={
                    "job_id": job_id,
                    "record_count": len(job_record_ids),
                },
            )
            for record_id in job_record_ids:
                self._record_job_ids.pop(record_id, None)

        log.debug("Marked %d known cloud records completed", len(unique_ids))
        return len(unique_ids)

    async def _build_step_record(
        self,
        *,
        session: SessionContext,
        step_id: int,
        messages: Any,
        response: Any,
        step_reward: float,
        reward: Optional[float] = None,
        request: Optional[str] = None,
        env_state: Optional[str] = None,
        terminated: bool = False,
        truncated: bool = False,
        is_trainable: bool = False,
        dataset: Optional[Any] = None,
        provider_meta: Optional[Dict[str, Any]] = None,
    ) -> tuple[Any, str]:
        env_key = f"{session.env_name}_{session.env_id}"

        # Optimization: session.message_history already holds previously processed
        # messages with S3 URLs substituted for base64.  Reuse that prefix and only
        # process images in the *new* messages appended since the last step, avoiding
        # redundant re-uploads of the same images on every cumulative call.
        if messages is None:
            full_messages = None
            session.message_history = []
        else:
            if not isinstance(messages, list):
                raise ValueError("messages must be a list or None")
            prev_count = len(session.message_history)
            if prev_count > 0 and len(messages) >= prev_count:
                new_messages = messages[prev_count:]
                new_processed, _ = await self._process_images(
                    new_messages, env_key, step_id
                )
                full_messages = list(session.message_history) + list(new_processed)
            else:
                # First step or unexpected message count — process everything normally.
                full_messages, _ = await self._process_images(
                    messages, env_key, step_id
                )
            session.message_history = full_messages
        
        # Generate deterministic record ID
        record_id = generate_deterministic_id({
            "session_id": session.session_id,
            "step_id": step_id,
            "llm_model": session.llm_model,
            "env_name": session.env_name
        })
        
        meta_json = {
            "source": "AIEvoBox",
            "group_id": session.group_id,
            "request": request,
            "env_state": env_state,
        }
        if dataset is not None:
            meta_json["dataset"] = dataset
        if provider_meta:
            meta_json.update(provider_meta)

        # Create LandingRecord
        record = LandingRecord(
            dataset_type=CLOUD_DATASET_TYPE,
            dt=date.today().isoformat(),
            id=record_id,
            session_id=session.session_id,
            step_id=step_id,
            env_id=session.env_id,
            job_id=session.job_id,
            created_at=int(time.time()),
            step_reward=step_reward,
            reward=reward,
            messages=self._messages_to_landing_value(full_messages),
            response=self._response_to_landing_value(response),
            ground_truth_answer=None,
            reference_answer=None,
            agent_model=session.llm_model,
            env_name=session.env_name,
            is_terminal=terminated or truncated,
            is_truncated=truncated,
            is_session_completed=terminated,
            # Training eligibility is assigned by a later, explicit workflow.
            # Every newly recorded trajectory step starts non-trainable.
            is_trainable=False,
            meta_json=json.dumps(meta_json, ensure_ascii=False, default=str)
        )
        
        return record, record_id

    async def update_session_step(
        self,
        session_id: str,
        step_id: int,
        updates: Dict[str, Any],
    ) -> int:
        """Update one cloud-backed session step by session_id and step_id."""
        await self.init()

        if self._enable_buffer:
            await self._flush_records()

        filter_query = self._build_session_step_filter(session_id, step_id)
        job_id = self._job_id_for_session(session_id)
        if not job_id:
            log.warning(
                "Updating landing session step without job_id; "
                "falling back to an all-bucket HASH update: session_id=%s step_id=%s",
                session_id,
                step_id,
            )
        normalized_updates = self._normalize_session_step_updates_for_cloud(
            updates,
            filter_query=filter_query,
            job_id=job_id,
        )
        if not normalized_updates:
            return 0

        result = await self._timed_db_call(
            "update_landing",
            self.client.update_landing,
            filter_query,
            normalized_updates,
            partition=job_id or None,
            trace_context={
                "session_id": session_id,
                "step_id": step_id,
                "field_count": len(normalized_updates),
            },
        )
        log.debug(
            "Submitted cloud session step update: session_id=%s step_id=%s result=%s",
            session_id,
            step_id,
            result,
        )
        return 1

    async def list_session_steps(
        self,
        session_id: str,
        *,
        checkout_latest: bool = False,
    ) -> List[Dict[str, Any]]:
        """Read one cloud-backed session in deterministic trajectory order."""
        await self.init()

        if self._enable_buffer:
            await self._flush_records()

        job_id = self._job_id_for_session(session_id)
        clauses = []
        if job_id:
            clauses.append(f"job_id = '{_escape_sql_literal(job_id)}'")
        else:
            log.warning(
                "Reading landing session steps without job_id; "
                "falling back to an all-bucket HASH query: session_id=%s",
                session_id,
            )
        clauses.append(f"session_id = '{_escape_sql_literal(session_id)}'")
        query = " AND ".join(clauses)
        columns = [
            "id",
            "session_id",
            "step_id",
            "env_name",
            "agent_model",
            "job_id",
            "messages",
            "response",
            "step_reward",
            "reward",
            "meta_json",
            "is_terminal",
            "is_truncated",
            "is_session_completed",
            "is_trainable",
            "created_at",
        ]
        cloud_rows = await self._timed_db_call(
            "filter_landing",
            self.client.query_data,
            filter_query=query,
            limit=10000,
            columns=columns,
            partition=job_id or None,
            checkout_latest=checkout_latest,
            deserialize_json=True,
            trace_context={"session_id": session_id},
        )
        if not cloud_rows:
            return []

        rows: List[Dict[str, Any]] = []
        for cloud_row in cloud_rows:
            row = {key: cloud_row.get(key) for key in columns}
            row["messages"] = _json_value(row.get("messages"), [])
            row["response"] = _json_value(
                row.get("response"),
                row.get("response"),
            )
            meta = _json_value(row.pop("meta_json", None), {})
            if not isinstance(meta, dict):
                meta = {}
            row["llm_model"] = row.pop("agent_model", None)
            env_state = _json_value(meta.get("env_state"), {})
            row["env_state"] = env_state if isinstance(env_state, dict) else {}
            row["group_id"] = meta.get("group_id")
            if "dataset" in meta:
                row["dataset"] = meta["dataset"]
            if row.get("is_trainable") is None:
                row["is_trainable"] = meta.get("is_trainable", False)
            rows.append(row)

        rows.sort(
            key=lambda row: (
                int(row.get("step_id") or 0),
                str(row.get("created_at") or ""),
                str(row.get("id") or ""),
            )
        )
        return rows

    async def record_evaluation_summary(
        self,
        session_id: str,
        step_id: int,
        reward: float,
        env_state: str,
        truncated: bool = False,
    ) -> int:
        """Persist an evaluation-only row when a session has no trainable step."""
        await self.init()
        session = self._sessions.get(session_id)
        if session is None:
            job_id = str(self.job_id or "")
            if not job_id:
                raise ValueError(
                    "job_id is required to persist a cloud evaluation summary"
                )
            log.warning(
                "Session context unavailable while recording evaluation summary; "
                "using configured job_id=%s session_id=%s",
                job_id,
                session_id,
            )
            session = SessionContext(
                session_id=session_id,
                env_id=session_id,
                env_name="gateway",
                llm_model="",
                job_id=job_id,
            )
            self._sessions[session_id] = session

        await self.record_step(
            session=session,
            step_id=step_id,
            messages=[],
            response="",
            step_reward=reward,
            reward=reward,
            env_state=env_state,
            terminated=True,
            truncated=truncated,
            is_trainable=False,
        )
        return 1

    async def mark_latest_session_completed(
        self,
        session_id: str,
        llm_model: Optional[str] = None,
        *,
        is_session_completed: bool = True,
        is_terminal: Optional[bool] = None,
    ) -> int:
        """Set the completion state of the latest cloud-backed trajectory row."""
        await self.init()
        completed = bool(is_session_completed)
        terminal = completed if is_terminal is None else bool(is_terminal)

        if self._enable_buffer:
            await self._flush_records()

        clauses = []
        job_id = self._job_id_for_session(session_id)
        if job_id:
            clauses.append(f"job_id = '{_escape_sql_literal(job_id)}'")
        else:
            log.warning(
                "Reading/updating latest landing session row without job_id; "
                "falling back to all HASH buckets: session_id=%s",
                session_id,
            )
        escaped_session_id = session_id.replace("'", "''")
        clauses.append(f"session_id = '{escaped_session_id}'")
        if llm_model:
            escaped_llm_model = llm_model.replace("'", "''")
            clauses.append(f"agent_model = '{escaped_llm_model}'")
        query = " AND ".join(clauses)

        rows = await self._timed_db_call(
            "filter_landing",
            self.client.query_data,
            filter_query=query,
            limit=1000,
            columns=[
                "step_id",
                "is_terminal",
                "is_session_completed",
                "meta_json",
                "agent_model",
            ],
            partition=job_id or None,
            checkout_latest=True,
            deserialize_json=True,
            trace_context={"session_id": session_id, "model": llm_model},
        )
        if not rows:
            return 0

        candidates: List[tuple[int, bool, bool, Any]] = []
        for row in rows:
            try:
                candidates.append(
                    (
                        int(row["step_id"]),
                        _truthy_bool(row.get("is_terminal")),
                        _truthy_bool(row.get("is_session_completed")),
                        row.get("meta_json"),
                    )
                )
            except Exception:
                continue
        if not candidates:
            return 0

        trajectory_candidates = [
            item for item in candidates if _is_trajectory_meta_json(item[3])
        ]
        latest_step_id, latest_terminal, latest_completed, _latest_meta_json = max(
            trajectory_candidates or candidates,
            key=lambda item: item[0],
        )
        if latest_completed == completed and latest_terminal == terminal:
            return 0

        update_query = self._build_session_step_filter(
            session_id,
            latest_step_id,
            llm_model=llm_model,
        )
        result = await self._timed_db_call(
            "update_landing",
            self.client.update_landing,
            update_query,
            {
                "is_session_completed": completed,
                "is_terminal": terminal,
                **({"step_reward": 0.0, "reward": None} if not completed else {}),
            },
            partition=job_id or None,
            trace_context={
                "session_id": session_id,
                "step_id": latest_step_id,
                "model": llm_model,
            },
        )
        log.debug(
            "Submitted cloud latest-session completion update: session_id=%s step_id=%s model=%s result=%s",
            session_id,
            latest_step_id,
            llm_model,
            result,
        )
        return 1

    async def close(self) -> None:
        """Clean up cloud clients"""
        if self._enable_buffer:
            await self._stop_buffer()

        if self.client and hasattr(self.client, 'close'):
            self.client.close()

        if self.env_manager and hasattr(self.env_manager, 'close'):
            self.env_manager.close()

        self.initialized = False
        log.debug("Cloud strategy closed")
                
    @property
    def buffer_stats(self) -> Optional[dict]:
        """Get buffer statistics"""
        return self._stats if self._enable_buffer else None

    def _build_session_step_filter(
        self,
        session_id: str,
        step_id: int,
        llm_model: Optional[str] = None,
    ) -> str:
        job_id = self._job_id_for_session(session_id)
        clauses = []
        if job_id:
            clauses.append(f"job_id = '{_escape_sql_literal(job_id)}'")
        escaped_session_id = session_id.replace("'", "''")
        clauses.append(f"session_id = '{escaped_session_id}'")
        clauses.append(f"step_id = {int(step_id)}")
        if llm_model:
            clauses.append(f"agent_model = '{_escape_sql_literal(llm_model)}'")
        return " AND ".join(clauses)

    def _job_id_for_session(self, session_id: str) -> str:
        session = self._sessions.get(session_id)
        job_id = session.job_id if session is not None else None
        return str(job_id or self.job_id or "")

    def _normalize_session_step_updates_for_cloud(
        self,
        updates: Dict[str, Any],
        *,
        filter_query: str,
        job_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not updates:
            return {}

        direct_field_map = {
            "session_id": "session_id",
            "step_id": "step_id",
            "env_id": "env_id",
            "env_name": "env_name",
            "llm_model": "agent_model",
            "job_id": "job_id",
            "step_reward": "step_reward",
            "reward": "reward",
            "total_reward": "reward",
            "is_terminal": "is_terminal",
            "is_truncated": "is_truncated",
            "truncated": "is_truncated",
            "is_session_completed": "is_session_completed",
        }
        meta_fields = {"group_id", "env_state", "is_trainable", "request"}
        blocked_fields = {"id", "created_at"}

        normalized: Dict[str, Any] = {}
        meta_updates: Dict[str, Any] = {}

        for field, value in updates.items():
            if field in blocked_fields:
                raise ValueError(f"SessionStep field cannot be updated in cloud mode: {field}")
            if field == "messages":
                normalized["messages"] = self._messages_to_landing_value(value)
            elif field == "response":
                normalized["response"] = self._response_to_landing_value(value)
            elif field in meta_fields:
                meta_updates[field] = value
            elif field == "meta_json":
                if isinstance(value, str):
                    normalized["meta_json"] = value
                else:
                    normalized["meta_json"] = json.dumps(value, ensure_ascii=False)
            elif field in direct_field_map:
                normalized[direct_field_map[field]] = value
            else:
                raise ValueError(f"Unknown SessionStep field for cloud update: {field}")

        if meta_updates:
            if "meta_json" in normalized:
                meta_json = _json_value(normalized["meta_json"], {})
                if not isinstance(meta_json, dict):
                    meta_json = {"source": "AIEvoBox"}
            else:
                meta_json = self._load_existing_meta_json(
                    filter_query,
                    job_id=job_id,
                )
            meta_json.update(meta_updates)
            normalized["meta_json"] = json.dumps(meta_json, ensure_ascii=False)

        return normalized

    def _load_existing_meta_json(
        self,
        filter_query: str,
        *,
        job_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        meta_json: Dict[str, Any] = {"source": "AIEvoBox"}
        if not job_id:
            log.warning(
                "Loading landing meta_json without job_id; "
                "falling back to an all-bucket HASH query"
            )
        try:
            rows = self.client.query_data(
                filter_query=filter_query,
                limit=1,
                columns=["meta_json"],
                partition=job_id or None,
                checkout_latest=True,
                deserialize_json=True,
            )
        except Exception as e:
            log.warning("Failed to load existing meta_json before cloud update: %s", e)
            return meta_json

        if not rows:
            return meta_json

        raw_meta = rows[0].get("meta_json")
        if not raw_meta:
            return meta_json

        parsed = _json_value(raw_meta, {})
        if isinstance(parsed, dict):
            meta_json.update(parsed)

        return meta_json

    def _messages_to_landing_value(self, value: Any) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, str):
            parsed = _json_value(value, None)
            if not isinstance(parsed, (dict, list)):
                raise ValueError("messages JSON string must contain an object or array")
            return value
        if not isinstance(value, (dict, list)):
            raise ValueError("messages update must be an object, array, or JSON string")
        return json.dumps(value, ensure_ascii=False, default=str)

    def _response_to_landing_value(self, value: Any) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, str):
            parsed = _json_value(value, None)
            if isinstance(parsed, (dict, list)):
                return value
            value = {"role": "assistant", "content": value}
        elif not isinstance(value, (dict, list)):
            value = {"role": "assistant", "content": str(value)}
        return json.dumps(value, ensure_ascii=False, default=str)

    def _convert_to_chat_messages(self, messages: List[Dict]) -> List[Any]:
        """Adapt extended messages only for legacy SDK consumers."""
        _load_wt_sdk()
        if ChatMessage is None or ContentItem is None:
            raise RuntimeError(
                "Installed wt_sdk does not expose legacy ChatMessage/ContentItem models"
            )

        result = []
        for message in messages:
            content_items = []
            for field in ("reasoning_content", "encrypted_content"):
                value = message.get(field)
                if isinstance(value, str) and value:
                    content_items.append(ContentItem(type=field, text=value))

            content = message.get("content")
            if isinstance(content, str):
                content_items.append(ContentItem(type="text", text=content))
            elif isinstance(content, list):
                for item in content:
                    if not isinstance(item, dict):
                        continue
                    if item.get("type") == "image_url":
                        content_items.append(
                            ContentItem(
                                type="image_url",
                                image_url={"url": item.get("image_url", {}).get("url", "")},
                            )
                        )
                    elif item.get("type") == "text":
                        content_items.append(
                            ContentItem(type="text", text=item.get("text", ""))
                        )
                    else:
                        content_items.append(
                            ContentItem(
                                type=str(item.get("type") or "provider_content"),
                                text=json.dumps(
                                    item,
                                    ensure_ascii=False,
                                    separators=(",", ":"),
                                    default=str,
                                ),
                            )
                        )

            kwargs: Dict[str, Any] = {
                "role": message.get("role", "user"),
                "content": content_items,
            }
            for field in ("name", "refusal", "tool_calls", "tool_call_id", "function_call"):
                if message.get(field) is not None:
                    kwargs[field] = message[field]
            result.append(ChatMessage(**kwargs))
        return result
    
    async def _process_images(
        self,
        messages: List[Dict],
        env_key: str,
        step_id: int
    ) -> tuple[List[Dict], List[str]]:
        """
        Process images in messages:
        1. Extract base64 images
        2. Upload to S3 with retry
        3. On failure: save locally as fallback
        4. Replace base64 with URL/path in messages
        """
        processed_messages = []
        uploaded_urls = []
        image_count = 0

        for msg_idx, message in enumerate(messages):
            content = message.get("content")

            # Skip non-list content or content without images
            if not isinstance(content, list):
                processed_messages.append(message)
                continue

            has_images = any(
                isinstance(item, dict) and item.get("type") == "image_url"
                for item in content
            )
            if not has_images:
                processed_messages.append(message)
                continue

            # Process images in content
            new_message = message.copy()
            new_content = []

            for item_idx, item in enumerate(content):
                if not isinstance(item, dict) or item.get("type") != "image_url":
                    new_content.append(item)
                    continue

                image_url = item.get("image_url", {}).get("url", "")

                # Check if base64
                match = re.match(r"data:image/(\w+);base64,(.+)", image_url)
                if not match:
                    new_content.append(item)
                    continue

                # Extract image data
                ext = match.group(1)
                b64_str = match.group(2)
                file_name = f"step_{step_id}_m{msg_idx}_i{image_count}.{ext}"

                # Upload with retry
                final_url = await self._upload_image_with_retry(
                    b64_str, env_key, file_name, ext
                )

                # Update item
                new_item = item.copy()
                new_item["image_url"] = item["image_url"].copy()
                new_item["image_url"]["url"] = final_url

                new_content.append(new_item)
                uploaded_urls.append(final_url)
                image_count += 1

            new_message["content"] = new_content
            processed_messages.append(new_message)

        return processed_messages, uploaded_urls

    async def _buffer_record(self, record: Any) -> None:
        await self._buffer_records([record])

    async def _buffer_records(self, records: List[Any]) -> None:
        should_flush = False

        async with self._buffer_lock:
            self._record_buffer.extend(records)
            self._stats["total_create_buffered"] += len(records)
            if len(self._record_buffer) >= self._buffer_size:
                should_flush = True

        if should_flush:
            self._create_flush_task(self._flush_records())

    def _create_flush_task(self, coro) -> None:
        task = asyncio.create_task(coro)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    async def _periodic_flush(self) -> None:
        try:
            while self._running:
                await asyncio.sleep(self._flush_interval)
                await self._flush_records()
        except asyncio.CancelledError:
            raise

    async def _stop_buffer(self) -> None:
        self._running = False

        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
            self._flush_task = None

        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)

        await self._flush_records()
        log.debug("Cloud buffer stopped: stats=%s", self._stats)

    async def _flush_records(self) -> int:
        async with self._flush_lock:
            async with self._buffer_lock:
                if not self._record_buffer:
                    return 0
                records = list(self._record_buffer)
                self._record_buffer.clear()

            try:
                await self._timed_db_call(
                    "ingest_landing_batch",
                    self.client.ingest_landing_batch,
                    records,
                    trace_context={"record_count": len(records), "buffer_flush": True},
                )
            except Exception as e:
                async with self._buffer_lock:
                    self._record_buffer = records + self._record_buffer
                log.error("Failed to flush %d cloud records: %s", len(records), e)
                raise

            self._stats["total_create_flushed"] += len(records)
            self._stats["flush_count"] += 1
            log.debug("Flushed %d cloud records", len(records))
            return len(records)
    
    async def _upload_image_with_retry(
        self,
        b64_str: str,
        env_key: str,
        file_name: str,
        ext: str
    ) -> str:
        """
        Upload image to S3 with retry logic.
        Falls back to local storage on failure.
        """
        # Decode base64
        try:
            img_data = base64.b64decode(b64_str)
        except Exception as e:
            log.error("Failed to decode base64: %s", e)
            return f"data:image/{ext};base64,{b64_str[:50]}..."  # Keep partial for debugging

        # Create local fallback path
        local_dir = os.path.join(self._local_fallback_dir, env_key)
        local_path = os.path.join(local_dir, file_name)

        # Try S3 upload with retry
        if self.s3_uploader:
            s3_key = f"aievobox/{self.job_id}/{env_key}/{file_name}"

            for attempt in range(MAX_UPLOAD_RETRIES):
                try:
                    # Save to temp file first
                    os.makedirs(local_dir, exist_ok=True)
                    with open(local_path, "wb") as f:
                        f.write(img_data)

                    # Upload to S3
                    s3_url = await asyncio.to_thread(
                        self.s3_uploader.upload_file,
                        file_path=local_path,
                        key=s3_key
                    )

                    if s3_url:
                        log.debug("Uploaded image to S3: %s", s3_url)
                        return s3_url

                except Exception as e:
                    wait_time = RETRY_BACKOFF_BASE * (2 ** attempt)
                    log.warning(
                        "S3 upload failed (attempt %d/%d): %s. Retrying in %.1fs",
                        attempt + 1, MAX_UPLOAD_RETRIES, e, wait_time
                    )
                    await asyncio.sleep(wait_time)

            log.error("S3 upload failed after %d retries. Using local fallback.", MAX_UPLOAD_RETRIES)

        # Fallback to local storage
        try:
            os.makedirs(local_dir, exist_ok=True)
            with open(local_path, "wb") as f:
                f.write(img_data)
            log.debug("Image saved locally: %s", local_path)
            return local_path
        except Exception as e:
            log.error("Local save also failed: %s", e)
            return f"[IMAGE_SAVE_FAILED:{file_name}]"
        
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
        
        results = self.client.pull_data(
            dataset_type=CLOUD_DATASET_TYPE,
            cursor=after_id,
            checkout_latest=True,
            where_sql="job_id = '{}' AND is_terminal = True".format(_escape_sql_literal(job_id)),
            limit=limit,
            deserialize_json=True,
        )
        
        if results is None or len(results) == 0:
            log.debug("No completed cloud steps to fetch: result_count=%s", 0 if results is None else len(results))
            return []
        
        cursor = self.client.extract_cursor(results)
        
        rows = []
        for _, row in results.iterrows():
            meta = _json_value(row.get("meta_json"), {})
            if not isinstance(meta, dict):
                meta = {}
            messages = _json_value(row.get("messages"), [])
            if not isinstance(messages, (dict, list)):
                messages = []
            response = _json_value(
                row.get("response"),
                row.get("response"),
            )
            rows.append(
                {
                    "step_pk": cursor,
                    "step_id": row["step_id"],
                    "env_name": row["env_name"],
                    "env_id": row["session_id"],
                    "env_state": meta.get("env_state"),
                    "prompt": self.normalize_messages(messages),
                    "request": meta.get("request"),
                    "response": _response_text(response),
                    "reward": row["reward"],
                    "step_reward": row["step_reward"],
                    "total_reward": row["reward"],
                    "session_id": row["session_id"],
                    "session_end_time": row["created_at"] if row["created_at"] else None,
                    "group_id": meta.get("group_id"),
                    "truncated": row["is_truncated"],
                    "is_session_completed": row["is_session_completed"], 
                }
            )
        return rows
        
    async def get_max_step_id(self, job_id: str) -> int:
        """Get maximum primary key for pagination"""
        await self.init()
        
        last_cursor = self.client.get_max_created_at(
            where_sql=(
                "dataset_type = '{}' AND job_id = '{}' AND is_terminal = True"
                .format(CLOUD_DATASET_TYPE, _escape_sql_literal(job_id))
            ),
        )
        
        return last_cursor

    # --- Helpers ---
    def extract_image_path(self, item: dict) -> str | None:
        """
        Extract image path:
        1. Priority: item["image_url"]["url"]
        2. Otherwise, try s3:// or http(s):// within item["text"]
        """
        
        image_url = item.get("image_url")
        if isinstance(image_url, dict):
            url = image_url.get("url")
            if url:
                return url

        text = item.get("text")
        if isinstance(text, str) and text.startswith(("s3://", "http://", "https://")):
            return text

        return None
    
    def download_image_as_base64(self, image_path: str) -> tuple[str, str | None]:
        """
        Use S3Downloader to download the image and convert it to base64

        Returns:
            (base64_string, media_type)
        """
        _load_wt_sdk()
        downloader = S3Downloader()

        suffix = os.path.splitext(image_path)[1]
        if not suffix:
            suffix = ".bin"

        with tempfile.TemporaryDirectory() as tmpdir:
            local_path = os.path.join(tmpdir, f"image{suffix}")
            downloader.download_file(image_path, local_path)

            with open(local_path, "rb") as f:
                image_bytes = f.read()

        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        media_type = suffix[1:]
        return image_base64, media_type
    
    def remove_none_and_empty(self, obj: Any) -> Any:
        if isinstance(obj, dict):
            cleaned = {}
            for k, v in obj.items():
                v = self.remove_none_and_empty(v)
                if v is not None and v != [] and v != {}:
                    cleaned[k] = v
            return cleaned

        if isinstance(obj, list):
            return [self.remove_none_and_empty(x) for x in obj]

        return obj
    
    def normalize_messages(self, messages: Any) -> list:
        if isinstance(messages, dict):
            messages = [messages]
        elif not isinstance(messages, list):
            raise TypeError(f"Unsupported messages type: {type(messages)}")

        for msg in messages:
            content = msg.get("content")
            if not isinstance(content, list):
                continue

            for item in content:
                if not isinstance(item, dict):
                    continue

                if item.get("type") == "image_url":
                    image_path = self.extract_image_path(item)
                    if image_path:
                        try:
                            image_base64, media_type = self.download_image_as_base64(image_path)
                            item.clear()
                            item["type"] = "image_url"
                            item["image_url"] = {
                                "url": f"data:image/{media_type};base64,{image_base64}"
                            }
                        except Exception as e:
                            item.clear()
                            item["type"] = "image_error"
                            item["error"] = str(e)
                            item["source"] = image_path

        messages = self.remove_none_and_empty(messages)
        return messages
