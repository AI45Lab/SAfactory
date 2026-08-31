import asyncio
from importlib import import_module
import json
import logging
import os
import time
import uuid
import base64
import tempfile
from typing import List, Dict, Optional, Any, Set
from datetime import date

from core.data_manager.cloud_delete_guard import CloudDeleteGuard
from core.data_manager.contracts import EnvironmentQuery, SessionStepQuery
from core.data_manager.strategy.base_strategy import StorageStrategy
from core.perf_trace import PerfTrace

log = logging.getLogger("cloud_strategy")

WTGatewayClient = None
GatewayConfig = None
EnvConfigManager = None
LandingRecord = None
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
        ):
            self.db_uri = db_uri
            self.landing_table = landing_table

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
    generate_deterministic_id = generate_deterministic_id or _deterministic_id
    S3Uploader = S3Uploader or _S3Uploader
    S3Downloader = S3Downloader or _S3Downloader


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


def _meta_json_object(meta_json: Any) -> Dict[str, Any]:
    """Return canonical metadata and flatten records written by the legacy schema."""
    meta = _json_value(meta_json, {})
    if not isinstance(meta, dict):
        return {}
    meta = dict(meta)
    legacy_state = _json_value(meta.pop("env_state", None), {})
    if isinstance(legacy_state, dict):
        legacy_state.update(meta)
        return legacy_state
    return meta


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
    Cloud DAO for environment config rows and LandingTable session-step rows.

    Callers provide complete logical rows. Image externalization is deliberately
    kept at the surrounding data-manager boundary, before rows reach this DAO.
    """

    def __init__(
        self,
        job_id: str,
        db_url: Optional[str] = None,
        enable_buffer: bool = False,
        buffer_size: int = 1,
        flush_interval: float = 1.0,
        landing_table: Optional[str] = None,
        env_config_table: str = "evaluation_env_config",
        dldb_model: Optional[str] = None,
        enable_dldb_timing_logs: bool = False,
        dldb_metrics_log_path: Optional[str] = None,
        confirm_cloud_delete_job_id: str = "",
        confirm_production: bool = False,
    ):
        self.db_url = str(db_url or "").strip()
        self.job_id = job_id
        self.initialized = False
        self.landing_table = str(landing_table or "").strip() or None
        self.env_config_table = env_config_table
        self.dldb_model = dldb_model
        self.enable_dldb_timing_logs = enable_dldb_timing_logs
        self.dldb_metrics_log_path = dldb_metrics_log_path
        self.confirm_cloud_delete_job_id = str(confirm_cloud_delete_job_id or "").strip()
        self.confirm_production = bool(confirm_production)

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
        self._record_job_ids: Dict[str, str] = {}

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
        self.landing_table = config.tables.landing_table
        self.db_url = str(config.tables.db_uri or self.db_url)

        try:
            self.client = WTGatewayClient(config)
            log.debug(
                "CloudStrategy initialized with landing_table=%s db_uri=%s",
                config.tables.landing_table,
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
                checkout_latest=True,
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
        if query.after_id:
            clauses.append(f"id > {int(query.after_id)}")
        filter_query = " AND ".join(clauses) or None
        page_size = max(100, query.limit or 1000)
        effective_offset = max(0, query.offset)
        normalized: List[Dict[str, Any]] = []
        scanned = 0
        while True:
            page = await asyncio.to_thread(
                self.env_manager.get_env_configs,
                limit=page_size,
                offset=effective_offset + scanned,
                filter_query=filter_query,
                checkout_latest=True,
            )
            if not page:
                break
            for config in page:
                row = self._normalize_env_config(config)
                if row.get("id") is None:
                    raise RuntimeError(
                        "cloud environment pagination requires EnvConfigManager "
                        "to return the physical id column"
                    )
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

    async def _preflight_destructive_delete(
        self,
        *,
        operation: str,
        job_id: str,
        landing_filter: str,
    ) -> List[Dict[str, Any]]:
        guard = CloudDeleteGuard(
            client=self.client,
            db_uri=self.db_url,
            landing_table=str(self.landing_table or ""),
            confirmed_job_id=self.confirm_cloud_delete_job_id,
            confirm_production=self.confirm_production,
        )
        return await guard.preflight(
            operation=operation,
            job_id=job_id,
            landing_filter=landing_filter,
        )

    async def delete_session_step_rows(self, query: SessionStepQuery) -> int:
        await self.init()
        job_id = str(query.job_id or "").strip()
        if query.record_id or query.record_ids:
            record_ids = tuple(dict.fromkeys(
                item for item in (query.record_id,) + query.record_ids if item
            ))
            quoted = ", ".join(
                f"'{_escape_sql_literal(item)}'" for item in record_ids if item
            )
            clauses = [f"id IN ({quoted})"]
            if job_id:
                clauses.insert(0, f"job_id = '{_escape_sql_literal(job_id)}'")
            landing_filter = " AND ".join(clauses)
            rows = await self._preflight_destructive_delete(
                operation="delete_session_step_rows",
                job_id=job_id,
                landing_filter=landing_filter,
            )
            await asyncio.to_thread(self.client.delete_landing, landing_filter)
            return len(rows)
        session_ids = list(query.session_ids)
        if query.session_id:
            session_ids.append(query.session_id)
        unique_session_ids = tuple(dict.fromkeys(item for item in session_ids if item))
        clauses = []
        if job_id:
            clauses.append(f"job_id = '{_escape_sql_literal(job_id)}'")
        if unique_session_ids:
            quoted_sessions = ", ".join(
                f"'{_escape_sql_literal(item)}'" for item in unique_session_ids
            )
            clauses.append(f"session_id IN ({quoted_sessions})")
        if not clauses:
            raise ValueError("job_id or session_ids is required for cloud deletion")
        landing_filter = " AND ".join(clauses)
        rows = await self._preflight_destructive_delete(
            operation="delete_session_step_rows",
            job_id=job_id,
            landing_filter=landing_filter,
        )
        await asyncio.to_thread(self.client.delete_landing, landing_filter)
        return len(rows)

    async def delete_job_rows(self, job_id: str) -> None:
        await self.init()
        rows = await self.list_environment_rows(EnvironmentQuery(job_id=job_id))
        landing_filter = f"job_id = '{_escape_sql_literal(job_id)}'"
        await self._preflight_destructive_delete(
            operation="delete_job_rows",
            job_id=job_id,
            landing_filter=landing_filter,
        )
        await asyncio.to_thread(
            self.client.delete_landing,
            landing_filter,
        )
        for row in rows:
            env_id = str(row.get("env_id") or "")
            if env_id and not await asyncio.to_thread(self.env_manager.delete_config, env_id):
                raise RuntimeError(f"failed to delete cloud env config env_id={env_id}")
            self._env_configs.pop(env_id, None)

    def _landing_record_from_row(self, row: Dict[str, Any]) -> tuple[Any, str]:
        record_id = str(row.get("record_id") or generate_deterministic_id({
            "job_id": row.get("job_id"),
            "session_id": row.get("session_id"),
            "step_id": row.get("step_id"),
            "llm_model": row.get("llm_model"),
        }))
        meta_json = _meta_json_object(row.get("meta_json"))
        meta_json.setdefault("source", "AIEvoBox")
        if row.get("group_id") not in (None, ""):
            meta_json["group_id"] = row["group_id"]
        if row.get("request") is not None:
            meta_json.setdefault("request", row["request"])
        record = LandingRecord(
            dataset_type=CLOUD_DATASET_TYPE,
            dt=date.today().isoformat(),
            id=record_id,
            session_id=str(row.get("session_id") or ""),
            step_id=int(row.get("step_id") or 0),
            env_id=str(row.get("env_id") or row.get("session_id") or ""),
            job_id=str(row.get("job_id") or self.job_id),
            created_at=int(row.get("created_at") or time.time()),
            step_reward=float(row.get("step_reward") or 0.0),
            reward=row.get("reward"),
            messages=self._messages_to_landing_value(row.get("messages", [])),
            response=self._response_to_landing_value(row.get("response", "")),
            ground_truth_answer=None,
            reference_answer=None,
            agent_model=str(row.get("llm_model") or ""),
            env_name=str(row.get("env_name") or ""),
            is_terminal=bool(row.get("is_terminal", False)),
            is_truncated=bool(row.get("is_truncated", False)),
            is_session_completed=bool(row.get("is_session_completed", False)),
            is_trainable=bool(row.get("is_trainable", False)),
            meta_json=json.dumps(meta_json, ensure_ascii=False, default=str),
        )
        return record, record_id

    async def insert_session_step_rows(
        self,
        rows: List[Dict[str, Any]],
    ) -> List[str]:
        """Persist caller-constructed rows without applying lifecycle policy."""
        await self.init()
        if not rows:
            return []
        records: List[Any] = []
        record_ids: List[str] = []
        for row in rows:
            record, record_id = self._landing_record_from_row(row)
            records.append(record)
            record_ids.append(record_id)
            job_id = str(row.get("job_id") or self.job_id)
            if job_id:
                self._record_job_ids[record_id] = job_id
        if self._enable_buffer:
            await self._buffer_records(records)
        else:
            await self._timed_db_call(
                "ingest_landing_batch",
                self.client.ingest_landing_batch,
                records,
                trace_context={"record_count": len(records)},
            )
        return record_ids

    async def list_session_step_rows(
        self,
        query: SessionStepQuery,
    ) -> List[Dict[str, Any]]:
        await self.init()
        if self._enable_buffer:
            await self._flush_records()
        clauses: List[str] = []
        if query.job_id:
            clauses.append(f"job_id = '{_escape_sql_literal(query.job_id)}'")
        if query.session_id:
            clauses.append(f"session_id = '{_escape_sql_literal(query.session_id)}'")
        if query.session_ids:
            values = ", ".join(f"'{_escape_sql_literal(item)}'" for item in query.session_ids)
            clauses.append(f"session_id IN ({values})")
        if query.record_id:
            clauses.append(f"id = '{_escape_sql_literal(query.record_id)}'")
        if query.record_ids:
            values = ", ".join(f"'{_escape_sql_literal(item)}'" for item in query.record_ids)
            clauses.append(f"id IN ({values})")
        if query.step_id is not None:
            clauses.append(f"step_id = {int(query.step_id)}")
        if query.llm_model:
            clauses.append(f"agent_model = '{_escape_sql_literal(query.llm_model)}'")
        if query.is_terminal is not None:
            clauses.append(f"is_terminal = {str(bool(query.is_terminal))}")
        if query.is_trainable is not None:
            clauses.append(f"is_trainable = {str(bool(query.is_trainable))}")
        if not clauses:
            raise ValueError("cloud session-step query requires at least one filter")

        columns = [
            "id", "session_id", "step_id", "env_id", "env_name", "agent_model",
            "job_id", "messages", "response", "step_reward", "reward", "meta_json",
            "is_terminal", "is_truncated", "is_session_completed", "is_trainable",
            "created_at",
        ]
        cloud_rows = await self._timed_db_call(
            "filter_landing",
            self.client.query_data,
            filter_query=" AND ".join(clauses),
            limit=query.limit or 10000,
            columns=columns,
            partition=query.job_id or None,
            checkout_latest=query.checkout_latest,
            deserialize_json=True,
            trace_context={"job_id": query.job_id, "session_id": query.session_id},
        )
        rows: List[Dict[str, Any]] = []
        for cloud_row in cloud_rows or []:
            meta_json = _meta_json_object(cloud_row.get("meta_json"))
            row = {key: cloud_row.get(key) for key in columns if key != "meta_json"}
            row["record_id"] = row.get("id")
            row["llm_model"] = row.pop("agent_model", None)
            row["messages"] = _json_value(row.get("messages"), [])
            row["response"] = _json_value(row.get("response"), row.get("response"))
            row["meta_json"] = meta_json
            row["group_id"] = meta_json.get("group_id")
            row["request"] = meta_json.get("request")
            rows.append(row)
            if row.get("record_id") and row.get("job_id"):
                self._record_job_ids[str(row["record_id"])] = str(row["job_id"])
        rows.sort(key=lambda row: (
            int(row.get("step_id") or 0),
            str(row.get("created_at") or ""),
            str(row.get("record_id") or ""),
        ))
        return rows

    async def update_session_step_rows(
        self,
        query: SessionStepQuery,
        updates: Dict[str, Any],
    ) -> int:
        await self.init()
        if self._enable_buffer:
            await self._flush_records()
        clauses: List[str] = []
        if query.job_id:
            clauses.append(f"job_id = '{_escape_sql_literal(query.job_id)}'")
        if query.session_id:
            clauses.append(f"session_id = '{_escape_sql_literal(query.session_id)}'")
        if query.session_ids:
            values = ", ".join(f"'{_escape_sql_literal(item)}'" for item in query.session_ids)
            clauses.append(f"session_id IN ({values})")
        if query.record_id:
            clauses.append(f"id = '{_escape_sql_literal(query.record_id)}'")
        if query.record_ids:
            values = ", ".join(f"'{_escape_sql_literal(item)}'" for item in query.record_ids)
            clauses.append(f"id IN ({values})")
        if query.step_id is not None:
            clauses.append(f"step_id = {int(query.step_id)}")
        if query.llm_model:
            clauses.append(f"agent_model = '{_escape_sql_literal(query.llm_model)}'")
        if not clauses:
            raise ValueError("cloud session-step update requires at least one filter")
        filter_query = " AND ".join(clauses)
        job_id = query.job_id or next(
            (self._record_job_ids.get(item) for item in query.record_ids if self._record_job_ids.get(item)),
            None,
        ) or (self._record_job_ids.get(query.record_id) if query.record_id else None)
        normalized = self._normalize_session_step_updates_for_cloud(
            updates,
            filter_query=filter_query,
            job_id=job_id,
        )
        if not normalized:
            return 0
        await self._timed_db_call(
            "update_landing",
            self.client.update_landing,
            filter_query,
            normalized,
            partition=job_id or None,
            trace_context={"job_id": job_id, "field_count": len(normalized)},
        )
        return len(query.record_ids) or int(bool(query.record_id)) or 1

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
            "is_trainable": "is_trainable",
        }
        meta_fields = {"group_id", "request"}
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
                normalized["meta_json"] = json.dumps(
                    _meta_json_object(value),
                    ensure_ascii=False,
                )
            elif field in direct_field_map:
                normalized[direct_field_map[field]] = value
            else:
                raise ValueError(f"Unknown SessionStep field for cloud update: {field}")

        if meta_updates:
            if "meta_json" in normalized:
                meta_json = _meta_json_object(normalized["meta_json"])
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

        meta_json.update(_meta_json_object(raw_meta))

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
            meta = _meta_json_object(row.get("meta_json"))
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
                    "meta_json": json.dumps(meta, ensure_ascii=False, default=str),
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
