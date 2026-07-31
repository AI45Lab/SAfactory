from __future__ import annotations

import json
from dataclasses import dataclass, fields
from typing import Any

import yaml

DEFAULT_SQLITE_DB_URL = "sqlite://env_trajs.db"


@dataclass(frozen=True)
class LLMRouteConfig:
    base_url: str
    api_key: str | None = None
    supports_stream: bool = True
    max_concurrency: int | None = None
    anthropic_compatibility: str = "native"
    anthropic_thinking_budget_tokens: int = 1024
    anthropic_max_tokens: int | None = None
    anthropic_interleaved_thinking: bool = False


@dataclass(frozen=True)
class GatewayConfig:
    listen_host: str = "0.0.0.0"
    listen_port: int = 8080
    base_session_path: str = "/v1/sessions"
    max_steps: int = -1
    max_inflight_requests: int = 2048
    max_active_streams: int = 1024
    max_queue_size: int = 4096
    upstream_max_connections: int = 512
    upstream_keepalive_connections: int = 128
    upstream_request_timeout_s: float = 300.0
    upstream_connect_timeout_s: float = 10.0
    upstream_http_proxy: str | None = None
    upstream_no_proxy: list[str] | str | None = None
    per_llm_route_max_concurrency: int = 256
    per_session_max_inflight: int = 8
    telemetry_mode: str = "strict"
    telemetry_loss_policy: str = "fail_closed"
    telemetry_write_timeout_s: float = 10.0
    payload_capture_policy: str = "full"
    payload_sample_rate: float = 1.0
    redact_sensitive_fields: bool = True
    telemetry_batch_size: int = 200
    telemetry_flush_interval_ms: int = 100
    telemetry_writer_count: int = 4
    telemetry_async_cloud_writes: bool = True
    request_log_enabled: bool = True
    request_log_path: str | None = "logs/gateway_requests.jsonl"
    request_log_max_bytes: int = 100 * 1024 * 1024
    request_log_backup_count: int = 5
    request_log_body_limit_bytes: int = 0
    micro_batch_window_ms: int = 0
    session_cache_ttl_s: int = 1800
    close_mode: str = "soft_close"
    drain_timeout_s: int = 30
    storage_type: str = "sqlite"
    storage_config: dict[str, Any] | None = None
    llm_routes: dict[str, LLMRouteConfig] | None = None

    def __post_init__(self) -> None:
        storage_type = str(self.storage_type or "").strip().lower()
        object.__setattr__(self, "storage_type", storage_type)
        object.__setattr__(
            self,
            "storage_config",
            _storage_config_for(storage_type, self.storage_config),
        )

        if not self.base_session_path.startswith("/"):
            raise ValueError("base_session_path must start with '/'")
        if self.max_steps < -1:
            raise ValueError("max_steps must be -1 or a non-negative integer")
        if self.storage_type not in {"sqlite", "cloud"}:
            raise ValueError("storage_type must be one of: sqlite, cloud")
        if self.telemetry_mode == "durable_async":
            raise ValueError(
                "telemetry_mode='durable_async' requires a durable outbox and is not implemented yet"
            )
        if self.telemetry_mode not in {"best_effort", "strict"}:
            raise ValueError("telemetry_mode must be one of: best_effort, strict")
        if self.telemetry_loss_policy not in {"drop_newest", "drop_oldest", "fail_closed"}:
            raise ValueError(
                "telemetry_loss_policy must be one of: drop_newest, drop_oldest, fail_closed"
            )
        if float(self.telemetry_write_timeout_s) <= 0.0:
            raise ValueError("telemetry_write_timeout_s must be positive")
        if int(self.telemetry_writer_count) <= 0:
            raise ValueError("telemetry_writer_count must be positive")
        if self.per_llm_route_max_concurrency <= 0:
            raise ValueError("per_llm_route_max_concurrency must be positive")
        if self.request_log_max_bytes < 0:
            raise ValueError("request_log_max_bytes must be non-negative")
        if self.request_log_backup_count < 0:
            raise ValueError("request_log_backup_count must be non-negative")
        if self.request_log_body_limit_bytes < 0:
            raise ValueError("request_log_body_limit_bytes must be non-negative")
def load_gateway_config(path: str | None = None) -> GatewayConfig:
    file_data = _load_file(path) if path else {}
    cfg = _dict_to_config(file_data)

    storage_config = _storage_config_for(cfg.storage_type, cfg.storage_config)
    llm_routes = cfg.llm_routes or _default_routes()

    return GatewayConfig(
        **{
            **cfg.__dict__,
            "storage_config": storage_config,
            "llm_routes": llm_routes,
        }
    )


def _load_file(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        if path.endswith(".json"):
            return json.load(fh)
        return yaml.safe_load(fh) or {}


def _dict_to_config(data: dict[str, Any]) -> GatewayConfig:
    normalized = dict(data)
    telemetry = normalized.pop("telemetry", None)
    if isinstance(telemetry, dict):
        normalized.setdefault("telemetry_mode", telemetry.get("mode"))
        normalized.setdefault("max_queue_size", telemetry.get("queue_max_size"))
        normalized.setdefault("telemetry_batch_size", telemetry.get("batch_size"))
        normalized.setdefault("telemetry_flush_interval_ms", telemetry.get("flush_interval_ms"))
        normalized.setdefault("telemetry_writer_count", telemetry.get("writer_count"))
        normalized.setdefault("telemetry_async_cloud_writes", telemetry.get("async_cloud_writes"))
        normalized.setdefault("telemetry_loss_policy", telemetry.get("loss_policy"))
        normalized.setdefault("telemetry_write_timeout_s", telemetry.get("write_timeout_s"))
        normalized.setdefault("payload_capture_policy", telemetry.get("capture_payload"))
        normalized.setdefault("payload_sample_rate", telemetry.get("payload_sample_rate"))
        normalized.setdefault("redact_sensitive_fields", telemetry.get("redact_sensitive_fields"))

    request_log = normalized.pop("request_log", None)
    if isinstance(request_log, dict):
        normalized.setdefault("request_log_enabled", request_log.get("enabled"))
        normalized.setdefault("request_log_path", request_log.get("path"))
        normalized.setdefault("request_log_max_bytes", request_log.get("max_bytes"))
        normalized.setdefault("request_log_backup_count", request_log.get("backup_count"))
        normalized.setdefault("request_log_body_limit_bytes", request_log.get("body_limit_bytes"))

    known = {field.name for field in fields(GatewayConfig)}
    kwargs = {key: value for key, value in normalized.items() if key in known and value is not None}

    if "base_session_path" in kwargs:
        kwargs["base_session_path"] = str(kwargs["base_session_path"]).rstrip("/")

    if "max_steps" in kwargs:
        kwargs["max_steps"] = int(kwargs["max_steps"])

    if "telemetry_write_timeout_s" in kwargs:
        kwargs["telemetry_write_timeout_s"] = float(kwargs["telemetry_write_timeout_s"])

    if "storage_type" in kwargs:
        kwargs["storage_type"] = str(kwargs["storage_type"]).strip().lower()

    if "llm_routes" in kwargs:
        kwargs["llm_routes"] = _routes_from_mapping(kwargs["llm_routes"])

    return GatewayConfig(**kwargs)


def _storage_config_for(storage_type: str, raw: Any) -> dict[str, Any]:
    if raw is None:
        storage_config: dict[str, Any] = {}
    elif isinstance(raw, dict):
        storage_config = dict(raw)
    else:
        raise ValueError("storage_config must be a mapping")

    if storage_type == "sqlite":
        storage_config.setdefault("db_url", DEFAULT_SQLITE_DB_URL)
    elif storage_type == "cloud":
        storage_config.pop("db_url", None)
        storage_config.pop("env_config_db_url", None)

    return storage_config


def _routes_from_mapping(raw: Any) -> dict[str, LLMRouteConfig]:
    if not isinstance(raw, dict):
        raise ValueError("llm_routes must be a mapping from model name to route config")

    routes: dict[str, LLMRouteConfig] = {}
    for model, value in raw.items():
        if not isinstance(value, dict):
            raise ValueError(f"llm_routes[{model!r}] must be a mapping")
        routes[str(model)] = _route_from_mapping(value)
    return routes


def _route_from_mapping(data: dict[str, Any]) -> LLMRouteConfig:
    if "base_url" not in data:
        raise ValueError("LLM route config requires base_url")
    anthropic_compatibility = str(
        data.get("anthropic_compatibility", "native")
    ).strip().lower()
    if anthropic_compatibility not in {
        "native",
        "adaptive_thinking",
        "fixed_thinking",
    }:
        raise ValueError(
            "anthropic_compatibility must be one of: "
            "native, adaptive_thinking, fixed_thinking"
        )
    thinking_budget = int(data.get("anthropic_thinking_budget_tokens", 1024))
    if thinking_budget < 1024:
        raise ValueError("anthropic_thinking_budget_tokens must be at least 1024")
    anthropic_max_tokens = data.get("anthropic_max_tokens")
    if anthropic_max_tokens is not None and int(anthropic_max_tokens) <= thinking_budget:
        raise ValueError(
            "anthropic_max_tokens must be greater than anthropic_thinking_budget_tokens"
        )
    anthropic_interleaved_thinking = data.get(
        "anthropic_interleaved_thinking", False
    )
    if not isinstance(anthropic_interleaved_thinking, bool):
        raise ValueError("anthropic_interleaved_thinking must be a boolean")
    return LLMRouteConfig(
        base_url=str(data["base_url"]).rstrip("/"),
        api_key=data.get("api_key"),
        supports_stream=bool(data.get("supports_stream", True)),
        max_concurrency=None
        if data.get("max_concurrency") is None
        else int(data["max_concurrency"]),
        anthropic_compatibility=anthropic_compatibility,
        anthropic_thinking_budget_tokens=thinking_budget,
        anthropic_max_tokens=None
        if anthropic_max_tokens is None
        else int(anthropic_max_tokens),
        anthropic_interleaved_thinking=anthropic_interleaved_thinking,
    )


def _default_routes() -> dict[str, LLMRouteConfig]:
    route = LLMRouteConfig(
        base_url="http://127.0.0.1:8001/v1",
    )
    return {"default": route}
