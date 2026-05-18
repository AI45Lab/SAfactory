from __future__ import annotations

import json
from dataclasses import dataclass, fields
from typing import Any

import yaml


@dataclass(frozen=True)
class LLMRouteConfig:
    base_url: str
    api_key: str | None = None
    supports_stream: bool = True
    max_concurrency: int | None = None


@dataclass(frozen=True)
class GatewayConfig:
    listen_host: str = "0.0.0.0"
    listen_port: int = 8080
    base_session_path: str = "/v1/sessions"
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
    payload_capture_policy: str = "full"
    payload_sample_rate: float = 1.0
    redact_sensitive_fields: bool = True
    telemetry_batch_size: int = 200
    telemetry_flush_interval_ms: int = 100
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
        if not self.base_session_path.startswith("/"):
            raise ValueError("base_session_path must start with '/'")
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

    storage_config = dict(cfg.storage_config or {})
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
        normalized.setdefault("telemetry_loss_policy", telemetry.get("loss_policy"))
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

    if "llm_routes" in kwargs:
        kwargs["llm_routes"] = _routes_from_mapping(kwargs["llm_routes"])

    return GatewayConfig(**kwargs)


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
    return LLMRouteConfig(
        base_url=str(data["base_url"]).rstrip("/"),
        api_key=data.get("api_key"),
        supports_stream=bool(data.get("supports_stream", True)),
        max_concurrency=None
        if data.get("max_concurrency") is None
        else int(data["max_concurrency"]),
    )


def _default_routes() -> dict[str, LLMRouteConfig]:
    route = LLMRouteConfig(
        base_url="http://127.0.0.1:8001/v1",
    )
    return {"default": route}
