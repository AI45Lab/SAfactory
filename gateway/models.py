from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass
class GatewaySessionBinding:
    session_id: str
    model: str
    upstream_base_url: str | None
    status: str
    last_seen_at: datetime
    request_count: int = 0
    error_count: int = 0
    active_request_count: int = 0
    active_stream_count: int = 0
    first_seen_at: datetime | None = None
    closed_at: datetime | None = None
    close_reason: str | None = None
    llm_step_count: int = 0
    truncated: bool = False
    truncate_reason: str | None = None
    stop_response_sent: bool = False

    def close(self, reason: str, closed_at: datetime) -> None:
        self.status = "closed"
        self.closed_at = closed_at
        self.close_reason = reason
        self.last_seen_at = closed_at

    def mark_truncated(self, reason: str, closed_at: datetime) -> None:
        self.truncated = True
        self.truncate_reason = reason
        self.close(reason, closed_at)


@dataclass(frozen=True)
class GatewayRequestContext:
    request_id: str
    session_id: str
    requested_model: str
    endpoint: str
    is_stream: bool
    created_at: datetime
    llm_step_index: int | None = None
    synthetic_stop: bool = False


@dataclass(frozen=True)
class GatewayTelemetryRecord:
    event_type: str
    request_id: str
    session_id: str
    seq_id: int
    endpoint: str
    requested_model: str
    upstream_base_url: str | None
    status_code: int
    error_type: str | None
    error_text: str | None
    is_stream: bool
    retry_count: int
    request_bytes: int | None
    response_bytes: int | None
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    ttft_ms: float | None
    output_chunk_count: int | None
    output_bytes: int | None
    upstream_latency_ms: float | None
    gateway_overhead_ms: float | None
    total_latency_ms: float
    finish_reason: str | None
    client_cancelled: bool
    upstream_cancelled: bool
    redaction_policy: str
    payload_sampled: bool
    messages: list[dict[str, Any]]
    response: str
    created_at: datetime
    completed_at: datetime
    llm_step_index: int | None = None
    max_steps: int = -1
    is_truncated: bool = False
    is_session_completed: bool = False
    truncate_reason: str | None = None
    synthetic_stop: bool = False
