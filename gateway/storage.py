from __future__ import annotations

import asyncio
import inspect
import json
import time
from dataclasses import dataclass
from typing import Any

from core.data_manager.manager import DataManager
from core.data_manager.strategy.base_strategy import SessionContext

from gateway.config import GatewayConfig
from gateway.models import GatewaySessionBinding, GatewayTelemetryRecord

GATEWAY_STORAGE_NAMESPACE = "gateway"


@dataclass
class _CachedSession:
    session: SessionContext
    last_access_monotonic: float


class GatewayStorage:
    def __init__(self, cfg: GatewayConfig, data_manager: DataManager):
        self.cfg = cfg
        self.data_manager = data_manager
        self._sessions: dict[str, _CachedSession] = {}
        self._lock = asyncio.Lock()

    @classmethod
    async def from_config(cls, cfg: GatewayConfig) -> "GatewayStorage":
        storage_config = dict(cfg.storage_config or {})
        if cfg.storage_type == "sqlite" and "db_url" not in storage_config:
            storage_config["db_url"] = "sqlite://env_trajs.db"
        manager = DataManager(
            job_id=GATEWAY_STORAGE_NAMESPACE,
            storage_type=cfg.storage_type,
            **storage_config,
        )
        await manager.init()
        return cls(cfg, manager)

    async def get_or_create_session(
        self,
        binding: GatewaySessionBinding,
        requested_model: str,
    ) -> SessionContext:
        await self._evict_expired()
        now = time.monotonic()
        async with self._lock:
            cached = self._sessions.get(binding.session_id)
            if cached is not None:
                cached.last_access_monotonic = now
                return cached.session

            maybe_session = self.data_manager.create_session(
                env_id=binding.session_id,
                env_name="gateway",
                llm_model=requested_model,
                group_id="",
            )
            session = await maybe_session if inspect.isawaitable(maybe_session) else maybe_session
            self._sessions[binding.session_id] = _CachedSession(
                session=session,
                last_access_monotonic=now,
            )
            return session

    async def record_inference_step(
        self,
        binding: GatewaySessionBinding,
        record: GatewayTelemetryRecord,
    ) -> None:
        session = await self.get_or_create_session(binding, record.requested_model)
        await self.data_manager.record_step(
            session=session,
            step_id=record.seq_id,
            messages=record.messages,
            response=record.response,
            step_reward=0.0,
            env_state=json.dumps(self._metadata(record), ensure_ascii=False, default=str),
            terminated=False,
            truncated=record.is_truncated,
            is_trainable=False,
        )

    async def record_session_close(
        self,
        binding: GatewaySessionBinding,
        record: GatewayTelemetryRecord,
    ) -> None:
        await self.record_inference_step(binding, record)

    async def close(self) -> None:
        await self.data_manager.close()

    async def _evict_expired(self) -> None:
        if self.cfg.session_cache_ttl_s <= 0:
            return
        cutoff = time.monotonic() - self.cfg.session_cache_ttl_s
        async with self._lock:
            expired = [
                session_id
                for session_id, cached in self._sessions.items()
                if cached.last_access_monotonic < cutoff
            ]
            for session_id in expired:
                self._sessions.pop(session_id, None)

    @staticmethod
    def _metadata(record: GatewayTelemetryRecord) -> dict[str, Any]:
        return {
            "event_type": record.event_type,
            "request_id": record.request_id,
            "session_id": record.session_id,
            "endpoint": record.endpoint,
            "requested_model": record.requested_model,
            "upstream_base_url": record.upstream_base_url,
            "status_code": record.status_code,
            "error_type": record.error_type,
            "error_text": record.error_text,
            "is_stream": record.is_stream,
            "retry_count": record.retry_count,
            "request_bytes": record.request_bytes,
            "response_bytes": record.response_bytes,
            "prompt_tokens": record.prompt_tokens,
            "completion_tokens": record.completion_tokens,
            "total_tokens": record.total_tokens,
            "ttft_ms": record.ttft_ms,
            "output_chunk_count": record.output_chunk_count,
            "output_bytes": record.output_bytes,
            "upstream_latency_ms": record.upstream_latency_ms,
            "gateway_overhead_ms": record.gateway_overhead_ms,
            "total_latency_ms": record.total_latency_ms,
            "finish_reason": record.finish_reason,
            "client_cancelled": record.client_cancelled,
            "upstream_cancelled": record.upstream_cancelled,
            "redaction_policy": record.redaction_policy,
            "payload_sampled": record.payload_sampled,
            "llm_step_index": record.llm_step_index,
            "max_steps": record.max_steps,
            "is_truncated": record.is_truncated,
            "is_session_completed": record.is_session_completed,
            "truncate_reason": record.truncate_reason if record.is_truncated else None,
            "synthetic_stop": record.synthetic_stop,
            "created_at": record.created_at.isoformat(),
            "completed_at": record.completed_at.isoformat(),
        }
