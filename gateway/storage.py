from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import defaultdict
from typing import Any

from core.data_manager.manager import DataManager
from core.perf_trace import PerfTrace

from gateway.anthropic_messages import (
    AnthropicMessageConversionError,
    normalize_anthropic_request,
)
from gateway.config import GatewayConfig
from gateway.models import GatewaySessionBinding, GatewayTelemetryRecord
from gateway.trajectory_builder import (
    build_gateway_step_row,
)

GATEWAY_STORAGE_NAMESPACE = "gateway"
log = logging.getLogger("gateway.storage")


class GatewayStorage:
    def __init__(self, cfg: GatewayConfig, data_manager: DataManager):
        self.cfg = cfg
        self.data_manager = data_manager
        self._needs_env_config: dict[str, bool] = {}
        self._job_probe_locks: defaultdict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    @classmethod
    async def from_config(cls, cfg: GatewayConfig) -> "GatewayStorage":
        storage_config = dict(cfg.storage_config or {})
        log.info(
            "Gateway storage from_config begin: storage_type=%s storage_config_keys=%s",
            cfg.storage_type,
            sorted(storage_config.keys()),
        )
        manager = DataManager(
            job_id=GATEWAY_STORAGE_NAMESPACE,
            storage_type=cfg.storage_type,
            **storage_config,
        )
        await manager.init()
        log.info("Gateway storage from_config complete: strategy=%s", manager.backend_name)
        return cls(cfg, manager)

    async def _has_environment_rows(self, job_id: str) -> bool:
        refs = await self.data_manager.list_environment_refs(
            job_id=job_id, is_deleted=False, limit=1,
        )
        return bool(refs)

    async def count_environment_rows(self, job_id: str) -> int:
        """Refresh this job's enrichment policy and report unfinished rows to the launcher."""
        async with self._job_probe_locks[job_id]:
            pending = await self.data_manager.list_environment_refs(
                job_id=job_id, finished=False, is_deleted=False,
            )
            # Finished environments still carry valid datasets on resume.
            self._needs_env_config[job_id] = bool(pending) or await self._has_environment_rows(job_id)
            return len(pending)

    async def bind_session_environment(self, binding: GatewaySessionBinding) -> None:
        """Resolve optional metadata; the job in the request URL owns the trajectory."""
        job_id = binding.job_id
        if not job_id:
            raise ValueError("Gateway session requires a job_id")

        try:
            if job_id not in self._needs_env_config:
                # Another replica or a direct client may not have run our preflight.
                async with self._job_probe_locks[job_id]:
                    if job_id not in self._needs_env_config:
                        self._needs_env_config[job_id] = await self._has_environment_rows(job_id)
            async with binding.environment_lock:
                if not self._needs_env_config[job_id]:
                    binding.dataset = None
                    binding.environment_checked = False
                    return
                if binding.environment_checked:
                    return
                rows = await self.data_manager.list_environment_rows(
                    job_id=job_id, env_id=binding.session_id, is_deleted=False, limit=1,
                )
                if rows:
                    row = rows[0]
                    params = row.get("env_params")
                    if isinstance(params, str):
                        params = _json_loads(params)
                    binding.env_name = row.get("env_name")
                    binding.group_id = row.get("group_id")
                    binding.dataset = params.get("dataset") if isinstance(params, dict) else None
                    binding.environment_checked = True
        except Exception:
            # A failed probe leaves the job unknown; a failed lookup is retried on
            # the next request. Neither prevents recording this inference.
            log.exception("Gateway environment lookup failed: job_id=%s session_id=%s", job_id, binding.session_id)

    async def record_inference_step(
        self,
        binding: GatewaySessionBinding,
        record: GatewayTelemetryRecord,
    ) -> None:
        await self.record_inference_steps_batch([(binding, record)])

    async def record_telemetry_batch(
        self,
        batch: list[tuple[GatewaySessionBinding, GatewayTelemetryRecord]],
    ) -> None:
        """Persist an ordered telemetry batch."""
        await self.record_inference_steps_batch(batch)

    async def record_inference_steps_batch(
        self,
        batch: list[tuple[GatewaySessionBinding, GatewayTelemetryRecord]],
    ) -> None:
        if not batch:
            return
        started = time.perf_counter()
        trace = PerfTrace(
            "gateway.storage.record_inference_steps_batch",
            logger=log,
            context={
                "record_count": len(batch),
                "session_count": len({record.session_id for _, record in batch}),
            },
        )
        log.info(
            "Gateway storage record_step batch begin: records=%d sessions=%d",
            len(batch),
            len({record.session_id for _, record in batch}),
        )
        try:
            steps: list[dict[str, Any]] = []
            with trace.span("session_context.prepare", operation="in_memory"):
                for binding, record in batch:
                    stored_messages: Any = _trajectory_messages(record)
                    stored_response: Any = record.response
                    provider_meta: dict[str, Any] | None = None
                    if self.cfg.storage_type == "cloud":
                        request_payload = _json_loads(record.request)
                        if record.endpoint == "messages":
                            response_payload = _json_loads(record.response)
                            provider_meta = {
                                "provider": "anthropic",
                                "request": (
                                    request_payload
                                    if request_payload is not None
                                    else record.request
                                ),
                                "response": (
                                    response_payload
                                    if response_payload is not None
                                    else record.response
                                ),
                            }
                        elif record.endpoint == "chat/completions":
                            stored_messages = _openai_request_messages(
                                request_payload,
                                fallback=record.messages,
                            )
                            stored_response = _chat_completion_output(record.response)
                        elif record.endpoint == "responses":
                            stored_messages = _responses_request_input(request_payload)
                            stored_response = _responses_output(record.response)
                    steps.append(build_gateway_step_row(
                        binding=binding,
                        record=record,
                        messages=stored_messages,
                        response=stored_response,
                        metadata=self._metadata(record),
                        dataset=binding.dataset,
                        provider_meta=provider_meta,
                    ))
            with trace.span("storage.record_steps_batch", table="session_steps"):
                await self.data_manager.insert_session_step_rows(steps)
            elapsed_ms = (time.perf_counter() - started) * 1000
            log.info(
                "Gateway storage record_step batch complete: records=%d elapsed_ms=%.2f",
                len(batch),
                elapsed_ms,
            )
            trace.emit_summary(status="success", elapsed_ms=elapsed_ms)
        except asyncio.CancelledError:
            trace.emit_summary(status="cancelled", error_type="CancelledError")
            raise
        except Exception as exc:
            trace.emit_summary(status="failed", error_type=type(exc).__name__, error=str(exc))
            raise

    async def clear_session_cache(self, session_ids: list[str], *, job_id: str) -> int:
        removed = await self.data_manager.clear_environment_cache(session_ids, job_id=job_id)
        removed += await self.data_manager.clear_session_cache(session_ids, job_id=job_id)
        return removed

    async def close(self) -> None:
        log.info("Gateway storage close begin")
        await self.data_manager.close()
        log.info("Gateway storage close complete")

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
            "request_method": record.request_method,
            "request_url": record.request_url,
            "request_headers": record.request_headers,
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
            "weight_version": _response_weight_version(record.response),
            "created_at": record.created_at.isoformat(),
            "completed_at": record.completed_at.isoformat(),
        }


def _trajectory_messages(record: GatewayTelemetryRecord) -> list[dict[str, Any]]:
    # A step's messages are the request-side conversation history. The current
    # assistant output belongs in step.response and will naturally appear in a
    # later step's messages when the client includes it in the next request.
    if record.endpoint == "messages":
        try:
            return normalize_anthropic_request(record.request)
        except AnthropicMessageConversionError as exc:
            log.warning(
                "Anthropic message normalization skipped: request_id=%s error=%s",
                record.request_id,
                exc,
            )
    return [dict(message) for message in record.messages]


def _openai_request_messages(
    request_payload: Any,
    *,
    fallback: list[dict[str, Any]],
) -> Any:
    if isinstance(request_payload, dict):
        messages = request_payload.get("messages")
        if isinstance(messages, list):
            return messages
    return [dict(message) for message in fallback]


def _responses_request_input(request_payload: Any) -> Any:
    if not isinstance(request_payload, dict):
        return []
    input_value = request_payload.get("input")
    if isinstance(input_value, list):
        return input_value
    if isinstance(input_value, dict):
        return [input_value]
    if input_value is None:
        return []
    return [{"role": "user", "content": input_value}]


def _chat_completion_output(response: Any) -> Any:
    payload = _json_loads(response)
    if not isinstance(payload, dict):
        return None
    choices = payload.get("choices")
    if not isinstance(choices, list):
        return None
    messages = [
        dict(choice["message"])
        for choice in choices
        if isinstance(choice, dict) and isinstance(choice.get("message"), dict)
    ]
    if len(messages) == 1:
        return messages[0]
    return messages or None


def _responses_output(response: Any) -> Any:
    payload = _json_loads(response)
    if not isinstance(payload, dict):
        return None
    output = payload.get("output")
    if isinstance(output, (dict, list)):
        return output
    if payload.get("type") in {"message", "function_call", "reasoning"}:
        return payload

    # Streaming summaries may only contain the aggregated model text. Keep the
    # derived message free of HTTP envelope fields.
    output_text = payload.get("output_text")
    reasoning_text = payload.get("reasoning_text")
    if not isinstance(output_text, str) and not isinstance(reasoning_text, str):
        return None
    message: dict[str, Any] = {"role": "assistant", "content": []}
    if isinstance(output_text, str):
        message["content"].append({"type": "output_text", "text": output_text})
    if isinstance(reasoning_text, str):
        message["reasoning"] = reasoning_text
    return message


def _json_loads(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return None


def _response_weight_version(response: str) -> Any:
    parsed = _json_loads(response)
    payloads = parsed if isinstance(parsed, list) else [parsed]
    for payload in reversed(payloads):
        if not isinstance(payload, dict):
            continue
        for key in ("metadata", "meta_info"):
            metadata = payload.get(key)
            if isinstance(metadata, dict) and metadata.get("weight_version") is not None:
                return metadata["weight_version"]
    return None
