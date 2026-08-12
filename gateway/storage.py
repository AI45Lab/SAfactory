from __future__ import annotations

import asyncio
import inspect
import json
import logging
import time
from dataclasses import dataclass
from typing import Any

from core.data_manager.manager import DataManager
from core.data_manager.strategy.base_strategy import SessionContext
from core.perf_trace import PerfTrace

from gateway.anthropic_messages import (
    AnthropicMessageConversionError,
    normalize_anthropic_request,
)
from gateway.config import GatewayConfig
from gateway.models import GatewaySessionBinding, GatewayTelemetryRecord

GATEWAY_STORAGE_NAMESPACE = "gateway"
log = logging.getLogger("gateway.storage")


@dataclass
class _CachedSession:
    session: SessionContext
    last_access_monotonic: float


@dataclass(frozen=True)
class _SessionEnvironment:
    job_id: str
    env_name: str
    group_id: str | None = None
    dataset: Any = None


class GatewayStorage:
    def __init__(self, cfg: GatewayConfig, data_manager: DataManager):
        self.cfg = cfg
        self.data_manager = data_manager
        self._sessions: dict[tuple[str, str], _CachedSession] = {}
        self._environments: dict[str, _SessionEnvironment] = {}
        self._patched_environment_sessions: set[str] = set()
        self._dataset_pending_sessions: set[str] = set()
        self._dataset_written_sessions: set[str] = set()
        self._latest_record_ids: dict[tuple[str, str], str] = {}
        self._lock = asyncio.Lock()

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
        log.info("Gateway storage from_config complete: strategy=%s", manager.strategy.__class__.__name__)
        return cls(cfg, manager)

    async def get_or_create_session(
        self,
        binding: GatewaySessionBinding,
        requested_model: str,
    ) -> SessionContext:
        await self.bind_session_environment(binding)
        await self._evict_expired()
        now = time.monotonic()
        cache_key = (binding.session_id, requested_model)
        async with self._lock:
            cached = self._sessions.get(cache_key)
            if cached is not None:
                self._apply_binding_to_session(cached.session, binding)
                cached.last_access_monotonic = now
                log.debug(
                    "Gateway storage session cache hit: session_id=%s model=%s job_id=%s env_name=%s",
                    binding.session_id,
                    requested_model,
                    cached.session.job_id,
                    cached.session.env_name,
                )
                return cached.session

            log.debug(
                "Gateway storage create session begin: session_id=%s model=%s job_id=%s env_name=%s group_id=%s",
                binding.session_id,
                requested_model,
                binding.job_id or GATEWAY_STORAGE_NAMESPACE,
                binding.env_name or "gateway",
                binding.group_id or "",
            )
            maybe_session = self.data_manager.create_session(
                env_id=binding.session_id,
                env_name=binding.env_name or "gateway",
                llm_model=requested_model,
                group_id=binding.group_id or "",
                job_id=binding.job_id or GATEWAY_STORAGE_NAMESPACE,
            )
            session = await maybe_session if inspect.isawaitable(maybe_session) else maybe_session
            self._sessions[cache_key] = _CachedSession(
                session=session,
                last_access_monotonic=now,
            )
            log.debug(
                "Gateway storage create session complete: session_id=%s model=%s job_id=%s env_name=%s",
                session.session_id,
                requested_model,
                session.job_id,
                session.env_name,
            )
            return session

    async def bind_session_environment(self, binding: GatewaySessionBinding) -> None:
        async with self._lock:
            cached_environment = self._environments.get(binding.session_id)
        if cached_environment is not None and binding.job_id and binding.env_name:
            return

        log.info("Gateway storage resolve environment begin: session_id=%s", binding.session_id)
        environment = await self._resolve_session_environment(binding.session_id)
        if environment is None:
            log.info("Gateway storage resolve environment miss: session_id=%s", binding.session_id)
            return

        binding.job_id = environment.job_id
        binding.env_name = environment.env_name
        binding.group_id = environment.group_id
        log.info(
            "Gateway storage resolved environment: session_id=%s job_id=%s env_name=%s group_id=%s",
            binding.session_id,
            binding.job_id,
            binding.env_name,
            binding.group_id,
        )

        async with self._lock:
            for (session_id, _model), cached in self._sessions.items():
                if session_id == binding.session_id:
                    self._apply_binding_to_session(cached.session, binding)
        await self._patch_session_steps_environment_once(binding.session_id, environment)

    async def _resolve_session_environment(self, session_id: str) -> _SessionEnvironment | None:
        async with self._lock:
            cached = self._environments.get(session_id)
        if cached is not None:
            return cached

        environment = await self._query_session_environment(session_id)
        if environment is None:
            return None

        async with self._lock:
            self._environments[session_id] = environment
        return environment

    async def _query_session_environment(self, session_id: str) -> _SessionEnvironment | None:
        trace = PerfTrace(
            "gateway.storage.environment_lookup",
            logger=log,
            context={
                "operation": "storage_lookup",
                "table": "job_environments",
                "session_id": session_id,
            },
        )
        try:
            log.debug("Gateway storage environment lookup begin: env_id=%s", session_id)
            with trace.span("storage.environment_lookup"):
                environment = await self._load_environment_config(session_id)
        except Exception as exc:
            trace.emit_summary(status="failed", error_type=type(exc).__name__, error=str(exc))
            log.warning("Failed to resolve gateway session environment for session_id=%s: %s", session_id, exc)
            return None

        resolved = self._environment_from_mapping(environment)
        if resolved is not None:
            log.debug(
                "Gateway storage environment lookup complete: env_id=%s job_id=%s env_name=%s",
                session_id,
                resolved.job_id,
                resolved.env_name,
            )
            trace.emit_summary(status="success", row_count=1, job_id=resolved.job_id, env_name=resolved.env_name)
        else:
            trace.emit_summary(status="miss", row_count=0)
        return resolved

    async def _load_environment_config(self, env_id: str) -> dict[str, Any] | None:
        lookup = getattr(self.data_manager, "get_environment_by_env_id", None)
        if callable(lookup):
            maybe_environment = lookup(env_id)
            environment = await maybe_environment if inspect.isawaitable(maybe_environment) else maybe_environment
            return environment if isinstance(environment, dict) else None

        get_all = getattr(self.data_manager, "get_all_environments", None)
        if not callable(get_all):
            return None

        maybe_environments = get_all()
        environments = await maybe_environments if inspect.isawaitable(maybe_environments) else maybe_environments
        if not isinstance(environments, list):
            return None

        for environment in environments:
            if not isinstance(environment, dict):
                continue
            if str(environment.get("env_id") or "") == env_id:
                return environment
        return None

    @staticmethod
    def _environment_from_mapping(environment: dict[str, Any] | None) -> _SessionEnvironment | None:
        if not environment:
            return None

        job_id = str(environment.get("job_id") or "").strip()
        env_name = str(environment.get("env_name") or "").strip()
        if not job_id or not env_name:
            return None

        group_id = environment.get("group_id")
        env_params = environment.get("env_params")
        if isinstance(env_params, str):
            try:
                env_params = json.loads(env_params)
            except (TypeError, ValueError):
                env_params = {}
        return _SessionEnvironment(
            job_id=job_id,
            env_name=env_name,
            group_id=str(group_id) if group_id not in (None, "") else None,
            dataset=env_params.get("dataset") if isinstance(env_params, dict) else None,
        )

    async def _patch_session_steps_environment_once(
        self,
        session_id: str,
        environment: _SessionEnvironment,
    ) -> None:
        async with self._lock:
            if session_id in self._patched_environment_sessions:
                return
            self._patched_environment_sessions.add(session_id)

        if self.cfg.storage_type == "cloud":
            log.debug(
                "Gateway storage patch session environment skipped: storage_type=cloud session_id=%s",
                session_id,
            )
            return

        try:
            trace = PerfTrace(
                "gateway.storage.patch_session_environment",
                logger=log,
                context={
                    "operation": "db_write",
                    "table": "session_steps",
                    "session_id": session_id,
                    "job_id": environment.job_id,
                    "env_name": environment.env_name,
                },
            )
            log.debug(
                "Gateway storage patch session environment begin: session_id=%s job_id=%s env_name=%s",
                session_id,
                environment.job_id,
                environment.env_name,
            )
            with trace.span("db_write.patch_session_environment"):
                updated = await self.data_manager.patch_session_environment(
                    session_id=session_id,
                    job_id=environment.job_id,
                    env_name=environment.env_name,
                    group_id=environment.group_id,
                )
            trace.emit_summary(status="success", updated_count=updated)
            log.debug("Gateway storage patch session environment complete: session_id=%s", session_id)
        except Exception as exc:
            async with self._lock:
                self._patched_environment_sessions.discard(session_id)
            if "trace" in locals():
                trace.emit_summary(status="failed", error_type=type(exc).__name__, error=str(exc))
            log.warning("Failed to patch session environment for session_id=%s: %s", session_id, exc)

    @staticmethod
    def _apply_binding_to_session(session: SessionContext, binding: GatewaySessionBinding) -> None:
        if binding.job_id:
            session.job_id = binding.job_id
        if binding.env_name:
            session.env_name = binding.env_name
        if binding.group_id:
            session.group_id = binding.group_id

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
        """Persist an ordered telemetry batch, flushing inference rows before close events."""
        inference_batch: list[tuple[GatewaySessionBinding, GatewayTelemetryRecord]] = []
        for binding, record in batch:
            if record.event_type != "gateway_session_close":
                inference_batch.append((binding, record))
                continue
            if inference_batch:
                await self.record_inference_steps_batch(inference_batch)
                inference_batch = []
            await self.record_session_close(binding, record)
        if inference_batch:
            await self.record_inference_steps_batch(inference_batch)

    async def record_inference_steps_batch(
        self,
        batch: list[tuple[GatewaySessionBinding, GatewayTelemetryRecord]],
    ) -> None:
        if not batch:
            return
        claimed_dataset_sessions: set[str] = set()
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
                    session = await self.get_or_create_session(binding, record.requested_model)
                    environment = await self._resolve_session_environment(record.session_id)
                    dataset = environment.dataset if environment is not None else None
                    attach_dataset = False
                    if dataset is not None and record.seq_id == 1:
                        async with self._lock:
                            if (
                                record.session_id not in self._dataset_pending_sessions
                                and record.session_id not in self._dataset_written_sessions
                            ):
                                self._dataset_pending_sessions.add(record.session_id)
                                claimed_dataset_sessions.add(record.session_id)
                                attach_dataset = True

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
                    step = {
                        "session": session,
                        "step_id": record.seq_id,
                        "messages": stored_messages,
                        "request": record.request,
                        "response": stored_response,
                        "step_reward": 0.0,
                        "env_state": json.dumps(self._metadata(record), ensure_ascii=False, default=str),
                        "terminated": False,
                        "truncated": record.is_truncated,
                        "is_trainable": False,
                    }
                    if attach_dataset:
                        step["dataset"] = dataset
                    if provider_meta is not None:
                        step["provider_meta"] = provider_meta
                    steps.append(step)
            with trace.span("storage.record_steps_batch", table="session_steps"):
                record_ids = await self.data_manager.record_steps_batch(steps)

            async with self._lock:
                self._dataset_pending_sessions.difference_update(claimed_dataset_sessions)
                self._dataset_written_sessions.update(claimed_dataset_sessions)
                for (_, record), record_id in zip(batch, record_ids):
                    if record_id:
                        self._latest_record_ids[(record.session_id, record.requested_model)] = record_id
            elapsed_ms = (time.perf_counter() - started) * 1000
            log.info(
                "Gateway storage record_step batch complete: records=%d elapsed_ms=%.2f",
                len(batch),
                elapsed_ms,
            )
            trace.emit_summary(status="success", elapsed_ms=elapsed_ms)
        except asyncio.CancelledError:
            async with self._lock:
                self._dataset_pending_sessions.difference_update(claimed_dataset_sessions)
            trace.emit_summary(status="cancelled", error_type="CancelledError")
            raise
        except Exception as exc:
            async with self._lock:
                self._dataset_pending_sessions.difference_update(claimed_dataset_sessions)
            trace.emit_summary(status="failed", error_type=type(exc).__name__, error=str(exc))
            raise

    async def record_session_close(
        self,
        binding: GatewaySessionBinding,
        record: GatewayTelemetryRecord,
    ) -> None:
        started = time.perf_counter()
        trace = PerfTrace(
            "gateway.storage.record_session_close",
            logger=log,
            context={
                "session_id": binding.session_id,
                "reason": binding.close_reason,
                "is_session_completed": record.is_session_completed,
            },
        )
        try:
            if self.cfg.storage_type == "cloud" and record.is_session_completed:
                async with self._lock:
                    record_ids = [
                        record_id
                        for (session_id, _model), record_id in self._latest_record_ids.items()
                        if session_id == binding.session_id
                    ]
                trace.update_context(record_id_count=len(record_ids), close_strategy="known_record_ids")
                if not record_ids:
                    elapsed_ms = (time.perf_counter() - started) * 1000
                    log.info(
                        "Gateway storage session_close skipped: session_id=%s has no persisted record IDs",
                        binding.session_id,
                    )
                    trace.emit_summary(status="success", elapsed_ms=elapsed_ms, updated_count=0)
                    return
                with trace.span(
                    "storage.mark_known_records_completed",
                    table="session_steps",
                    record_count=len(record_ids),
                ):
                    updated_count = await self.data_manager.mark_records_completed(record_ids)
                elapsed_ms = (time.perf_counter() - started) * 1000
                log.info(
                    "Gateway storage session_close complete: session_id=%s updated_count=%d elapsed_ms=%.2f",
                    binding.session_id,
                    updated_count,
                    elapsed_ms,
                )
                trace.emit_summary(status="success", elapsed_ms=elapsed_ms, updated_count=updated_count)
                return

            with trace.span("models_for_session"):
                models = await self._models_for_session(binding)
            trace.update_context(model_count=len(models), models=models)
            log.info(
                "Gateway storage session_close begin: session_id=%s models=%s reason=%s completed=%s",
                binding.session_id,
                models,
                binding.close_reason,
                record.is_session_completed,
            )
            if not models:
                with trace.span(
                    "mark_latest_session_completed_without_model",
                    operation="db_write",
                    table="session_steps",
                ):
                    await self.data_manager.mark_latest_session_completed(
                        session_id=binding.session_id,
                        is_session_completed=record.is_session_completed,
                    )
                elapsed_ms = (time.perf_counter() - started) * 1000
                trace.emit_summary(status="success", elapsed_ms=elapsed_ms, updated_without_model=True)
                log.info(
                    "Gateway storage session_close complete: session_id=%s updated_without_model elapsed_ms=%.2f",
                    binding.session_id,
                    elapsed_ms,
                )
                return

            updated_count = 0
            if self.cfg.storage_type == "cloud" and len(models) > 1:
                with trace.span(
                    "mark_latest_session_completed_models",
                    operation="db_write",
                    table="session_steps",
                    model_count=len(models),
                ):
                    update_counts = await asyncio.gather(
                        *(
                            self.data_manager.mark_latest_session_completed(
                                session_id=binding.session_id,
                                llm_model=model,
                                is_session_completed=record.is_session_completed,
                            )
                            for model in models
                        )
                    )
                updated_count = sum(update_counts)
            else:
                for model in models:
                    with trace.span(
                        "mark_latest_session_completed",
                        operation="db_write",
                        table="session_steps",
                        model=model,
                    ):
                        updated_count += await self.data_manager.mark_latest_session_completed(
                            session_id=binding.session_id,
                            llm_model=model,
                            is_session_completed=record.is_session_completed,
                        )

            if updated_count == 0:
                with trace.span(
                    "mark_latest_session_completed_fallback",
                    operation="db_write",
                    table="session_steps",
                ):
                    await self.data_manager.mark_latest_session_completed(
                        session_id=binding.session_id,
                        is_session_completed=record.is_session_completed,
                    )
            elapsed_ms = (time.perf_counter() - started) * 1000
            log.info(
                "Gateway storage session_close complete: session_id=%s updated_count=%d elapsed_ms=%.2f",
                binding.session_id,
                updated_count,
                elapsed_ms,
            )
            trace.emit_summary(status="success", elapsed_ms=elapsed_ms, updated_count=updated_count)
        except asyncio.CancelledError:
            trace.emit_summary(status="cancelled", error_type="CancelledError")
            raise
        except Exception as exc:
            trace.emit_summary(status="failed", error_type=type(exc).__name__, error=str(exc))
            raise

    async def close(self) -> None:
        log.info("Gateway storage close begin")
        await self.data_manager.close()
        log.info("Gateway storage close complete")

    async def _models_for_session(self, binding: GatewaySessionBinding) -> list[str]:
        models = set(binding.llm_step_count_by_model)
        if binding.model:
            models.add(binding.model)

        async with self._lock:
            models.update(
                model
                for session_id, model in self._sessions
                if session_id == binding.session_id and model
            )

        return sorted(models)

    async def _evict_expired(self) -> None:
        if self.cfg.session_cache_ttl_s <= 0:
            return
        cutoff = time.monotonic() - self.cfg.session_cache_ttl_s
        async with self._lock:
            expired = [
                cache_key
                for cache_key, cached in self._sessions.items()
                if cached.last_access_monotonic < cutoff
            ]
            for cache_key in expired:
                self._sessions.pop(cache_key, None)
                self._latest_record_ids.pop(cache_key, None)

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
