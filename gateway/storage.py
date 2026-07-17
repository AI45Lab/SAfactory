from __future__ import annotations

import asyncio
import inspect
import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Any

from core.data_manager.manager import DataManager
from core.data_manager.strategy.base_strategy import SessionContext
from core.perf_trace import PerfTrace

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


class GatewayStorage:
    def __init__(self, cfg: GatewayConfig, data_manager: DataManager):
        self.cfg = cfg
        self.data_manager = data_manager
        self._sessions: dict[tuple[str, str], _CachedSession] = {}
        self._environments: dict[str, _SessionEnvironment] = {}
        self._patched_environment_sessions: set[str] = set()
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
        if binding.job_id and binding.env_name:
            log.debug(
                "Gateway storage bind environment skipped: session_id=%s job_id=%s env_name=%s",
                binding.session_id,
                binding.job_id,
                binding.env_name,
            )
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
        return _SessionEnvironment(
            job_id=job_id,
            env_name=env_name,
            group_id=str(group_id) if group_id not in (None, "") else None,
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
                    steps.append(
                        {
                            "session": session,
                            "step_id": record.seq_id,
                            "messages": _trajectory_messages(record),
                            "response": "",
                            "step_reward": 0.0,
                            "env_state": json.dumps(self._metadata(record), ensure_ascii=False, default=str),
                            "terminated": False,
                            "truncated": record.is_truncated,
                            "is_trainable": False,
                        }
                    )
            with trace.span("storage.record_steps_batch", table="session_steps"):
                record_ids = await self.data_manager.record_steps_batch(steps)

            async with self._lock:
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
            trace.emit_summary(status="cancelled", error_type="CancelledError")
            raise
        except Exception as exc:
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
            context={"session_id": binding.session_id, "reason": binding.close_reason},
        )
        try:
            if self.cfg.storage_type == "cloud":
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
                "Gateway storage session_close begin: session_id=%s models=%s reason=%s",
                binding.session_id,
                models,
                binding.close_reason,
            )
            if not models:
                with trace.span(
                    "mark_latest_session_completed_without_model",
                    operation="db_write",
                    table="session_steps",
                ):
                    await self.data_manager.mark_latest_session_completed(session_id=binding.session_id)
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
                        )

            if updated_count == 0:
                with trace.span(
                    "mark_latest_session_completed_fallback",
                    operation="db_write",
                    table="session_steps",
                ):
                    await self.data_manager.mark_latest_session_completed(session_id=binding.session_id)
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


THINK_TAG_RE = re.compile(r"<think>(.*?)</think>", re.IGNORECASE | re.DOTALL)


def _trajectory_messages(record: GatewayTelemetryRecord) -> list[dict[str, Any]]:
    messages = [dict(message) for message in record.messages]
    if record.error_text:
        return messages
    messages.extend(_assistant_messages_from_response(record.response))
    return messages


def _assistant_messages_from_response(response: str) -> list[dict[str, Any]]:
    if not response:
        return []

    parsed = _json_loads(response)
    if isinstance(parsed, dict):
        return _assistant_messages_from_payload(parsed)
    if isinstance(parsed, list):
        message = _assistant_message_from_stream(parsed)
        return [message] if message else []

    if isinstance(response, str):
        if response.lstrip().startswith("data:"):
            message = _assistant_message_from_stream(response)
            return [message] if message else []
        message = _assistant_message_from_parts(response)
        return [message] if message else []
    return []


def _assistant_messages_from_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []

    choices = payload.get("choices")
    if isinstance(choices, list):
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            raw_message = choice.get("message")
            if isinstance(raw_message, dict):
                message = _assistant_message_from_chat_message(raw_message)
                if message:
                    messages.append(message)
                    continue
            raw_delta = choice.get("delta")
            if isinstance(raw_delta, dict):
                message = _assistant_message_from_chat_message(raw_delta)
                if message:
                    messages.append(message)
        if messages:
            return messages

    message = _assistant_message_from_response_payload(payload)
    if message:
        return [message]

    stream_message = _assistant_message_from_stream(payload.get("stream_text"))
    return [stream_message] if stream_message else []


def _assistant_message_from_chat_message(raw_message: dict[str, Any]) -> dict[str, Any] | None:
    content = raw_message.get("content")
    reasoning = _first_string(raw_message, ("think", "reasoning", "reasoning_content"))
    message = _assistant_message_from_parts(content if isinstance(content, str) else None, reasoning)

    if message is None:
        message = {"role": "assistant"}
    if content is not None and not isinstance(content, str):
        message["content"] = content

    tool_calls = raw_message.get("tool_calls")
    if isinstance(tool_calls, list) and tool_calls:
        message["tool_calls"] = tool_calls

    function_call = raw_message.get("function_call")
    if isinstance(function_call, dict) and function_call:
        message["function_call"] = function_call

    return message if len(message) > 1 else None


def _assistant_message_from_response_payload(payload: dict[str, Any]) -> dict[str, Any] | None:
    content = _first_string(payload, ("output_text", "text", "content"))
    reasoning = _first_string(payload, ("think", "reasoning_text", "reasoning", "reasoning_content"))
    output_content, output_reasoning = _extract_responses_output(payload.get("output"))

    if output_content:
        content = (content or "") + output_content
    if output_reasoning:
        reasoning = "\n\n".join(part for part in (reasoning, output_reasoning) if part)

    return _assistant_message_from_parts(content, reasoning)


def _assistant_message_from_stream(stream_text: Any) -> dict[str, Any] | None:
    if isinstance(stream_text, str):
        events = _iter_sse_events(stream_text)
    elif isinstance(stream_text, list):
        events = (event for event in stream_text if isinstance(event, dict))
    else:
        return None

    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    for event in events:
        _collect_event_text(event, content_parts, reasoning_parts)
    return _assistant_message_from_parts("".join(content_parts), "\n\n".join(reasoning_parts))


def _collect_event_text(
    event: dict[str, Any],
    content_parts: list[str],
    reasoning_parts: list[str],
) -> None:
    choices = event.get("choices")
    if isinstance(choices, list):
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            for key in ("message", "delta"):
                payload = choice.get(key)
                if isinstance(payload, dict):
                    _collect_message_text(payload, content_parts, reasoning_parts)

    event_type = event.get("type")
    delta = event.get("delta")
    if event_type == "response.output_text.delta" and isinstance(delta, str):
        content_parts.append(delta)
    elif event_type == "response.reasoning_text.delta" and isinstance(delta, str):
        reasoning_parts.append(delta)


def _collect_message_text(
    payload: dict[str, Any],
    content_parts: list[str],
    reasoning_parts: list[str],
) -> None:
    content = payload.get("content")
    if isinstance(content, str):
        content_parts.append(content)
    reasoning = _first_string(payload, ("think", "reasoning", "reasoning_content"))
    if reasoning:
        reasoning_parts.append(reasoning)


def _extract_responses_output(output: Any) -> tuple[str, str]:
    if not isinstance(output, list):
        return "", ""

    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    for item in output:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type == "reasoning":
            reasoning = _first_string(item, ("text", "content", "summary"))
            if reasoning:
                reasoning_parts.append(reasoning)
            continue
        if item_type == "message":
            content = item.get("content")
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict):
                        text = _first_string(part, ("text", "content"))
                        if text:
                            content_parts.append(text)
            else:
                text = _first_string(item, ("text", "content"))
                if text:
                    content_parts.append(text)
    return "".join(content_parts), "\n\n".join(reasoning_parts)


def _assistant_message_from_parts(
    content: str | None = None,
    reasoning: str | None = None,
) -> dict[str, Any] | None:
    content = content or ""
    tag_think, clean_content = _split_think_tags(content)
    think = "\n\n".join(part.strip() for part in (reasoning, tag_think) if part and part.strip())

    message: dict[str, Any] = {"role": "assistant"}
    if clean_content.strip():
        message["content"] = clean_content.strip()
    elif content or think:
        message["content"] = ""
    if think:
        message["think"] = think
    return message if len(message) > 1 else None


def _split_think_tags(content: str) -> tuple[str, str]:
    matches = [match.group(1).strip() for match in THINK_TAG_RE.finditer(content)]
    if not matches:
        return "", content
    clean_content = THINK_TAG_RE.sub("", content)
    return "\n\n".join(match for match in matches if match), clean_content


def _iter_sse_events(stream_text: str):
    for raw_line in stream_text.splitlines():
        line = raw_line.strip()
        if not line.startswith("data:"):
            continue
        data = line[len("data:") :].strip()
        if not data or data == "[DONE]":
            continue
        parsed = _json_loads(data)
        if isinstance(parsed, dict):
            yield parsed


def _json_loads(value: str) -> Any:
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


def _first_string(payload: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str):
            return value
    return ""
