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


def _row_value(row: Any, key: str, index: int) -> Any:
    keys = row.keys() if hasattr(row, "keys") else ()
    if key in keys:
        return row[key]
    return row[index]


class GatewayStorage:
    def __init__(self, cfg: GatewayConfig, data_manager: DataManager):
        self.cfg = cfg
        self.data_manager = data_manager
        self._sessions: dict[tuple[str, str], _CachedSession] = {}
        self._environments: dict[str, _SessionEnvironment] = {}
        self._patched_environment_sessions: set[str] = set()
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
        await self.bind_session_environment(binding)
        await self._evict_expired()
        now = time.monotonic()
        cache_key = (binding.session_id, requested_model)
        async with self._lock:
            cached = self._sessions.get(cache_key)
            if cached is not None:
                self._apply_binding_to_session(cached.session, binding)
                cached.last_access_monotonic = now
                return cached.session

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
            return session

    async def bind_session_environment(self, binding: GatewaySessionBinding) -> None:
        if binding.job_id and binding.env_name:
            return

        environment = await self._resolve_session_environment(binding.session_id)
        if environment is None:
            return

        binding.job_id = environment.job_id
        binding.env_name = environment.env_name
        binding.group_id = environment.group_id

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

        environment = await asyncio.to_thread(self._query_session_environment, session_id)
        if environment is None:
            return None

        async with self._lock:
            self._environments[session_id] = environment
        return environment

    def _query_session_environment(self, session_id: str) -> _SessionEnvironment | None:
        try:
            conn = self.data_manager.get_sync_connection()
        except Exception as exc:
            log.debug("Cannot open sync storage connection for session environment lookup: %s", exc)
            return None

        if conn is None:
            return None

        try:
            cursor = conn.execute(
                """
                SELECT job_id, env_name, group_id
                FROM job_environments
                WHERE env_id = ? AND is_deleted = 0
                ORDER BY id DESC
                LIMIT 1
                """,
                (session_id,),
            )
            row = cursor.fetchone()
        except Exception as exc:
            log.warning("Failed to resolve gateway session environment for session_id=%s: %s", session_id, exc)
            return None
        finally:
            try:
                conn.close()
            except Exception:
                pass

        if row is None:
            return None

        job_id = str(_row_value(row, "job_id", 0))
        env_name = str(_row_value(row, "env_name", 1))
        group_id = _row_value(row, "group_id", 2)
        return _SessionEnvironment(
            job_id=job_id,
            env_name=env_name,
            group_id=str(group_id) if group_id is not None else None,
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

        try:
            await asyncio.to_thread(self._patch_session_steps_environment, session_id, environment)
        except Exception:
            async with self._lock:
                self._patched_environment_sessions.discard(session_id)
            raise

    def _patch_session_steps_environment(
        self,
        session_id: str,
        environment: _SessionEnvironment,
    ) -> None:
        try:
            conn = self.data_manager.get_sync_connection()
        except Exception as exc:
            log.debug("Cannot open sync storage connection for session step environment patch: %s", exc)
            return

        if conn is None:
            return

        try:
            if environment.group_id is None:
                conn.execute(
                    """
                    UPDATE session_steps
                    SET job_id = ?, env_name = ?
                    WHERE session_id = ?
                      AND (
                        job_id IS NULL OR job_id != ?
                        OR env_name IS NULL OR env_name != ?
                      )
                    """,
                    (
                        environment.job_id,
                        environment.env_name,
                        session_id,
                        environment.job_id,
                        environment.env_name,
                    ),
                )
            else:
                conn.execute(
                    """
                    UPDATE session_steps
                    SET job_id = ?, env_name = ?, group_id = ?
                    WHERE session_id = ?
                      AND (
                        job_id IS NULL OR job_id != ?
                        OR env_name IS NULL OR env_name != ?
                        OR group_id IS NULL OR group_id != ?
                      )
                    """,
                    (
                        environment.job_id,
                        environment.env_name,
                        environment.group_id,
                        session_id,
                        environment.job_id,
                        environment.env_name,
                        environment.group_id,
                    ),
                )
            conn.commit()
        except Exception as exc:
            log.warning("Failed to patch session_steps environment for session_id=%s: %s", session_id, exc)
        finally:
            try:
                conn.close()
            except Exception:
                pass

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
        session = await self.get_or_create_session(binding, record.requested_model)
        await self.data_manager.record_step(
            session=session,
            step_id=record.seq_id,
            messages=_trajectory_messages(record),
            response="",
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
        models = await self._models_for_session(binding)
        if not models:
            await self.data_manager.mark_latest_session_completed(session_id=binding.session_id)
            return

        updated_count = 0
        for model in models:
            updated_count += await self.data_manager.mark_latest_session_completed(
                session_id=binding.session_id,
                llm_model=model,
            )

        if updated_count == 0:
            await self.data_manager.mark_latest_session_completed(session_id=binding.session_id)

    async def close(self) -> None:
        await self.data_manager.close()

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


def _first_string(payload: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str):
            return value
    return ""
