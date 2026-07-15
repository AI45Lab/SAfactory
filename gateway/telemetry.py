from __future__ import annotations

import asyncio
import json
import logging
import random
import time
import zlib
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from core.perf_trace import PerfTrace
from gateway.config import GatewayConfig
from gateway.llm_router import LLMRouteTarget
from gateway.models import GatewayRequestContext, GatewaySessionBinding, GatewayTelemetryRecord
from gateway.storage import GatewayStorage

log = logging.getLogger("gateway.telemetry")

SENSITIVE_KEY_PARTS = ("authorization", "api_key", "token", "password", "secret")


@dataclass(frozen=True)
class StreamTelemetryStats:
    ttft_ms: float | None = None
    output_chunk_count: int | None = None
    output_bytes: int | None = None
    client_cancelled: bool = False
    upstream_cancelled: bool = False


class TelemetryRecorder:
    def __init__(self, cfg: GatewayConfig, storage: GatewayStorage):
        self.cfg = cfg
        self.storage = storage
        self._async_writes = bool(cfg.storage_type == "cloud" and cfg.telemetry_async_cloud_writes)
        writer_count = int(cfg.telemetry_writer_count) if self._async_writes else 1
        queue_capacity = max(writer_count, int(cfg.max_queue_size))
        base_capacity, extra = divmod(queue_capacity, writer_count)
        self._queues: list[asyncio.Queue[tuple[GatewaySessionBinding, GatewayTelemetryRecord]]] = [
            asyncio.Queue(maxsize=base_capacity + (1 if index < extra else 0))
            for index in range(writer_count)
        ]
        self._seq_by_session_model: dict[tuple[str, str], int] = {}
        self._lock = asyncio.Lock()
        self._writer_tasks: list[asyncio.Task] = []
        self._running = False
        self.dropped_total = 0
        self.dropped_by_reason: defaultdict[str, int] = defaultdict(int)
        self.flushed_total = 0
        self._request_totals: defaultdict[tuple[str, int], int] = defaultdict(int)
        self._duration_sum_ms: defaultdict[tuple[str, str], float] = defaultdict(float)
        self._duration_count: defaultdict[tuple[str, str], int] = defaultdict(int)
        self._ttft_sum_ms: defaultdict[str, float] = defaultdict(float)
        self._ttft_count: defaultdict[str, int] = defaultdict(int)
        self._session_truncated_total: defaultdict[str, int] = defaultdict(int)
        self._synthetic_stop_total: defaultdict[str, int] = defaultdict(int)
        self._truncated_sessions: set[tuple[str, str]] = set()

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        if self._async_writes or self.cfg.telemetry_mode != "strict":
            self._writer_tasks = [
                asyncio.create_task(self._writer_loop(index), name=f"gateway-telemetry-writer-{index}")
                for index in range(len(self._queues))
            ]
        log.info(
            "Gateway telemetry recorder started: mode=%s async_writes=%s writers=%d loss_policy=%s "
            "queue_max_size=%d batch_size=%d flush_interval_ms=%d",
            self.cfg.telemetry_mode,
            self._async_writes,
            len(self._writer_tasks),
            self.cfg.telemetry_loss_policy,
            self.cfg.max_queue_size,
            self.cfg.telemetry_batch_size,
            self.cfg.telemetry_flush_interval_ms,
        )

    async def stop(self) -> None:
        log.info("Gateway telemetry recorder stopping: queued=%d", self.queue_depth())
        self._running = False
        if self._writer_tasks:
            try:
                drain_timeout_s = max(30.0, float(self.cfg.telemetry_write_timeout_s) * 2)
                await asyncio.wait_for(
                    asyncio.gather(*(queue.join() for queue in self._queues)),
                    timeout=drain_timeout_s,
                )
            except asyncio.TimeoutError:
                log.error("Gateway telemetry drain timed out: queued=%d", self.queue_depth())
            finally:
                for task in self._writer_tasks:
                    task.cancel()
                await asyncio.gather(*self._writer_tasks, return_exceptions=True)
                self._writer_tasks = []
        log.info(
            "Gateway telemetry recorder stopped: flushed_total=%d dropped_total=%d",
            self.flushed_total,
            self.dropped_total,
        )

    @property
    def async_writes_enabled(self) -> bool:
        return self._async_writes

    async def enqueue_success(
        self,
        ctx: GatewayRequestContext,
        binding: GatewaySessionBinding,
        target: LLMRouteTarget,
        request_body: dict[str, Any],
        response_body: dict[str, Any] | None,
        latency_ms: float,
        upstream_latency_ms: float | None = None,
        stream_stats: StreamTelemetryStats | None = None,
    ) -> None:
        await self._record_binding(binding, target, error=False)
        record = await self._build_record(
            ctx=ctx,
            binding=binding,
            target=target,
            request_body=request_body,
            response_body=response_body,
            status_code=200,
            latency_ms=latency_ms,
            upstream_latency_ms=upstream_latency_ms,
            stream_stats=stream_stats,
        )
        await self._enqueue(binding, record)

    async def enqueue_failure(
        self,
        ctx: GatewayRequestContext,
        binding: GatewaySessionBinding,
        target: LLMRouteTarget | None,
        request_body: dict[str, Any],
        error_text: str,
        status_code: int,
        latency_ms: float,
        upstream_latency_ms: float | None = None,
        stream_stats: StreamTelemetryStats | None = None,
        response_body: dict[str, Any] | None = None,
    ) -> None:
        await self._record_binding(binding, target, error=True)
        record = await self._build_record(
            ctx=ctx,
            binding=binding,
            target=target,
            request_body=request_body,
            response_body=response_body,
            status_code=status_code,
            latency_ms=latency_ms,
            upstream_latency_ms=upstream_latency_ms,
            error_text=error_text,
            error_type=_status_error_type(status_code),
            stream_stats=stream_stats,
        )
        await self._enqueue(binding, record)

    async def enqueue_session_close(self, binding: GatewaySessionBinding) -> None:
        now = datetime.now(timezone.utc)
        seq_id = await self._next_seq(binding.session_id, binding.model or "")
        record = GatewayTelemetryRecord(
            event_type="gateway_session_close",
            request_id=f"close_{binding.session_id}_{seq_id}",
            session_id=binding.session_id,
            seq_id=seq_id,
            endpoint="session/close",
            requested_model=binding.model or "",
            upstream_base_url=binding.upstream_base_url,
            status_code=200,
            error_type=None,
            error_text=None,
            is_stream=False,
            retry_count=0,
            request_bytes=None,
            response_bytes=None,
            prompt_tokens=None,
            completion_tokens=None,
            total_tokens=None,
            ttft_ms=None,
            output_chunk_count=None,
            output_bytes=None,
            upstream_latency_ms=None,
            gateway_overhead_ms=None,
            total_latency_ms=0.0,
            finish_reason=binding.close_reason,
            client_cancelled=False,
            upstream_cancelled=False,
            redaction_policy="sensitive_keys" if self.cfg.redact_sensitive_fields else "none",
            payload_sampled=False,
            messages=[],
            response=binding.close_reason or "gateway_close",
            created_at=now,
            completed_at=now,
            max_steps=self.cfg.max_steps,
            is_truncated=binding.truncated,
            is_session_completed=True,
            truncate_reason=binding.truncate_reason,
        )
        await self._enqueue(binding, record)

    async def record_synthetic_stop(
        self,
        ctx: GatewayRequestContext,
        binding: GatewaySessionBinding,
        reason: str,
    ) -> None:
        async with self._lock:
            self._synthetic_stop_total[reason] += 1
            truncated_key = (ctx.session_id, ctx.requested_model)
            if binding.is_model_truncated(ctx.requested_model) and truncated_key not in self._truncated_sessions:
                self._truncated_sessions.add(truncated_key)
                self._session_truncated_total[binding.model_truncate_reason(ctx.requested_model) or reason] += 1

    async def _writer_loop(self, writer_index: int) -> None:
        queue = self._queues[writer_index]
        interval_s = max(0.001, self.cfg.telemetry_flush_interval_ms / 1000)
        batch_size = max(1, int(self.cfg.telemetry_batch_size))
        try:
            while self._running or not queue.empty():
                try:
                    first = await asyncio.wait_for(queue.get(), timeout=interval_s)
                except asyncio.TimeoutError:
                    continue

                batch = [first]
                deadline = asyncio.get_running_loop().time() + interval_s
                while len(batch) < batch_size:
                    try:
                        batch.append(queue.get_nowait())
                        continue
                    except asyncio.QueueEmpty:
                        remaining = deadline - asyncio.get_running_loop().time()
                        if remaining <= 0:
                            break
                    try:
                        batch.append(await asyncio.wait_for(queue.get(), timeout=remaining))
                    except asyncio.TimeoutError:
                        break

                try:
                    await self._write_batch(batch, writer_index=writer_index)
                    self.flushed_total += len(batch)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    log.exception(
                        "Gateway telemetry batch write failed: writer=%d records=%d",
                        writer_index,
                        len(batch),
                    )
                    for _binding, _record in batch:
                        await self._drop("write_failed")
                finally:
                    for _ in batch:
                        queue.task_done()
        except asyncio.CancelledError:
            raise

    async def flush_once(self, *, drain_all: bool = False) -> None:
        for writer_index, queue in enumerate(self._queues):
            limit = queue.qsize() if drain_all else self.cfg.telemetry_batch_size
            batch: list[tuple[GatewaySessionBinding, GatewayTelemetryRecord]] = []
            for _ in range(max(0, limit)):
                try:
                    batch.append(queue.get_nowait())
                except asyncio.QueueEmpty:
                    break
            if not batch:
                continue
            try:
                await self._write_batch(batch, writer_index=writer_index)
                self.flushed_total += len(batch)
            finally:
                for _ in batch:
                    queue.task_done()

    def queue_depth(self) -> int:
        return sum(queue.qsize() for queue in self._queues)

    def should_reject_new_requests(self, session_id: str | None = None) -> bool:
        queues = [self._queue_for_session(session_id)] if session_id else self._queues
        return (
            (self._async_writes or self.cfg.telemetry_mode != "strict")
            and self.cfg.telemetry_loss_policy == "fail_closed"
            and any(queue.full() for queue in queues)
        )

    async def metrics_snapshot(self) -> dict[str, Any]:
        async with self._lock:
            return {
                "request_totals": dict(self._request_totals),
                "duration_sum_ms": dict(self._duration_sum_ms),
                "duration_count": dict(self._duration_count),
                "ttft_sum_ms": dict(self._ttft_sum_ms),
                "ttft_count": dict(self._ttft_count),
                "dropped_by_reason": dict(self.dropped_by_reason),
                "session_truncated_total": dict(self._session_truncated_total),
                "synthetic_stop_total": dict(self._synthetic_stop_total),
            }

    async def _record_binding(
        self,
        binding: GatewaySessionBinding,
        target: LLMRouteTarget | None,
        *,
        error: bool,
    ) -> None:
        binding.request_count += 1
        if error:
            binding.error_count += 1
        if target is not None:
            binding.model = target.route_model
            binding.upstream_base_url = target.base_url
        binding.last_seen_at = datetime.now(timezone.utc)

    async def _next_seq(self, session_id: str, model: str) -> int:
        async with self._lock:
            key = (session_id, model)
            next_seq = self._seq_by_session_model.get(key, 0) + 1
            self._seq_by_session_model[key] = next_seq
            return next_seq

    async def _enqueue(
        self,
        binding: GatewaySessionBinding,
        record: GatewayTelemetryRecord,
    ) -> None:
        async with self._lock:
            self._request_totals[(record.endpoint, record.status_code)] += 1
            duration_key = (record.endpoint, record.requested_model)
            self._duration_sum_ms[duration_key] += record.total_latency_ms
            self._duration_count[duration_key] += 1
            if record.ttft_ms is not None:
                self._ttft_sum_ms[record.requested_model] += record.ttft_ms
                self._ttft_count[record.requested_model] += 1

        if self.cfg.telemetry_mode == "strict" and not self._async_writes:
            started = time.perf_counter()
            log.info(
                "Gateway telemetry strict write begin: event_type=%s request_id=%s session_id=%s seq_id=%s model=%s",
                record.event_type,
                record.request_id,
                record.session_id,
                record.seq_id,
                record.requested_model,
            )
            await self._write_record(binding, record)
            self.flushed_total += 1
            log.info(
                "Gateway telemetry strict write complete: event_type=%s request_id=%s elapsed_ms=%.2f flushed_total=%d",
                record.event_type,
                record.request_id,
                (time.perf_counter() - started) * 1000,
                self.flushed_total,
            )
            return

        queue = self._queue_for_session(record.session_id)
        if self._async_writes and self.cfg.telemetry_mode == "strict":
            await queue.put((binding, record))
            log.info(
                "Gateway telemetry submitted: event_type=%s request_id=%s session_id=%s seq_id=%s queued=%d",
                record.event_type,
                record.request_id,
                record.session_id,
                record.seq_id,
                self.queue_depth(),
            )
            return

        policy = self.cfg.telemetry_loss_policy
        if queue.full():
            if policy == "drop_newest":
                await self._drop("drop_newest")
                return
            if policy == "drop_oldest":
                try:
                    queue.get_nowait()
                    queue.task_done()
                except asyncio.QueueEmpty:
                    pass
                await self._drop("drop_oldest")
            elif policy == "fail_closed":
                await self._drop("fail_closed_queue_full")
                return

        try:
            queue.put_nowait((binding, record))
            log.info(
                "Gateway telemetry queued: event_type=%s request_id=%s session_id=%s seq_id=%s queued=%d",
                record.event_type,
                record.request_id,
                record.session_id,
                record.seq_id,
                self.queue_depth(),
            )
        except asyncio.QueueFull:
            if policy == "drop_oldest":
                try:
                    queue.get_nowait()
                    queue.task_done()
                    queue.put_nowait((binding, record))
                    await self._drop("drop_oldest")
                    return
                except asyncio.QueueEmpty:
                    pass
                except asyncio.QueueFull:
                    pass
            await self._drop("queue_full")

    def _queue_for_session(
        self,
        session_id: str,
    ) -> asyncio.Queue[tuple[GatewaySessionBinding, GatewayTelemetryRecord]]:
        if len(self._queues) == 1:
            return self._queues[0]
        shard = zlib.crc32(session_id.encode("utf-8")) % len(self._queues)
        return self._queues[shard]

    async def _write_batch(
        self,
        batch: list[tuple[GatewaySessionBinding, GatewayTelemetryRecord]],
        *,
        writer_index: int,
    ) -> None:
        if not batch:
            return
        started = time.perf_counter()
        log.info(
            "Gateway telemetry batch write begin: writer=%d records=%d queued=%d",
            writer_index,
            len(batch),
            self.queue_depth(),
        )
        if self._async_writes:
            # A timeout around asyncio.to_thread cannot stop the underlying SDK call.
            # Fixed writers bound concurrency, so let each cloud batch finish instead
            # of leaking timed-out writes into the default executor.
            await self.storage.record_telemetry_batch(batch)
        else:
            for binding, record in batch:
                await self._write_record(binding, record)
        log.info(
            "Gateway telemetry batch write complete: writer=%d records=%d elapsed_ms=%.2f",
            writer_index,
            len(batch),
            (time.perf_counter() - started) * 1000,
        )

    async def _write_record(
        self,
        binding: GatewaySessionBinding,
        record: GatewayTelemetryRecord,
    ) -> None:
        async def _write() -> None:
            if record.event_type == "gateway_session_close":
                await self.storage.record_session_close(binding, record)
            else:
                await self.storage.record_inference_step(binding, record)

        timeout_s = max(0.001, float(self.cfg.telemetry_write_timeout_s))
        trace = PerfTrace(
            "gateway.telemetry.write_record",
            logger=log,
            context={
                "operation": "db_write",
                "event_type": record.event_type,
                "request_id": record.request_id,
                "session_id": record.session_id,
                "seq_id": record.seq_id,
                "model": record.requested_model,
                "telemetry_mode": self.cfg.telemetry_mode,
            },
        )
        try:
            with trace.span("storage_write_record", timeout_s=timeout_s):
                await asyncio.wait_for(_write(), timeout=timeout_s)
            trace.emit_summary(status="success")
        except asyncio.TimeoutError:
            trace.emit_summary(status="timeout", timeout_s=timeout_s)
            log.error(
                "Gateway telemetry write timed out: event_type=%s request_id=%s session_id=%s timeout_s=%.1f",
                record.event_type,
                record.request_id,
                record.session_id,
                timeout_s,
            )
            raise
        except Exception as exc:
            trace.emit_summary(status="failed", error_type=type(exc).__name__, error=str(exc))
            raise

    async def _drop(self, reason: str) -> None:
        async with self._lock:
            self.dropped_total += 1
            self.dropped_by_reason[reason] += 1
        log.warning("Gateway telemetry dropped record: reason=%s dropped_total=%d", reason, self.dropped_total)

    async def _build_record(
        self,
        *,
        ctx: GatewayRequestContext,
        binding: GatewaySessionBinding,
        target: LLMRouteTarget | None,
        request_body: dict[str, Any],
        response_body: dict[str, Any] | None,
        status_code: int,
        latency_ms: float,
        upstream_latency_ms: float | None,
        error_type: str | None = None,
        error_text: str | None = None,
        stream_stats: StreamTelemetryStats | None = None,
    ) -> GatewayTelemetryRecord:
        completed_at = datetime.now(timezone.utc)
        payload_sampled = self._should_capture_payload(status_code >= 400)
        safe_request_body = _redact(request_body) if self.cfg.redact_sensitive_fields else request_body
        safe_response_body = _redact(response_body) if self.cfg.redact_sensitive_fields else response_body
        usage = response_body.get("usage", {}) if response_body else {}

        ttft_ms = stream_stats.ttft_ms if stream_stats else None
        output_chunk_count = stream_stats.output_chunk_count if stream_stats else None
        output_bytes = stream_stats.output_bytes if stream_stats else None
        client_cancelled = stream_stats.client_cancelled if stream_stats else False
        upstream_cancelled = stream_stats.upstream_cancelled if stream_stats else False
        is_truncated = self.cfg.max_steps >= 0 and ctx.llm_step_index == self.cfg.max_steps
        if is_truncated:
            binding.mark_model_truncated(ctx.requested_model, "max_steps_reached", completed_at)
            await self._record_truncated_session(
                ctx.session_id,
                ctx.requested_model,
                "max_steps_reached",
            )

        return GatewayTelemetryRecord(
            event_type="gateway_inference",
            request_id=ctx.request_id,
            session_id=ctx.session_id,
            seq_id=await self._next_seq(ctx.session_id, ctx.requested_model),
            endpoint=ctx.endpoint,
            requested_model=ctx.requested_model,
            upstream_base_url=target.base_url if target else None,
            status_code=status_code,
            error_type=error_type,
            error_text=error_text,
            is_stream=ctx.is_stream,
            retry_count=0,
            request_bytes=_json_size(request_body),
            response_bytes=output_bytes if output_bytes is not None else _json_size(response_body),
            prompt_tokens=_usage_int(usage, "prompt_tokens", "input_tokens"),
            completion_tokens=_usage_int(usage, "completion_tokens", "output_tokens"),
            total_tokens=_usage_int(usage, "total_tokens"),
            ttft_ms=ttft_ms,
            output_chunk_count=output_chunk_count,
            output_bytes=output_bytes,
            upstream_latency_ms=upstream_latency_ms,
            gateway_overhead_ms=max(0.0, latency_ms - upstream_latency_ms) if upstream_latency_ms else None,
            total_latency_ms=latency_ms,
            finish_reason=_finish_reason(response_body),
            client_cancelled=client_cancelled,
            upstream_cancelled=upstream_cancelled,
            redaction_policy="sensitive_keys" if self.cfg.redact_sensitive_fields else "none",
            payload_sampled=payload_sampled,
            messages=_messages_for_record(ctx.endpoint, safe_request_body) if payload_sampled else [],
            response=_response_for_record(safe_response_body, error_text, payload_sampled),
            created_at=ctx.created_at,
            completed_at=completed_at,
            llm_step_index=ctx.llm_step_index,
            max_steps=self.cfg.max_steps,
            is_truncated=is_truncated,
            is_session_completed=is_truncated,
            truncate_reason="max_steps_reached" if is_truncated else None,
            synthetic_stop=ctx.synthetic_stop,
        )

    def _should_capture_payload(self, failed: bool) -> bool:
        policy = self.cfg.payload_capture_policy
        if policy == "full":
            return True
        if policy == "failed_only":
            return failed
        if policy == "sampled":
            return random.random() < self.cfg.payload_sample_rate
        return False

    async def _record_truncated_session(self, session_id: str, model: str, reason: str) -> None:
        async with self._lock:
            key = (session_id, model)
            if key in self._truncated_sessions:
                return
            self._truncated_sessions.add(key)
            self._session_truncated_total[reason] += 1


def _status_error_type(status_code: int) -> str:
    if status_code == 429:
        return "rate_limited"
    if status_code == 499:
        return "client_cancelled"
    if status_code == 503:
        return "unavailable"
    if status_code == 504:
        return "timeout"
    if status_code >= 500:
        return "gateway_or_upstream_error"
    return "request_error"


def _usage_int(usage: dict[str, Any], *keys: str) -> int | None:
    for key in keys:
        value = usage.get(key)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                return None
    return None


def _finish_reason(body: dict[str, Any] | None) -> str | None:
    if not body:
        return None
    choices = body.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            return first.get("finish_reason")
    if body.get("status") in {"completed", "failed", "cancelled"}:
        return body.get("status")
    return None


def _json_size(body: dict[str, Any] | None) -> int | None:
    if body is None:
        return None
    return len(json.dumps(body, ensure_ascii=False, default=str).encode("utf-8"))


def _messages_for_record(endpoint: str, body: dict[str, Any]) -> list[dict[str, Any]]:
    if endpoint == "chat/completions":
        messages = body.get("messages")
        if isinstance(messages, list):
            return [message for message in messages if isinstance(message, dict)]
    if endpoint == "responses" and "input" in body:
        return [{"role": "user", "content": body["input"]}]
    return [{"role": "user", "content": json.dumps(body, ensure_ascii=False, default=str)}]


def _response_for_record(
    body: dict[str, Any] | None,
    error_text: str | None,
    payload_sampled: bool,
) -> str:
    if error_text:
        return error_text
    if not body:
        return ""
    if payload_sampled:
        return json.dumps(body, ensure_ascii=False, default=str)
    summary = {
        "id": body.get("id"),
        "object": body.get("object"),
        "status": body.get("status"),
        "finish_reason": _finish_reason(body),
    }
    usage = body.get("usage")
    if isinstance(usage, dict):
        summary["usage"] = usage
    return json.dumps({k: v for k, v in summary.items() if v is not None}, ensure_ascii=False)


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        redacted = {}
        for key, item in value.items():
            if any(part in str(key).lower() for part in SENSITIVE_KEY_PARTS):
                redacted[key] = "[REDACTED]"
            else:
                redacted[key] = _redact(item)
        return redacted
    if isinstance(value, list):
        return [_redact(item) for item in value]
    return value
