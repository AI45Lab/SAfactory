from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import replace
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from gateway.admission_control import AdmissionController, AdmissionRejected
from gateway.config import GatewayConfig, load_gateway_config
from gateway.inference_forwarder import (
    InferenceForwarder,
    SessionClosedError,
    StreamForwardContext,
)
from gateway.llm_router import LLMRouteTarget, LLMRouter
from gateway.models import GatewayRequestContext, GatewaySessionBinding
from gateway.request_logger import GatewayRequestLogger
from gateway.session_resolver import SessionResolver
from gateway.storage import GatewayStorage
from gateway.telemetry import StreamTelemetryStats, TelemetryRecorder

log = logging.getLogger("gateway.app")


def create_app(cfg: GatewayConfig | None = None, storage: GatewayStorage | None = None) -> FastAPI:
    cfg = cfg or load_gateway_config()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.gateway_ready = False
        app.state.gateway_draining = False
        app.state.gateway_config = cfg
        app.state.gateway_storage = storage or await GatewayStorage.from_config(cfg)
        app.state.gateway_router = LLMRouter(cfg)
        app.state.gateway_admission = AdmissionController(cfg)
        app.state.gateway_forwarder = InferenceForwarder(cfg)
        app.state.gateway_resolver = SessionResolver(cfg)
        app.state.gateway_request_logger = GatewayRequestLogger(cfg)
        app.state.gateway_request_logger.start()
        app.state.gateway_telemetry = TelemetryRecorder(cfg, app.state.gateway_storage)
        await app.state.gateway_telemetry.start()
        app.state.gateway_ready = True
        try:
            yield
        finally:
            app.state.gateway_ready = False
            app.state.gateway_draining = True
            app.state.gateway_admission.draining = True
            await _wait_for_drain(app, cfg.drain_timeout_s)
            await app.state.gateway_telemetry.stop()
            await app.state.gateway_forwarder.close()
            app.state.gateway_request_logger.close()
            await app.state.gateway_storage.close()

    app = FastAPI(
        title="AIEvo API Gateway",
        version="0.2.0",
        lifespan=lifespan,
    )
    session_root = cfg.base_session_path.rstrip("/")
    standard_openai_root = _standard_openai_root(session_root)

    async def handle_inference_request(
        request: Request,
        *,
        endpoint: str,
        path_session_id: str,
    ) -> Response:
        started = time.perf_counter()
        payload: dict[str, Any] = {}
        ctx: GatewayRequestContext | None = None
        binding: GatewaySessionBinding | None = None
        target: LLMRouteTarget | None = None
        route_reserved = False
        release_in_finally = True

        resolver: SessionResolver = request.app.state.gateway_resolver
        admission: AdmissionController = request.app.state.gateway_admission
        router: LLMRouter = request.app.state.gateway_router
        forwarder: InferenceForwarder = request.app.state.gateway_forwarder
        telemetry: TelemetryRecorder = request.app.state.gateway_telemetry
        storage: GatewayStorage = request.app.state.gateway_storage
        request_logger: GatewayRequestLogger = request.app.state.gateway_request_logger

        try:
            payload = await request.json()
            if not isinstance(payload, dict):
                raise ValueError("request body must be a JSON object")

            ctx = await resolver.resolve(
                payload,
                endpoint=endpoint,
                path_session_id=path_session_id,
            )
            binding = await resolver.get_or_create_binding(ctx)
            await storage.bind_session_environment(binding)
            if binding.status == "closed":
                if binding.is_model_truncated(ctx.requested_model):
                    ctx = replace(ctx, synthetic_stop=True)
                    reason = binding.model_truncate_reason(ctx.requested_model) or "max_steps_reached"
                    return await _return_synthetic_stop(
                        ctx=ctx,
                        binding=binding,
                        payload=payload,
                        reason=reason,
                        cfg=request.app.state.gateway_config,
                        request_logger=request_logger,
                        telemetry=telemetry,
                    )
                raise SessionClosedError(f"session {ctx.session_id} is closed")
            if binding.is_model_truncated(ctx.requested_model):
                ctx = replace(ctx, synthetic_stop=True)
                reason = binding.model_truncate_reason(ctx.requested_model) or "max_steps_reached"
                return await _return_synthetic_stop(
                    ctx=ctx,
                    binding=binding,
                    payload=payload,
                    reason=reason,
                    cfg=request.app.state.gateway_config,
                    request_logger=request_logger,
                    telemetry=telemetry,
                )

            if cfg.max_steps < 0 or binding.step_count_for(ctx.requested_model) < cfg.max_steps:
                if telemetry.should_reject_new_requests():
                    raise AdmissionRejected("telemetry queue is full", 503)
                target = await router.select_target(ctx, binding)

            decision = await admission.acquire_request(ctx, binding, target)
            if decision.action == "stop":
                ctx = replace(ctx, synthetic_stop=True)
                return await _return_synthetic_stop(
                    ctx=ctx,
                    binding=binding,
                    payload=payload,
                    reason=decision.stop_reason or "max_steps_reached",
                    cfg=request.app.state.gateway_config,
                    request_logger=request_logger,
                    telemetry=telemetry,
                )
            ctx = replace(ctx, llm_step_index=decision.llm_step_index)

            if target is None:
                target = await router.select_target(ctx, binding)
            route_reserved = True
            await router.on_acquire(target.route_model, is_stream=ctx.is_stream)

            headers = forwarder.build_upstream_headers(target)
            await request_logger.log_request(ctx, binding, target, payload)
            if ctx.is_stream:
                opened = await _open_stream(forwarder, target, endpoint, payload, headers)
                release_in_finally = False
                return StreamingResponse(
                    _stream_and_finalize(
                        opened=opened,
                        started=started,
                        ctx=ctx,
                        binding=binding,
                        target=target,
                        payload=payload,
                        router=router,
                        telemetry=telemetry,
                        request_logger=request_logger,
                        admission=admission,
                    ),
                    status_code=opened.status_code,
                    media_type=opened.media_type,
                )

            result = await _forward_json(forwarder, target, endpoint, payload, headers)
            latency_ms = (time.perf_counter() - started) * 1000
            await request_logger.log_response(
                ctx,
                binding,
                target,
                status_code=result.status_code,
                response_body=result.body,
                latency_ms=latency_ms,
                upstream_latency_ms=result.upstream_latency_ms,
            )
            await router.mark_route_result(
                target.route_model,
                True,
                result.upstream_latency_ms,
                result.status_code,
            )
            await telemetry.enqueue_success(
                ctx,
                binding,
                target,
                payload,
                result.body,
                latency_ms,
                upstream_latency_ms=result.upstream_latency_ms,
            )
            return JSONResponse(result.body, status_code=result.status_code)
        except Exception as exc:
            latency_ms = (time.perf_counter() - started) * 1000
            status_code, error_body = forwarder.normalize_error(exc)
            if target is not None:
                await router.mark_route_result(target.route_model, False, latency_ms, status_code)
            await request_logger.log_error(
                endpoint=endpoint,
                path_session_id=path_session_id,
                request_body=payload,
                error_body=error_body,
                error_text=str(exc),
                status_code=status_code,
                latency_ms=latency_ms,
                ctx=ctx,
                binding=binding,
                target=target,
            )
            if ctx is not None and binding is not None:
                await telemetry.enqueue_failure(
                    ctx,
                    binding,
                    target,
                    payload,
                    str(exc),
                    status_code,
                    latency_ms,
                )
            return JSONResponse(error_body, status_code=status_code)
        finally:
            if release_in_finally:
                if route_reserved and target is not None and ctx is not None:
                    await router.on_release(target.route_model, is_stream=ctx.is_stream)
                await admission.release(ctx, binding, target)

    async def handle_session_chat_completions(session_id: str, request: Request) -> Response:
        return await handle_inference_request(
            request,
            endpoint="chat/completions",
            path_session_id=session_id,
        )

    async def handle_session_responses(session_id: str, request: Request) -> Response:
        return await handle_inference_request(
            request,
            endpoint="responses",
            path_session_id=session_id,
        )

    async def handle_standard_chat_completions(request: Request) -> Response:
        return await handle_standard_inference_request(
            request,
            endpoint="chat/completions",
        )

    async def handle_standard_responses(request: Request) -> Response:
        return await handle_standard_inference_request(
            request,
            endpoint="responses",
        )

    async def handle_standard_inference_request(
        request: Request,
        *,
        endpoint: str,
    ) -> Response:
        started = time.perf_counter()
        payload: dict[str, Any] = {}
        target: LLMRouteTarget | None = None
        route_reserved = False
        release_in_finally = True
        is_stream = False

        router: LLMRouter = request.app.state.gateway_router
        forwarder: InferenceForwarder = request.app.state.gateway_forwarder

        try:
            payload = await request.json()
            if not isinstance(payload, dict):
                raise ValueError("request body must be a JSON object")

            requested_model = payload.get("model")
            if not isinstance(requested_model, str) or not requested_model:
                raise ValueError("request body requires a non-empty model field")

            is_stream = bool(payload.get("stream", False))
            target = await router.select_standard_target(
                requested_model=requested_model,
                is_stream=is_stream,
            )
            route_reserved = True
            await router.on_acquire(target.route_model, is_stream=is_stream)

            headers = forwarder.build_upstream_headers(target)
            if is_stream:
                opened = await _open_stream(forwarder, target, endpoint, payload, headers)
                release_in_finally = False
                return StreamingResponse(
                    _standard_stream_and_finalize(
                        opened=opened,
                        started=started,
                        target=target,
                        router=router,
                        is_stream=is_stream,
                    ),
                    status_code=opened.status_code,
                    media_type=opened.media_type,
                )

            result = await _forward_json(forwarder, target, endpoint, payload, headers)
            await router.mark_route_result(
                target.route_model,
                True,
                result.upstream_latency_ms,
                result.status_code,
            )
            return JSONResponse(result.body, status_code=result.status_code)
        except Exception as exc:
            latency_ms = (time.perf_counter() - started) * 1000
            status_code, error_body = forwarder.normalize_error(exc)
            if target is not None:
                await router.mark_route_result(target.route_model, False, latency_ms, status_code)
            return JSONResponse(error_body, status_code=status_code)
        finally:
            if release_in_finally and route_reserved and target is not None:
                await router.on_release(target.route_model, is_stream=is_stream)

    async def get_session_status(session_id: str) -> Response:
        resolver: SessionResolver = app.state.gateway_resolver
        status = await resolver.get_status(session_id)
        if status is None:
            return JSONResponse({"error": {"type": "not_found", "message": "session not found"}}, status_code=404)
        return JSONResponse(status)

    async def close_session(session_id: str, request: Request) -> dict[str, Any]:
        resolver: SessionResolver = app.state.gateway_resolver
        telemetry: TelemetryRecorder = app.state.gateway_telemetry
        reason = "gateway_close"
        try:
            body = await request.json()
            if isinstance(body, dict) and body.get("reason"):
                reason = str(body["reason"])
        except Exception:
            pass
        binding = await resolver.close_session(session_id, reason=reason)
        await telemetry.enqueue_session_close(binding)
        return {"session_id": session_id, "status": binding.status}

    app.add_api_route(
        f"{session_root}/{{session_id}}/chat/completions",
        handle_session_chat_completions,
        methods=["POST"],
    )
    app.add_api_route(
        f"{session_root}/{{session_id}}/responses",
        handle_session_responses,
        methods=["POST"],
    )
    app.add_api_route(
        f"{standard_openai_root}/chat/completions",
        handle_standard_chat_completions,
        methods=["POST"],
    )
    app.add_api_route(
        f"{standard_openai_root}/responses",
        handle_standard_responses,
        methods=["POST"],
    )
    app.add_api_route(
        f"{session_root}/{{session_id}}",
        get_session_status,
        methods=["GET"],
    )
    app.add_api_route(
        f"{session_root}/{{session_id}}/close",
        close_session,
        methods=["POST"],
    )

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz() -> Response:
        if not app.state.gateway_ready:
            return JSONResponse({"status": "draining" if app.state.gateway_draining else "starting"}, status_code=503)
        return JSONResponse(
            {
                "status": "ready",
                "storage_type": cfg.storage_type,
                "storage_config": _ready_storage_config(cfg),
                "max_steps": cfg.max_steps,
            }
        )

    @app.get("/metrics")
    async def metrics() -> Response:
        admission: AdmissionController = app.state.gateway_admission
        router: LLMRouter = app.state.gateway_router
        telemetry: TelemetryRecorder = app.state.gateway_telemetry
        admission_snapshot = await admission.snapshot()
        route_snapshot = await router.snapshot()
        telemetry_snapshot = await telemetry.metrics_snapshot()
        lines = [
            "# TYPE gateway_inflight_requests gauge",
            f"gateway_inflight_requests {admission_snapshot['inflight_requests']}",
            "# TYPE gateway_active_streams gauge",
            f"gateway_active_streams {admission_snapshot['active_streams']}",
            "# TYPE gateway_max_steps gauge",
            f"gateway_max_steps {cfg.max_steps}",
            "# TYPE gateway_telemetry_queue_depth gauge",
            f"gateway_telemetry_queue_depth {telemetry.queue_depth()}",
            "# TYPE gateway_telemetry_dropped_total counter",
        ]
        if telemetry_snapshot["dropped_by_reason"]:
            for reason, count in telemetry_snapshot["dropped_by_reason"].items():
                lines.append(f'gateway_telemetry_dropped_total{{reason="{_metric_label(reason)}"}} {count}')
        else:
            lines.append('gateway_telemetry_dropped_total{reason="none"} 0')

        lines.append("# TYPE gateway_session_truncated_total counter")
        if telemetry_snapshot["session_truncated_total"]:
            for reason, count in telemetry_snapshot["session_truncated_total"].items():
                lines.append(f'gateway_session_truncated_total{{reason="{_metric_label(reason)}"}} {count}')
        else:
            lines.append('gateway_session_truncated_total{reason="none"} 0')

        lines.append("# TYPE gateway_synthetic_stop_total counter")
        if telemetry_snapshot["synthetic_stop_total"]:
            for reason, count in telemetry_snapshot["synthetic_stop_total"].items():
                lines.append(f'gateway_synthetic_stop_total{{reason="{_metric_label(reason)}"}} {count}')
        else:
            lines.append('gateway_synthetic_stop_total{reason="none"} 0')

        lines.extend(
            [
                "# TYPE gateway_requests_accepted_total counter",
                f"gateway_requests_accepted_total {admission_snapshot['accepted_total']}",
                "# TYPE gateway_requests_rejected_total counter",
                f"gateway_requests_rejected_total {admission_snapshot['rejected_total']}",
                "# TYPE gateway_inference_requests_total counter",
            ]
        )
        for (endpoint, status_code), count in telemetry_snapshot["request_totals"].items():
            lines.append(
                f'gateway_inference_requests_total{{endpoint="{_metric_label(endpoint)}",status_code="{status_code}"}} {count}'
            )

        lines.append("# TYPE gateway_request_duration_seconds summary")
        for (endpoint, model), count in telemetry_snapshot["duration_count"].items():
            total_ms = telemetry_snapshot["duration_sum_ms"][(endpoint, model)]
            label = f'endpoint="{_metric_label(endpoint)}",model="{_metric_label(model)}"'
            lines.append(f"gateway_request_duration_seconds_sum{{{label}}} {total_ms / 1000}")
            lines.append(f"gateway_request_duration_seconds_count{{{label}}} {count}")

        lines.append("# TYPE gateway_ttft_seconds summary")
        for model, count in telemetry_snapshot["ttft_count"].items():
            label = f'model="{_metric_label(model)}"'
            lines.append(f"gateway_ttft_seconds_sum{{{label}}} {telemetry_snapshot['ttft_sum_ms'][model] / 1000}")
            lines.append(f"gateway_ttft_seconds_count{{{label}}} {count}")

        lines.append("# TYPE gateway_llm_route_inflight gauge")
        for route_model, state in route_snapshot.items():
            label = f'model="{_metric_label(route_model)}"'
            lines.append(f"gateway_llm_route_inflight{{{label}}} {state['inflight_requests']}")
            lines.append(f"gateway_llm_route_error_rate{{{label}}} {state['recent_error_rate']}")
        return Response("\n".join(lines) + "\n", media_type="text/plain; version=0.0.4")

    return app


async def _forward_json(
    forwarder: InferenceForwarder,
    target: LLMRouteTarget,
    endpoint: str,
    payload: dict[str, Any],
    headers: dict[str, str],
):
    if endpoint == "chat/completions":
        return await forwarder.forward_chat(target, payload, headers)
    if endpoint == "responses":
        return await forwarder.forward_responses(target, payload, headers)
    raise ValueError(f"unsupported endpoint {endpoint}")


async def _open_stream(
    forwarder: InferenceForwarder,
    target: LLMRouteTarget,
    endpoint: str,
    payload: dict[str, Any],
    headers: dict[str, str],
) -> StreamForwardContext:
    if endpoint == "chat/completions":
        return await forwarder.open_chat_stream(target, payload, headers)
    if endpoint == "responses":
        return await forwarder.open_responses_stream(target, payload, headers)
    raise ValueError(f"unsupported endpoint {endpoint}")


async def _return_synthetic_stop(
    *,
    ctx: GatewayRequestContext,
    binding: GatewaySessionBinding,
    payload: dict[str, Any],
    reason: str,
    cfg: GatewayConfig,
    request_logger: GatewayRequestLogger,
    telemetry: TelemetryRecorder,
) -> Response:
    binding.mark_model_stop_response_sent(ctx.requested_model)
    await telemetry.record_synthetic_stop(ctx, binding, reason)
    await request_logger.log_stop_request(ctx, binding, payload, reason)
    return build_synthetic_stop_response(
        endpoint=ctx.endpoint,
        model=ctx.requested_model,
        session_id=ctx.session_id,
        reason=reason,
        max_steps=cfg.max_steps,
        llm_step_count=binding.step_count_for(ctx.requested_model),
        is_stream=ctx.is_stream,
    )


def build_synthetic_stop_response(
    *,
    endpoint: str,
    model: str,
    session_id: str,
    reason: str,
    max_steps: int,
    llm_step_count: int,
    is_stream: bool = False,
) -> Response:
    message = (
        f"SAFACTORY_STOP: {reason}. Session {session_id} model {model} has reached max_steps={max_steps} "
        f"after {llm_step_count} real LLM request(s). "
        "Stop the task and return final status."
    )
    created = int(time.time())

    if endpoint == "chat/completions":
        body = {
            "id": "safactory-stop-session",
            "object": "chat.completion",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": message},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
        if is_stream:
            return StreamingResponse(
                _chat_stop_events(body, message),
                status_code=200,
                media_type="text/event-stream",
            )
        return JSONResponse(body, status_code=200)

    if endpoint == "responses":
        body = {
            "id": "safactory-stop-session",
            "object": "response",
            "created_at": created,
            "status": "completed",
            "model": model,
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": message}],
                }
            ],
            "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        }
        if is_stream:
            return StreamingResponse(
                _responses_stop_events(body, message),
                status_code=200,
                media_type="text/event-stream",
            )
        return JSONResponse(body, status_code=200)

    raise ValueError(f"unsupported endpoint {endpoint}")


async def _chat_stop_events(body: dict[str, Any], message: str) -> AsyncIterator[bytes]:
    chunk_base = {
        "id": body["id"],
        "object": "chat.completion.chunk",
        "created": body["created"],
        "model": body["model"],
    }
    yield _sse_data(
        {
            **chunk_base,
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": message}}],
        }
    )
    yield _sse_data(
        {
            **chunk_base,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": body["usage"],
        }
    )
    yield b"data: [DONE]\n\n"


async def _responses_stop_events(body: dict[str, Any], message: str) -> AsyncIterator[bytes]:
    yield _sse_data({"type": "response.output_text.delta", "delta": message})
    yield _sse_data({"type": "response.completed", "response": body})
    yield b"data: [DONE]\n\n"


def _sse_data(payload: dict[str, Any]) -> bytes:
    return f"data: {json.dumps(payload, ensure_ascii=False, default=str)}\n\n".encode("utf-8")


async def _stream_and_finalize(
    *,
    opened: StreamForwardContext,
    started: float,
    ctx: GatewayRequestContext,
    binding: GatewaySessionBinding,
    target: LLMRouteTarget,
    payload: dict[str, Any],
    router: LLMRouter,
    telemetry: TelemetryRecorder,
    request_logger: GatewayRequestLogger,
    admission: AdmissionController,
) -> AsyncIterator[bytes]:
    first_chunk_at: float | None = None
    chunk_count = 0
    output_bytes = 0
    status_code = opened.status_code
    error_text: str | None = None
    client_cancelled = False
    upstream_cancelled = False
    stream_response_body: dict[str, Any] = {}
    stream_metadata_buffer = ""
    stream_choice_states: dict[int, dict[str, Any]] = {}
    stream_text_parts: list[str] = []
    stream_total_bytes = 0
    stream_capture = request_logger.new_stream_capture()

    try:
        async for chunk in opened.response.content.iter_any():
            if chunk:
                stream_capture.append(chunk)
                stream_total_bytes += len(chunk)
                stream_text_parts.append(chunk.decode("utf-8", errors="replace"))
                if first_chunk_at is None:
                    first_chunk_at = time.perf_counter()
                chunk_count += 1
                output_bytes += len(chunk)
                stream_metadata_buffer = _collect_stream_metadata(
                    chunk,
                    stream_metadata_buffer,
                    stream_response_body,
                    stream_choice_states,
                )
            yield chunk
    except asyncio.CancelledError:
        client_cancelled = True
        status_code = 499
        error_text = "client cancelled streaming response"
        raise
    except Exception as exc:
        upstream_cancelled = True
        status_code = 502
        error_text = str(exc)
        raise
    finally:
        opened.response.close()
        latency_ms = (time.perf_counter() - started) * 1000
        ttft_ms = (first_chunk_at - started) * 1000 if first_chunk_at is not None else None
        stats = StreamTelemetryStats(
            ttft_ms=ttft_ms,
            output_chunk_count=chunk_count,
            output_bytes=output_bytes,
            client_cancelled=client_cancelled,
            upstream_cancelled=upstream_cancelled,
        )
        ok = status_code < 400
        await router.mark_route_result(target.route_model, ok, latency_ms, status_code)
        try:
            stream_body = stream_capture.snapshot()
            telemetry_response_body = _stream_response_body_for_telemetry(
                summary=stream_response_body,
                choice_states=stream_choice_states,
                stream_text="".join(stream_text_parts),
                stream_total_bytes=stream_total_bytes,
                stream_truncated=False,
            )
            await request_logger.log_stream_response(
                ctx,
                binding,
                target,
                status_code=status_code,
                stream_body=stream_body,
                stream_summary=stream_response_body,
                latency_ms=latency_ms,
                upstream_latency_ms=opened.upstream_latency_ms,
                ttft_ms=ttft_ms,
                output_chunk_count=chunk_count,
                client_cancelled=client_cancelled,
                upstream_cancelled=upstream_cancelled,
                error_text=error_text,
            )
            if ok:
                await telemetry.enqueue_success(
                    ctx,
                    binding,
                    target,
                    payload,
                    telemetry_response_body,
                    latency_ms,
                    upstream_latency_ms=opened.upstream_latency_ms,
                    stream_stats=stats,
                )
            else:
                await telemetry.enqueue_failure(
                    ctx,
                    binding,
                    target,
                    payload,
                    error_text or "stream failed",
                    status_code,
                    latency_ms,
                    upstream_latency_ms=opened.upstream_latency_ms,
                    stream_stats=stats,
                    response_body=telemetry_response_body,
                )
        finally:
            await router.on_release(target.route_model, is_stream=ctx.is_stream)
            await admission.release(ctx, binding, target)


async def _standard_stream_and_finalize(
    *,
    opened: StreamForwardContext,
    started: float,
    target: LLMRouteTarget,
    router: LLMRouter,
    is_stream: bool,
) -> AsyncIterator[bytes]:
    status_code = opened.status_code
    try:
        async for chunk in opened.response.content.iter_any():
            yield chunk
    except asyncio.CancelledError:
        status_code = 499
        raise
    except Exception:
        status_code = 502
        raise
    finally:
        opened.response.close()
        latency_ms = (time.perf_counter() - started) * 1000
        await router.mark_route_result(target.route_model, status_code < 400, latency_ms, status_code)
        await router.on_release(target.route_model, is_stream=is_stream)


def _standard_openai_root(session_root: str) -> str:
    if session_root.endswith("/sessions"):
        parent = session_root[: -len("/sessions")]
        return parent or "/v1"
    return "/v1"


async def _wait_for_drain(app: FastAPI, drain_timeout_s: int) -> None:
    admission: AdmissionController = app.state.gateway_admission
    deadline = time.monotonic() + max(0, drain_timeout_s)
    while time.monotonic() < deadline:
        snapshot = await admission.snapshot()
        if snapshot["inflight_requests"] <= 0 and snapshot["active_streams"] <= 0:
            return
        await asyncio.sleep(0.05)


def _ready_storage_config(cfg: GatewayConfig) -> dict[str, Any]:
    storage_config = dict(cfg.storage_config or {})
    public: dict[str, Any] = {}
    if "db_url" in storage_config:
        public["db_url"] = storage_config["db_url"]
    return public


def _metric_label(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _collect_stream_metadata(
    chunk: bytes,
    buffer: str,
    summary: dict[str, Any],
    choice_states: dict[int, dict[str, Any]],
) -> str:
    buffer += chunk.decode("utf-8", errors="ignore")
    while "\n" in buffer:
        line, buffer = buffer.split("\n", 1)
        line = line.strip()
        if not line or line.startswith(":") or not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            summary.setdefault("status", "completed")
            continue
        try:
            event = json.loads(data)
        except ValueError:
            continue
        if isinstance(event, dict):
            _merge_stream_event(summary, event, choice_states)
    if len(buffer) > 65536:
        return buffer[-4096:]
    return buffer


def _merge_stream_event(
    summary: dict[str, Any],
    event: dict[str, Any],
    choice_states: dict[int, dict[str, Any]] | None = None,
) -> None:
    for key in ("id", "object"):
        value = event.get(key)
        if value is not None:
            summary.setdefault(key, value)

    usage = event.get("usage")
    if isinstance(usage, dict):
        summary["usage"] = usage

    choices = event.get("choices")
    if isinstance(choices, list):
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            if choice_states is not None:
                _merge_chat_completion_choice(choice_states, choice)
            if choice.get("finish_reason"):
                index = _choice_index(choice)
                summary["choices"] = [{"index": index, "finish_reason": choice["finish_reason"]}]

    response = event.get("response")
    if isinstance(response, dict):
        for key in ("id", "object", "status"):
            value = response.get(key)
            if value is not None:
                summary[key] = value
        response_usage = response.get("usage")
        if isinstance(response_usage, dict):
            summary["usage"] = response_usage

    event_type = event.get("type")
    if event_type == "response.completed":
        summary["status"] = "completed"
    elif event_type == "response.failed":
        summary["status"] = "failed"
    elif event_type == "response.cancelled":
        summary["status"] = "cancelled"
    elif event_type == "response.output_text.delta" and isinstance(event.get("delta"), str):
        summary.setdefault("output_text_parts", []).append(event["delta"])
    elif event_type == "response.reasoning_text.delta" and isinstance(event.get("delta"), str):
        summary.setdefault("reasoning_text_parts", []).append(event["delta"])


def _merge_chat_completion_choice(choice_states: dict[int, dict[str, Any]], choice: dict[str, Any]) -> None:
    index = _choice_index(choice)
    state = choice_states.setdefault(
        index,
        {
            "index": index,
            "role": "assistant",
            "content_parts": [],
            "reasoning_parts": [],
            "tool_calls": {},
            "function_call": {"name_parts": [], "arguments_parts": []},
            "finish_reason": None,
        },
    )

    message = choice.get("message")
    if isinstance(message, dict):
        _merge_message_payload(state, message)

    delta = choice.get("delta")
    if isinstance(delta, dict):
        _merge_message_payload(state, delta)

    if choice.get("finish_reason"):
        state["finish_reason"] = choice["finish_reason"]


def _merge_message_payload(state: dict[str, Any], payload: dict[str, Any]) -> None:
    role = payload.get("role")
    if isinstance(role, str) and role:
        state["role"] = role

    content = payload.get("content")
    if isinstance(content, str):
        state.setdefault("content_parts", []).append(content)
    elif content is not None:
        state["content"] = content

    for key in ("reasoning", "reasoning_content"):
        reasoning = payload.get(key)
        if isinstance(reasoning, str):
            state.setdefault("reasoning_parts", []).append(reasoning)
        elif reasoning is not None:
            state[key] = reasoning

    tool_calls = payload.get("tool_calls")
    if isinstance(tool_calls, list):
        for raw_tool_call in tool_calls:
            if isinstance(raw_tool_call, dict):
                _merge_tool_call_delta(state.setdefault("tool_calls", {}), raw_tool_call)

    function_call = payload.get("function_call")
    if isinstance(function_call, dict):
        _merge_function_call_delta(state.setdefault("function_call", {}), function_call)


def _merge_tool_call_delta(tool_calls: dict[int, dict[str, Any]], delta: dict[str, Any]) -> None:
    index = _safe_int(delta.get("index"), len(tool_calls))
    state = tool_calls.setdefault(
        index,
        {
            "index": index,
            "id_parts": [],
            "type": None,
            "function": {"name_parts": [], "arguments_parts": []},
        },
    )

    if isinstance(delta.get("id"), str):
        state.setdefault("id_parts", []).append(delta["id"])
    if isinstance(delta.get("type"), str):
        state["type"] = delta["type"]

    function = delta.get("function")
    if isinstance(function, dict):
        _merge_function_call_delta(state.setdefault("function", {}), function)


def _merge_function_call_delta(state: dict[str, Any], delta: dict[str, Any]) -> None:
    name = delta.get("name")
    if isinstance(name, str):
        state.setdefault("name_parts", []).append(name)
    arguments = delta.get("arguments")
    if isinstance(arguments, str):
        state.setdefault("arguments_parts", []).append(arguments)


def _stream_response_body_for_telemetry(
    *,
    summary: dict[str, Any],
    choice_states: dict[int, dict[str, Any]],
    stream_text: str,
    stream_total_bytes: int,
    stream_truncated: bool,
) -> dict[str, Any]:
    body = dict(summary)
    if "output_text_parts" in body:
        body["output_text"] = "".join(str(part) for part in body.pop("output_text_parts"))
    if "reasoning_text_parts" in body:
        body["reasoning_text"] = "".join(str(part) for part in body.pop("reasoning_text_parts"))
    if choice_states:
        body["choices"] = [_finalize_choice_state(choice_states[index]) for index in sorted(choice_states)]
    body["stream_text"] = stream_text
    body["stream_total_bytes"] = stream_total_bytes
    body["stream_truncated"] = stream_truncated
    return body


def _finalize_choice_state(state: dict[str, Any]) -> dict[str, Any]:
    message: dict[str, Any] = {"role": state.get("role") or "assistant"}
    content_parts = state.get("content_parts") or []
    if content_parts:
        message["content"] = "".join(str(part) for part in content_parts)
    elif "content" in state:
        message["content"] = state["content"]

    reasoning_parts = state.get("reasoning_parts") or []
    if reasoning_parts:
        message["reasoning"] = "".join(str(part) for part in reasoning_parts)
    elif "reasoning" in state:
        message["reasoning"] = state["reasoning"]
    elif "reasoning_content" in state:
        message["reasoning_content"] = state["reasoning_content"]

    tool_calls = _finalize_tool_calls(state.get("tool_calls") or {})
    if tool_calls:
        message["tool_calls"] = tool_calls

    function_call = _finalize_function_call(state.get("function_call") or {})
    if function_call:
        message["function_call"] = function_call

    choice = {
        "index": _safe_int(state.get("index"), 0),
        "message": message,
    }
    if state.get("finish_reason"):
        choice["finish_reason"] = state["finish_reason"]
    return choice


def _finalize_tool_calls(tool_calls: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    finalized: list[dict[str, Any]] = []
    for index in sorted(tool_calls):
        state = tool_calls[index]
        item: dict[str, Any] = {"index": index}
        id_parts = state.get("id_parts") or []
        if id_parts:
            item["id"] = "".join(str(part) for part in id_parts)
        if state.get("type"):
            item["type"] = state["type"]
        function = _finalize_function_call(state.get("function") or {})
        if function:
            item["function"] = function
        finalized.append(item)
    return finalized


def _finalize_function_call(state: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    name_parts = state.get("name_parts") or []
    if name_parts:
        result["name"] = "".join(str(part) for part in name_parts)
    argument_parts = state.get("arguments_parts") or []
    if argument_parts:
        result["arguments"] = "".join(str(part) for part in argument_parts)
    return result


def _choice_index(choice: dict[str, Any]) -> int:
    return _safe_int(choice.get("index"), 0)


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


app = create_app()
