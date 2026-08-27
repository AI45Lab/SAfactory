from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator, Coroutine
from contextlib import asynccontextmanager
from dataclasses import replace
from typing import Any
from urllib.parse import parse_qsl, urlencode

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from core.perf_trace import PerfTrace
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


def _without_beta_query(query: str) -> str | None:
    filtered = [
        (name, value)
        for name, value in parse_qsl(query, keep_blank_values=True)
        if name != "beta"
    ]
    return urlencode(filtered, doseq=True) or None


def create_app(cfg: GatewayConfig | None = None, storage: GatewayStorage | None = None) -> FastAPI:
    cfg = cfg or load_gateway_config()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        log.info(
            "Gateway startup begin: storage_type=%s telemetry_mode=%s max_steps=%s routes=%s",
            cfg.storage_type,
            cfg.telemetry_mode,
            cfg.max_steps,
            sorted((cfg.llm_routes or {}).keys()),
        )
        app.state.gateway_ready = False
        app.state.gateway_draining = False
        app.state.gateway_config = cfg
        log.info("Gateway storage init begin")
        app.state.gateway_storage = storage or await GatewayStorage.from_config(cfg)
        log.info("Gateway storage init complete")
        app.state.gateway_router = LLMRouter(cfg)
        app.state.gateway_admission = AdmissionController(cfg)
        app.state.gateway_forwarder = InferenceForwarder(cfg)
        app.state.gateway_resolver = SessionResolver(cfg)
        app.state.gateway_request_logger = GatewayRequestLogger(cfg)
        app.state.gateway_request_logger.start()
        app.state.gateway_telemetry = TelemetryRecorder(cfg, app.state.gateway_storage)
        app.state.gateway_stream_finalize_tasks: set[asyncio.Task[None]] = set()
        log.info("Gateway telemetry start begin")
        await app.state.gateway_telemetry.start()
        log.info("Gateway telemetry start complete")
        app.state.gateway_ready = True
        log.info("Gateway startup complete: ready=true")
        try:
            yield
        finally:
            log.info("Gateway shutdown begin: drain_timeout_s=%s", cfg.drain_timeout_s)
            app.state.gateway_ready = False
            app.state.gateway_draining = True
            app.state.gateway_admission.draining = True
            await _wait_for_drain(app, cfg.drain_timeout_s)
            await _shutdown_stream_finalize_tasks(
                app.state.gateway_stream_finalize_tasks,
                cfg.drain_timeout_s,
            )
            log.info("Gateway drain complete; stopping telemetry and clients")
            await app.state.gateway_telemetry.stop()
            await app.state.gateway_forwarder.close()
            app.state.gateway_request_logger.close()
            await app.state.gateway_storage.close()
            log.info("Gateway shutdown complete")

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
        headers: dict[str, str] = {}
        route_reserved = False
        release_in_finally = True
        trace_status: str | None = None
        trace_extra: dict[str, Any] = {}
        trace = PerfTrace(
            "gateway.session_request",
            logger=log,
            context={"endpoint": endpoint, "path_session_id": path_session_id},
        )

        resolver: SessionResolver = request.app.state.gateway_resolver
        admission: AdmissionController = request.app.state.gateway_admission
        router: LLMRouter = request.app.state.gateway_router
        forwarder: InferenceForwarder = request.app.state.gateway_forwarder
        telemetry: TelemetryRecorder = request.app.state.gateway_telemetry
        storage: GatewayStorage = request.app.state.gateway_storage
        request_logger: GatewayRequestLogger = request.app.state.gateway_request_logger

        try:
            with trace.span("parse_request_json"):
                payload = await request.json()
            if not isinstance(payload, dict):
                raise ValueError("request body must be a JSON object")

            with trace.span("resolve_request"):
                ctx = await resolver.resolve(
                    payload,
                    endpoint=endpoint,
                    path_session_id=path_session_id,
                )
            trace.update_context(
                request_id=ctx.request_id,
                session_id=ctx.session_id,
                model=ctx.requested_model,
                stream=ctx.is_stream,
            )
            log.info(
                "Gateway request resolved: request_id=%s session_id=%s endpoint=%s model=%s stream=%s",
                ctx.request_id,
                ctx.session_id,
                ctx.endpoint,
                ctx.requested_model,
                ctx.is_stream,
            )
            with trace.span("get_or_create_binding"):
                binding = await resolver.get_or_create_binding(ctx)
            log.debug("Gateway request binding loaded: request_id=%s session_id=%s", ctx.request_id, ctx.session_id)
            with trace.span("bind_session_environment"):
                await storage.bind_session_environment(binding)
            log.info(
                "Gateway request start: request_id=%s session_id=%s endpoint=%s model=%s stream=%s "
                "binding_status=%s step_count=%s queue_depth=%d",
                ctx.request_id,
                ctx.session_id,
                ctx.endpoint,
                ctx.requested_model,
                ctx.is_stream,
                binding.status,
                binding.step_count_for(ctx.requested_model),
                telemetry.queue_depth(),
            )
            log.debug(
                "Gateway binding resolved: request_id=%s job_id=%s env_name=%s group_id=%s "
                "active_requests=%d active_streams=%d",
                ctx.request_id,
                binding.job_id,
                binding.env_name,
                binding.group_id,
                binding.active_request_count,
                binding.active_stream_count,
            )
            if binding.status == "closed":
                if binding.is_model_truncated(ctx.requested_model):
                    ctx = replace(ctx, synthetic_stop=True)
                    reason = binding.model_truncate_reason(ctx.requested_model) or "max_steps_reached"
                    log.info(
                        "Gateway synthetic stop for closed truncated session: request_id=%s reason=%s",
                        ctx.request_id,
                        reason,
                    )
                    with trace.span("synthetic_stop_response", reason=reason):
                        response = await _return_synthetic_stop(
                            ctx=ctx,
                            binding=binding,
                            payload=payload,
                            reason=reason,
                            cfg=request.app.state.gateway_config,
                            request_logger=request_logger,
                            telemetry=telemetry,
                        )
                    trace_status = "synthetic_stop"
                    trace_extra = {"reason": reason, "status_code": 200}
                    return response
                raise SessionClosedError(f"session {ctx.session_id} is closed")
            if binding.is_model_truncated(ctx.requested_model):
                ctx = replace(ctx, synthetic_stop=True)
                reason = binding.model_truncate_reason(ctx.requested_model) or "max_steps_reached"
                log.info("Gateway synthetic stop: request_id=%s reason=%s", ctx.request_id, reason)
                with trace.span("synthetic_stop_response", reason=reason):
                    response = await _return_synthetic_stop(
                        ctx=ctx,
                        binding=binding,
                        payload=payload,
                        reason=reason,
                        cfg=request.app.state.gateway_config,
                        request_logger=request_logger,
                        telemetry=telemetry,
                    )
                trace_status = "synthetic_stop"
                trace_extra = {"reason": reason, "status_code": 200}
                return response

            if cfg.max_steps < 0 or binding.step_count_for(ctx.requested_model) < cfg.max_steps:
                if telemetry.should_reject_new_requests(ctx.session_id):
                    raise AdmissionRejected("telemetry queue is full", 503)
                with trace.span("select_target_initial"):
                    target = await router.select_target(ctx, binding)
                log.debug(
                    "Gateway target selected: request_id=%s route_model=%s base_url=%s max_concurrency=%d",
                    ctx.request_id,
                    target.route_model,
                    target.base_url,
                    target.max_concurrency,
                )

            with trace.span("admission_acquire"):
                decision = await admission.acquire_request(ctx, binding, target)
            log.debug(
                "Gateway admission decision: request_id=%s action=%s llm_step_index=%s reason=%s",
                ctx.request_id,
                decision.action,
                decision.llm_step_index,
                decision.stop_reason,
            )
            if decision.action == "stop":
                ctx = replace(ctx, synthetic_stop=True)
                log.info(
                    "Gateway synthetic stop from admission: request_id=%s reason=%s",
                    ctx.request_id,
                    decision.stop_reason or "max_steps_reached",
                )
                reason = decision.stop_reason or "max_steps_reached"
                with trace.span("synthetic_stop_response", reason=reason):
                    response = await _return_synthetic_stop(
                        ctx=ctx,
                        binding=binding,
                        payload=payload,
                        reason=reason,
                        cfg=request.app.state.gateway_config,
                        request_logger=request_logger,
                        telemetry=telemetry,
                    )
                trace_status = "synthetic_stop"
                trace_extra = {"reason": reason, "status_code": 200}
                return response
            ctx = replace(ctx, llm_step_index=decision.llm_step_index)

            if target is None:
                with trace.span("select_target_fallback"):
                    target = await router.select_target(ctx, binding)
            route_reserved = True
            trace.update_context(route_model=target.route_model, upstream_base_url=target.base_url)
            with trace.span("route_acquire"):
                await router.on_acquire(target.route_model, is_stream=ctx.is_stream)
            log.debug(
                "Gateway route acquired: request_id=%s route_model=%s stream=%s",
                ctx.request_id,
                target.route_model,
                ctx.is_stream,
            )
            # The Shanhai/Bedrock route rejects Claude Code's beta query flag.
            # Preserve any other native Anthropic query parameters.
            anthropic_query_string = (
                _without_beta_query(request.url.query)
                if endpoint == "messages"
                else None
            )

            headers = (
                forwarder.build_anthropic_headers(
                    target,
                    request.headers,
                    session_id=ctx.session_id,
                    request_id=ctx.request_id,
                    llm_step_index=ctx.llm_step_index,
                )
                if endpoint == "messages"
                else forwarder.build_upstream_headers(target, session_id=ctx.session_id)
            )
            with trace.span("request_log_write"):
                await request_logger.log_request(ctx, binding, target, payload)
            if ctx.is_stream:
                log.info(
                    "Gateway upstream stream open begin: request_id=%s route_model=%s endpoint=%s",
                    ctx.request_id,
                    target.route_model,
                    endpoint,
                )
                with trace.span("upstream_stream_open"):
                    opened = await _open_stream(
                        forwarder,
                        target,
                        endpoint,
                        payload,
                        headers,
                        anthropic_query_string=anthropic_query_string,
                    )
                trace.mark(
                    "upstream_stream_opened",
                    status_code=opened.status_code,
                    upstream_latency_ms=opened.upstream_latency_ms,
                    upstream_open_latency_ms=opened.upstream_latency_ms,
                )
                log.info(
                    "Gateway upstream stream opened: request_id=%s status=%d upstream_latency_ms=%.2f",
                    ctx.request_id,
                    opened.status_code,
                    opened.upstream_latency_ms,
                )
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
                        request_headers=headers,
                        trace=trace,
                        finalize_tasks=request.app.state.gateway_stream_finalize_tasks,
                    ),
                    status_code=opened.status_code,
                    media_type=opened.media_type,
                )

            log.info(
                "Gateway upstream request begin: request_id=%s route_model=%s endpoint=%s",
                ctx.request_id,
                target.route_model,
                endpoint,
            )
            with trace.span("upstream_json_forward"):
                result = await _forward_json(
                    forwarder,
                    target,
                    endpoint,
                    payload,
                    headers,
                    anthropic_query_string=anthropic_query_string,
                )
            latency_ms = (time.perf_counter() - started) * 1000
            trace.mark(
                "upstream_json_complete",
                status_code=result.status_code,
                upstream_latency_ms=result.upstream_latency_ms,
                total_latency_ms=latency_ms,
            )
            log.info(
                "Gateway upstream request complete: request_id=%s status=%d upstream_latency_ms=%.2f total_latency_ms=%.2f",
                ctx.request_id,
                result.status_code,
                result.upstream_latency_ms,
                latency_ms,
            )
            with trace.span("request_log_response"):
                await request_logger.log_response(
                    ctx,
                    binding,
                    target,
                    status_code=result.status_code,
                    response_body=result.body,
                    latency_ms=latency_ms,
                    upstream_latency_ms=result.upstream_latency_ms,
                )
            with trace.span("router_mark_success"):
                await router.mark_route_result(
                    target.route_model,
                    True,
                    result.upstream_latency_ms,
                    result.status_code,
                )
            with trace.span("telemetry_enqueue_success"):
                await telemetry.enqueue_success(
                    ctx,
                    binding,
                    target,
                    payload,
                    result.body,
                    latency_ms,
                    upstream_latency_ms=result.upstream_latency_ms,
                    request_headers=headers,
                )
            log.info(
                "Gateway request complete: request_id=%s session_id=%s model=%s status=%d total_latency_ms=%.2f",
                ctx.request_id,
                ctx.session_id,
                ctx.requested_model,
                result.status_code,
                latency_ms,
            )
            trace_status = "success" if result.status_code < 400 else "failed"
            trace_extra = {"status_code": result.status_code, "total_latency_ms": latency_ms}
            return JSONResponse(result.body, status_code=result.status_code)
        except asyncio.CancelledError:
            trace_status = "cancelled"
            trace_extra = {"error_type": "CancelledError"}
            raise
        except Exception as exc:
            latency_ms = (time.perf_counter() - started) * 1000
            upstream_latency_ms = _upstream_latency_ms_from_exception(exc)
            status_code, error_body = forwarder.normalize_error(exc)
            log.warning(
                "Gateway request failed: request_id=%s session_id=%s endpoint=%s model=%s status=%d "
                "latency_ms=%.2f upstream_latency_ms=%s error=%s",
                ctx.request_id if ctx else None,
                ctx.session_id if ctx else path_session_id,
                endpoint,
                ctx.requested_model if ctx else None,
                status_code,
                latency_ms,
                f"{upstream_latency_ms:.2f}" if upstream_latency_ms is not None else None,
                exc,
            )
            if target is not None:
                with trace.span("router_mark_failure"):
                    await router.mark_route_result(target.route_model, False, latency_ms, status_code)
            with trace.span("request_log_error"):
                await request_logger.log_error(
                    endpoint=endpoint,
                    path_session_id=path_session_id,
                    request_body=payload,
                    error_body=error_body,
                    error_text=str(exc),
                    status_code=status_code,
                    latency_ms=latency_ms,
                    upstream_latency_ms=upstream_latency_ms,
                    ctx=ctx,
                    binding=binding,
                    target=target,
                )
            if ctx is not None and binding is not None:
                with trace.span("telemetry_enqueue_failure"):
                    await telemetry.enqueue_failure(
                        ctx,
                        binding,
                        target,
                        payload,
                        str(exc),
                        status_code,
                        latency_ms,
                        upstream_latency_ms=upstream_latency_ms,
                        response_body=error_body,
                        request_headers=headers,
                    )
            trace_status = "failed"
            trace_extra = {
                "status_code": status_code,
                "total_latency_ms": latency_ms,
                "upstream_latency_ms": upstream_latency_ms,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            return JSONResponse(error_body, status_code=status_code)
        finally:
            if release_in_finally:
                if route_reserved and target is not None and ctx is not None:
                    with trace.span("route_release"):
                        await router.on_release(target.route_model, is_stream=ctx.is_stream)
                    log.debug(
                        "Gateway route released: request_id=%s route_model=%s stream=%s",
                        ctx.request_id,
                        target.route_model,
                        ctx.is_stream,
                    )
                with trace.span("admission_release"):
                    await admission.release(ctx, binding, target)
                if trace_status is not None:
                    trace.emit_summary(status=trace_status, **trace_extra)

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

    async def handle_session_anthropic_messages(session_id: str, request: Request) -> Response:
        return await handle_inference_request(
            request,
            endpoint="messages",
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
        trace_status: str | None = None
        trace_extra: dict[str, Any] = {}
        trace = PerfTrace(
            "gateway.standard_request",
            logger=log,
            context={"endpoint": endpoint},
        )

        router: LLMRouter = request.app.state.gateway_router
        forwarder: InferenceForwarder = request.app.state.gateway_forwarder

        try:
            with trace.span("parse_request_json"):
                payload = await request.json()
            if not isinstance(payload, dict):
                raise ValueError("request body must be a JSON object")

            requested_model = payload.get("model")
            if not isinstance(requested_model, str) or not requested_model:
                raise ValueError("request body requires a non-empty model field")

            is_stream = bool(payload.get("stream", False))
            trace.update_context(model=requested_model, stream=is_stream)
            log.info(
                "Gateway standard request start: endpoint=%s model=%s stream=%s",
                endpoint,
                requested_model,
                is_stream,
            )
            with trace.span("select_standard_target"):
                target = await router.select_standard_target(
                    requested_model=requested_model,
                    is_stream=is_stream,
                )
            trace.update_context(route_model=target.route_model, upstream_base_url=target.base_url)
            route_reserved = True
            with trace.span("route_acquire"):
                await router.on_acquire(target.route_model, is_stream=is_stream)

            headers = forwarder.build_upstream_headers(target)
            if is_stream:
                log.info(
                    "Gateway standard upstream stream open begin: endpoint=%s model=%s",
                    endpoint,
                    requested_model,
                )
                with trace.span("upstream_stream_open"):
                    opened = await _open_stream(forwarder, target, endpoint, payload, headers)
                trace.mark(
                    "upstream_stream_opened",
                    status_code=opened.status_code,
                    upstream_latency_ms=opened.upstream_latency_ms,
                    upstream_open_latency_ms=opened.upstream_latency_ms,
                )
                log.info(
                    "Gateway standard upstream stream opened: endpoint=%s model=%s status=%d upstream_latency_ms=%.2f",
                    endpoint,
                    requested_model,
                    opened.status_code,
                    opened.upstream_latency_ms,
                )
                release_in_finally = False
                return StreamingResponse(
                    _standard_stream_and_finalize(
                        opened=opened,
                        started=started,
                        target=target,
                        router=router,
                        is_stream=is_stream,
                        trace=trace,
                    ),
                    status_code=opened.status_code,
                    media_type=opened.media_type,
                )

            with trace.span("upstream_json_forward"):
                result = await _forward_json(forwarder, target, endpoint, payload, headers)
            latency_ms = (time.perf_counter() - started) * 1000
            trace.mark(
                "upstream_json_complete",
                status_code=result.status_code,
                upstream_latency_ms=result.upstream_latency_ms,
                total_latency_ms=latency_ms,
            )
            with trace.span("router_mark_success"):
                await router.mark_route_result(
                    target.route_model,
                    True,
                    result.upstream_latency_ms,
                    result.status_code,
                )
            log.info(
                "Gateway standard request complete: endpoint=%s model=%s status=%d upstream_latency_ms=%.2f total_latency_ms=%.2f",
                endpoint,
                requested_model,
                result.status_code,
                result.upstream_latency_ms,
                latency_ms,
            )
            trace_status = "success" if result.status_code < 400 else "failed"
            trace_extra = {"status_code": result.status_code, "total_latency_ms": latency_ms}
            return JSONResponse(result.body, status_code=result.status_code)
        except asyncio.CancelledError:
            trace_status = "cancelled"
            trace_extra = {"error_type": "CancelledError"}
            raise
        except Exception as exc:
            latency_ms = (time.perf_counter() - started) * 1000
            upstream_latency_ms = _upstream_latency_ms_from_exception(exc)
            status_code, error_body = forwarder.normalize_error(exc)
            log.warning(
                "Gateway standard request failed: endpoint=%s model=%s status=%d latency_ms=%.2f upstream_latency_ms=%s error=%s",
                endpoint,
                payload.get("model") if isinstance(payload, dict) else None,
                status_code,
                latency_ms,
                f"{upstream_latency_ms:.2f}" if upstream_latency_ms is not None else None,
                exc,
            )
            if target is not None:
                with trace.span("router_mark_failure"):
                    await router.mark_route_result(target.route_model, False, latency_ms, status_code)
            trace_status = "failed"
            trace_extra = {
                "status_code": status_code,
                "total_latency_ms": latency_ms,
                "upstream_latency_ms": upstream_latency_ms,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            return JSONResponse(error_body, status_code=status_code)
        finally:
            if release_in_finally and route_reserved and target is not None:
                with trace.span("route_release"):
                    await router.on_release(target.route_model, is_stream=is_stream)
            if release_in_finally and trace_status is not None:
                trace.emit_summary(status=trace_status, **trace_extra)

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
        completion_mode = "complete"
        try:
            body = await request.json()
            if isinstance(body, dict):
                if body.get("reason"):
                    reason = str(body["reason"])
                completion_mode = str(body.get("completion_mode") or completion_mode).strip().lower()
        except Exception:
            pass
        if completion_mode not in {"complete", "seal", "abort"}:
            raise HTTPException(
                status_code=400,
                detail="completion_mode must be 'complete', 'seal', or 'abort'",
            )
        binding = await resolver.close_session(session_id, reason=reason)
        log.info(
            "Gateway session close requested: session_id=%s reason=%s completion_mode=%s",
            session_id,
            reason,
            completion_mode,
        )
        drained = True
        if cfg.close_mode == "soft_close":
            drained = await _wait_for_session_drain(binding, cfg.drain_timeout_s)

        telemetry_status = "queued" if telemetry.async_writes_enabled else "flushed"
        try:
            await telemetry.enqueue_session_close(
                binding,
                is_session_completed=completion_mode == "complete",
            )
        except asyncio.TimeoutError:
            telemetry_status = "timeout"
            log.warning(
                "Gateway session close telemetry timed out; returning closed status: "
                "session_id=%s reason=%s timeout_s=%.1f",
                session_id,
                reason,
                cfg.telemetry_write_timeout_s,
            )
        return {
            "session_id": session_id,
            "status": binding.status,
            "drained": drained,
            "telemetry_status": telemetry_status,
            "completion_mode": completion_mode,
        }

    async def clear_session_cache(payload: dict[str, Any]) -> dict[str, Any]:
        raw_session_ids = payload.get("session_ids")
        if not isinstance(raw_session_ids, list):
            raise HTTPException(status_code=400, detail="session_ids must be a non-empty list")
        session_ids = list(dict.fromkeys(
            item.strip()
            for item in raw_session_ids
            if isinstance(item, str) and item.strip()
        ))
        if not session_ids:
            raise HTTPException(status_code=400, detail="session_ids must be a non-empty list")

        resolver: SessionResolver = app.state.gateway_resolver
        removed = await resolver.clear_session_cache(session_ids)
        log.info("Gateway session cache cleared: sessions=%d removed=%d", len(session_ids), removed)
        return {"session_ids": session_ids, "removed": removed}

    app.add_api_route(
        f"{session_root}/cache/cleanup",
        clear_session_cache,
        methods=["POST"],
    )
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
        f"{session_root}/{{session_id}}/v1/messages",
        handle_session_anthropic_messages,
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
    *,
    anthropic_query_string: str | None = None,
):
    if endpoint == "chat/completions":
        return await forwarder.forward_chat(target, payload, headers)
    if endpoint == "responses":
        return await forwarder.forward_responses(target, payload, headers)
    if endpoint == "messages":
        if anthropic_query_string is None:
            return await forwarder.forward_anthropic_messages(target, payload, headers)
        return await forwarder.forward_anthropic_messages(
            target,
            payload,
            headers,
            query_string=anthropic_query_string,
        )
    raise ValueError(f"unsupported endpoint {endpoint}")


async def _open_stream(
    forwarder: InferenceForwarder,
    target: LLMRouteTarget,
    endpoint: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    *,
    anthropic_query_string: str | None = None,
) -> StreamForwardContext:
    if endpoint == "chat/completions":
        return await forwarder.open_chat_stream(target, payload, headers)
    if endpoint == "responses":
        return await forwarder.open_responses_stream(target, payload, headers)
    if endpoint == "messages":
        if anthropic_query_string is None:
            return await forwarder.open_anthropic_messages_stream(
                target,
                payload,
                headers,
            )
        return await forwarder.open_anthropic_messages_stream(
            target,
            payload,
            headers,
            query_string=anthropic_query_string,
        )
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


def _upstream_latency_ms_from_exception(exc: Exception) -> float | None:
    value = getattr(exc, "upstream_latency_ms", None)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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
    request_headers: dict[str, str],
    trace: PerfTrace,
    finalize_tasks: set[asyncio.Task[None]],
) -> AsyncIterator[bytes]:
    first_chunk_at: float | None = None
    chunk_count = 0
    output_bytes = 0
    status_code = opened.status_code
    error_text: str | None = None
    client_cancelled = False
    upstream_cancelled = False
    stream_response_body: dict[str, Any] = {}
    stream_metadata_buffer = b""
    stream_choice_states: dict[int, dict[str, Any]] = {}
    stream_text_parts: list[str] = []
    stream_total_bytes = 0
    stream_capture = request_logger.new_stream_capture()

    try:
        log.info(
            "Gateway stream proxy start: request_id=%s session_id=%s model=%s status=%d",
            ctx.request_id,
            ctx.session_id,
            ctx.requested_model,
            opened.status_code,
        )
        trace.mark("stream_proxy_start", status_code=opened.status_code)
        async for chunk in opened.response.content.iter_any():
            if chunk:
                stream_capture.append(chunk)
                stream_total_bytes += len(chunk)
                if ctx.endpoint != "messages":
                    stream_text_parts.append(chunk.decode("utf-8", errors="replace"))
                if first_chunk_at is None:
                    first_chunk_at = time.perf_counter()
                    log.info(
                        "Gateway stream first chunk: request_id=%s ttft_ms=%.2f bytes=%d",
                        ctx.request_id,
                        (first_chunk_at - started) * 1000,
                        len(chunk),
                    )
                    trace.mark(
                        "stream_first_chunk",
                        ttft_ms=(first_chunk_at - started) * 1000,
                        bytes=len(chunk),
                    )
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
        log.warning("Gateway stream client cancelled: request_id=%s", ctx.request_id)
        raise
    except Exception as exc:
        upstream_cancelled = True
        status_code = 502
        error_text = str(exc)
        log.warning("Gateway stream upstream failed: request_id=%s error=%s", ctx.request_id, exc)
        raise
    finally:
        opened.response.close()
        latency_ms = (time.perf_counter() - started) * 1000
        upstream_stream_total_ms = (time.perf_counter() - opened.upstream_started_perf) * 1000
        ttft_ms = (first_chunk_at - started) * 1000 if first_chunk_at is not None else None
        stats = StreamTelemetryStats(
            ttft_ms=ttft_ms,
            output_chunk_count=chunk_count,
            output_bytes=output_bytes,
            client_cancelled=client_cancelled,
            upstream_cancelled=upstream_cancelled,
        )
        ok = status_code < 400

        async def finalize() -> None:
            try:
                log.info(
                    "Gateway stream finalize begin: request_id=%s status=%d chunks=%d bytes=%d total_latency_ms=%.2f",
                    ctx.request_id,
                    status_code,
                    chunk_count,
                    output_bytes,
                    latency_ms,
                )
                trace.mark(
                    "stream_finalize_begin",
                    status_code=status_code,
                    chunk_count=chunk_count,
                    output_bytes=output_bytes,
                    total_latency_ms=latency_ms,
                    upstream_open_latency_ms=opened.upstream_latency_ms,
                    upstream_stream_total_ms=upstream_stream_total_ms,
                    ttft_ms=ttft_ms,
                )
                stream_body = stream_capture.snapshot()
                stream_text = "".join(stream_text_parts)
                telemetry_response_body = (
                    _anthropic_stream_response_body_for_telemetry(stream_response_body)
                    if ctx.endpoint == "messages"
                    else _stream_response_body_for_telemetry(
                        summary=stream_response_body,
                        choice_states=stream_choice_states,
                        stream_text=stream_text,
                        stream_total_bytes=stream_total_bytes,
                        stream_truncated=False,
                    )
                )
                # Anthropic stays provider-native both on the wire and in the
                # trajectory; only the SSE event framing is removed.
                trajectory_response_text = (
                    json.dumps(
                        telemetry_response_body,
                        ensure_ascii=False,
                        default=str,
                    )
                    if ctx.endpoint == "messages"
                    else None
                )

                if ok:
                    with trace.span("telemetry_enqueue_stream_success"):
                        await telemetry.enqueue_success(
                            ctx,
                            binding,
                            target,
                            payload,
                            telemetry_response_body,
                            latency_ms,
                            upstream_latency_ms=upstream_stream_total_ms,
                            stream_stats=stats,
                            request_headers=request_headers,
                            response_text=trajectory_response_text,
                        )
                else:
                    with trace.span("telemetry_enqueue_stream_failure"):
                        await telemetry.enqueue_failure(
                            ctx,
                            binding,
                            target,
                            payload,
                            error_text or "stream failed",
                            status_code,
                            latency_ms,
                            upstream_latency_ms=upstream_stream_total_ms,
                            stream_stats=stats,
                            response_body=telemetry_response_body,
                            request_headers=request_headers,
                            response_text=trajectory_response_text,
                        )

                with trace.span("router_mark_stream_result"):
                    await router.mark_route_result(target.route_model, ok, latency_ms, status_code)
                with trace.span("request_log_stream_response"):
                    await request_logger.log_stream_response(
                        ctx,
                        binding,
                        target,
                        status_code=status_code,
                        stream_body=stream_body,
                        stream_summary=stream_response_body,
                        latency_ms=latency_ms,
                        upstream_latency_ms=upstream_stream_total_ms,
                        ttft_ms=ttft_ms,
                        output_chunk_count=chunk_count,
                        client_cancelled=client_cancelled,
                        upstream_cancelled=upstream_cancelled,
                        error_text=error_text,
                        upstream_open_latency_ms=opened.upstream_latency_ms,
                        upstream_stream_total_ms=upstream_stream_total_ms,
                    )
                log.info(
                    "Gateway stream finalize complete: request_id=%s status=%d telemetry_recorded=true",
                    ctx.request_id,
                    status_code,
                )
                trace.mark("stream_finalize_complete", status_code=status_code)
            finally:
                try:
                    with trace.span("route_release"):
                        await router.on_release(target.route_model, is_stream=ctx.is_stream)
                finally:
                    with trace.span("admission_release"):
                        await admission.release(ctx, binding, target)
                trace.update_context(
                    final_status_code=status_code,
                    stream_chunk_count=chunk_count,
                    stream_output_bytes=output_bytes,
                )
                trace.emit_summary(
                    status=(
                        "client_cancelled"
                        if client_cancelled
                        else "upstream_failed"
                        if upstream_cancelled
                        else "success"
                        if ok
                        else "failed"
                    ),
                    status_code=status_code,
                    total_latency_ms=latency_ms,
                    upstream_open_latency_ms=opened.upstream_latency_ms,
                    upstream_stream_total_ms=upstream_stream_total_ms,
                    ttft_ms=ttft_ms,
                )
                log.debug("Gateway stream resources released: request_id=%s", ctx.request_id)

        await _await_stream_finalize_cancel_safe(
            finalize(),
            request_id=ctx.request_id,
            finalize_tasks=finalize_tasks,
        )


async def _await_stream_finalize_cancel_safe(
    finalize: Coroutine[Any, Any, None],
    *,
    request_id: str,
    finalize_tasks: set[asyncio.Task[None]],
) -> None:
    task = asyncio.create_task(finalize, name=f"gateway-stream-finalize-{request_id}")
    finalize_tasks.add(task)

    def _on_done(done: asyncio.Task[None]) -> None:
        finalize_tasks.discard(done)
        if done.cancelled():
            log.error("Gateway stream finalize cancelled: request_id=%s", request_id)
            return
        error = done.exception()
        if error is not None:
            log.error(
                "Gateway stream finalize failed: request_id=%s error=%s",
                request_id,
                error,
                exc_info=(type(error), error, error.__traceback__),
            )

    task.add_done_callback(_on_done)
    try:
        await asyncio.shield(task)
    except asyncio.CancelledError:
        log.warning(
            "Gateway stream response task cancelled; finalize continues in background: request_id=%s",
            request_id,
        )
        raise


async def _standard_stream_and_finalize(
    *,
    opened: StreamForwardContext,
    started: float,
    target: LLMRouteTarget,
    router: LLMRouter,
    is_stream: bool,
    trace: PerfTrace,
) -> AsyncIterator[bytes]:
    status_code = opened.status_code
    first_chunk_at: float | None = None
    chunk_count = 0
    output_bytes = 0
    client_cancelled = False
    upstream_failed = False
    try:
        log.info(
            "Gateway standard stream proxy start: route_model=%s status=%d",
            target.route_model,
            opened.status_code,
        )
        trace.mark("stream_proxy_start", status_code=opened.status_code)
        async for chunk in opened.response.content.iter_any():
            if chunk:
                if first_chunk_at is None:
                    first_chunk_at = time.perf_counter()
                    trace.mark(
                        "stream_first_chunk",
                        ttft_ms=(first_chunk_at - started) * 1000,
                        bytes=len(chunk),
                    )
                chunk_count += 1
                output_bytes += len(chunk)
            yield chunk
    except asyncio.CancelledError:
        status_code = 499
        client_cancelled = True
        log.warning("Gateway standard stream client cancelled: route_model=%s", target.route_model)
        raise
    except Exception:
        status_code = 502
        upstream_failed = True
        log.warning("Gateway standard stream failed: route_model=%s", target.route_model, exc_info=True)
        raise
    finally:
        opened.response.close()
        latency_ms = (time.perf_counter() - started) * 1000
        upstream_stream_total_ms = (time.perf_counter() - opened.upstream_started_perf) * 1000
        ttft_ms = (first_chunk_at - started) * 1000 if first_chunk_at is not None else None
        with trace.span("router_mark_stream_result"):
            await router.mark_route_result(target.route_model, status_code < 400, latency_ms, status_code)
        with trace.span("route_release"):
            await router.on_release(target.route_model, is_stream=is_stream)
        trace.update_context(
            final_status_code=status_code,
            stream_chunk_count=chunk_count,
            stream_output_bytes=output_bytes,
        )
        trace.emit_summary(
            status=(
                "client_cancelled"
                if client_cancelled
                else "upstream_failed"
                if upstream_failed
                else "success"
                if status_code < 400
                else "failed"
            ),
            status_code=status_code,
            total_latency_ms=latency_ms,
            upstream_open_latency_ms=opened.upstream_latency_ms,
            upstream_stream_total_ms=upstream_stream_total_ms,
            ttft_ms=ttft_ms,
        )
        log.info(
            "Gateway standard stream complete: route_model=%s status=%d total_latency_ms=%.2f",
            target.route_model,
            status_code,
            latency_ms,
        )


def _standard_openai_root(session_root: str) -> str:
    if session_root.endswith("/sessions"):
        parent = session_root[: -len("/sessions")]
        return parent or "/v1"
    return "/v1"


async def _wait_for_drain(app: FastAPI, drain_timeout_s: int) -> None:
    admission: AdmissionController = app.state.gateway_admission
    deadline = time.monotonic() + max(0, drain_timeout_s)
    next_log_at = 0.0
    while time.monotonic() < deadline:
        snapshot = await admission.snapshot()
        if snapshot["inflight_requests"] <= 0 and snapshot["active_streams"] <= 0:
            return
        now = time.monotonic()
        if now >= next_log_at:
            log.info(
                "Gateway draining: inflight_requests=%d active_streams=%d",
                snapshot["inflight_requests"],
                snapshot["active_streams"],
            )
            next_log_at = now + 1.0
        await asyncio.sleep(0.05)


async def _shutdown_stream_finalize_tasks(
    finalize_tasks: set[asyncio.Task[None]],
    timeout_s: float,
) -> None:
    tasks = set(finalize_tasks)
    if not tasks:
        return

    log.info("Gateway waiting for stream finalization tasks: count=%d", len(tasks))
    _, pending = await asyncio.wait(tasks, timeout=max(0.0, float(timeout_s)))
    if pending:
        log.error(
            "Gateway stream finalization timed out; cancelling unfinished tasks: count=%d",
            len(pending),
        )
        for task in pending:
            task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


async def _wait_for_session_drain(binding: GatewaySessionBinding, drain_timeout_s: int) -> bool:
    deadline = time.monotonic() + max(0, drain_timeout_s)
    next_log_at = 0.0
    while binding.active_request_count > 0 or binding.active_stream_count > 0:
        if time.monotonic() >= deadline:
            log.warning(
                "Gateway session close drain timed out: session_id=%s active_requests=%d active_streams=%d",
                binding.session_id,
                binding.active_request_count,
                binding.active_stream_count,
            )
            return False
        now = time.monotonic()
        if now >= next_log_at:
            log.info(
                "Gateway session close draining: session_id=%s active_requests=%d active_streams=%d",
                binding.session_id,
                binding.active_request_count,
                binding.active_stream_count,
            )
            next_log_at = now + 1.0
        await asyncio.sleep(0.05)
    return True


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
    buffer: bytes,
    summary: dict[str, Any],
    choice_states: dict[int, dict[str, Any]],
) -> bytes:
    buffer += chunk
    while b"\n" in buffer:
        line, buffer = buffer.split(b"\n", 1)
        line = line.strip()
        if not line or line.startswith(b":") or not line.startswith(b"data:"):
            continue
        data = line[5:].strip()
        if data == b"[DONE]":
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
    event_type = event.get("type")
    if event_type in {
        "message_start",
        "content_block_start",
        "content_block_delta",
        "message_delta",
    }:
        _merge_anthropic_stream_event(summary, event)
        return

    for key in ("id", "object"):
        value = event.get(key)
        if value is not None:
            summary.setdefault(key, value)

    usage = event.get("usage")
    if isinstance(usage, dict):
        summary["usage"] = usage

    metadata = event.get("metadata")
    if isinstance(metadata, dict):
        summary["metadata"] = metadata

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
        for key in ("id", "object", "status", "output"):
            value = response.get(key)
            if value is not None:
                summary[key] = value
        response_usage = response.get("usage")
        if isinstance(response_usage, dict):
            summary["usage"] = response_usage

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


def _merge_anthropic_stream_event(summary: dict[str, Any], event: dict[str, Any]) -> None:
    event_type = event.get("type")
    if event_type == "message_start":
        message = event.get("message")
        if not isinstance(message, dict):
            return
        summary["anthropic_message"] = dict(message)
        usage = message.get("usage")
        if isinstance(usage, dict):
            summary["usage"] = dict(usage)
        return

    if event_type == "content_block_start":
        block = event.get("content_block")
        if not isinstance(block, dict):
            return
        index = _safe_int(event.get("index"), 0)
        state = summary.setdefault("anthropic_blocks", {}).setdefault(index, {})
        state.update(block)
        for key in ("text", "thinking", "signature"):
            value = block.get(key)
            if isinstance(value, str) and value:
                state.setdefault(f"{key}_parts", []).append(value)
        if "input" in block:
            state["input"] = block["input"]
        return

    if event_type == "content_block_delta":
        delta = event.get("delta")
        if not isinstance(delta, dict):
            return
        index = _safe_int(event.get("index"), 0)
        state = summary.setdefault("anthropic_blocks", {}).setdefault(index, {})
        delta_type = delta.get("type")
        if delta_type == "citations_delta" and isinstance(delta.get("citation"), dict):
            state.setdefault("citations", []).append(delta["citation"])
            return
        field_by_delta_type = {
            "text_delta": ("text", "text"),
            "thinking_delta": ("thinking", "thinking"),
            "signature_delta": ("signature", "signature"),
            "input_json_delta": ("partial_json", "input_json"),
        }
        source_and_target = field_by_delta_type.get(delta_type)
        if source_and_target is None:
            return
        source, target = source_and_target
        state.setdefault(
            "type",
            {
                "text_delta": "text",
                "thinking_delta": "thinking",
                "signature_delta": "thinking",
                "input_json_delta": "tool_use",
            }[delta_type],
        )
        value = delta.get(source)
        if isinstance(value, str):
            state.setdefault(f"{target}_parts", []).append(value)
        return

    if event_type == "message_delta":
        delta = event.get("delta")
        if isinstance(delta, dict):
            summary.setdefault("anthropic_message", {}).update(delta)
        usage = event.get("usage")
        if isinstance(usage, dict):
            summary.setdefault("usage", {}).update(usage)


def _anthropic_stream_response_body_for_telemetry(summary: dict[str, Any]) -> dict[str, Any]:
    message = dict(summary.get("anthropic_message") or {})
    message.setdefault("type", "message")
    message.setdefault("role", "assistant")
    content: list[dict[str, Any]] = []
    blocks = summary.get("anthropic_blocks")
    if isinstance(blocks, dict):
        for index in sorted(blocks):
            block = blocks[index]
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            item = {
                key: value
                for key, value in block.items()
                if not key.endswith("_parts")
                and not (key == "input" and block_type == "tool_use")
            }
            if block_type == "text":
                item["text"] = "".join(str(part) for part in block.get("text_parts", []))
            elif block_type == "thinking":
                item["thinking"] = "".join(
                    str(part) for part in block.get("thinking_parts", [])
                )
                item["signature"] = "".join(
                    str(part) for part in block.get("signature_parts", [])
                )
            elif block_type == "tool_use":
                arguments = "".join(str(part) for part in block.get("input_json_parts", []))
                try:
                    item["input"] = json.loads(arguments) if arguments else block.get("input", {})
                except json.JSONDecodeError:
                    item["input"] = block.get("input", {})
            content.append(item)
    message["content"] = content
    usage = summary.get("usage")
    if isinstance(usage, dict):
        message["usage"] = dict(usage)
    return message


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
            "extra_message_fields": {},
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

    if "content" in payload:
        content = payload.get("content")
        if isinstance(content, str):
            state.setdefault("content_parts", []).append(content)
        else:
            state["content"] = content

    for key in ("reasoning", "reasoning_content"):
        reasoning = payload.get(key)
        if isinstance(reasoning, str):
            state.setdefault("reasoning_parts", []).append(reasoning)
        elif key in payload:
            state[key] = reasoning

    tool_calls = payload.get("tool_calls")
    if isinstance(tool_calls, list):
        for raw_tool_call in tool_calls:
            if isinstance(raw_tool_call, dict):
                _merge_tool_call_delta(state.setdefault("tool_calls", {}), raw_tool_call)

    function_call = payload.get("function_call")
    if isinstance(function_call, dict):
        _merge_function_call_delta(state.setdefault("function_call", {}), function_call)

    known_fields = {
        "role",
        "content",
        "reasoning",
        "reasoning_content",
        "tool_calls",
        "function_call",
    }
    extra_fields = state.setdefault("extra_message_fields", {})
    for key, value in payload.items():
        if key in known_fields:
            continue
        if isinstance(value, str) and isinstance(extra_fields.get(key), str):
            extra_fields[key] += value
        else:
            extra_fields[key] = value


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

    message.update(state.get("extra_message_fields") or {})

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
