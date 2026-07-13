from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Dict

from clusters.rjob_cluster import (
    RJOB_FAILED_STATUSES,
    RJOB_SUCCEEDED_STATUSES,
    RJobClusterBackend,
    merge_env_dicts as _merge_dicts,
)
from core.perf_trace import PerfTrace

from .episode_common import (
    json_for_log,
    normalize_result,
    parse_result_artifact,
    parse_result_output,
    request_env,
    request_payload,
    result_artifact_candidates,
    result_artifact_path,
    tail,
)
from .types import SimulationAgentLease, SimulationStartRequest, SimulationStartResult

log = logging.getLogger("manager.rjob_episode_runner")


class RJobEpisodeRunner:
    """Runs one episode as one RJob job."""

    def __init__(self, *, timeout_s: float) -> None:
        self.timeout_s = float(timeout_s)
        self._cluster = RJobClusterBackend()

    async def start(
        self,
        lease: SimulationAgentLease,
        request: SimulationStartRequest,
    ) -> SimulationStartResult:
        cfg = dict(lease.runtime_config or {})
        poll_interval_s = float(cfg.get("poll_interval_s", 5.0) or 5.0)
        timeout_s = float(request.agent_start_timeout_s or self.timeout_s)
        trace = PerfTrace(
            "rjob_episode.start",
            logger=log,
            context={
                "job_id": request.job_id,
                "session_id": request.session_id,
                "group_id": request.group_id,
                "agent_name": lease.agent_name,
                "agent_id": lease.agent_id,
                "row_id": lease.row_id,
                "runtime": lease.runtime,
                "result_mode": lease.result_mode,
                "image": lease.image,
                "resource_name": lease.resource_name,
                "timeout_s": timeout_s,
                "poll_interval_s": poll_interval_s,
            },
        )

        submitted_name = ""
        terminal_status = "Unknown"
        logs_text = ""
        logs_error = ""
        result: SimulationStartResult | None = None
        timings_ms: Dict[str, float] = {}
        status_poll_count = 0
        summary_status = "failed"
        summary_extra: Dict[str, Any] = {}
        episode_started = time.perf_counter()
        try:
            if not lease.image:
                raise RuntimeError(f"RJob lease missing image: {lease.agent_name}/{lease.agent_id}")
            if not lease.run_command:
                raise RuntimeError(f"RJob lease missing run_command: {lease.agent_name}/{lease.agent_id}")

            with trace.span("resolve_gateway_base_url"):
                gateway_base_url = self._cluster.resolve_gateway_base_url(cfg, request)
            with trace.span("build_request_payload"):
                request_params, payload = request_payload(request)
            with trace.span("client_init"):
                client = self._cluster.client(cfg)

            rjob_name = self._cluster.build_job_name(cfg, lease, request)
            trace.update_context(requested_rjob_name=rjob_name)
            with trace.span("build_job", requested_rjob_name=rjob_name):
                merged_env = _merge_dicts(
                    cfg.get("env"),
                    request_env(
                        request,
                        payload,
                        gateway_base_url=gateway_base_url,
                        containerize_local_gateway=False,
                    ),
                )
                job = self._cluster.build_job(
                    cfg=cfg,
                    lease=lease,
                    request=request,
                    rjob_name=rjob_name,
                    env=merged_env,
                )
                submit_kwargs = self._cluster.submit_kwargs(cfg)
            trace.update_context(
                dry_run=bool(submit_kwargs.get("dry_run")),
                predict_only=bool(submit_kwargs.get("predict_only")),
            )

            log.debug(
                "RJob submit params: env=%s agent_id=%s rjob=%s params=%s",
                lease.agent_name,
                lease.agent_id,
                rjob_name,
                self._safe_json_for_log(
                    {
                        "request": request_params,
                        "image": lease.image,
                        "workdir": lease.workdir,
                        "run_command": lease.run_command,
                        "result_artifact_path": result_artifact_path(request),
                        "submit": submit_kwargs,
                        "resources": cfg.get("resources") or cfg.get("default_resources") or {},
                        "mount_config": cfg.get("mount_config") or [],
                    }
                ),
            )

            started = time.perf_counter()
            with trace.span("submit_job", requested_rjob_name=rjob_name):
                submitted = await self._cluster.submit_job(client, job, submit_kwargs)
            timings_ms["rjob_submit_ms"] = _elapsed_ms(started)
            trace.update_context(rjob_submit_ms=timings_ms["rjob_submit_ms"])
            submitted_name = str(submitted or rjob_name).strip()
            trace.update_context(submitted_rjob_name=submitted_name)
            trace.mark(
                "job_submitted",
                submitted_rjob_name=submitted_name,
                rjob_submit_ms=timings_ms["rjob_submit_ms"],
            )
            if not submitted_name and (submit_kwargs.get("dry_run") or submit_kwargs.get("predict_only")):
                result = SimulationStartResult(
                    session_id=request.session_id,
                    status="succeeded",
                    total_reward=0.0,
                    step_count=0,
                    terminated=True,
                    truncated=False,
                    metrics={"runtime": "rjob", "dry_run": bool(submit_kwargs.get("dry_run"))},
                )
                summary_status = result.status
                return result
            if not submitted_name:
                raise RuntimeError("RJobClient.submit returned empty job name")

            started = time.perf_counter()
            with trace.span("wait_terminal", submitted_rjob_name=submitted_name):
                terminal_status, status_poll_count = await self._cluster.wait_terminal(
                    client,
                    submitted_name,
                    timeout_s=timeout_s,
                    poll_interval_s=poll_interval_s,
                    trace=trace,
                )
            timings_ms["rjob_wait_terminal_ms"] = _elapsed_ms(started)
            trace.update_context(rjob_status=terminal_status, status_poll_count=status_poll_count)
            try:
                started = time.perf_counter()
                with trace.span("fetch_logs", submitted_rjob_name=submitted_name, terminal_status=terminal_status):
                    logs_text = await self._cluster.logs_text(client, submitted_name)
                timings_ms["rjob_fetch_logs_ms"] = _elapsed_ms(started)
                trace.mark("logs_fetched", log_chars=len(logs_text))
            except Exception as exc:
                logs_error = str(exc)
                timings_ms["rjob_fetch_logs_ms"] = _elapsed_ms(started)
                trace.mark("logs_fetch_failed", error=logs_error, error_type=type(exc).__name__)

            started = time.perf_counter()
            with trace.span("parse_result", submitted_rjob_name=submitted_name, terminal_status=terminal_status):
                result = self._result_from_terminal_status(
                    terminal_status=terminal_status,
                    logs_text=logs_text,
                    lease=lease,
                    request=request,
                    job_name=submitted_name,
                )
            if logs_error:
                result.metrics = dict(result.metrics or {})
                result.metrics["logs_error"] = logs_error
            timings_ms["rjob_parse_result_ms"] = _elapsed_ms(started)
            summary_status = result.status
            return result
        except asyncio.CancelledError:
            summary_status = "cancelled"
            summary_extra = {"error_type": "CancelledError"}
            trace.update_context(error_type="CancelledError")
            raise
        except TimeoutError as exc:
            terminal_status = str(getattr(exc, "last_status", terminal_status) or terminal_status)
            status_poll_count = int(getattr(exc, "poll_count", status_poll_count) or status_poll_count)
            if submitted_name:
                started = time.perf_counter()
                with trace.span("stop_timed_out_job", submitted_rjob_name=submitted_name):
                    await self._cluster.stop_job(client, submitted_name)
                timings_ms["rjob_stop_ms"] = _elapsed_ms(started)
                started = time.perf_counter()
                with trace.span("fetch_timeout_logs", submitted_rjob_name=submitted_name):
                    logs_text = await self._cluster.logs_text(client, submitted_name, suppress_errors=True)
                timings_ms["rjob_fetch_timeout_logs_ms"] = _elapsed_ms(started)
            result = SimulationStartResult(
                session_id=request.session_id,
                status="failed",
                total_reward=0.0,
                step_count=0,
                terminated=True,
                truncated=False,
                error_text=str(exc),
                metrics={
                    "runtime": "rjob",
                    "rjob_name": submitted_name or rjob_name,
                    "rjob_status": terminal_status,
                    "logs_tail": tail(logs_text),
                },
            )
            summary_status = result.status
            summary_extra = {
                "error_type": type(exc).__name__,
                "error": str(exc),
                "timeout_layer": "rjob_wait_terminal",
            }
            return result
        except Exception as exc:
            summary_status = "failed"
            summary_extra = {"error_type": type(exc).__name__, "error": str(exc)}
            trace.update_context(error_type=type(exc).__name__, error=str(exc))
            raise
        finally:
            if submitted_name:
                cleanup_started = time.perf_counter()
                with trace.span(
                    "cleanup_job",
                    submitted_rjob_name=submitted_name,
                    terminal_status=terminal_status,
                    cleanup_on_finish=bool(cfg.get("cleanup_on_finish", True)),
                    keep_failed_jobs=bool(cfg.get("keep_failed_jobs", False)),
                ):
                    await self._cluster.cleanup_job(
                        client,
                        submitted_name,
                        cfg=cfg,
                        terminal_status=terminal_status,
                    )
                timings_ms["rjob_cleanup_ms"] = _elapsed_ms(cleanup_started)
            timings_ms["rjob_total_ms"] = _elapsed_ms(episode_started)
            trace.update_context(**timings_ms)
            if result is not None:
                self._attach_timing_metrics(
                    result,
                    trace=trace,
                    timings_ms=timings_ms,
                    submitted_name=submitted_name,
                    terminal_status=terminal_status,
                    status_poll_count=status_poll_count,
                )
                trace.update_context(
                    result_status=result.status,
                    result_reward=result.total_reward,
                    result_step_count=result.step_count,
                    result_truncated=result.truncated,
                )
            trace.emit_summary(
                status=summary_status,
                submitted_rjob_name=submitted_name or None,
                terminal_status=terminal_status,
                status_poll_count=status_poll_count or None,
                **summary_extra,
            )

    async def close(self) -> None:
        return

    def _result_from_terminal_status(
        self,
        *,
        terminal_status: str,
        logs_text: str,
        lease: SimulationAgentLease,
        request: SimulationStartRequest,
        job_name: str,
    ) -> SimulationStartResult:
        result_mode = str(lease.result_mode or "json").strip().lower()
        if terminal_status in RJOB_SUCCEEDED_STATUSES and result_mode == "exit_code":
            return SimulationStartResult(
                session_id=request.session_id,
                status="succeeded",
                total_reward=0.0,
                step_count=0,
                terminated=True,
                truncated=False,
                metrics={
                    "runtime": "rjob",
                    "rjob_name": job_name,
                    "rjob_status": terminal_status,
                    "result_mode": result_mode,
                    "logs_tail": tail(logs_text),
                },
            )

        try:
            body = parse_result_output(logs_text)
            result = normalize_result(body, session_id=request.session_id)
            result_source = "stdout"
            artifact_source_path = ""
            stdout_parse_error = ""
        except Exception as exc:
            stdout_parse_error = str(exc)
            try:
                body, artifact_path = parse_result_artifact(request)
                result = normalize_result(body, session_id=request.session_id)
                result_source = "artifact"
                artifact_source_path = str(artifact_path)
            except Exception as artifact_exc:
                artifact_path_text = result_artifact_path(request)
                candidates = [str(path) for path in result_artifact_candidates(request, artifact_path_text)]
                error_text = (
                    f"RJob {job_name} finished with status={terminal_status}, "
                    f"but no SimulationStartResult JSON could be parsed: {stdout_parse_error}; "
                    f"artifact_error={artifact_exc}"
                )
                if candidates:
                    error_text += f"; artifact_candidates={candidates}"
                return SimulationStartResult(
                    session_id=request.session_id,
                    status="failed",
                    total_reward=0.0,
                    step_count=0,
                    terminated=True,
                    truncated=False,
                    error_text=error_text,
                    metrics={
                        "runtime": "rjob",
                        "rjob_name": job_name,
                        "rjob_status": terminal_status,
                        "result_artifact_path": artifact_path_text,
                        "logs_tail": tail(logs_text),
                    },
                )

        result.metrics = dict(result.metrics or {})
        result.metrics.update(
            {
                "runtime": "rjob",
                "rjob_name": job_name,
                "rjob_status": terminal_status,
                "logs_tail": tail(logs_text),
                "result_source": result_source,
            }
        )
        if artifact_source_path:
            result.metrics["result_artifact_path"] = artifact_source_path
        if stdout_parse_error:
            result.metrics["stdout_parse_error"] = stdout_parse_error

        if terminal_status in RJOB_FAILED_STATUSES and result.status == "succeeded":
            result.status = "failed"
            result.error_text = f"RJob {job_name} failed with status={terminal_status}"
        return result

    @staticmethod
    def _safe_json_for_log(value: Any) -> str:
        scrubbed = json.loads(json_for_log(value))
        if isinstance(scrubbed, dict):
            submit = scrubbed.get("submit")
            if isinstance(submit, dict):
                for key in ("access_key", "secret_key"):
                    if key in submit:
                        submit[key] = "***"
        return json_for_log(scrubbed)

    @staticmethod
    def _attach_timing_metrics(
        result: SimulationStartResult,
        *,
        trace: PerfTrace,
        timings_ms: Dict[str, float],
        submitted_name: str,
        terminal_status: str,
        status_poll_count: int,
    ) -> None:
        result.metrics = dict(result.metrics or {})
        result.metrics.update(
            {
                "runtime": "rjob",
                "rjob_trace_id": trace.trace_id,
                "rjob_started_at_utc": trace.started_at_utc,
                "rjob_name": submitted_name or result.metrics.get("rjob_name"),
                "rjob_status": terminal_status,
                "rjob_status_poll_count": status_poll_count,
                **timings_ms,
            }
        )


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 3)
