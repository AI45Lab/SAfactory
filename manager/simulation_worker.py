from __future__ import annotations

import asyncio
import inspect
import logging
import os
import time
from collections import deque
from typing import Any, Dict, Optional

import httpx

from core.data_manager.load_yaml import materialize_dataset_env_params
from core.data_manager.contracts import SessionContext
from core.data_manager.manager import DataManager
from core.perf_trace import PerfTrace
from core.runtime_metadata import strip_internal_env_params
from evaluator.eval_types import EvalRequest, EvalResult, EvalStatus
from evaluator.gateway_client import GatewayClient
from evaluator.reward_committer import RewardCommitter
from evaluator.rule_evaluator import discover_rule_eval_spec
from evaluator.service import EvaluationService

# Structured timing log (rl/timing_log.py). The launcher subprocess does not
# have rl/ on PYTHONPATH (only AIEVOBOX_ROOT), so add it defensively.
try:
    from timing_log import emit as _timing_emit  # type: ignore
except Exception:  # pragma: no cover - import path fixup
    import os as _os
    import sys as _sys
    _rl_dir = _os.path.join(_os.environ.get("AIEVOBOX_ROOT", ""), "rl")
    if _rl_dir and _rl_dir not in _sys.path:
        _sys.path.insert(0, _rl_dir)
    try:
        from timing_log import emit as _timing_emit  # type: ignore
    except Exception:
        _timing_emit = None  # type: ignore

from .agent_start_client import AgentStartClient
from .session_lifecycle import complete_latest_session_step
from .simulation_lease_pool import SimulationLeasePool
from .types import (
    SimulationAgentLease,
    SimulationRunConfig,
    SimulationRunSummary,
    SimulationStartRequest,
    SimulationStartResult,
)

log = logging.getLogger("manager.simulation_worker")


def _iso_to_epoch(s: str | None) -> float | None:
    """Parse an ISO-8601 timestamp (from gateway session status) to epoch seconds."""
    if not s:
        return None
    try:
        from datetime import datetime

        return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
    except Exception:
        return None


class _SimulationCircuitBreaker:
    _TIMEOUT_MARKERS = (
        "timed out",
        "timeoutexpired",
        "command timed out",
        "docker exec timed out",
    )

    def __init__(self, cfg: SimulationRunConfig) -> None:
        self.enabled = bool(cfg.circuit_breaker_enabled)
        self.window = max(1, int(cfg.circuit_breaker_window or 1))
        self.min_samples = max(1, min(self.window, int(cfg.circuit_breaker_min_samples or 1)))
        self.failure_rate_threshold = min(1.0, max(0.0, float(cfg.circuit_breaker_failure_rate)))
        self.timeout_rate_threshold = min(1.0, max(0.0, float(cfg.circuit_breaker_timeout_rate)))
        self.consecutive_timeout_limit = max(1, int(cfg.circuit_breaker_consecutive_timeouts or 1))
        self._samples: deque[tuple[bool, bool]] = deque(maxlen=self.window)
        self._consecutive_timeouts = 0
        self._opened = asyncio.Event()
        self._reason = ""
        self._lock = asyncio.Lock()

    def is_open(self) -> bool:
        return self._opened.is_set()

    def reason(self) -> str:
        return self._reason

    async def wait_open(self) -> None:
        await self._opened.wait()

    async def record(self, result: SimulationStartResult) -> bool:
        if not self.enabled or self._opened.is_set():
            return False
        if result.truncated or str(result.status or "").lower() == "truncated":
            return False

        failed = str(result.status or "").lower() != "succeeded"
        timed_out = self._is_timeout(result)
        async with self._lock:
            if self._opened.is_set():
                return False
            self._samples.append((failed, timed_out))
            self._consecutive_timeouts = self._consecutive_timeouts + 1 if timed_out else 0

            reason = self._evaluate_locked()
            if not reason:
                return False
            self._reason = reason
            self._opened.set()
            return True

    def _evaluate_locked(self) -> str:
        if self._consecutive_timeouts >= self.consecutive_timeout_limit:
            return f"consecutive_timeouts={self._consecutive_timeouts}"

        sample_count = len(self._samples)
        if sample_count < self.min_samples:
            return ""

        failures = sum(1 for failed, _ in self._samples if failed)
        timeouts = sum(1 for _, timed_out in self._samples if timed_out)
        failure_rate = failures / sample_count
        timeout_rate = timeouts / sample_count
        if failure_rate >= self.failure_rate_threshold:
            return (
                f"failure_rate={failure_rate:.3f} "
                f"threshold={self.failure_rate_threshold:.3f} samples={sample_count}"
            )
        if timeout_rate >= self.timeout_rate_threshold:
            return (
                f"timeout_rate={timeout_rate:.3f} "
                f"threshold={self.timeout_rate_threshold:.3f} samples={sample_count}"
            )
        return ""

    @classmethod
    def _is_timeout(cls, result: SimulationStartResult) -> bool:
        if bool(result.truncated):
            return True
        metrics = result.metrics if isinstance(result.metrics, dict) else {}
        if str(metrics.get("timeout_layer") or "").strip():
            return True
        text = " ".join(
            [
                str(result.error_text or ""),
                str(metrics.get("error") or ""),
                str(metrics.get("error_text") or ""),
            ]
        ).lower()
        return any(marker in text for marker in cls._TIMEOUT_MARKERS)


class SimulationWorkerGroup:
    def __init__(
        self,
        lease_pool: SimulationLeasePool,
        data_manager: DataManager,
        agent_start_client: AgentStartClient,
        cfg: SimulationRunConfig,
        gateway_client: GatewayClient | None = None,
        evaluation_service: EvaluationService | None = None,
        reward_committer: RewardCommitter | None = None,
    ) -> None:
        self.lease_pool = lease_pool
        self.data_manager = data_manager
        self.agent_start_client = agent_start_client
        self.cfg = cfg
        self.gateway_client = gateway_client
        self.evaluation_service = evaluation_service
        self.reward_committer = reward_committer
        self.worker_count = self._derive_worker_count()
        self._results: Dict[str, SimulationStartResult] = {}
        self._results_lock = asyncio.Lock()
        self._circuit_breaker = _SimulationCircuitBreaker(cfg)
        # Number of episodes currently running (acquired a lease, not yet
        # released). Used for the periodic active_envs timing snapshot.
        self._active_episodes = 0
        self._active_lock = asyncio.Lock()

    async def run_all(self) -> SimulationRunSummary:
        log.info(
            "simulation workers starting: warm_pool_size=%d workers=%d",
            self.lease_pool.pool_size,
            self.worker_count,
        )
        tasks = [
            asyncio.create_task(self._worker_loop(worker_id), name=f"simulation-worker-{worker_id}")
            for worker_id in range(self.worker_count)
        ]
        # Periodic snapshot of concurrent active episodes for capacity /
        # GPU-ratio analysis (env pods vs inference GPUs).
        snapshot_task = asyncio.create_task(self._active_snapshot_loop(), name="active-envs-snapshot")
        tasks.append(snapshot_task)
        cancelled = False
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            cancelled = True
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        except Exception:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

        async with self._results_lock:
            results = dict(self._results)

        if not results:
            return SimulationRunSummary(
                job_id=self.cfg.job_id,
                status="failed_no_episodes",
                total_episodes=0,
                succeeded_episodes=0,
                truncated_episodes=0,
                failed_episodes=0,
                cancelled=cancelled,
                results={},
            )

        succeeded = sum(1 for result in results.values() if result.status == "succeeded")
        truncated = sum(1 for result in results.values() if result.status == "truncated")
        failed = len(results) - succeeded - truncated
        if cancelled:
            status = "cancelled"
        elif failed:
            status = "completed_with_failures"
        elif truncated:
            status = "completed_with_truncations"
        else:
            status = "succeeded"
        return SimulationRunSummary(
            job_id=self.cfg.job_id,
            status=status,
            total_episodes=len(results),
            succeeded_episodes=succeeded,
            truncated_episodes=truncated,
            failed_episodes=failed,
            cancelled=cancelled,
            results={key: result.total_reward for key, result in results.items()},
        )

    async def _active_snapshot_loop(self) -> None:
        """Emit periodic active-env snapshots for capacity / GPU-ratio analysis."""
        interval = float(os.environ.get("SAFACTORY_ACTIVE_SNAPSHOT_INTERVAL_S", "10") or 10)
        try:
            while True:
                await asyncio.sleep(interval)
                async with self._active_lock:
                    active = self._active_episodes
                if _timing_emit is not None:
                    _timing_emit(
                        "active_envs",
                        active_episodes=active,
                        pool_size=self.lease_pool.pool_size,
                        worker_count=self.worker_count,
                    )
        except asyncio.CancelledError:
            return

    async def _worker_loop(self, worker_id: int) -> None:
        while True:
            trace = PerfTrace(
                "simulation_worker.episode",
                logger=log,
                context={
                    "job_id": self.cfg.job_id,
                    "worker_id": worker_id,
                    "evaluation_enabled": self.evaluation_service is not None,
                },
            )
            with trace.span("acquire_lease"):
                lease = await self._acquire_lease_or_stop(worker_id)
            if lease is None:
                log.debug("worker=%d: lease pool exhausted", worker_id)
                trace.emit_summary(
                    status="lease_pool_exhausted",
                    circuit_breaker_reason=self._circuit_breaker.reason() or None,
                )
                return

            agent_key = f"{lease.agent_name}_{lease.agent_id}"
            async with self._active_lock:
                self._active_episodes += 1
            trace.update_context(
                agent_key=agent_key,
                agent_name=lease.agent_name,
                agent_id=lease.agent_id,
                group_id=lease.group_id,
                row_id=lease.row_id,
                runtime=lease.runtime,
                resource_name=lease.resource_name or lease.container_name or lease.container_id,
                reuse_container=lease.reuse_container,
            )
            started = time.perf_counter()
            session: SessionContext | None = None
            result: SimulationStartResult | None = None
            release_reusable: bool | None = None
            cancelled = False
            gateway_finalized = self.gateway_client is None
            try:
                log.debug(
                    "worker=%d acquired agent=%s runtime=%s resource=%s reuse=%s",
                    worker_id,
                    agent_key,
                    lease.runtime,
                    lease.resource_name or lease.container_name or lease.container_id,
                    lease.reuse_container,
                )
                with trace.span("create_session"):
                    session = await self._create_session(lease)
                trace.update_context(session_id=session.session_id)
                request = self._build_start_request(lease, session, worker_id)
                trace.mark(
                    "start_request_built",
                    max_steps=request.max_steps,
                    storage_type=request.storage_type,
                    record_mode=request.record_mode,
                )
                with trace.span("agent_rollout"):
                    result = await self._run_one_episode(lease, session, request, worker_id)
                metrics = result.metrics if isinstance(result.metrics, dict) else {}
                if result.truncated or str(metrics.get("timeout_layer") or "") in {
                    "docker_exec",
                    "sandbox_command",
                    "rjob_wait_terminal",
                }:
                    result.status = "truncated"
                    result.truncated = True
                trace.update_context(
                    rollout_status=result.status,
                    rollout_steps=result.step_count,
                    rollout_reward=result.total_reward,
                    rollout_truncated=result.truncated,
                )
                result.total_reward = None
                completion_mode = self._gateway_completion_mode(result)
                trace.update_context(gateway_completion_mode=completion_mode)
                if self.gateway_client is not None:
                    with trace.span("gateway_finalize"):
                        gateway_finalized = await self._finalize_gateway_session(
                            result,
                            completion_mode=completion_mode,
                            worker_id=worker_id,
                            agent_key=agent_key,
                            trace=trace,
                        )

                if not gateway_finalized:
                    result.status = "failed"
                    result.error_text = result.error_text or "gateway session finalization failed"
                    release_reusable = False
                elif completion_mode == "abort":
                    release_reusable = False
                elif result.truncated:
                    if self.reward_committer is None:
                        raise RuntimeError("reward committer is required for truncated sessions")
                    with trace.span("reward_commit_truncated"):
                        await self.reward_committer.commit(
                            session_id=result.session_id,
                            eval_result=EvalResult(
                                session_id=result.session_id,
                                status=EvalStatus.TRUNCATED.value,
                                normalized_score_10=0.0,
                                reason=result.error_text or "agent rollout timed out",
                                artifacts={"metrics": dict(result.metrics or {})},
                            ),
                        )
                    result.total_reward = 0.0
                    with trace.span("mark_environment_finished"):
                        await self.data_manager.mark_environment_finished(lease.agent_id)
                    release_reusable = False
                elif self.evaluation_service is not None and self.reward_committer is not None:
                    with trace.span("eval_discover_rule"):
                        public_env_params = materialize_dataset_env_params(
                            strip_internal_env_params(lease.env_params)
                        )
                        eval_spec = discover_rule_eval_spec(
                            agent_name=lease.agent_name,
                            env_root=self.cfg.agent_root,
                        )
                    if eval_spec is not None:
                        _eval_timeout = float(lease.env_params.get("rule_evaluator_timeout_s") or 0)
                        if _eval_timeout > 0:
                            eval_spec.timeout_s = _eval_timeout
                    trace.mark("eval_spec_resolved", eval_spec_found=eval_spec is not None)
                    if eval_spec is not None:
                        log.debug(
                            "worker=%d agent=%s rule evaluator resolved: %s",
                            worker_id,
                            agent_key,
                            eval_spec.eval_id,
                        )
                    else:
                        log.debug("worker=%d agent=%s has no rule evaluator; skip evaluation", worker_id, agent_key)

                    if eval_spec is not None:
                        eval_request = EvalRequest(
                            job_id=self.cfg.job_id,
                            session_id=result.session_id,
                            lease=lease,
                            start_result=result,
                            env_params=public_env_params,
                            eval_spec=eval_spec,
                        )
                        _eval_started = time.perf_counter()
                        with trace.span("evaluation_service"):
                            eval_result = await self.evaluation_service.evaluate(eval_request)
                        _eval_elapsed = time.perf_counter() - _eval_started
                        trace.update_context(
                            eval_status=eval_result.status,
                            eval_score=eval_result.normalized_score_10,
                        )
                        # Rule-evaluator timing: for PatchEval this is the
                        # apply-patch + PoC + unit-test cost, which for
                        # compiled CVE projects can dominate the episode.
                        if _timing_emit is not None:
                            _eval_artifacts = eval_result.artifacts if isinstance(eval_result.artifacts, dict) else {}
                            _patch = _eval_artifacts.get("patch") or ""
                            _patch_lines = _patch.count("\n") + 1 if _patch else 0
                            _timing_emit(
                                "eval",
                                session_id=result.session_id,
                                env_name=lease.agent_name,
                                group_id=lease.group_id,
                                cve_id=_eval_artifacts.get("cve_id"),
                                eval_status=eval_result.status,
                                score=eval_result.normalized_score_10,
                                raw_score=eval_result.raw_score,
                                reason=eval_result.reason,
                                eval_elapsed_s=round(_eval_elapsed, 3),
                                strict_success=_eval_artifacts.get("strict_success"),
                                poc_passed=_eval_artifacts.get("poc_passed"),
                                unit_tests_passed=_eval_artifacts.get("unit_tests_passed"),
                                validation_type=_eval_artifacts.get("validation_type"),
                                patch_produced=bool(_patch),
                                patch_lines=_patch_lines or None,
                            )
                        if eval_result.status == "succeeded":
                            with trace.span("reward_commit"):
                                await self.reward_committer.commit(
                                    session_id=result.session_id,
                                    eval_result=eval_result,
                                )
                            result.total_reward = eval_result.normalized_score_10
                            with trace.span("mark_environment_finished"):
                                await self.data_manager.mark_environment_finished(lease.agent_id)
                        else:
                            result.status = "failed"
                            result.error_text = eval_result.error_text or eval_result.reason
                        release_reusable = result.status == "succeeded" and eval_result.status == "succeeded"
                    else:
                        result.status = "failed"
                        result.error_text = (
                            f"rule evaluator not found for environment {lease.agent_name!r}"
                        )
                        release_reusable = False
                else:
                    await complete_latest_session_step(
                        self.data_manager,
                        session_id=result.session_id,
                        job_id=self.cfg.job_id,
                        llm_model=self.cfg.llm_model,
                    )
                    await self.data_manager.mark_environment_finished(lease.agent_id)
                    release_reusable = None

                with trace.span("store_result"):
                    async with self._results_lock:
                        self._results[agent_key] = result
            except asyncio.CancelledError:
                cancelled = True
                release_reusable = False
                trace.update_context(error_type="CancelledError", error="worker task cancelled")
                raise
            except Exception as exc:
                trace.update_context(error_type=type(exc).__name__, error=str(exc))
                log.warning("worker=%d agent=%s failed before summary: %s", worker_id, agent_key, exc, exc_info=True)
                if result is None:
                    result = SimulationStartResult(
                        session_id=session.session_id if session is not None else lease.agent_id,
                        status="failed",
                        total_reward=None,
                        step_count=0,
                        terminated=True,
                        truncated=False,
                        error_text=str(exc),
                        metrics={},
                    )
                else:
                    result.status = "failed"
                    result.error_text = str(exc)
                release_reusable = False
                with trace.span("store_failed_result"):
                    async with self._results_lock:
                        self._results[agent_key] = result
            finally:
                try:
                    if result is not None:
                        with trace.span("record_circuit_result"):
                            await self._record_circuit_result(result, worker_id=worker_id, agent_key=agent_key)
                    with trace.span("lease_pool_done"):
                        await self.lease_pool.done(lease, result, reusable=release_reusable)
                except Exception as exc:
                    trace.update_context(release_error_type=type(exc).__name__, release_error=str(exc))
                    log.exception("worker=%d agent=%s critical error in lease_pool.done()", worker_id, agent_key)
                finally:
                    async with self._active_lock:
                        self._active_episodes = max(0, self._active_episodes - 1)
                if cancelled:
                    trace.update_context(final_status="cancelled")
                    trace.emit_summary(status="cancelled")

            elapsed = time.perf_counter() - started
            if result is not None:
                trace.update_context(
                    final_status=result.status,
                    final_reward=result.total_reward,
                    final_step_count=result.step_count,
                )
            trace.emit_summary(status=result.status if result is not None else "failed")
            log.info(
                "worker=%d agent=%s finished status=%s reward=%s time=%.2fs",
                worker_id,
                agent_key,
                result.status,
                result.total_reward,
                elapsed,
            )

            # Structured per-episode timing record for offline analysis.
            # rjob_*_ms come from RJobEpisodeRunner._attach_timing_metrics.
            # gw_first_seen_ts / gw_closed_ts come from the gateway session
            # status captured during _finalize_gateway_session. Together with
            # rjob_submit_ts they yield:
            #   env_startup_s   = first LLM call - rjob submit  (pod boot + agent init)
            #   env_active_s    = session close - first LLM call (active rollout)
            #   env_lifecycle_s = session close - rjob submit   (full env lifetime)
            if _timing_emit is not None and result is not None:
                ep_metrics = result.metrics if isinstance(result.metrics, dict) else {}
                _rjob_submit_ts = ep_metrics.get("rjob_submit_ts")
                _gw_first_ts = ep_metrics.get("gw_first_seen_ts")
                _gw_closed_ts = ep_metrics.get("gw_closed_ts")
                _env_startup = (
                    round(_gw_first_ts - _rjob_submit_ts, 3)
                    if _gw_first_ts is not None and _rjob_submit_ts is not None
                    else None
                )
                _env_active = (
                    round(_gw_closed_ts - _gw_first_ts, 3)
                    if _gw_closed_ts is not None and _gw_first_ts is not None
                    else None
                )
                _env_lifecycle = (
                    round(_gw_closed_ts - _rjob_submit_ts, 3)
                    if _gw_closed_ts is not None and _rjob_submit_ts is not None
                    else None
                )
                _timing_emit(
                    "episode",
                    worker_id=worker_id,
                    env_name=lease.agent_name,
                    agent_id=lease.agent_id,
                    group_id=lease.group_id,
                    session_id=session.session_id if session is not None else lease.agent_id,
                    runtime=lease.runtime,
                    status=result.status,
                    reward=result.total_reward,
                    step_count=result.step_count,
                    truncated=bool(result.truncated),
                    episode_elapsed_s=round(elapsed, 3),
                    rjob_submit_ms=ep_metrics.get("rjob_submit_ms"),
                    rjob_wait_terminal_ms=ep_metrics.get("rjob_wait_terminal_ms"),
                    rjob_fetch_logs_ms=ep_metrics.get("rjob_fetch_logs_ms"),
                    rjob_parse_result_ms=ep_metrics.get("rjob_parse_result_ms"),
                    rjob_cleanup_ms=ep_metrics.get("rjob_cleanup_ms"),
                    rjob_total_ms=ep_metrics.get("rjob_total_ms"),
                    rjob_status=ep_metrics.get("rjob_status"),
                    rjob_name=ep_metrics.get("rjob_name"),
                    # Absolute epoch seconds at RJob submit; join with the
                    # gateway's first llm_step ts (same session_id) to derive
                    # env-startup time = first_llm_call - rjob_submit.
                    rjob_submit_ts=ep_metrics.get("rjob_submit_ts"),
                    # Per-env startup / active / lifecycle durations (seconds),
                    # derived from gateway session timing + rjob_submit_ts.
                    env_startup_s=_env_startup,
                    env_active_s=_env_active,
                    env_lifecycle_s=_env_lifecycle,
                    gw_first_seen_ts=_gw_first_ts,
                    gw_closed_ts=_gw_closed_ts,
                )

    async def _acquire_lease_or_stop(self, worker_id: int) -> SimulationAgentLease | None:
        del worker_id
        if self._circuit_breaker.is_open():
            return None
        return await self.lease_pool.acquire()

    async def _record_circuit_result(
        self,
        result: SimulationStartResult,
        *,
        worker_id: int,
        agent_key: str,
    ) -> None:
        opened = await self._circuit_breaker.record(result)
        if not opened:
            return
        reason = self._circuit_breaker.reason()
        log.error(
            "worker=%d agent=%s opened simulation circuit breaker: %s; stopping pool refills and draining active work",
            worker_id,
            agent_key,
            reason,
        )
        await self.lease_pool.stop_refills(reason=f"circuit_breaker:{reason}")

    async def _run_one_episode(
        self,
        lease: SimulationAgentLease,
        session: SessionContext,
        request: SimulationStartRequest,
        worker_id: int,
    ) -> SimulationStartResult:
        try:
            result = await self.agent_start_client.start(lease, request)
        except Exception as exc:
            log.warning(
                "worker=%d runtime=%s env=%s agent_id=%s start failed: %s",
                worker_id,
                lease.runtime,
                lease.agent_name,
                lease.agent_id,
                exc,
            )
            result = SimulationStartResult(
                session_id=session.session_id,
                status="failed",
                total_reward=None,
                step_count=0,
                terminated=True,
                truncated=False,
                error_text=str(exc),
                metrics={},
            )

        if result.status != "succeeded":
            logs_tail = ""
            metrics = result.metrics if isinstance(result.metrics, dict) else {}
            if metrics.get("logs_tail"):
                logs_tail = self._tail(str(metrics.get("logs_tail")))
            log.warning(
                "worker=%d runtime=%s env=%s agent_id=%s returned status=%s error=%s | rjob=%s | LOGS_TAIL:\n%s\n[/LOGS_TAIL]",
                worker_id,
                lease.runtime,
                lease.agent_name,
                lease.agent_id,
                result.status,
                self._tail(result.error_text or ""),
                metrics.get("rjob_name", ""),
                logs_tail,
            )

        return result

    async def _finalize_gateway_session(
        self,
        result: SimulationStartResult,
        *,
        completion_mode: str,
        worker_id: int,
        agent_key: str,
        trace: PerfTrace | None = None,
    ) -> bool:
        if self.gateway_client is None:
            return True
        reason = {
            "complete": "rollout_finished",
            "seal": "rollout_sealed",
            "abort": "system_error",
        }[completion_mode]
        try:
            if trace is None:
                await self.gateway_client.close_session(
                    result.session_id,
                    reason=reason,
                    completion_mode=completion_mode,
                )
                await self.gateway_client.wait_telemetry_flush(result.session_id)
            else:
                with trace.span("gateway_close_session"):
                    await self.gateway_client.close_session(
                        result.session_id,
                        reason=reason,
                        completion_mode=completion_mode,
                    )
                with trace.span("gateway_wait_telemetry_flush"):
                    await self.gateway_client.wait_telemetry_flush(result.session_id)
            # Capture gateway-side session timing for per-env startup /
            # lifecycle analysis. first_seen_at = wall-clock of the FIRST LLM
            # call reaching the gateway (= env ready / agent started making
            # requests); closed_at = when we closed the session. Joined with
            # the worker's rjob_submit_ts (in result.metrics) to derive
            # env_startup_s / env_active_s / env_lifecycle_s in the episode
            # record. Best-effort: a failure here must not fail finalization.
            try:
                status = await self.gateway_client.get_session_status(result.session_id)
                if isinstance(status, dict):
                    result.metrics = dict(result.metrics or {})
                    result.metrics["gw_first_seen_ts"] = _iso_to_epoch(status.get("first_seen_at"))
                    result.metrics["gw_closed_ts"] = _iso_to_epoch(status.get("closed_at"))
            except Exception as exc:
                log.debug(
                    "worker=%d agent=%s gateway session status fetch failed: %s",
                    worker_id,
                    agent_key,
                    exc,
                )
            return True
        except httpx.HTTPError as exc:
            log.warning(
                "worker=%d agent=%s gateway session finalization failed; preserving rollout status=%s "
                "session_id=%s error=%s",
                worker_id,
                agent_key,
                result.status,
                result.session_id,
                exc,
            )
            return False

    @staticmethod
    def _gateway_completion_mode(result: SimulationStartResult) -> str:
        metrics = result.metrics if isinstance(result.metrics, dict) else {}
        if result.truncated or result.status == "truncated":
            return "seal"
        if str(metrics.get("timeout_layer") or "") in {
            "docker_exec",
            "sandbox_command",
            "rjob_wait_terminal",
        }:
            return "seal"
        if str(metrics.get("rjob_status") or "") in {"Failed", "Stopped", "Killed"}:
            return "abort"
        if result.status == "succeeded":
            return "seal"
        return "abort"

    def _build_start_request(
        self,
        lease: SimulationAgentLease,
        session: SessionContext,
        worker_id: int,
    ) -> SimulationStartRequest:
        storage_config: Dict[str, Any] = {}
        if self.cfg.storage_type == "sqlite":
            storage_config["db_url"] = self.cfg.db_url
        public_env_params = materialize_dataset_env_params(
            strip_internal_env_params(lease.env_params)
        )

        return SimulationStartRequest(
            job_id=self.cfg.job_id,
            session_id=session.session_id,
            group_id=lease.group_id,
            gateway_base_url=self.cfg.gateway_base_url,
            model=self.cfg.llm_model,
            temperature=self.cfg.llm_temperature,
            max_steps=self.cfg.max_steps,
            storage_type=self.cfg.storage_type,
            env_params=public_env_params,
            storage_config=storage_config,
            agent_start_timeout_s=self.cfg.agent_start_timeout_s,
            record_mode="agent_runtime",
            agent_name=lease.agent_name,
            agent_id=lease.agent_id,
            metadata={
                "worker_id": worker_id,
                "row_id": lease.row_id,
                "image": lease.image,
                "runtime": lease.runtime,
                "resource_id": lease.resource_id,
                "resource_name": lease.resource_name,
                "container_id": lease.container_id,
                "container_name": lease.container_name,
                "workdir": lease.workdir,
                "reuse_container": lease.reuse_container,
                "agent_name": lease.agent_name,
                "agent_id": lease.agent_id,
                "env_params": strip_internal_env_params(lease.env_params),
            },
        )

    async def _create_session(self, lease: SimulationAgentLease) -> SessionContext:
        maybe_session = self.data_manager.create_session(
            env_id=lease.agent_id,
            env_name=lease.agent_name,
            llm_model=self.cfg.llm_model,
            group_id=lease.group_id,
            job_id=self.cfg.job_id,
        )
        if inspect.isawaitable(maybe_session):
            return await maybe_session
        return maybe_session

    def _derive_worker_count(self) -> int:
        worker_count = max(1, int(getattr(self.lease_pool, "pool_size", 1) or 1))
        if self.cfg.max_workers is not None:
            worker_count = max(1, min(worker_count, int(self.cfg.max_workers)))
        return worker_count

    @staticmethod
    def _tail(value: str, limit: int = 1000) -> str:
        return (value or "").strip()[-int(limit):]
