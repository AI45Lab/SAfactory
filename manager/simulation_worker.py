from __future__ import annotations

import asyncio
import inspect
import logging
import time
from collections import deque
from typing import Any, Dict, Optional

import httpx

from core.data_manager.manager import DataManager, SessionContext
from core.perf_trace import PerfTrace
from core.runtime_metadata import strip_internal_env_params
from evaluator.eval_types import EvalRequest, parse_eval_specs
from evaluator.gateway_client import GatewayClient
from evaluator.markdown_eval_resolver import MarkdownEvalTaskResolver
from evaluator.reward_committer import RewardCommitter
from evaluator.rule_eval_resolver import resolve_rule_eval_specs
from evaluator.run_registry import InMemoryRunRegistry
from evaluator.service import EvaluationService

from .agent_start_client import AgentStartClient
from .simulation_lease_pool import SimulationLeasePool
from .types import (
    SimulationAgentLease,
    SimulationRunConfig,
    SimulationRunSummary,
    SimulationStartRequest,
    SimulationStartResult,
)

log = logging.getLogger("manager.simulation_worker")


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
        registry: InMemoryRunRegistry | None = None,
        gateway_client: GatewayClient | None = None,
        evaluation_service: EvaluationService | None = None,
        reward_committer: RewardCommitter | None = None,
    ) -> None:
        self.lease_pool = lease_pool
        self.data_manager = data_manager
        self.agent_start_client = agent_start_client
        self.cfg = cfg
        self.registry = registry
        self.gateway_client = gateway_client
        self.evaluation_service = evaluation_service
        self.reward_committer = reward_committer
        self.markdown_eval_resolver = MarkdownEvalTaskResolver(
            strict=self.cfg.strict_eval_tasks,
        )
        self.worker_count = self._derive_worker_count()
        self._results: Dict[str, SimulationStartResult] = {}
        self._results_lock = asyncio.Lock()
        self._circuit_breaker = _SimulationCircuitBreaker(cfg)

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
                failed_episodes=0,
                cancelled=cancelled,
                results={},
            )

        succeeded = sum(1 for result in results.values() if result.status == "succeeded")
        failed = len(results) - succeeded
        if cancelled:
            status = "cancelled"
        else:
            status = "succeeded" if failed == 0 else "completed_with_failures"
        return SimulationRunSummary(
            job_id=self.cfg.job_id,
            status=status,
            total_episodes=len(results),
            succeeded_episodes=succeeded,
            failed_episodes=failed,
            cancelled=cancelled,
            results={key: result.total_reward for key, result in results.items()},
        )

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
                log.info("worker=%d: lease pool exhausted", worker_id)
                trace.emit_summary(
                    status="lease_pool_exhausted",
                    circuit_breaker_reason=self._circuit_breaker.reason() or None,
                )
                return

            agent_key = f"{lease.agent_name}_{lease.agent_id}"
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
            run_failed = False
            cancelled = False
            try:
                log.info(
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
                if self.registry is not None:
                    with trace.span("registry_create_run"):
                        await self.registry.create_run(
                            job_id=self.cfg.job_id,
                            session_id=session.session_id,
                            metadata={
                                "agent_name": lease.agent_name,
                                "agent_id": lease.agent_id,
                                "group_id": lease.group_id,
                                "row_id": lease.row_id,
                                "image": lease.image,
                                "runtime": lease.runtime,
                                "resource_id": lease.resource_id,
                                "resource_name": lease.resource_name,
                                "container_id": lease.container_id,
                                "container_name": lease.container_name,
                            },
                        )
                        await self.registry.mark_running(session.session_id)

                with trace.span("agent_rollout"):
                    result = await self._run_one_episode(lease, session, request, worker_id)
                trace.update_context(
                    rollout_status=result.status,
                    rollout_steps=result.step_count,
                    rollout_reward=result.total_reward,
                    rollout_truncated=result.truncated,
                )
                if self.registry is not None:
                    with trace.span("registry_rollout_finished"):
                        await self.registry.mark_rollout_finished(session.session_id, result)

                if self.gateway_client is not None:
                    with trace.span("gateway_finalize"):
                        await self._finalize_gateway_session(
                            result,
                            worker_id=worker_id,
                            agent_key=agent_key,
                            trace=trace,
                        )

                if self.evaluation_service is None or self.reward_committer is None:
                    release_reusable = False
                else:
                    with trace.span("eval_resolve_specs"):
                        public_env_params = strip_internal_env_params(lease.env_params)
                        eval_specs = self.markdown_eval_resolver.resolve_specs(lease.env_params)
                        if not eval_specs:
                            eval_specs = parse_eval_specs(
                                public_env_params,
                                default_specs=getattr(self.evaluation_service, "default_specs", []),
                            )
                        if not eval_specs:
                            eval_specs = resolve_rule_eval_specs(
                                lease.env_params,
                                agent_name=lease.agent_name,
                            )
                    trace.mark("eval_specs_resolved", eval_spec_count=len(eval_specs))
                    if eval_specs:
                        log.info(
                            "worker=%d agent=%s evaluation specs resolved: %s",
                            worker_id,
                            agent_key,
                            [spec.eval_id for spec in eval_specs],
                        )
                    else:
                        log.debug("worker=%d agent=%s has no eval specs; skip evaluation", worker_id, agent_key)

                    if eval_specs:
                        eval_request = EvalRequest(
                            job_id=self.cfg.job_id,
                            session_id=result.session_id,
                            lease=lease,
                            start_result=result,
                            env_params=public_env_params,
                            eval_specs=eval_specs,
                        )
                        if self.registry is not None:
                            with trace.span("registry_awaiting_eval"):
                                await self.registry.mark_awaiting_eval(result.session_id, eval_request)
                        with trace.span("evaluation_service"):
                            eval_result = await self.evaluation_service.evaluate(eval_request)
                        trace.update_context(
                            eval_status=eval_result.status,
                            eval_score=eval_result.normalized_score_10,
                        )
                        with trace.span("reward_commit"):
                            await self.reward_committer.commit(
                                session_id=result.session_id,
                                eval_result=eval_result,
                            )
                        if eval_result.status == "succeeded":
                            if self.registry is not None:
                                with trace.span("registry_reward_committed"):
                                    await self.registry.mark_reward_committed(
                                        result.session_id,
                                        eval_result.normalized_score_10,
                                    )
                            result.total_reward = eval_result.normalized_score_10
                        else:
                            result.total_reward = 0.0
                            result.status = "failed"
                            result.error_text = eval_result.error_text or eval_result.reason
                        release_reusable = result.status == "succeeded" and eval_result.status == "succeeded"
                    else:
                        release_reusable = None

                with trace.span("store_result"):
                    async with self._results_lock:
                        self._results[agent_key] = result
            except asyncio.CancelledError:
                run_failed = True
                cancelled = True
                release_reusable = False
                trace.update_context(error_type="CancelledError", error="worker task cancelled")
                raise
            except Exception as exc:
                run_failed = True
                trace.update_context(error_type=type(exc).__name__, error=str(exc))
                log.warning("worker=%d agent=%s failed before summary: %s", worker_id, agent_key, exc, exc_info=True)
                if result is None:
                    result = SimulationStartResult(
                        session_id=session.session_id if session is not None else lease.agent_id,
                        status="failed",
                        total_reward=0.0,
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
                if self.registry is not None and session is not None:
                    with trace.span("registry_mark_failed"):
                        await self.registry.mark_failed(session.session_id, str(exc))
                with trace.span("store_failed_result"):
                    async with self._results_lock:
                        self._results[agent_key] = result
            finally:
                try:
                    if result is not None:
                        with trace.span("record_circuit_result"):
                            await self._record_circuit_result(result, worker_id=worker_id, agent_key=agent_key)
                    if self.registry is not None and session is not None and not run_failed:
                        with trace.span("registry_releasing_container"):
                            await self.registry.mark_releasing_container(session.session_id)
                    with trace.span("lease_pool_done"):
                        await self.lease_pool.done(lease, result, reusable=release_reusable)
                    if self.registry is not None and session is not None and not run_failed:
                        with trace.span("registry_done"):
                            await self.registry.mark_done(session.session_id)
                except Exception as exc:
                    trace.update_context(release_error_type=type(exc).__name__, release_error=str(exc))
                    log.exception("worker=%d agent=%s critical error in lease_pool.done()", worker_id, agent_key)
                    if self.registry is not None and session is not None:
                        await self.registry.mark_failed(session.session_id, f"container release failed: {exc}")
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
                "worker=%d agent=%s finished status=%s reward=%.6f time=%.2fs",
                worker_id,
                agent_key,
                result.status,
                result.total_reward,
                elapsed,
            )

    async def _acquire_lease_or_stop(self, worker_id: int) -> SimulationAgentLease | None:
        del worker_id
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
            "worker=%d agent=%s opened simulation circuit breaker: %s; continuing until data is exhausted",
            worker_id,
            agent_key,
            reason,
        )

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
                "worker=%d agent=%s/%s OpenClaw start failed: %s",
                worker_id,
                lease.agent_name,
                lease.agent_id,
                exc,
            )
            result = SimulationStartResult(
                session_id=session.session_id,
                status="failed",
                total_reward=0.0,
                step_count=0,
                terminated=True,
                truncated=False,
                error_text=str(exc),
                metrics={},
            )

        if result.status != "succeeded":
            log.warning(
                "worker=%d agent=%s/%s returned status=%s error=%s",
                worker_id,
                lease.agent_name,
                lease.agent_id,
                result.status,
                self._tail(result.error_text or ""),
            )

        return result

    async def _finalize_gateway_session(
        self,
        result: SimulationStartResult,
        *,
        worker_id: int,
        agent_key: str,
        trace: PerfTrace | None = None,
    ) -> None:
        if self.gateway_client is None:
            return
        try:
            if trace is None:
                await self.gateway_client.close_session(result.session_id, reason="rollout_finished")
                await self.gateway_client.wait_telemetry_flush(result.session_id)
            else:
                with trace.span("gateway_close_session"):
                    await self.gateway_client.close_session(result.session_id, reason="rollout_finished")
                with trace.span("gateway_wait_telemetry_flush"):
                    await self.gateway_client.wait_telemetry_flush(result.session_id)
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

    def _build_start_request(
        self,
        lease: SimulationAgentLease,
        session: SessionContext,
        worker_id: int,
    ) -> SimulationStartRequest:
        storage_config: Dict[str, Any] = {}
        if self.cfg.storage_type == "sqlite":
            storage_config["db_url"] = self.cfg.db_url

        return SimulationStartRequest(
            job_id=self.cfg.job_id,
            session_id=session.session_id,
            group_id=lease.group_id,
            gateway_base_url=self.cfg.gateway_base_url,
            model=self.cfg.llm_model,
            temperature=self.cfg.llm_temperature,
            max_steps=self.cfg.max_steps,
            storage_type=self.cfg.storage_type,
            env_params=strip_internal_env_params(lease.env_params),
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
