from __future__ import annotations

import asyncio
import inspect
import logging
import time
from typing import Any, Dict, Optional

from core.data_manager.manager import DataManager, SessionContext
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
            eval_task_dir_name=self.cfg.eval_task_dir_name,
            strict=self.cfg.strict_eval_tasks,
        )
        self.worker_count = self._derive_worker_count()
        self._results: Dict[str, SimulationStartResult] = {}
        self._results_lock = asyncio.Lock()

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
        status = "cancelled" if cancelled else ("succeeded" if failed == 0 else "completed_with_failures")
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
            lease = await self.lease_pool.acquire()
            if lease is None:
                log.info("worker=%d: lease pool exhausted", worker_id)
                return

            agent_key = f"{lease.agent_name}_{lease.agent_id}"
            started = time.perf_counter()
            session: SessionContext | None = None
            result: SimulationStartResult | None = None
            release_reusable: bool | None = None
            run_failed = False
            try:
                log.info(
                    "worker=%d acquired agent=%s runtime=%s resource=%s reuse=%s",
                    worker_id,
                    agent_key,
                    lease.runtime,
                    lease.resource_name or lease.container_name or lease.container_id,
                    lease.reuse_container,
                )
                session = await self._create_session(lease)
                request = self._build_start_request(lease, session, worker_id)
                if self.registry is not None:
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

                result = await self._run_one_episode(lease, session, request, worker_id)
                if self.registry is not None:
                    await self.registry.mark_rollout_finished(session.session_id, result)

                if self.gateway_client is not None:
                    await self.gateway_client.close_session(result.session_id, reason="rollout_finished")
                    await self.gateway_client.wait_telemetry_flush(result.session_id)

                if self.evaluation_service is None or self.reward_committer is None:
                    release_reusable = False
                else:
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
                            await self.registry.mark_awaiting_eval(result.session_id, eval_request)
                        eval_result = await self.evaluation_service.evaluate(eval_request)
                        await self.reward_committer.commit(
                            session_id=result.session_id,
                            eval_result=eval_result,
                        )
                        if eval_result.status == "succeeded":
                            if self.registry is not None:
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

                async with self._results_lock:
                    self._results[agent_key] = result
            except Exception as exc:
                run_failed = True
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
                    await self.registry.mark_failed(session.session_id, str(exc))
                async with self._results_lock:
                    self._results[agent_key] = result
            finally:
                try:
                    if self.registry is not None and session is not None and not run_failed:
                        await self.registry.mark_releasing_container(session.session_id)
                    await self.lease_pool.done(lease, result, reusable=release_reusable)
                    if self.registry is not None and session is not None and not run_failed:
                        await self.registry.mark_done(session.session_id)
                except Exception as exc:
                    log.exception("worker=%d agent=%s critical error in lease_pool.done()", worker_id, agent_key)
                    if self.registry is not None and session is not None:
                        await self.registry.mark_failed(session.session_id, f"container release failed: {exc}")

            elapsed = time.perf_counter() - started
            log.info(
                "worker=%d agent=%s finished status=%s reward=%.6f time=%.2fs",
                worker_id,
                agent_key,
                result.status,
                result.total_reward,
                elapsed,
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

    def _build_start_request(
        self,
        lease: SimulationAgentLease,
        session: SessionContext,
        worker_id: int,
    ) -> SimulationStartRequest:
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
            storage_config={"db_url": self.cfg.db_url},
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
