from __future__ import annotations

import asyncio
import logging
import time
from typing import Protocol

from core.perf_trace import PerfTrace
from evaluator.eval_types import (
    EvalRequest,
    EvalResult,
    EvalSpec,
    EvalStatus,
    Trajectory,
    merge_eval_results,
    parse_eval_specs,
    validate_eval_specs,
)
from evaluator.trajectory_reader import TrajectoryReader

log = logging.getLogger("evaluator.service")


class RuleEvaluationBackend(Protocol):
    async def evaluate(
        self,
        *,
        request: EvalRequest,
        spec: EvalSpec,
        trajectory: Trajectory,
    ) -> EvalResult:
        ...


class EvaluationService:
    def __init__(
        self,
        *,
        trajectory_reader: TrajectoryReader | None,
        backend: RuleEvaluationBackend,
        max_concurrency: int = 64,
        default_specs: list[EvalSpec | dict] | None = None,
    ) -> None:
        self.trajectory_reader = trajectory_reader
        self.backend = backend
        self.max_concurrency = max(1, int(max_concurrency or 1))
        self.default_specs = default_specs or []
        self._sem = asyncio.Semaphore(self.max_concurrency)

    async def evaluate(self, request: EvalRequest) -> EvalResult:
        started_at = time.perf_counter()
        trace = PerfTrace(
            "evaluator.evaluate",
            logger=log,
            context={
                "job_id": request.job_id,
                "session_id": request.session_id,
                "max_concurrency": self.max_concurrency,
            },
        )
        try:
            with trace.span("resolve_specs"):
                specs = request.eval_specs or parse_eval_specs(
                    request.env_params,
                    default_specs=self.default_specs,
                )
                validate_eval_specs(specs)
            trace.mark(
                "specs_validated",
                spec_count=len(specs),
                eval_ids=[spec.eval_id for spec in specs],
            )

            with trace.span("read_trajectory"):
                trajectory = await self._read_trajectory(request)
            trace.mark(
                "trajectory_ready",
                sealed=trajectory.sealed,
                step_count=len(trajectory.steps),
                warning_count=len(trajectory.warnings),
            )

            tasks = [self._evaluate_one_spec(request, spec, trajectory) for spec in specs]
            with trace.span("rule_evaluate_all", spec_count=len(specs)):
                results = await asyncio.gather(*tasks)
            with trace.span("merge_results"):
                final_result = merge_eval_results(results, specs)
                final_result.session_id = request.session_id

            log.info(
                "EVAL complete: session=%s status=%s score=%.4f specs=%d elapsed=%.2fs",
                request.session_id,
                final_result.status,
                final_result.normalized_score_10,
                len(specs),
                time.perf_counter() - started_at,
            )
            trace.emit_summary(
                status=final_result.status,
                score=final_result.normalized_score_10,
                reason=final_result.reason,
            )
            return final_result
        except asyncio.CancelledError:
            trace.emit_summary(status="cancelled", error_type="CancelledError")
            raise
        except Exception as exc:
            trace.emit_summary(status="failed", error_type=type(exc).__name__, error=str(exc))
            raise

    async def _evaluate_one_spec(
        self,
        request: EvalRequest,
        spec: EvalSpec,
        trajectory: Trajectory,
    ) -> EvalResult:
        async with self._sem:
            try:
                return await asyncio.wait_for(
                    self.backend.evaluate(request=request, spec=spec, trajectory=trajectory),
                    timeout=spec.timeout_s,
                )
            except asyncio.TimeoutError as exc:
                log.warning(
                    "EVAL RULE timeout: session=%s eval_id=%s timeout_s=%.2f",
                    request.session_id,
                    spec.eval_id,
                    spec.timeout_s,
                )
                return EvalResult.failed(
                    session_id=request.session_id,
                    eval_id=spec.eval_id,
                    method=spec.method.value,
                    status=EvalStatus.TIMEOUT.value,
                    reason="rule evaluator timed out",
                    error_text=str(exc),
                )
            except Exception as exc:
                log.exception(
                    "EVAL RULE failed: session=%s eval_id=%s",
                    request.session_id,
                    spec.eval_id,
                )
                return EvalResult.failed(
                    session_id=request.session_id,
                    eval_id=spec.eval_id,
                    method=spec.method.value,
                    reason="rule evaluator failed",
                    error_text=str(exc),
                )

    async def _read_trajectory(self, request: EvalRequest) -> Trajectory:
        if self.trajectory_reader is None:
            return Trajectory(session_id=request.session_id)
        return await self.trajectory_reader.wait_until_sealed(request.session_id)
