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
    ) -> None:
        self.trajectory_reader = trajectory_reader
        self.backend = backend
        self.max_concurrency = max(1, int(max_concurrency or 1))
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
            spec = request.eval_spec
            if spec is None:
                raise ValueError("rule evaluator spec is required")
            trace.mark(
                "spec_ready",
                eval_id=spec.eval_id,
            )

            with trace.span("read_trajectory"):
                trajectory = await self._read_trajectory(request)
            trace.mark(
                "trajectory_ready",
                sealed=trajectory.sealed,
                step_count=len(trajectory.steps),
                warning_count=len(trajectory.warnings),
            )

            with trace.span("rule_evaluate", eval_id=spec.eval_id):
                final_result = await self._evaluate_rule(request, spec, trajectory)
            final_result.session_id = request.session_id

            log.info(
                "EVAL complete: session=%s status=%s score=%.4f elapsed=%.2fs reason=%s error_text=%s",
                request.session_id,
                final_result.status,
                final_result.normalized_score_10,
                time.perf_counter() - started_at,
                final_result.reason,
                final_result.error_text,
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

    async def _evaluate_rule(
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
