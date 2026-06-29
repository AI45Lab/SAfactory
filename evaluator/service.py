from __future__ import annotations

import asyncio
import logging
import time
from typing import Protocol

from core.perf_trace import PerfTrace
from evaluator.backends.base import EvaluationBackend
from evaluator.eval_types import (
    EvalMethod,
    EvalRequest,
    EvalResult,
    EvalSpec,
    EvalStatus,
    Trajectory,
    coerce_eval_method,
    merge_eval_results,
    parse_eval_specs,
    validate_eval_specs,
)
from evaluator.trajectory_reader import TrajectoryReader

log = logging.getLogger("evaluator.service")


class RunRegistryLike(Protocol):
    async def mark_evaluating(self, session_id: str) -> None:
        ...

    async def mark_eval_finished(self, session_id: str, result: EvalResult) -> None:
        ...


class EvaluationService:
    def __init__(
        self,
        *,
        trajectory_reader: TrajectoryReader | None,
        backends: dict[EvalMethod | str, EvaluationBackend],
        registry: RunRegistryLike | None = None,
        max_concurrency: int = 64,
        fail_policy: str = "zero_reward",
        default_specs: list[EvalSpec | dict] | None = None,
    ) -> None:
        self.registry = registry
        self.trajectory_reader = trajectory_reader
        self.backends = {coerce_eval_method(method): backend for method, backend in backends.items()}
        self.max_concurrency = max(1, int(max_concurrency or 1))
        self.fail_policy = fail_policy
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
                "fail_policy": self.fail_policy,
            },
        )
        try:
            if self.registry is not None:
                with trace.span("registry_mark_evaluating"):
                    await self.registry.mark_evaluating(request.session_id)

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
                methods=[spec.method.value for spec in specs],
            )
            log.info(
                "EVAL FLOW start: session=%s job=%s specs=%s fail_policy=%s",
                request.session_id,
                request.job_id,
                [
                    {
                        "eval_id": spec.eval_id,
                        "method": spec.method.value,
                        "weight": spec.weight,
                        "timeout_s": spec.timeout_s,
                    }
                    for spec in specs
                ],
                self.fail_policy,
            )
            with trace.span("read_trajectory"):
                trajectory = await self._read_trajectory(request)
            trace.mark(
                "trajectory_ready",
                sealed=trajectory.sealed,
                step_count=len(trajectory.steps),
                warning_count=len(trajectory.warnings),
            )
            log.info(
                "EVAL FLOW trajectory ready: session=%s sealed=%s steps=%d warnings=%s",
                request.session_id,
                trajectory.sealed,
                len(trajectory.steps),
                trajectory.warnings,
            )

            tasks = [self._evaluate_one_spec(request, spec, trajectory) for spec in specs]
            with trace.span("backend_evaluate_all", spec_count=len(specs)):
                results = await asyncio.gather(*tasks)
            trace.mark(
                "backend_results_ready",
                result_count=len(results),
                failed_count=sum(1 for result in results if result.status != EvalStatus.SUCCEEDED.value),
            )
            log.info(
                "EVAL FLOW backend results: session=%s results=%s",
                request.session_id,
                [
                    {
                        "eval_id": result.eval_id,
                        "method": result.method,
                        "status": result.status,
                        "score": result.normalized_score_10,
                        "reason": result.reason,
                    }
                    for result in results
                ],
            )
            with trace.span("merge_results"):
                final_result = merge_eval_results(results, specs, fail_policy=self.fail_policy)
                final_result.session_id = request.session_id
            if self.registry is not None:
                with trace.span("registry_mark_eval_finished"):
                    await self.registry.mark_eval_finished(request.session_id, final_result)
            log.info(
                "EVAL FLOW complete: session=%s status=%s score=%.4f elapsed=%.2fs reason=%s",
                request.session_id,
                final_result.status,
                final_result.normalized_score_10,
                time.perf_counter() - started_at,
                final_result.reason,
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
        backend = self._select_backend(spec)
        async with self._sem:
            try:
                log.info(
                    "EVAL SPEC start: session=%s eval_id=%s method=%s backend=%s timeout_s=%.2f",
                    request.session_id,
                    spec.eval_id,
                    spec.method.value,
                    backend.__class__.__name__,
                    spec.timeout_s,
                )
                started_at = time.perf_counter()
                result = await asyncio.wait_for(
                    backend.evaluate(request=request, spec=spec, trajectory=trajectory),
                    timeout=spec.timeout_s,
                )
                log.info(
                    "EVAL SPEC complete: session=%s eval_id=%s method=%s status=%s score=%.4f elapsed=%.2fs",
                    request.session_id,
                    spec.eval_id,
                    spec.method.value,
                    result.status,
                    result.normalized_score_10,
                    time.perf_counter() - started_at,
                )
                return result
            except asyncio.TimeoutError as exc:
                log.warning(
                    "EVAL SPEC timeout: session=%s eval_id=%s method=%s timeout_s=%.2f",
                    request.session_id,
                    spec.eval_id,
                    spec.method.value,
                    spec.timeout_s,
                )
                return EvalResult.failed(
                    session_id=request.session_id,
                    eval_id=spec.eval_id,
                    method=spec.method.value,
                    status=EvalStatus.TIMEOUT.value,
                    reason="eval spec timed out",
                    error_text=str(exc),
                )
            except Exception as exc:
                log.exception(
                    "EVAL SPEC failed: session=%s eval_id=%s method=%s",
                    request.session_id,
                    spec.eval_id,
                    spec.method.value,
                )
                return EvalResult.failed(
                    session_id=request.session_id,
                    eval_id=spec.eval_id,
                    method=spec.method.value,
                    reason="eval spec failed",
                    error_text=str(exc),
                )

    def _select_backend(self, spec: EvalSpec) -> EvaluationBackend:
        try:
            return self.backends[spec.method]
        except KeyError as exc:
            raise KeyError(f"no backend registered for eval method {spec.method.value}") from exc

    async def _read_trajectory(self, request: EvalRequest) -> Trajectory:
        if self.trajectory_reader is None:
            log.info("EVAL FLOW trajectory reader disabled: session=%s", request.session_id)
            return Trajectory(session_id=request.session_id)
        log.info("EVAL FLOW waiting for sealed trajectory: session=%s", request.session_id)
        return await self.trajectory_reader.wait_until_sealed(request.session_id)
