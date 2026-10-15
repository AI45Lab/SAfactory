"""Evaluate a Sandbox episode exclusively from its downloaded result file."""
from __future__ import annotations

import asyncio
from pathlib import Path

from core.result_artifact import read_result_file
from evaluator.eval_types import EvalRequest, EvalResult, EvalSpec, Trajectory


class ResultFileEvaluatorBackend:
    async def evaluate(
        self, *, request: EvalRequest, spec: EvalSpec, trajectory: Trajectory
    ) -> EvalResult:
        del trajectory
        start_result = request.start_result
        path = getattr(start_result, "result_artifact_path", None)
        if not path:
            raise ValueError("Sandbox evaluation requires a downloaded result_artifact_path")
        lease = request.lease
        attempt_id = str((getattr(lease, "runtime_config", None) or {}).get("attempt_id") or "")
        body = await asyncio.to_thread(
            read_result_file, Path(path), job_id=request.job_id,
            session_id=request.session_id, attempt_id=attempt_id,
        )
        evaluation = body["evaluation"]
        artifacts = dict(evaluation["artifacts"])
        artifacts["normalizer_id"] = evaluation["normalizer_id"]
        artifacts["result_artifact_path"] = str(path)
        if (getattr(start_result, "metrics", None) or {}).get("execution_error"):
            return EvalResult.failed(
                session_id=request.session_id, eval_id=evaluation["eval_id"],
                method=evaluation["method"], reason="Sandbox execution did not finish successfully",
                error_text=str(start_result.metrics["execution_error"]), artifacts=artifacts,
            )
        if evaluation["status"] not in {"succeeded", "truncated"}:
            return EvalResult.failed(
                session_id=request.session_id, eval_id=evaluation["eval_id"],
                method=evaluation["method"], status=evaluation["status"],
                reason=evaluation["reason"], error_text=evaluation["error_text"], artifacts=artifacts,
            )
        return EvalResult(
            session_id=request.session_id, status=evaluation["status"],
            normalized_score_10=evaluation["normalized_score_10"],
            raw_score=evaluation.get("raw_score"), reason=evaluation["reason"],
            artifacts=artifacts, error_text=evaluation["error_text"],
            eval_id=evaluation["eval_id"], method=evaluation["method"],
        )
