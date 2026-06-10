from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from evaluator.eval_types import EvalResult, EvalRequest, EvalSpec, EvalStatus, Trajectory


async def evaluate_rule(
    *,
    request: EvalRequest,
    spec: EvalSpec,
    trajectory: Trajectory,
) -> EvalResult:
    metrics = _start_metrics(request)
    insucess_rate = _float_or_none(
        metrics.get("insucess_rate", metrics.get("insuccess_rate"))
    )
    metric_payloads: list[dict[str, Any]] = []

    if insucess_rate is None:
        metric_payloads = _read_metric_payloads(metrics.get("metrics_files"))
        values = [
            _float_or_none(item.get("insucess_rate", item.get("insuccess_rate")))
            for item in metric_payloads
        ]
        values = [value for value in values if value is not None]
        if values:
            insucess_rate = sum(values) / len(values)

    if insucess_rate is None:
        return EvalResult.failed(
            session_id=request.session_id,
            eval_id=spec.eval_id,
            method=spec.method.value,
            reason="openrt metrics did not contain insucess_rate",
            artifacts={
                "bench": "openrt",
                "metrics": metrics,
                "metric_payload_count": len(metric_payloads),
            },
        )

    normalized_score = _clamp(float(insucess_rate) * 10.0, 0.0, 10.0)
    return EvalResult(
        session_id=request.session_id,
        eval_id=spec.eval_id,
        method=spec.method.value,
        status=EvalStatus.SUCCEEDED.value,
        raw_score=float(insucess_rate),
        normalized_score_10=round(normalized_score, 6),
        reason="openrt rule evaluator: reward = insucess_rate * 10",
        artifacts={
            "bench": "openrt",
            "metric": "insucess_rate",
            "insucess_rate": float(insucess_rate),
            "metrics": metrics,
            "metric_payload_count": len(metric_payloads),
        },
    )


def _start_metrics(request: EvalRequest) -> dict[str, Any]:
    start_result = getattr(request, "start_result", None)
    metrics = getattr(start_result, "metrics", None)
    return dict(metrics) if isinstance(metrics, dict) else {}


def _read_metric_payloads(value: Any) -> list[dict[str, Any]]:
    paths: list[str]
    if isinstance(value, str):
        paths = [value]
    elif isinstance(value, (list, tuple)):
        paths = [str(item) for item in value if str(item).strip()]
    else:
        paths = []

    payloads: list[dict[str, Any]] = []
    for raw_path in paths:
        for path in _candidate_paths(raw_path):
            try:
                body = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if isinstance(body, dict):
                payloads.append(body)
                break
    return payloads


def _candidate_paths(raw_path: str) -> list[Path]:
    path = Path(raw_path)
    candidates = [path]
    text = str(path)
    marker = "/app/results/"
    if text.startswith(marker):
        candidates.append(Path.cwd() / "results" / text[len(marker) :])
    return candidates


def _float_or_none(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
