from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from typing import Any


class EvalMethod(str, Enum):
    RULE_EVALUATOR = "rule_evaluator"


class EvalStatus(str, Enum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass
class EvalSpec:
    eval_id: str
    method: EvalMethod | str = EvalMethod.RULE_EVALUATOR
    timeout_s: float = 60.0
    rule_evaluator: str | None = None
    weight: float = 1.0

    def __post_init__(self) -> None:
        self.method = coerce_eval_method(self.method)
        self.timeout_s = float(self.timeout_s)
        self.weight = float(self.weight)
        if self.rule_evaluator is not None:
            self.rule_evaluator = str(self.rule_evaluator).strip() or None


@dataclass
class EvalRequest:
    job_id: str
    session_id: str
    lease: Any | None = None
    start_result: Any | None = None
    env_params: dict[str, Any] = field(default_factory=dict)
    eval_specs: list[EvalSpec] = field(default_factory=list)


@dataclass
class EvalResult:
    session_id: str
    status: str
    normalized_score_10: float
    raw_score: float | None = None
    reason: str = ""
    method_results: list[dict[str, Any]] = field(default_factory=list)
    artifacts: dict[str, Any] = field(default_factory=dict)
    error_text: str | None = None
    eval_id: str | None = None
    method: str | None = None

    @classmethod
    def failed(
        cls,
        *,
        session_id: str,
        reason: str,
        eval_id: str | None = None,
        method: str | None = None,
        status: str = EvalStatus.FAILED.value,
        error_text: str | None = None,
        artifacts: dict[str, Any] | None = None,
    ) -> "EvalResult":
        return cls(
            session_id=session_id,
            status=status,
            normalized_score_10=0.0,
            reason=reason,
            error_text=error_text or reason,
            artifacts=artifacts or {},
            eval_id=eval_id,
            method=method,
        )

    def to_method_result(self) -> dict[str, Any]:
        data = to_jsonable(self)
        data.pop("method_results", None)
        return data


@dataclass
class Trajectory:
    session_id: str
    steps: list[dict[str, Any]] = field(default_factory=list)
    final_response: str | None = None
    token_usage: dict[str, int] = field(default_factory=dict)
    raw_rows: list[dict[str, Any]] = field(default_factory=list)
    sealed: bool = False
    warnings: list[str] = field(default_factory=list)


def parse_eval_specs(
    env_params: dict[str, Any] | None,
    *,
    default_specs: list[EvalSpec | dict[str, Any]] | None = None,
) -> list[EvalSpec]:
    env_params = env_params or {}
    evaluation = env_params.get("evaluation")
    evaluation = evaluation if isinstance(evaluation, dict) else {}
    raw_specs = (
        env_params.get("eval")
        or evaluation.get("eval")
        or evaluation.get("specs")
        or default_specs
        or []
    )
    if isinstance(raw_specs, dict):
        raw_specs = [raw_specs]
    return [coerce_eval_spec(item) for item in raw_specs]


def coerce_eval_spec(value: EvalSpec | dict[str, Any]) -> EvalSpec:
    if isinstance(value, EvalSpec):
        return value
    data = dict(value)
    data.setdefault("method", EvalMethod.RULE_EVALUATOR)
    data.setdefault("eval_id", str(data["method"]))
    return EvalSpec(**data)


def coerce_eval_method(value: EvalMethod | str) -> EvalMethod:
    if isinstance(value, EvalMethod):
        return value
    text = str(value).strip().lower()
    if text in {"rule", "rule_eval", "rule_judge", "rule_evaluator"}:
        return EvalMethod.RULE_EVALUATOR
    raise ValueError(
        f"unsupported eval method {value!r}; this build only supports rule_evaluator"
    )


def validate_eval_specs(specs: list[EvalSpec]) -> None:
    seen: set[str] = set()
    for spec in specs:
        if not spec.eval_id:
            raise ValueError("eval_id must not be empty")
        if spec.eval_id in seen:
            raise ValueError(f"duplicate eval_id: {spec.eval_id}")
        seen.add(spec.eval_id)
        if spec.timeout_s <= 0:
            raise ValueError(f"{spec.eval_id}: timeout_s must be > 0")
        if spec.weight < 0:
            raise ValueError(f"{spec.eval_id}: weight must be >= 0")


def merge_eval_results(results: list[EvalResult], specs: list[EvalSpec]) -> EvalResult:
    if not results:
        return EvalResult.failed(session_id="", reason="no eval results")

    session_id = results[0].session_id
    weight_by_id = {spec.eval_id: spec.weight for spec in specs}
    succeeded: list[tuple[EvalResult, float]] = []
    for index, result in enumerate(results):
        eval_id = result.eval_id or (specs[index].eval_id if index < len(specs) else None)
        weight = weight_by_id.get(eval_id or "", specs[index].weight if index < len(specs) else 1.0)
        if result.status == EvalStatus.SUCCEEDED.value and weight > 0:
            succeeded.append((result, weight))

    method_results = [result.to_method_result() for result in results]
    artifacts = {"method_count": len(results)}
    if not succeeded:
        reason = "all eval specs failed"
        return EvalResult(
            session_id=session_id,
            status=EvalStatus.FAILED.value,
            normalized_score_10=0.0,
            reason=reason,
            method_results=method_results,
            artifacts=artifacts,
            error_text=reason,
        )

    total_weight = sum(weight for _, weight in succeeded)
    score = sum(result.normalized_score_10 * weight for result, weight in succeeded) / total_weight
    failed_count = len(results) - len(succeeded)
    return EvalResult(
        session_id=session_id,
        status=EvalStatus.SUCCEEDED.value,
        normalized_score_10=_clamp_score_10(score),
        raw_score=score,
        reason=f"merged {len(succeeded)} succeeded eval result(s), {failed_count} failed",
        method_results=method_results,
        artifacts=artifacts,
    )


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return to_jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(item) for item in value]
    return value


def _clamp_score_10(value: float) -> float:
    return max(0.0, min(10.0, float(value)))
