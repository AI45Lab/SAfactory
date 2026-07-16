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

    def __post_init__(self) -> None:
        self.eval_id = str(self.eval_id or "").strip()
        self.method = coerce_eval_method(self.method)
        self.timeout_s = float(self.timeout_s)
        if self.rule_evaluator is not None:
            self.rule_evaluator = str(self.rule_evaluator).strip() or None
        if not self.eval_id:
            raise ValueError("eval_id must not be empty")
        if self.timeout_s <= 0:
            raise ValueError(f"{self.eval_id}: timeout_s must be > 0")


@dataclass
class EvalRequest:
    job_id: str
    session_id: str
    lease: Any | None = None
    start_result: Any | None = None
    env_params: dict[str, Any] = field(default_factory=dict)
    eval_spec: EvalSpec | None = None


@dataclass
class EvalResult:
    session_id: str
    status: str
    normalized_score_10: float
    raw_score: float | None = None
    reason: str = ""
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


@dataclass
class Trajectory:
    session_id: str
    steps: list[dict[str, Any]] = field(default_factory=list)
    final_response: str | None = None
    token_usage: dict[str, int] = field(default_factory=dict)
    raw_rows: list[dict[str, Any]] = field(default_factory=list)
    sealed: bool = False
    warnings: list[str] = field(default_factory=list)


def coerce_eval_method(value: EvalMethod | str) -> EvalMethod:
    if isinstance(value, EvalMethod):
        return value
    text = str(value).strip().lower()
    if text == EvalMethod.RULE_EVALUATOR.value:
        return EvalMethod.RULE_EVALUATOR
    raise ValueError(
        f"unsupported eval method {value!r}; this build only supports rule_evaluator"
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
