"""Portable, standard-library-only contract for authoritative result files.

This file is also uploaded next to Sandbox runners and imported as
``result_artifact``. Keep benchmark and host runtime imports out of it.
"""
from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "safactory.result.v2"
_RUN_STATUSES = {"succeeded", "failed", "timeout", "truncated"}
_EVAL_STATUSES = {"succeeded", "failed", "timeout", "truncated"}
_SCORED_STATUSES = {"succeeded", "truncated"}


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _number(value: Any, name: str, *, bounded: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite or (bounded and not 0 <= value <= 10):
        raise ValueError(f"{name} must be {'in [0, 10]' if bounded else 'finite'}")


def _validate_result(
    payload: Any, *, job_id: str, session_id: str, attempt_id: str
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("result JSON must be an object")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION!r}")
    for name, expected in (("job_id", job_id), ("session_id", session_id), ("attempt_id", attempt_id)):
        _text(expected, f"expected {name}")
        if payload.get(name) != expected:
            raise ValueError(f"result {name} does not match this execution")
    status = payload.get("status")
    if not isinstance(status, str) or status not in _RUN_STATUSES:
        raise ValueError("result status is missing or unsupported")
    for name in ("terminated", "truncated"):
        if type(payload.get(name)) is not bool:
            raise ValueError(f"result {name} must be a boolean")
    if not (payload["terminated"] or payload["truncated"]):
        raise ValueError("a final result must be terminated or truncated")
    if payload["truncated"] != (status in {"timeout", "truncated"}):
        raise ValueError("result status and truncated disagree")
    count = payload.get("step_count")
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError("step_count must be a non-negative integer")
    if "total_reward" not in payload:
        raise ValueError("result total_reward is required (null is allowed)")
    if payload["total_reward"] is not None:
        _number(payload["total_reward"], "total_reward")
    if "error_text" not in payload or not (payload["error_text"] is None or isinstance(payload["error_text"], str)):
        raise ValueError("result error_text must be a string or null")
    if not isinstance(payload.get("metrics"), dict):
        raise ValueError("result metrics must be an object")

    evaluation = payload.get("evaluation")
    if not isinstance(evaluation, dict):
        raise ValueError("result evaluation must be an object")
    if evaluation.get("session_id") != session_id:
        raise ValueError("evaluation session_id does not match this execution")
    eval_status = evaluation.get("status")
    if not isinstance(eval_status, str) or eval_status not in _EVAL_STATUSES:
        raise ValueError("evaluation status is missing or unsupported")
    if eval_status == "succeeded" and status != "succeeded":
        raise ValueError("successful evaluation requires successful execution")
    if eval_status == "truncated" and not payload["truncated"]:
        raise ValueError("truncated evaluation requires truncated execution")
    for name in ("eval_id", "method", "normalizer_id"):
        _text(evaluation.get(name), f"evaluation.{name}")
    if not isinstance(evaluation.get("reason"), str):
        raise ValueError("evaluation.reason must be a string")
    if not isinstance(evaluation.get("artifacts"), dict):
        raise ValueError("evaluation.artifacts must be an object")
    if "error_text" not in evaluation or not (
        evaluation["error_text"] is None or isinstance(evaluation["error_text"], str)
    ):
        raise ValueError("evaluation.error_text must be a string or null")
    if eval_status in _SCORED_STATUSES:
        _number(evaluation.get("normalized_score_10"), "evaluation.normalized_score_10", bounded=True)
    elif evaluation.get("normalized_score_10") is not None:
        # A failure placeholder never becomes a score, but must still be valid JSON data.
        _number(evaluation["normalized_score_10"], "evaluation.normalized_score_10", bounded=True)
    if evaluation.get("raw_score") is not None:
        _number(evaluation["raw_score"], "evaluation.raw_score")
    return payload


def finalize_result(
    result: Mapping[str, Any], *, env_params: Mapping[str, Any],
    scorer: Callable[..., Mapping[str, Any]], identity: Mapping[str, str],
) -> dict[str, Any]:
    """Call the environment scorer once and attach its completed evaluation."""
    payload = dict(result)
    trusted = {name: _text(identity.get(name), name) for name in ("job_id", "session_id", "attempt_id")}
    payload.update(schema_version=SCHEMA_VERSION, **trusted)
    try:
        value = scorer(payload, env_params=env_params)
        if not isinstance(value, Mapping):
            raise TypeError("scorer must return a mapping")
        evaluation = dict(value)
        if "session_id" in evaluation and evaluation["session_id"] != trusted["session_id"]:
            raise ValueError("scorer returned another session_id")
        evaluation["session_id"] = trusted["session_id"]
        payload["evaluation"] = evaluation
        _validate_result(payload, **trusted)
    except Exception as exc:
        payload["evaluation"] = {
            "session_id": trusted["session_id"], "status": "failed",
            "eval_id": "scorer_error", "method": "rule_evaluator",
            "normalizer_id": "scorer_error", "raw_score": None,
            "reason": "environment scoring failed", "artifacts": {},
            "error_text": f"{type(exc).__name__}: {exc}",
        }
        # Invalid execution fields remain errors; do not manufacture successful rollout data.
        _validate_result(payload, **trusted)
    return payload


def write_result_atomic(path: str | Path, payload: Mapping[str, Any]) -> None:
    """Validate and durably publish one complete result without exposing partial JSON."""
    body = dict(payload)
    _validate_result(body, **{name: body.get(name) for name in ("job_id", "session_id", "attempt_id")})
    write_json_atomic(path, body)


def write_json_atomic(path: str | Path, payload: Mapping[str, Any]) -> None:
    """Durably publish a JSON object; callers own protocol and identity validation."""
    content = json.dumps(dict(payload), ensure_ascii=False, allow_nan=False).encode("utf-8")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp", delete=False) as stream:
            temporary = stream.name
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        temporary = None
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)


def update_json_atomic(
    path: str | Path, updates: Mapping[str, Any], *, identity: Mapping[str, str]
) -> None:
    """Merge an execution reference after checking its trusted identity.

    Replacement is atomic; callers serialize updates belonging to one execution.
    Missing references and identity changes fail before any file is replaced.
    """
    expected = {name: _text(identity.get(name), name) for name in ("job_id", "session_id", "attempt_id")}
    with Path(path).open("r", encoding="utf-8") as stream:
        reference = json.load(stream, object_pairs_hook=_unique_object, parse_constant=_reject_constant, parse_float=_finite_float)
    if not isinstance(reference, dict):
        raise ValueError("execution reference must be a JSON object")
    for name, value in expected.items():
        if reference.get(name) != value:
            raise ValueError(f"execution reference {name} mismatch")
    merged = {**reference, **updates}
    for name, value in expected.items():
        if merged.get(name) != value:
            raise ValueError(f"execution reference update cannot change {name}")
    write_json_atomic(path, merged)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> Any:
    raise ValueError(f"invalid JSON numeric constant: {value}")


def _finite_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("JSON number is not finite")
    return number


def read_result_file(
    path: str | Path, *, job_id: str, session_id: str, attempt_id: str
) -> dict[str, Any]:
    """Read the authoritative local file; never infer missing identity or scores."""
    with Path(path).open("r", encoding="utf-8") as stream:
        body = json.load(stream, object_pairs_hook=_unique_object, parse_constant=_reject_constant, parse_float=_finite_float)
    return _validate_result(body, job_id=job_id, session_id=session_id, attempt_id=attempt_id)
