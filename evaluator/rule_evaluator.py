from __future__ import annotations

import asyncio
import importlib
import importlib.util
import inspect
import logging
from pathlib import Path
from typing import Any

from evaluator.eval_types import EvalRequest, EvalResult, EvalSpec, EvalStatus, Trajectory
from evaluator.rule_eval_resolver import resolve_rule_evaluator_locator

log = logging.getLogger("evaluator.rule_evaluator")


class RuleEvaluatorBackend:
    """Load and run trusted environment-local rule evaluators."""

    def __init__(self, *, env_root: str | Path = "env") -> None:
        self.env_root = Path(env_root).expanduser().resolve(strict=False)
        self._cache: dict[str, Any] = {}

    async def evaluate(
        self,
        *,
        request: EvalRequest,
        spec: EvalSpec,
        trajectory: Trajectory,
    ) -> EvalResult:
        try:
            locator = self._resolve_locator(request=request, spec=spec)
            evaluator, source = self._load_locator(locator)
            log.info(
                "EVAL RULE start: session=%s eval_id=%s source=%s",
                request.session_id,
                spec.eval_id,
                source,
            )
            value = await self._invoke(
                evaluator,
                request=request,
                spec=spec,
                trajectory=trajectory,
            )
            result = self._coerce_result(value, request=request, spec=spec)
            result.artifacts.setdefault("rule_evaluator_source", source)
            log.info(
                "EVAL RULE complete: session=%s eval_id=%s status=%s score=%.4f",
                request.session_id,
                spec.eval_id,
                result.status,
                result.normalized_score_10,
            )
            return result
        except asyncio.CancelledError:
            raise
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

    def _resolve_locator(self, *, request: EvalRequest, spec: EvalSpec) -> str:
        configured = str(spec.rule_evaluator or "").strip()
        if configured and configured.lower() not in {"1", "true", "yes", "default", "auto"}:
            return configured

        raw_env_params = _raw_env_params(request)
        locator = resolve_rule_evaluator_locator(
            raw_env_params,
            agent_name=_agent_name(request),
            env_root=self.env_root,
        )
        if not locator:
            raise FileNotFoundError(
                "no rule evaluator configured or discovered for "
                f"agent={_agent_name(request)!r}"
            )
        return locator

    def _load_locator(self, locator: str) -> tuple[Any, str]:
        text = str(locator or "").strip()
        if not text:
            raise ValueError("empty rule evaluator locator")
        if text.startswith("file:"):
            return self._load_file(Path(text[len("file:") :]))
        if text.endswith(".py") or text.startswith("/") or text.startswith("."):
            return self._load_file(Path(text))

        module_name, _, attr = text.partition(":")
        module_name = module_name.strip()
        attr = attr.strip()
        if not module_name.startswith("env."):
            raise ValueError(
                "rule evaluator module locator must start with 'env.' "
                f"or point to a local file, got {locator!r}"
            )
        source = f"{module_name}:{attr}" if attr else module_name
        if source not in self._cache:
            module = importlib.import_module(module_name)
            self._cache[source] = _select_evaluator(module, attr=attr)
        return self._cache[source], source

    def _load_file(self, path: Path) -> tuple[Any, str]:
        resolved = path.expanduser().resolve(strict=False)
        if not resolved.is_file():
            raise FileNotFoundError(f"rule evaluator file not found: {resolved}")
        source = f"file:{resolved}"
        if source not in self._cache:
            module_name = "safactory_rule_evaluator_" + str(abs(hash(str(resolved))))
            module_spec = importlib.util.spec_from_file_location(module_name, resolved)
            if module_spec is None or module_spec.loader is None:
                raise ImportError(f"cannot import rule evaluator file: {resolved}")
            module = importlib.util.module_from_spec(module_spec)
            module_spec.loader.exec_module(module)
            self._cache[source] = _select_evaluator(module, attr="")
        return self._cache[source], source

    async def _invoke(
        self,
        evaluator: Any,
        *,
        request: EvalRequest,
        spec: EvalSpec,
        trajectory: Trajectory,
    ) -> Any:
        fn = getattr(evaluator, "evaluate", None) or evaluator
        if not callable(fn):
            raise TypeError(f"rule evaluator is not callable: {type(evaluator).__name__}")

        if inspect.iscoroutinefunction(fn) or inspect.iscoroutinefunction(
            getattr(fn, "__call__", None)
        ):
            value = _call_evaluator(fn, request=request, spec=spec, trajectory=trajectory)
        else:
            value = await asyncio.to_thread(
                _call_evaluator,
                fn,
                request=request,
                spec=spec,
                trajectory=trajectory,
            )
        if inspect.isawaitable(value):
            return await value
        return value

    def _coerce_result(self, value: Any, *, request: EvalRequest, spec: EvalSpec) -> EvalResult:
        if isinstance(value, EvalResult):
            value.session_id = value.session_id or request.session_id
            value.eval_id = value.eval_id or spec.eval_id
            value.method = value.method or spec.method.value
            value.normalized_score_10 = _clamp_score(value.normalized_score_10)
            return value

        if isinstance(value, dict):
            score = _float_or_none(
                value.get("normalized_score_10", value.get("score", value.get("reward")))
            )
            if score is None:
                raise ValueError("rule evaluator dict result requires normalized_score_10/score/reward")
            return EvalResult(
                session_id=str(value.get("session_id") or request.session_id),
                eval_id=str(value.get("eval_id") or spec.eval_id),
                method=str(value.get("method") or spec.method.value),
                status=str(value.get("status") or EvalStatus.SUCCEEDED.value),
                normalized_score_10=_clamp_score(score),
                raw_score=_float_or_none(value.get("raw_score")),
                reason=str(value.get("reason") or "rule evaluator score"),
                artifacts=dict(value.get("artifacts") or {}),
                error_text=None if value.get("error_text") is None else str(value.get("error_text")),
            )

        score = _float_or_none(value)
        if score is None:
            raise TypeError(f"unsupported rule evaluator result type: {type(value).__name__}")
        return EvalResult(
            session_id=request.session_id,
            eval_id=spec.eval_id,
            method=spec.method.value,
            status=EvalStatus.SUCCEEDED.value,
            normalized_score_10=_clamp_score(score),
            raw_score=score,
            reason="rule evaluator score",
        )


def _call_evaluator(
    fn: Any,
    *,
    request: EvalRequest,
    spec: EvalSpec,
    trajectory: Trajectory,
) -> Any:
    if _supports_keyword_call(fn):
        return fn(request=request, spec=spec, trajectory=trajectory)
    return fn(request, spec, trajectory)


def _select_evaluator(module: Any, *, attr: str) -> Any:
    if attr:
        evaluator = getattr(module, attr)
    else:
        evaluator = (
            getattr(module, "RuleEvaluator", None)
            or getattr(module, "evaluate_rule", None)
            or getattr(module, "evaluate", None)
        )
    if evaluator is None:
        raise AttributeError(
            "rule evaluator module must export RuleEvaluator, evaluate_rule, or evaluate"
        )
    return evaluator() if inspect.isclass(evaluator) else evaluator


def _supports_keyword_call(fn: Any) -> bool:
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        return True
    params = signature.parameters
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values()):
        return True
    return {"request", "spec", "trajectory"}.issubset(params)


def _raw_env_params(request: EvalRequest) -> dict[str, Any]:
    lease = getattr(request, "lease", None)
    env_params = getattr(lease, "env_params", None)
    if isinstance(env_params, dict):
        return env_params
    return request.env_params if isinstance(request.env_params, dict) else {}


def _agent_name(request: EvalRequest) -> str:
    lease = getattr(request, "lease", None)
    return str(
        getattr(lease, "agent_name", None)
        or request.env_params.get("task_family")
        or ""
    ).strip()


def _float_or_none(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _clamp_score(value: float) -> float:
    return max(0.0, min(10.0, float(value)))
