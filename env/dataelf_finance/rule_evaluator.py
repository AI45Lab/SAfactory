"""DataElf finance rule evaluator: fixed SAfactory protocol shell."""

from __future__ import annotations

import asyncio
import inspect
import logging
import math
import os
from pathlib import Path
from typing import Any

log = logging.getLogger("safactory.dataelf_finance.rule")

# dataelf_eval root -> {benchmark: REWARD_CALCULATOR export}; registry
# modules are trusted code from the mounted dataelf_eval checkout, cached
# per process like the framework caches loaded rule evaluator modules.
_REGISTRY_CACHE: dict[str, dict[str, Any]] = {}


async def evaluate_rule(*, request: Any, spec: Any, trajectory: Any) -> Any:
    """Evaluate one finished episode; never reruns the case."""
    from evaluator.eval_types import EvalResult, EvalStatus

    try:
        case = _build_case(request)
    except Exception as exc:
        return _failed(request, spec, "cannot build evaluation case", error_text=str(exc))

    try:
        entry, origin = await asyncio.to_thread(_resolve_evaluator, case)
    except Exception as exc:
        return _failed(
            request, spec, str(exc),
            artifacts={"bench": "dataelf_finance", "benchmark": case["benchmark"]},
        )

    try:
        score = await asyncio.to_thread(_invoke_case_evaluator, entry, case)
    except Exception as exc:
        log.exception("case evaluation crashed: benchmark=%s session=%s", case["benchmark"], request.session_id)
        return _failed(
            request, spec, f"case evaluation crashed for benchmark {case['benchmark']!r}",
            error_text=str(exc), artifacts=_case_audit(case, origin),
        )

    status = str(score.get("status") or "succeeded")
    extra = score.get("artifacts")
    if extra is not None and not isinstance(extra, dict):
        log.warning(
            "case score artifacts is not a dict (%s); dropped",
            type(extra).__name__,
        )
        extra = None
    artifacts = {**_json_safe(extra or {}), "bench": "dataelf_finance", "benchmark": case["benchmark"]}
    if status != "succeeded":
        return _failed(
            request, spec, str(score.get("reason") or "case evaluation failed"),
            error_text=score.get("error_text"), artifacts=artifacts,
        )
    try:
        score_10 = float(score["normalized_score_10"])
    except (KeyError, TypeError, ValueError) as exc:
        return _failed(
            request, spec, "case score dict is missing numeric normalized_score_10",
            error_text=str(exc), artifacts=artifacts,
        )
    if not math.isfinite(score_10):
        # NaN would clamp to 10.0 (max/min comparisons are False for NaN)
        # and silently award a perfect reward.
        return _failed(
            request, spec, "case score normalized_score_10 is not finite",
            error_text=repr(score_10), artifacts=artifacts,
        )
    return EvalResult(
        session_id=request.session_id,
        eval_id=spec.eval_id,
        method=spec.method.value,
        status=EvalStatus.SUCCEEDED.value,
        normalized_score_10=max(0.0, min(10.0, score_10)),
        raw_score=_float_or_none(score.get("raw_score")),
        reason=str(score.get("reason") or "case evaluation score"),
        artifacts={**artifacts, "score_kind": str(score.get("score_kind") or ""), "official": bool(score.get("official"))},
    )


# ---------------------------------------------------------------- case build


def _build_case(request: Any) -> dict[str, Any]:
    env_params = request.env_params if isinstance(request.env_params, dict) else {}
    native = env_params.get("dataelf")
    native = native if isinstance(native, dict) else {}
    eval_cfg = env_params.get("dataelf_eval")
    eval_cfg = eval_cfg if isinstance(eval_cfg, dict) else {}
    metrics = getattr(request.start_result, "metrics", None)
    metrics = dict(metrics) if isinstance(metrics, dict) else {}
    row = env_params.get("dataset")
    row = row if isinstance(row, dict) else {}

    benchmark = str(native.get("benchmark") or metrics.get("benchmark") or "").strip()
    if not benchmark:
        raise ValueError("env_params.dataelf.benchmark is required to dispatch evaluation")

    root = _first_text(eval_cfg.get("root"), os.environ.get("DATAEVAL_ROOT"))
    if root and not Path(root).expanduser().is_dir():
        raise FileNotFoundError(f"dataelf_eval root not found: {root}")
    eval_params = dict(eval_cfg)
    if root:
        eval_params["root"] = str(Path(root).expanduser().resolve())

    return {
        "benchmark": benchmark,
        "question_id": _first_text(metrics.get("case_id"), row.get("id"), row.get("task_id")),
        "entity_id": _first_text(metrics.get("entity_id"), row.get("entity_id")) or None,
        "objective": _first_text(metrics.get("objective"), row.get("task"), row.get("question")),
        "row": row,
        "summary": metrics.get("summary"),
        "key_findings": metrics.get("key_findings"),
        "artifacts": {
            "result_file": _host_path(metrics.get("native_output_path"), eval_params),
            "pi_events": _host_path(metrics.get("pi_events_path"), eval_params),
            "workspace": _host_path(metrics.get("native_workspace"), eval_params),
        },
        "eval_params": eval_params,
        "session_id": str(getattr(request, "session_id", "") or ""),
        "job_id": str(getattr(request, "job_id", "") or ""),
    }


def _host_path(container_path: Any, eval_params: dict[str, Any]) -> str:
    """Rewrite a container-side artifact path to the host-side mount."""
    text = str(container_path or "")
    container_root = str(eval_params.get("container_results_root") or "").rstrip("/")
    host_root = str(eval_params.get("host_results_root") or "").rstrip("/")
    if text and container_root and host_root and text.startswith(container_root + "/"):
        return str(Path(host_root) / text[len(container_root) + 1:])
    return text


# ---------------------------------------------------------------- dispatcher


def _resolve_evaluator(case: dict[str, Any]) -> tuple[Any, str]:
    """Return (reward-calculator entry, origin label) for the benchmark."""
    eval_params = case["eval_params"]
    root = eval_params.get("root")

    # Scoring policy never travels through env_params: judge endpoints and
    # reward modes belong to the benchmark-side reward calculator; official
    # scoring is not SAfactory's business at all.  Stale keys are ignored
    # loudly, not silently.
    for stale_key in ("purpose", "judge", "fallback_judge"):
        if stale_key in eval_params:
            log.warning(
                "env_params.dataelf_eval.%s is ignored; the rule evaluator only "
                "dispatches benchmark-side reward calculators (judge endpoints "
                "and official scoring belong to the dataelf_eval side)",
                stale_key,
            )

    registry = _load_registry(root)
    if case["benchmark"] in registry:
        return registry[case["benchmark"]], f"dataelf_eval_reward:{case['benchmark']}"

    # RL training reward only: no generic fallback — a heuristic-free
    # number would silently poison training, and an unregistered benchmark
    # must fail loudly instead of being silently mis-scored.
    known = f"; known reward calculators: {sorted(registry)}" if registry else ""
    hint = " (register benchmarks/<name>/reward_calculator.py exporting REWARD_CALCULATOR"
    if not root:
        hint += "; set dataelf_eval.root to the host-side dataelf_eval checkout"
    hint += ")"
    raise ValueError(
        f"benchmark {case['benchmark']!r} has no registered reward calculator{known}{hint}"
    )


def _load_registry(root: Any) -> dict[str, Any]:
    """Load ``<root>/benchmarks/*/reward_calculator.py`` exporting
    ``REWARD_CALCULATOR`` — the reward registry only; official case
    evaluators are never loaded on the SAfactory side."""
    root_text = str(root or "").strip()
    if not root_text:
        return {}
    root_path = Path(root_text).expanduser().resolve()
    cache_key = str(root_path)
    if cache_key in _REGISTRY_CACHE:
        return _REGISTRY_CACHE[cache_key]
    registry: dict[str, Any] = {}
    benchmarks_dir = root_path / "benchmarks"
    if benchmarks_dir.is_dir():
        for mod_path in sorted(benchmarks_dir.glob("*/reward_calculator.py")):
            try:
                module = _load_module(f"dataelf_eval_reward_{mod_path.parent.name}", mod_path)
                entry = getattr(module, "REWARD_CALCULATOR", None)
            except Exception as exc:
                log.warning("skipping reward calculator %s: %s", mod_path, exc)
                continue
            if entry is None:
                log.warning("reward calculator %s exports no REWARD_CALCULATOR; skipped", mod_path)
                continue
            registry[mod_path.parent.name] = entry
    _REGISTRY_CACHE[cache_key] = registry
    return registry


def _load_module(name: str, path: Path) -> Any:
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import reward calculator file: {path}")
    import sys

    existing = sys.modules.pop(name, None)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        if existing is not None:
            sys.modules[name] = existing
        raise
    return module


def _invoke_case_evaluator(entry: Any, case: dict[str, Any]) -> dict[str, Any]:
    evaluator = entry() if inspect.isclass(entry) else entry
    fn = getattr(evaluator, "evaluate_case", None)
    if callable(fn):
        value = fn(case)
    elif callable(evaluator):
        value = evaluator(case)
    else:
        raise TypeError(f"reward calculator is not callable: {type(evaluator).__name__}")
    if not isinstance(value, dict):
        raise TypeError(f"reward calculator must return a dict, got {type(value).__name__}")
    return value


# ---------------------------------------------------------------- helpers


def _case_audit(case: dict[str, Any], origin: str) -> dict[str, Any]:
    return {
        "bench": "dataelf_finance",
        "benchmark": case["benchmark"],
        "evaluator": origin,
        "question_id": case.get("question_id"),
        "artifacts": dict(case.get("artifacts") or {}),
    }


def _failed(request: Any, spec: Any, reason: str, *, error_text: Any = None, artifacts: dict[str, Any] | None = None) -> Any:
    from evaluator.eval_types import EvalResult

    return EvalResult.failed(
        session_id=request.session_id,
        eval_id=spec.eval_id,
        method=spec.method.value,
        reason=reason,
        error_text=None if error_text is None else str(error_text),
        artifacts=artifacts or {},
    )


def _float_or_none(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _json_safe(value: Any) -> Any:
    """Coerce case-score artifacts to JSON-safe values (RewardCommitter
    ``json.dumps`` has no ``default=``; one numpy array would drop the
    reward of a finished episode)."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return str(value)


def _first_text(*values: Any) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""
