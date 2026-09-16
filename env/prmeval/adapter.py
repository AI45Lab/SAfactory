"""PRMEval-specific adapter for the fixed SAfactory runner protocol.

The runner owns the SAfactory contract; this module owns only the native
PRMEval invocation.  It deliberately evaluates the one dataset row supplied
in ``request['env_params']['dataset']`` and never loops over the source file.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import tempfile
from typing import Any


def run_case(
    request: dict[str, Any],
    task: dict[str, Any],
    session_url: str,
) -> tuple[dict[str, Any], int]:
    """Run one PRMEval trajectory and return ``(metrics, step_count)``.

    ``prmeval`` is imported lazily so the protocol shell can be imported and
    contract-tested on a workstation without the benchmark wheel installed.
    """
    if not isinstance(task, dict):
        raise TypeError("env_params.dataset must be a JSON object")
    env_params = request.get("env_params")
    if not isinstance(env_params, dict):
        raise TypeError("env_params must be a JSON object")

    native_config = env_params.get("prmeval")
    if not isinstance(native_config, dict):
        raise ValueError("env_params.prmeval must be a JSON object")

    prepared_task = _prepare_trajectory(task, env_params)
    session_id = _required_text(request.get("session_id"), "session_id")
    job_id = _required_text(request.get("job_id"), "job_id")
    results_root = _first_text(
        env_params.get("results_root"),
        os.environ.get("SAFACTORY_RESULTS_ROOT"),
        "/tmp/safactory-prmeval-results",
    )
    run_name = _safe_path_part(f"{job_id}-{session_id}")

    # Keep the temporary source row private to this episode.  A fixed path
    # would race when env_num/pool-size is greater than one.
    with tempfile.TemporaryDirectory(prefix="safactory-prmeval-") as temp_dir:
        source_path = Path(temp_dir) / "trajectory.jsonl"
        source_path.write_text(
            json.dumps(prepared_task, ensure_ascii=False) + "\n", encoding="utf-8"
        )

        config = copy.deepcopy(native_config)
        sampling = config.setdefault("sampling", {})
        if not isinstance(sampling, dict):
            raise TypeError("env_params.prmeval.sampling must be a JSON object")
        sampling["paths"] = [str(source_path)]
        sampling["max_trajectories"] = 1
        sampling.setdefault("eval_types", ["progress"])

        infer = config.setdefault("infer", {})
        if not isinstance(infer, dict):
            raise TypeError("env_params.prmeval.infer must be a JSON object")
        infer["model_id"] = _first_text(
            os.environ.get("SAFACTORY_ROUTE_MODEL"), request.get("model")
        )
        infer["base_url"] = session_url.rstrip("/")
        infer.setdefault("api_key", os.environ.get("OPENAI_API_KEY", "EMPTY"))
        if request.get("temperature") is not None:
            infer["temperature"] = float(request["temperature"])
        options = infer.setdefault("options", {})
        if not isinstance(options, dict):
            raise TypeError("env_params.prmeval.infer.options must be a JSON object")
        options.setdefault("keep_base_url", True)

        output_dir = Path(results_root) / run_name
        config["output_dir"] = str(output_dir.parent)
        config["run_name"] = output_dir.name
        config.setdefault("resume", False)

        # Import only after the request/config has been validated.  This keeps
        # malformed requests deterministic and makes local protocol tests cheap.
        from prmeval.core import EvalConfig, Evaluator  # type: ignore

        summary = Evaluator(EvalConfig.model_validate(config)).run()

    metrics = _flatten_summary(summary)
    metrics.update(
        {
            "bench": "prmeval",
            "case_id": task.get("id") or task.get("sample_id") or task.get("task_id"),
            "native_output_dir": str(output_dir),
            "session_id": session_id,
        }
    )
    progress = metrics.get("mse")
    step_count = _step_count(summary, default=1)
    if progress is not None:
        metrics["native_score_available"] = True
    return metrics, step_count


def _prepare_trajectory(task: dict[str, Any], env_params: dict[str, Any]) -> dict[str, Any]:
    """Resolve relative media references without changing the source row."""
    prepared = copy.deepcopy(task)
    frames = prepared.get("frames")
    if isinstance(frames, str) and not Path(frames).is_absolute():
        media_root = _first_text(env_params.get("media_root"), env_params.get("frames_root"))
        if not media_root:
            raise ValueError("relative frames path requires env_params.media_root")
        prepared["frames"] = str((Path(media_root) / frames).resolve())
    return prepared


def _flatten_summary(summary: Any) -> dict[str, Any]:
    if not isinstance(summary, dict):
        raise TypeError("PRMEval summary must be a JSON object")
    metrics = summary.get("metrics") if isinstance(summary.get("metrics"), dict) else {}
    progress = metrics.get("progress") if isinstance(metrics.get("progress"), dict) else {}
    flattened: dict[str, Any] = {
        "native_summary": summary,
        "coverage": summary.get("coverage", {}),
        "mse": progress.get("mse"),
        "pearson": progress.get("pearson"),
        "num_samples": progress.get("num_samples"),
        "bench_output_path": summary.get("details") or summary.get("predictions"),
    }
    details = summary.get("details")
    if details:
        prediction = _first_prediction(Path(str(details)))
        if prediction is not None:
            flattened["predicted_progress"] = prediction
    return {key: value for key, value in flattened.items() if value is not None}


def _first_prediction(path: Path) -> list[float] | None:
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                values = ((row.get("prediction") or {}).get("values"))
                if isinstance(values, list):
                    return [float(value) for value in values]
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None
    return None


def _step_count(summary: Any, *, default: int) -> int:
    coverage = summary.get("coverage") if isinstance(summary, dict) else {}
    for key in ("executed", "successful"):
        try:
            value = int(coverage.get(key))
        except (AttributeError, TypeError, ValueError):
            continue
        if value >= 0:
            return value
    return default


def _required_text(value: Any, name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"SimulationStartRequest missing {name}")
    return text


def _first_text(*values: Any) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _safe_path_part(value: Any) -> str:
    text = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(value))
    return text.strip("._") or "episode"
