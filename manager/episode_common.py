from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict
from urllib.parse import urlsplit, urlunsplit

from .types import SimulationStartRequest, SimulationStartResult

RUNNER_DIAGNOSTIC_PREFIX = "SAFACTORY_OPENCLAW_DIAGNOSTIC "
RESULT_JSON_PREFIX = "SAFACTORY_RESULT_JSON "
RESULT_PATH_ENV = "SAFACTORY_RESULT_PATH"
DEFAULT_RESULT_ROOT = "/app/results"
RESULT_FILENAME = "safactory_result.json"


def request_payload(request: SimulationStartRequest) -> tuple[Dict[str, Any], str]:
    data = asdict(request)
    return data, json.dumps(data, ensure_ascii=False)


def request_env(
    request: SimulationStartRequest,
    payload: str,
    *,
    gateway_base_url: str = "",
    containerize_local_gateway: bool,
) -> Dict[str, str]:
    env_params = request.env_params if isinstance(request.env_params, dict) else {}
    dataset = env_params.get("dataset") if isinstance(env_params.get("dataset"), dict) else {}

    route_model = first_text(dataset.get("route_model"), env_params.get("route_model"), request.model)
    model_ref = first_text(dataset.get("model_ref"), env_params.get("model_ref"))
    if not model_ref and route_model:
        model_ref = route_model if "/" in route_model else f"safactory/{route_model}"

    base_url = str(gateway_base_url or request.gateway_base_url).rstrip("/")
    gateway_session_url = f"{base_url}/{request.session_id}"
    runtime_session_url = (
        containerize_local_gateway_url(gateway_session_url)
        if containerize_local_gateway
        else gateway_session_url
    )

    values: Dict[str, Any] = {
        "SAFACTORY_START_REQUEST_JSON": payload,
        "SAFACTORY_JOB_ID": request.job_id,
        "SAFACTORY_SESSION_ID": request.session_id,
        "SAFACTORY_AGENT_NAME": request.agent_name,
        "SAFACTORY_AGENT_ID": request.agent_id,
        "SAFACTORY_TASK_ID": dataset.get("task_id"),
        "SAFACTORY_TASK_PATH": dataset.get("task_path"),
        "SAFACTORY_CATEGORY": dataset.get("category"),
        "SAFACTORY_MODEL_REF": model_ref,
        "SAFACTORY_ROUTE_MODEL": route_model,
        "SAFACTORY_NATIVE_PARALLEL": first_text(
            dataset.get("native_parallel"),
            env_params.get("native_parallel"),
            "1",
        ),
        "SAFACTORY_OUTPUT_SUBDIR": env_params.get("output_subdir"),
        RESULT_PATH_ENV: result_artifact_path(request),
        "SAFACTORY_GATEWAY_BASE_URL": base_url,
        "SAFACTORY_GATEWAY_SESSION_URL": gateway_session_url,
        "SAFACTORY_GATEWAY_SESSION_URL_CONTAINER": runtime_session_url,
        "OPENROUTER_BASE_URL": runtime_session_url,
    }
    return {key: env_text(value) for key, value in values.items() if env_text(value) != ""}


def first_text(*values: Any) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def env_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def containerize_local_gateway_url(url: str) -> str:
    try:
        parts = urlsplit(str(url))
    except Exception:
        return str(url)
    if parts.hostname not in {"127.0.0.1", "localhost", "::1"}:
        return str(url)
    netloc = "host.docker.internal"
    if parts.port is not None:
        netloc = f"{netloc}:{parts.port}"
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))


def result_artifact_path(request: SimulationStartRequest) -> str:
    env_params = request.env_params if isinstance(request.env_params, dict) else {}
    dataset = env_params.get("dataset") if isinstance(env_params.get("dataset"), dict) else {}

    explicit = first_text(
        dataset.get("safactory_result_path"),
        env_params.get("safactory_result_path"),
    )
    if explicit:
        return explicit

    root = first_text(
        dataset.get("safactory_results_root"),
        env_params.get("safactory_results_root"),
        dataset.get("results_root"),
        env_params.get("results_root"),
        DEFAULT_RESULT_ROOT,
    ).rstrip("/")
    return "/".join(
        [
            root or DEFAULT_RESULT_ROOT,
            safe_path_part(request.job_id),
            safe_path_part(request.session_id),
            RESULT_FILENAME,
        ]
    )


def result_artifact_candidates(request: SimulationStartRequest, artifact_path: str | None = None) -> list[Path]:
    raw = str(artifact_path or result_artifact_path(request) or "").strip()
    if not raw:
        return []

    candidates: list[Path] = []

    def add(path: Path) -> None:
        if path not in candidates:
            candidates.append(path)

    path = Path(raw)
    add(path)

    marker_mappings = (
        ("/app/results/", Path.cwd() / "results"),
        ("/workspace/Safactory/results/", Path.cwd() / "results"),
    )
    for marker, local_root in marker_mappings:
        if raw.startswith(marker):
            add(local_root / raw[len(marker) :])

    marker = "/results/"
    if marker in raw:
        add(Path.cwd() / "results" / raw.split(marker, 1)[1])

    return candidates


def parse_result_artifact(
    request: SimulationStartRequest,
    artifact_path: str | None = None,
) -> tuple[Dict[str, Any], Path]:
    candidates = result_artifact_candidates(request, artifact_path=artifact_path)
    errors: list[str] = []
    for path in candidates:
        try:
            if not path.is_file():
                errors.append(f"{path}: not found")
                continue
            return json.loads(path.read_text(encoding="utf-8")), path
        except json.JSONDecodeError as exc:
            errors.append(f"{path}: invalid JSON: {exc.msg}")
        except OSError as exc:
            errors.append(f"{path}: {exc}")

    detail = "; ".join(errors) if errors else "no candidate paths"
    raise RuntimeError(f"SimulationStartResult artifact could not be read: {detail}")


def parse_result_output(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    if not text:
        raise RuntimeError("runner entrypoint returned empty output; expected SimulationStartResult JSON")

    for line in reversed(text.splitlines()):
        line = line.strip()
        if not line:
            continue
        if line.startswith(RESULT_JSON_PREFIX):
            line = line[len(RESULT_JSON_PREFIX) :].strip()
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue

    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"runner entrypoint returned non-JSON output: {tail(text)}") from exc


def normalize_result(result: Any, *, session_id: str) -> SimulationStartResult:
    body = to_dict(result)
    metrics = body.get("metrics")
    if not isinstance(metrics, dict):
        metrics = {}
    return SimulationStartResult(
        session_id=str(session_id),
        status=str(body.get("status") or "succeeded"),
        total_reward=float(body.get("total_reward", 0.0) or 0.0),
        step_count=int(body.get("step_count", 0) or 0),
        terminated=bool(body.get("terminated", False)),
        truncated=bool(body.get("truncated", False)),
        error_text=None if body.get("error_text") is None else str(body.get("error_text")),
        metrics=metrics,
    )


def to_dict(result: Any) -> Dict[str, Any]:
    if isinstance(result, dict):
        return result
    if is_dataclass(result):
        return asdict(result)
    if hasattr(result, "model_dump"):
        return result.model_dump(mode="json")
    raise TypeError(f"Unsupported runner entrypoint result type: {type(result).__name__}")


def json_for_log(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def safe_path_part(value: str) -> str:
    text = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in str(value or "").strip())
    return text.strip("._") or "item"


def tail(value: str, limit: int = 1000) -> str:
    return (value or "").strip()[-int(limit):]
