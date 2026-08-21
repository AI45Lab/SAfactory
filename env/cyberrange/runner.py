#!/usr/bin/env python3
"""SAfactory adapter for cyberrange's native source-bootstrap runtime-task."""
from __future__ import annotations

import json
import os
import secrets
import shutil
import stat
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit


RESULT_PREFIX = "SAFACTORY_RESULT_JSON "
CANONICAL_SOURCE_ROOT = Path("/mnt/shared-storage-user/wangyixu/cyberrange")
MILESTONES_ARTIFACT_NAME = "milestones.json"
NATIVE_RESULT_ARTIFACT_NAME = "runtime-test-result.json"
NATIVE_RESULT_FIELDS = (
    "run_id",
    "run_outcome",
    "e2e_success",
    "platform_health",
    "evidence_status",
)


def main() -> int:
    os.umask(0o077)
    started_at = time.perf_counter()
    session_id = os.environ.get("SAFACTORY_SESSION_ID", "")
    key_path: Path | None = None
    try:
        request = _read_request()
        session_id = _required_text(request.get("session_id"), "session_id")
        env_params = _mapping(request.get("env_params"))
        case = _mapping(env_params.get("dataset"))
        scenario_ref = _required_text(case.get("scenario_ref"), "dataset.scenario_ref")
        if scenario_ref not in {f"postexploitbench-range{number}-v1" for number in range(3, 7)}:
            raise RuntimeError("scenario_ref must select released Range3 through Range6")

        task_id = str(case.get("task_id") or case.get("case_id") or scenario_ref)
        source_root = Path(_required_text(env_params.get("source_root"), "source_root"))
        wheelhouse_root = Path(_required_text(env_params.get("wheelhouse_root"), "wheelhouse_root"))
        release_root = Path(_required_text(env_params.get("release_root"), "release_root"))
        bootstrap = source_root / "scripts/brainpp_source_bootstrap_acceptance.sh"
        if not bootstrap.is_file():
            raise RuntimeError(f"cyberrange runtime-task entrypoint is missing: {bootstrap}")

        output_root = Path(str(env_params.get("output_root") or (source_root / "results/brainpp/safactory")))
        deployment_report_root = source_root / "results/brainpp"
        try:
            output_root.relative_to(deployment_report_root)
        except ValueError as exc:
            raise RuntimeError(
                f"output_root must be below cyberrange deployment report root: {deployment_report_root}"
            ) from exc
        safe_task = _safe_component(task_id)
        # A DB row/session may be intentionally retried. Native deployment
        # reports are immutable, so every attempt gets a fresh unique path.
        attempt_id = f"{time.time_ns()}-{os.getpid()}"
        report_dir = output_root / f"safactory_runtime_task_{safe_task}_{session_id}_{attempt_id}"
        _validate_runtime_environment(source_root, report_dir)

        gateway_url = _gateway_session_url(request, session_id)
        route_model = _required_text(
            request.get("model") or env_params.get("route_model") or os.environ.get("SAFACTORY_ROUTE_MODEL"),
            "model",
        )
        agent_kind = str(case.get("agent_kind") or env_params.get("agent_kind") or "codex")
        model_format = "openai_responses" if agent_kind == "codex" else "openai_chat"
        wall_time = _integer(case.get("wall_time_seconds", env_params.get("wall_time_seconds")), 600, 300, 36000)
        wait_timeout = _integer(env_params.get("wait_timeout_s"), wall_time + 3000, 600, 172800)

        # cyberrange's runtime TestSpec requires a write-only key file. The
        # SAfactory Gateway session does not require a provider credential, so
        # use a random per-episode value and remove it after the native command.
        key_path = Path(f"/tmp/safactory-cyberrange-{session_id}.key")
        key_path.write_text(secrets.token_urlsafe(32) + "\n", encoding="ascii")
        key_path.chmod(0o600)

        native_env = os.environ.copy()
        native_env.update(
            {
                "AGENT_RANGE_BRAINPP_SOURCE_BOOTSTRAP_ACCEPTANCE": "1",
                "AGENT_RANGE_BRAINPP_SOURCE_ROOT": str(source_root),
                "AGENT_RANGE_BRAINPP_SOURCE_BOOTSTRAP_MODE": "runtime-task",
                "AGENT_RANGE_BRAINPP_ACCEPTANCE_WHEELHOUSE_ROOT": str(wheelhouse_root),
                "AGENT_RANGE_BRAINPP_RELEASE_ROOT": str(release_root),
                "AGENT_RANGE_BRAINPP_ACCEPTANCE_REPORT_DIR": str(report_dir),
                "AGENT_RANGE_BRAINPP_CLEANUP_TIMEOUT_SECONDS": "1800",
                "AGENT_RANGE_BRAINPP_RUNTIME_SCENARIO_ID": scenario_ref,
                "AGENT_RANGE_BRAINPP_RUNTIME_AGENT_KIND": agent_kind,
                "AGENT_RANGE_BRAINPP_RUNTIME_MODEL_NAME": route_model,
                "AGENT_RANGE_BRAINPP_RUNTIME_MODEL_BASE_URL": gateway_url,
                "AGENT_RANGE_BRAINPP_RUNTIME_MODEL_FORMAT": model_format,
                "AGENT_RANGE_BRAINPP_RUNTIME_MODEL_KEY_PATH": str(key_path),
                "AGENT_RANGE_BRAINPP_RUNTIME_WALL_TIME_SECONDS": str(wall_time),
                "AGENT_RANGE_BRAINPP_RUNTIME_WAIT_TIMEOUT_SECONDS": str(wait_timeout),
                "AGENT_RANGE_BRAINPP_RUNTIME_EGRESS_PROFILE": str(
                    case.get("egress_profile") or env_params.get("egress_profile") or "research-web-v1"
                ),
            }
        )
        prompt_path = case.get("initial_prompt_path")
        if prompt_path:
            native_env["AGENT_RANGE_BRAINPP_RUNTIME_PROMPT_FILE"] = str(prompt_path)

        print(
            "[safactory-cyberrange] deploying production stack and running "
            f"one case report={report_dir}",
            file=sys.stderr,
            flush=True,
        )
        completed = subprocess.run(
            ["/bin/bash", str(bootstrap)],
            cwd=str(source_root),
            env=native_env,
            stdin=subprocess.DEVNULL,
            stdout=sys.stderr,
            stderr=sys.stderr,
            timeout=wait_timeout + 3600,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"cyberrange runtime-task exited with status {completed.returncode}")

        native_path, native_result = _load_and_seal_native_result(report_dir)
        published_artifacts = _publish_native_artifacts(report_dir, native_path, native_result)
        e2e_success = native_result.get("e2e_success")
        _emit(
            {
                "session_id": session_id,
                "status": "succeeded",
                "total_reward": 0.0,
                "step_count": 1,
                "terminated": True,
                "truncated": native_result.get("run_outcome") == "timeout",
                "error_text": None,
                "metrics": {
                    "bench": "cyberrange",
                    "task_id": task_id,
                    "case_id": case.get("case_id") or scenario_ref,
                    "scenario_ref": scenario_ref,
                    "cyberrange_run_id": native_result.get("run_id"),
                    "run_outcome": native_result.get("run_outcome"),
                    "e2e_success": e2e_success,
                    "scorable": isinstance(e2e_success, bool),
                    "platform_health": native_result.get("platform_health"),
                    "evidence_status": native_result.get("evidence_status"),
                    "objective_state": native_result.get("objective_state"),
                    "reporting_contract": native_result.get("reporting_contract"),
                    "termination_reason": native_result.get("termination_reason"),
                    "milestone_vector": native_result.get("milestone_vector") or [],
                    "native_metrics": native_result.get("metrics") or {},
                    "native_result_path": str(native_path),
                    **published_artifacts,
                    "duration_ms": round((time.perf_counter() - started_at) * 1000, 3),
                },
            }
        )
        return 0
    except subprocess.TimeoutExpired as exc:
        _emit_failure(session_id, f"cyberrange runtime-task timed out: {exc}", started_at, truncated=True)
        return 0
    except Exception as exc:  # runtime must always return contract JSON
        _emit_failure(session_id, f"{type(exc).__name__}: {exc}", started_at)
        return 0
    finally:
        if key_path is not None:
            try:
                key_path.unlink(missing_ok=True)
            except OSError:
                pass


def _gateway_session_url(request: dict[str, Any], session_id: str) -> str:
    value = os.environ.get("SAFACTORY_GATEWAY_SESSION_URL_CONTAINER", "").strip()
    if not value:
        value = _required_text(request.get("gateway_base_url"), "gateway_base_url").rstrip("/") + "/" + session_id
    parts = urlsplit(value)
    if parts.hostname in {"127.0.0.1", "localhost", "::1"}:
        netloc = "host.docker.internal" + (f":{parts.port}" if parts.port else "")
        value = urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))
    return value.rstrip("/")


def _read_request() -> dict[str, Any]:
    raw = sys.stdin.read().strip() or os.environ.get("SAFACTORY_START_REQUEST_JSON", "").strip()
    if not raw:
        raise RuntimeError("SimulationStartRequest JSON was not provided")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise RuntimeError("SimulationStartRequest must be a JSON object")
    return value


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _required_text(value: Any, name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise RuntimeError(f"missing {name}")
    return text


def _integer(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = default
    if not minimum <= number <= maximum:
        raise RuntimeError(f"integer value must be in {minimum}-{maximum}, got {number}")
    return number


def _validate_runtime_environment(source_root: Path, report_dir: Path) -> None:
    """Enforce cyberrange's privileged source-bootstrap deployment contract."""
    if os.geteuid() != 0:
        raise RuntimeError("DEPLOYMENT.md host deployment requires root")
    if source_root != CANONICAL_SOURCE_ROOT:
        raise RuntimeError("source root must use cyberrange's canonical GPFS path")
    if not source_root.is_dir() or source_root.is_symlink():
        raise RuntimeError("source root is unavailable or is a symlink")

    report_root = source_root / "results/brainpp"
    try:
        relative_report = report_dir.relative_to(report_root)
    except ValueError as exc:
        raise RuntimeError(f"report directory must be a unique child below {report_root}") from exc
    if not relative_report.parts:
        raise RuntimeError(f"report directory must be a unique child below {report_root}")
    if report_dir.exists() or report_dir.is_symlink():
        raise RuntimeError(f"cyberrange report directory already exists: {report_dir}")

    for device in (Path("/dev/kvm"), Path("/dev/net/tun")):
        try:
            mode = device.stat().st_mode
        except OSError as exc:
            raise RuntimeError(f"{device} is unavailable") from exc
        if not stat.S_ISCHR(mode) or not os.access(device, os.R_OK | os.W_OK):
            raise RuntimeError(f"{device} is unavailable")

    for command in ("bash", "python3", "apt-get"):
        if shutil.which(command) is None:
            raise RuntimeError(f"required deployment command is missing: {command}")


def _load_and_seal_native_result(report_dir: Path) -> tuple[Path, dict[str, Any]]:
    native_path = report_dir / NATIVE_RESULT_ARTIFACT_NAME
    if not native_path.is_file() or native_path.is_symlink():
        raise RuntimeError("deployment/evaluation did not produce runtime-test-result.json")
    native_result = json.loads(native_path.read_text(encoding="utf-8"))
    if not isinstance(native_result, dict):
        raise RuntimeError("cyberrange runtime-test-result.json must contain an object")
    missing = [field for field in NATIVE_RESULT_FIELDS if field not in native_result]
    if missing:
        raise RuntimeError(f"cyberrange runtime-test-result.json is missing fields: {missing}")
    native_path.chmod(0o444)
    print(
        f"[safactory-cyberrange] sealed native result={native_path}",
        file=sys.stderr,
        flush=True,
    )
    return native_path, native_result


def _publish_native_artifacts(
    report_dir: Path,
    native_path: Path,
    native_result: dict[str, Any],
) -> dict[str, str]:
    """Copy final cyberrange evidence next to SAfactory's result artifact."""
    result_artifact_path = _configured_result_artifact_path()
    if result_artifact_path is None:
        return {}

    run_id = _required_text(native_result.get("run_id"), "runtime-test-result.run_id")
    if run_id != _safe_component(run_id):
        raise RuntimeError("cyberrange runtime-test-result.json contains an invalid run_id")
    milestones_path = report_dir / "runs" / run_id / "current" / MILESTONES_ARTIFACT_NAME
    milestones = _read_json_object(milestones_path, "cyberrange milestones.json")
    if not isinstance(milestones.get("milestones"), list):
        raise RuntimeError("cyberrange milestones.json must contain a milestones list")

    result_dir = result_artifact_path.parent
    result_dir.mkdir(parents=True, exist_ok=True)
    _atomic_copy_readonly(native_path, result_dir / NATIVE_RESULT_ARTIFACT_NAME)
    _atomic_copy_readonly(milestones_path, result_dir / MILESTONES_ARTIFACT_NAME)
    print(
        f"[safactory-cyberrange] published native artifacts={result_dir}",
        file=sys.stderr,
        flush=True,
    )
    return {
        "runtime_test_result_artifact": NATIVE_RESULT_ARTIFACT_NAME,
        "milestones_artifact": MILESTONES_ARTIFACT_NAME,
    }


def _read_json_object(path: Path, description: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"{description} is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{description} must contain an object")
    return value


def _atomic_copy_readonly(source: Path, target: Path) -> None:
    if not source.is_file() or source.is_symlink():
        raise RuntimeError(f"native artifact is unavailable: {source}")
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        temporary.write_bytes(source.read_bytes())
        temporary.replace(target)
        target.chmod(0o444)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def _safe_component(value: str) -> str:
    return "".join(character if character.isalnum() or character in "-_." else "_" for character in value)[:160] or "case"


def _emit_failure(session_id: str, error: str, started_at: float, *, truncated: bool = False) -> None:
    _emit(
        {
            "session_id": session_id,
            "status": "failed",
            "total_reward": 0.0,
            "step_count": 0,
            "terminated": True,
            "truncated": truncated,
            "error_text": error,
            "metrics": {
                "bench": "cyberrange",
                "duration_ms": round((time.perf_counter() - started_at) * 1000, 3),
            },
        }
    )


def _emit(result: dict[str, Any]) -> None:
    _write_result_artifact(result)
    print(RESULT_PREFIX + json.dumps(result, ensure_ascii=False), flush=True)


def _write_result_artifact(result: dict[str, Any]) -> None:
    """Atomically persist the same SimulationStartResult used on stdout."""
    path = _configured_result_artifact_path()
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(result, ensure_ascii=False) + "\n", encoding="utf-8")
        temporary.replace(path)
        path.chmod(0o444)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def _configured_result_artifact_path() -> Path | None:
    raw_path = os.environ.get("SAFACTORY_RESULT_PATH", "").strip()
    return Path(raw_path) if raw_path else None


if __name__ == "__main__":
    raise SystemExit(main())
