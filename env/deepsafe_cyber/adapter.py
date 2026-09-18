"""SAfactory adapter for the deepsafe-cyber PatchEval workspace image.

One SAfactory episode is mapped to one native ``deepsafe-cyber run --task``
invocation. Native execution and scoring remain in the supplied benchmark
image; this module only prepares a session-scoped provider file, waits for the
nested Docker daemon, invokes the CLI, and translates its published envelope
into SAfactory metrics.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
from typing import Any
from urllib.parse import urlsplit


CLI_RESULT_SCHEMA = "deepsafe-cyber.cli-result.v2"
PUBLISHED_RESULT_SCHEMA = "deepsafe-cyber.published-result.v2"
PUBLISHED_RESULT_KEYS = {"schema", "run_id", "terminal", "evaluation", "cleanup", "artifacts"}
_CVE_RE = re.compile(r"^CVE-\d{4}-\d{4,}$")


def run_case(
    request: dict[str, Any],
    task: dict[str, Any],
    session_url: str,
) -> tuple[dict[str, Any], int]:
    """Run exactly one PatchEval row and return ``(metrics, step_count)``."""
    if not isinstance(request, dict):
        raise TypeError("request must be a JSON object")
    if not isinstance(task, dict):
        raise TypeError("env_params.dataset must be a JSON object")
    env_params = request.get("env_params")
    if not isinstance(env_params, dict):
        raise TypeError("env_params must be a JSON object")

    native = env_params.get("deepsafe_cyber")
    native = copy.deepcopy(native) if isinstance(native, dict) else {}
    cve_id = _cve_id(task)
    model = _required_text(request.get("model"), "request.model")
    provider_type = _text(native.get("provider_type"), "openai-responses")
    provider_name = _text(native.get("provider_credential_profile"), "safactory-gateway")
    model_profile = _text(native.get("model_profile"), "safactory-gateway-responses")
    session_url = session_url.rstrip("/")
    if not session_url:
        raise ValueError("Gateway session URL is empty")

    workspace = Path(_text(native.get("workspace"), "/opt/deepsafe-cyber"))
    results_root = Path(
        _text(native.get("results_root"), env_params.get("results_root"), "/opt/deepsafe-cyber/runs")
    )
    if not results_root.is_absolute():
        raise ValueError("deepsafe_cyber.results_root must be absolute")
    native_runs_root = results_root / _text(native.get("runs_subdir"), "safactory")
    native_runs_root.mkdir(parents=True, exist_ok=True)

    run_yaml_template = Path(_text(native.get("run_yaml_template"), "/opt/operator/run.yaml"))
    if not run_yaml_template.is_file():
        raise FileNotFoundError(f"deepsafe-cyber run YAML template not found: {run_yaml_template}")

    health_timeout_s = _positive(native.get("docker_health_timeout_s"), 600.0)
    task_time_limit_s = _positive(native.get("task_time_limit_s"), 18000.0)
    preflight_timeout_s = _positive(native.get("environment_preflight_timeout_s"), 3600.0)
    command_timeout_s = task_time_limit_s + preflight_timeout_s + 600.0

    _wait_for_nested_docker(timeout_s=health_timeout_s)

    # The provider file is deliberately private to this episode. It never
    # contains an operator secret: Gateway authentication is session-scoped and
    # the placeholder below only satisfies deepsafe's 32-byte key validation.
    with _temporary_episode_dir() as episode_dir:
        provider_path = episode_dir / "providers.yaml"
        provider_document = _provider_document(
            session_url=session_url,
            model=model,
            provider_type=provider_type,
            provider_name=provider_name,
            model_profile=model_profile,
        )
        _write_root_private_yaml(provider_path, provider_document)

        run_yaml_path = episode_dir / "run.yaml"
        run_yaml_document = run_yaml_template.read_text(encoding="utf-8")
        _validate_run_yaml(
            run_yaml_document,
            provider_name=provider_name,
            model_profile=model_profile,
        )
        shutil.copyfile(run_yaml_template, run_yaml_path)

        command = [
            _text(
                native.get("binary"),
                "/opt/deepsafe-cyber/.runtime/bin/deepsafe-cyber",
            ),
            "run",
            "--config",
            str(run_yaml_path),
            "--runs-root",
            str(native_runs_root),
            "--task",
            cve_id,
        ]
        process_env = _native_environment(session_url, provider_path)
        process = _run_native(
            command,
            cwd=workspace,
            env=process_env,
            timeout_s=command_timeout_s,
        )

        cli_result = _parse_cli_result(process.stdout)
        result_directory = _resolve_result_directory(native_runs_root, cli_result.get("result_ref"))
        result_path = result_directory / "public" / "result.json"
        published = _read_published_result(result_path)
        if str(published.get("run_id") or "") != str(cli_result.get("run_id") or ""):
            raise ValueError(
                "native run_id mismatch: "
                f"cli={cli_result.get('run_id')!r} published={published.get('run_id')!r}"
            )

        terminal = _object(published.get("terminal"))
        evaluation = _object(published.get("evaluation"))
        artifacts = _object(published.get("artifacts"))
        eval_log = _object(artifacts.get("eval_log"))
        terminal_code = _text(terminal.get("code")).upper()
        if not terminal_code:
            raise ValueError("published result is missing terminal.code")

        evidence_path = (
            result_directory / "private" / "evidence" / "patcheval" / "patcheval-score-evidence.json"
        )
        metrics: dict[str, Any] = {
            "bench": "deepsafe-cyber",
            "benchmark": "patcheval",
            "variant": "verified",
            "dataset_profile": "patcheval-verified-v1",
            "cve_id": cve_id,
            "run_id": published.get("run_id"),
            "result_ref": cli_result.get("result_ref"),
            "result_path": str(result_path),
            "result_directory": str(result_directory),
            "terminal_code": terminal_code,
            "terminal_reason": _first_text(terminal.get("reason"), terminal.get("message")),
            "model_verdict": evaluation.get("model_verdict"),
            "eval_log_ref": eval_log.get("ref"),
            "score_evidence_path": str(evidence_path) if evidence_path.is_file() else None,
            "cleanup_status": _object(published.get("cleanup")).get("status"),
            "native_exit_code": process.returncode,
            "native_stdout_schema": cli_result.get("schema"),
            "published_schema": published.get("schema"),
            "execution_failed": terminal_code not in {"PASS", "MODEL_INCORRECT"},
        }
        return {key: value for key, value in metrics.items() if value is not None}, 1


def _run_native(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    timeout_s: float,
) -> subprocess.CompletedProcess[str]:
    """Run the native CLI with all stdout/stderr captured away from the runner."""
    return subprocess.run(
        command,
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
    )


def _wait_for_nested_docker(*, timeout_s: float) -> None:
    """Wait until the image's nested dockerd accepts ``docker info``."""
    deadline = time.monotonic() + timeout_s
    last_error = ""
    while time.monotonic() < deadline:
        try:
            process = subprocess.run(
                ["docker", "info"],
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
            if process.returncode == 0:
                return
            last_error = (process.stderr or process.stdout or "").strip()[-1000:]
        except subprocess.TimeoutExpired as exc:
            last_error = f"docker info timed out after {exc.timeout:.0f}s"
        except OSError as exc:
            last_error = str(exc)
        time.sleep(2.0)
    raise RuntimeError(f"nested Docker daemon was not ready after {timeout_s:.0f}s: {last_error}")


def _provider_document(
    *,
    session_url: str,
    model: str,
    provider_type: str,
    provider_name: str,
    model_profile: str,
) -> str:
    if provider_type not in {"openai-responses", "openai-chat", "anthropic-messages"}:
        raise ValueError(f"unsupported deepsafe provider type: {provider_type}")
    if not provider_name or not model_profile:
        raise ValueError("provider profile names must not be empty")
    # dsh-headless expects the native Responses profile while supplying its
    # Responses-to-Chat bridge; the actual upstream is always this session URL.
    return (
        "providers:\n"
        f"  {_yaml_quote(provider_name)}:\n"
        f"    type: {_yaml_quote(provider_type)}\n"
        f"    model_profile: {_yaml_quote(model_profile)}\n"
        f"    base_url: {_yaml_quote(session_url)}\n"
        "    api_key: \"safactory-gateway-session-scoped-placeholder-key\"\n"
        f"    model: {_yaml_quote(model)}\n"
    )


def _write_root_private_yaml(path: Path, document: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(document)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(path, 0o600)
    try:
        os.chown(path, 0, 0)
    except OSError:
        if path.stat().st_uid != 0:
            raise


def _validate_run_yaml(document: str, *, provider_name: str, model_profile: str) -> None:
    """Fail early if the immutable native config drifts from provider profiles."""
    expected = {
        "dataset_profile": "patcheval-verified-v1",
        "harness_profile": "dsh-headless",
        "model_profile": model_profile,
        "provider_credential_profile": provider_name,
    }
    actual = {key: _yaml_scalar(document, key) for key in expected}
    drifted = {
        key: (expected[key], actual.get(key, ""))
        for key in expected
        if actual.get(key) != expected[key]
    }
    for key, expected_value in (("benchmark.name", "patcheval"), ("benchmark.variant", "verified")):
        scalar_key = key.rsplit(".", 1)[1]
        actual_value = _yaml_scalar(document, scalar_key)
        if actual_value != expected_value:
            drifted[key] = (expected_value, actual_value)
    if drifted:
        details = ", ".join(f"{key}={expected!r}/{actual!r}" for key, (expected, actual) in drifted.items())
        raise ValueError(f"deepsafe-cyber run YAML profiles are inconsistent: {details}")


def _native_environment(session_url: str, provider_path: Path) -> dict[str, str]:
    env = {key: value for key, value in os.environ.items() if isinstance(value, str)}
    env["DEEPSAFE_CYBER_PROVIDER_CONFIG"] = str(provider_path)
    host = _session_host(session_url)
    for key in ("NO_PROXY", "no_proxy"):
        values = [item.strip() for item in env.get(key, "").split(",") if item.strip()]
        if host and host not in values:
            values.append(host)
        env[key] = ",".join(values)
    return env


def _parse_cli_result(stdout: str) -> dict[str, Any]:
    for raw in reversed((stdout or "").splitlines()):
        if not raw.strip():
            continue
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and value.get("schema") == CLI_RESULT_SCHEMA:
            if not value.get("run_id") or not value.get("result_ref"):
                raise ValueError("cli-result.v2 is missing run_id or result_ref")
            return value
    raise ValueError(f"native stdout did not contain {CLI_RESULT_SCHEMA}")


def _read_published_result(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read native published result {path}: {exc}") from exc
    if not isinstance(value, dict) or value.get("schema") != PUBLISHED_RESULT_SCHEMA:
        raise ValueError(f"published result has unexpected schema at {path}")
    missing = PUBLISHED_RESULT_KEYS.difference(value)
    if missing:
        raise ValueError(f"published result is missing keys: {sorted(missing)}")
    return value


def _resolve_result_directory(runs_root: Path, result_ref: Any) -> Path:
    text = _required_text(result_ref, "cli-result.v2 result_ref")
    candidate = (runs_root / text).resolve()
    try:
        candidate.relative_to(runs_root.resolve())
    except ValueError as exc:
        raise ValueError(f"native result_ref escapes runs root: {text!r}") from exc
    # CLI result_ref is documented relative to --runs-root, but versions differ:
    # some return `<run_id>`, while 0.1.0 returns `<run_id>/public/result.json`.
    # Normalize the latter to the run directory before appending the envelope
    # path; otherwise the adapter looks for the duplicated `public/public`.
    if candidate.name == "result.json" and candidate.parent.name == "public":
        return candidate.parent.parent
    return candidate


def _session_host(session_url: str) -> str:
    host = urlsplit(session_url).hostname or ""
    return host.lower().strip("[]")


def _cve_id(task: dict[str, Any]) -> str:
    value = _required_text(task.get("task"), "dataset.task").upper()
    if not _CVE_RE.fullmatch(value):
        raise ValueError(f"dataset.task is not a canonical CVE identifier: {value!r}")
    return value


def _temporary_episode_dir() -> "_TempDir":
    return _TempDir("safactory-deepsafe-cyber-")


class _TempDir:
    def __init__(self, prefix: str) -> None:
        self._prefix = prefix
        self._path: Path | None = None

    def __enter__(self) -> Path:
        base = Path(os.environ.get("SAFACTORY_DEEPSAFE_TMPDIR", "/tmp"))
        path = base / f"{self._prefix}{os.getpid()}-{time.time_ns()}"
        path.mkdir(parents=True, mode=0o700)
        self._path = path
        return path

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self._path is not None:
            shutil.rmtree(self._path, ignore_errors=True)


def _object(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _first_text(*values: Any) -> str:
    for value in values:
        text = _text(value)
        if text:
            return text
    return ""


def _required_text(value: Any, name: str) -> str:
    text = _text(value)
    if not text:
        raise ValueError(f"missing required value: {name}")
    return text


def _text(*values: Any) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _positive(value: Any, default: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        result = default
    if result <= 0:
        raise ValueError(f"timeout values must be positive, got {value!r}")
    return result


def _yaml_quote(value: Any) -> str:
    return json.dumps(str(value), ensure_ascii=False)


def _yaml_scalar(document: str, key: str) -> str:
    match = re.search(
        rf"^\s*{re.escape(key)}:\s*(\".*?\"|'.*?'|[^#\n]+)\s*$",
        document,
        re.MULTILINE,
    )
    if not match:
        return ""
    value = match.group(1).strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        value = value[1:-1]
    return value
