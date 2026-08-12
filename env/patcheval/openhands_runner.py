#!/usr/bin/env python3
"""SAfactory adapter for the OpenHands CLI PatchEval baseline."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


DEFAULT_TIMEOUT_S = 2700.0
DEFAULT_INSTALL_TIMEOUT_S = 900.0
MAX_LOG_CHARS = 32_000


def main() -> int:
    started_at = time.perf_counter()
    request = _read_request()
    session_id = _required_text(request.get("session_id"), "session_id")
    cve_id = ""

    try:
        env_params = request.get("env_params") if isinstance(request.get("env_params"), dict) else {}
        dataset = env_params.get("dataset") if isinstance(env_params.get("dataset"), dict) else {}
        cve_id = _required_text(dataset.get("cve_id"), "cve_id").upper()
        work_dir = Path(_required_text(dataset.get("work_dir"), "work_dir"))
        problem_statement = _required_text(dataset.get("problem_statement"), "problem_statement")
        if not work_dir.is_dir():
            raise RuntimeError(f"PatchEval work directory does not exist: {work_dir}")

        _hide_evaluation_artifacts()
        _prepare_repository(work_dir)
        timeout_s = _positive_float(request.get("agent_start_timeout_s"), DEFAULT_TIMEOUT_S)
        install_timeout_s = _positive_float(
            os.environ.get("PATCHEVAL_OPENHANDS_INSTALL_TIMEOUT_S"),
            DEFAULT_INSTALL_TIMEOUT_S,
        )
        executable = _ensure_openhands(install_timeout_s)
        execution = _run_openhands(
            executable=executable,
            work_dir=work_dir,
            session_id=session_id,
            cve_id=cve_id,
            problem_statement=problem_statement,
            timeout_s=timeout_s,
        )
        patch, patch_source = _extract_patch(work_dir)
        _write_result(
            {
                "session_id": session_id,
                "status": "succeeded",
                "total_reward": 0.0,
                "step_count": 1,
                "terminated": True,
                "truncated": execution["timed_out"],
                "error_text": None if patch.strip() else "OpenHands did not generate a patch",
                "metrics": {
                    "bench": "patcheval",
                    "protocol": "openhands_cli_headless",
                    "setting": "agent-exp1",
                    "agent_framework": "openhands",
                    "cve_id": cve_id,
                    "patch": patch,
                    "patch_generated": bool(patch.strip()),
                    "patch_source": patch_source,
                    "openhands_exit_code": execution["exit_code"],
                    "openhands_timed_out": execution["timed_out"],
                    "openhands_log": _trim_log(execution["output"]),
                    "duration_ms": round((time.perf_counter() - started_at) * 1000, 3),
                },
            }
        )
    except Exception as exc:
        _write_result(
            {
                "session_id": session_id,
                "status": "failed",
                "total_reward": 0.0,
                "step_count": 0,
                "terminated": True,
                "truncated": isinstance(exc, subprocess.TimeoutExpired),
                "error_text": str(exc),
                "metrics": {
                    "bench": "patcheval",
                    "protocol": "openhands_cli_headless",
                    "agent_framework": "openhands",
                    "cve_id": cve_id or None,
                    "patch": "",
                    "infrastructure_error": True,
                    "duration_ms": round((time.perf_counter() - started_at) * 1000, 3),
                },
            }
        )
    return 0


def _ensure_openhands(timeout_s: float) -> str:
    existing = shutil.which("openhands")
    if existing:
        return existing
    install_dir = Path("/opt/openhands")
    install_dir.mkdir(parents=True, exist_ok=True)
    script = Path("/tmp/install-openhands.sh")
    _run(["curl", "-fsSL", "https://install.openhands.dev/install.sh", "-o", str(script)], timeout_s)
    env = os.environ.copy()
    env["OPENHANDS_INSTALL_DIR"] = str(install_dir)
    install = _run(["bash", str(script)], timeout_s, env=env, check=False)
    candidates = (
        install_dir / "openhands",
        Path("/usr/local/bin/openhands"),
        Path("/root/.local/bin/openhands"),
        Path.home() / ".local" / "bin" / "openhands",
        Path("/root/.openhands/bin/openhands"),
    )
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    raise RuntimeError(
        "OpenHands installation completed but no executable was found. "
        f"installer exit={install.returncode}; output={_trim_log(install.stdout + install.stderr)}"
    )


def _run_openhands(
    *,
    executable: str,
    work_dir: Path,
    session_id: str,
    cve_id: str,
    problem_statement: str,
    timeout_s: float,
) -> dict[str, Any]:
    gateway_base = _required_text(
        os.environ.get("PATCHEVAL_OPENHANDS_GATEWAY_BASE_URL"),
        "PATCHEVAL_OPENHANDS_GATEWAY_BASE_URL",
    ).rstrip("/")
    route_model = _required_text(
        os.environ.get("PATCHEVAL_OPENHANDS_MODEL"),
        "PATCHEVAL_OPENHANDS_MODEL",
    )
    task = (
        f"You are fixing {cve_id} in this repository. {problem_statement}\n\n"
        "Work only in the repository. Implement and validate the fix. Do not inspect or modify "
        "PatchEval evaluator scripts, test.patch, fix.patch, or other benchmark artifacts. "
        "Leave the final code changes in the git working tree."
    )
    env = os.environ.copy()
    env.update(
        {
            "HOME": "/tmp/openhands-home",
            "LLM_MODEL": f"openai/{route_model}",
            "LLM_API_KEY": "safactory",
            "LLM_BASE_URL": f"{gateway_base}/{session_id}",
        }
    )
    command = [
        executable,
        "--headless",
        "--json",
        "--always-approve",
        "--override-with-envs",
        "--exit-without-confirmation",
        "--task",
        task,
    ]
    try:
        completed = _run(command, timeout_s, cwd=work_dir, check=False, env=env)
    except subprocess.TimeoutExpired as exc:
        output = (exc.stdout or "") + (exc.stderr or "")
        return {"exit_code": None, "output": output, "timed_out": True}
    return {
        "exit_code": completed.returncode,
        "output": completed.stdout + completed.stderr,
        "timed_out": False,
    }


def _hide_evaluation_artifacts() -> None:
    secret_dir = Path("/tmp/patcheval-secret")
    secret_dir.mkdir(parents=True, exist_ok=True)
    for patch_path in Path("/workspace").glob("*.patch"):
        if patch_path.name == "test.patch":
            shutil.move(str(patch_path), str(secret_dir / patch_path.name))
        else:
            patch_path.unlink(missing_ok=True)


def _prepare_repository(work_dir: Path) -> None:
    status = _run(["git", "status", "--porcelain"], 60, cwd=work_dir, check=False)
    if status.returncode != 0:
        raise RuntimeError(f"PatchEval work directory is not a git repository: {work_dir}")
    if status.stdout.strip():
        _run(["git", "add", "-A"], 60, cwd=work_dir)
        _run(["git", "commit", "--no-verify", "-m", "PatchEval OpenHands baseline"], 120, cwd=work_dir, check=False)


def _extract_patch(work_dir: Path) -> tuple[str, str]:
    diff = _run(["git", "diff", "HEAD", "--", "."], 120, cwd=work_dir, check=False)
    return (diff.stdout, "git_diff") if diff.stdout.strip() else ("", "")


def _read_request() -> dict[str, Any]:
    raw = os.environ.get("SAFACTORY_START_REQUEST_JSON") or sys.stdin.read()
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("start request must be a JSON object")
    return value


def _write_result(value: dict[str, Any]) -> None:
    print(json.dumps(value, ensure_ascii=False))


def _run(
    args: list[str],
    timeout_s: float,
    *,
    cwd: Path = Path("/workspace"),
    check: bool = True,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=max(1.0, timeout_s),
        check=check,
        env=env,
    )


def _required_text(value: Any, name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{name} is required")
    return text


def _positive_float(value: Any, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _trim_log(value: str) -> str:
    return str(value or "")[-MAX_LOG_CHARS:]


if __name__ == "__main__":
    raise SystemExit(main())
