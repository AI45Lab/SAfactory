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
DEFAULT_MAX_OUTPUT_TOKENS = 8192
MAX_LOG_CHARS = 32_000

# Live log file on shared storage so the training node can `tail -f` and see
# exactly where OpenHands is stuck (subprocess.run(capture_output=True) buffers
# everything in memory until exit, so a hang shows nothing). Opened in main().
_LOG_FILE = None


def _log(msg: str) -> None:
    if _LOG_FILE is None:
        return
    try:
        _LOG_FILE.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")
        _LOG_FILE.flush()
    except Exception:
        pass


def _open_log(session_id: str):
    global _LOG_FILE
    candidates = []
    result_path = os.environ.get("SAFACTORY_RESULT_PATH", "")
    if result_path:
        candidates.append(os.path.join(os.path.dirname(result_path), "openhands.log"))
    subdir = os.environ.get("SAFACTORY_OUTPUT_SUBDIR", "")
    if subdir:
        candidates.append(os.path.join(subdir, "openhands.log"))
    candidates.append(f"/tmp/openhands-{session_id}.log")
    for path in candidates:
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            f = open(path, "a", buffering=1)
            _LOG_FILE = f
            _log(f"===== openhands_runner start session={session_id} pid={os.getpid()} =====")
            _log(f"log file: {path}")
            return
        except Exception:
            continue


def _block_github_cdn() -> None:
    """Point github.com + raw.githubusercontent.com at 127.0.0.1 in /etc/hosts
    so the openhands binary's startup calls (version check via
    raw.githubusercontent.com, public-skills git clone via github.com) fail fast
    (ECONNREFUSED) instead of hanging on the slow/flaky GitHub CDN from RJob
    pods. The binary catches these errors and proceeds to the LLM calls. Only
    done when using the pre-installed local binary (PATCHEVAL_OPENHANDS_BIN),
    since the curl installer path needs real github.com access.
    """
    if not os.environ.get("PATCHEVAL_OPENHANDS_BIN", "").strip():
        return
    hosts = ["raw.githubusercontent.com", "github.com", "objects.githubusercontent.com"]
    try:
        with open("/etc/hosts", "r") as f:
            current = f.read()
        additions = [f"127.0.0.1 {h}" for h in hosts if h not in current]
        if not additions:
            _log("block_github: already blocked in /etc/hosts")
            return
        with open("/etc/hosts", "a") as f:
            f.write("\n# safactory: fast-fail openhands startup github calls\n")
            f.write("\n".join(additions) + "\n")
        _log(f"block_github: added to /etc/hosts: {additions}")
    except Exception as exc:
        _log(f"block_github: failed to edit /etc/hosts: {exc!r}")


def main() -> int:
    started_at = time.perf_counter()
    request = _read_request()
    session_id = _required_text(request.get("session_id"), "session_id")
    _open_log(session_id)
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
    # 1) Explicit override: a pre-installed openhands binary on shared storage.
    _log("ensure_openhands: step 1) PATCHEVAL_OPENHANDS_BIN override")
    override = os.environ.get("PATCHEVAL_OPENHANDS_BIN", "").strip()
    if override and Path(override).is_file() and os.access(override, os.X_OK):
        _log(f"ensure_openhands: using override {override}")
        return override
    # 2) Already on PATH (e.g. baked into the env image).
    _log("ensure_openhands: step 2) which openhands")
    existing = shutil.which("openhands")
    if existing:
        _log(f"ensure_openhands: using PATH binary {existing}")
        return existing
    # 3) pip install from the cluster's internal PyPI mirror (no external egress
    #    needed). RJob env pods can reach mirrors.h.pjlab.org.cn (same domain as
    #    the image registry they already pull from). PIP_INDEX_URL is set in the
    #    start yaml env so pip uses the internal mirror instead of pypi.org.
    import sys
    _log("ensure_openhands: step 3) pip install openhands-ai")
    try:
        _run_streamed([sys.executable, "-m", "pip", "install", "openhands-ai"], timeout_s)
    except Exception as exc:
        _log(f"ensure_openhands: pip install raised {exc!r}")
    found = shutil.which("openhands")
    if found:
        _log(f"ensure_openhands: using pip-installed binary {found}")
        return found
    # 4) Last resort: download + run the official installer (needs external egress).
    _log("ensure_openhands: step 4) curl install.openhands.dev installer")
    install_dir = Path("/opt/openhands")
    install_dir.mkdir(parents=True, exist_ok=True)
    script = Path("/tmp/install-openhands.sh")
    _run_streamed(["curl", "-fsSL", "https://install.openhands.dev/install.sh", "-o", str(script)], timeout_s)
    env = os.environ.copy()
    env["OPENHANDS_INSTALL_DIR"] = str(install_dir)
    install_rc, install_out = _run_streamed(["bash", str(script)], timeout_s, env=env)
    _log(f"ensure_openhands: installer exit={install_rc}")
    candidates = (
        install_dir / "openhands",
        Path("/usr/local/bin/openhands"),
        Path("/root/.local/bin/openhands"),
        Path.home() / ".local" / "bin" / "openhands",
        Path("/root/.openhands/bin/openhands"),
    )
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            _log(f"ensure_openhands: using installer binary {candidate}")
            return str(candidate)
    _log("ensure_openhands: NO binary found after all steps")
    raise RuntimeError(
        "OpenHands installation completed but no executable was found. "
        f"installer exit={install_rc}; output={_trim_log(install_out)}"
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
    # Prefer the dynamic gateway URL injected by the launcher
    # (SAFACTORY_GATEWAY_BASE_URL, derived from AIEVOBOX_GATEWAY_HOST) over the
    # static PATCHEVAL_OPENHANDS_GATEWAY_BASE_URL baked into the start yaml,
    # which can go stale when the training pod IP changes between runs.
    gateway_base = _required_text(
        os.environ.get("SAFACTORY_GATEWAY_BASE_URL")
        or os.environ.get("PATCHEVAL_OPENHANDS_GATEWAY_BASE_URL"),
        "SAFACTORY_GATEWAY_BASE_URL",
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
    # OpenHands defaults max_output_tokens to 0 ("auto-detect from model"), but
    # auto-detection fails for the custom gateway-routed model, so no max_tokens
    # is sent in the LLM request and sglang falls back to a tiny default (~128),
    # truncating generation mid-tool-call (finish_reason=length). Set an
    # explicit cap so tool calls + reasoning render fully.
    max_output_tokens = _positive_int(
        os.environ.get("PATCHEVAL_OPENHANDS_MAX_OUTPUT_TOKENS"),
        DEFAULT_MAX_OUTPUT_TOKENS,
    )
    env = os.environ.copy()
    env.update(
        {
            "HOME": "/tmp/openhands-home",
            "LLM_MODEL": f"openai/{route_model}",
            "LLM_API_KEY": "safactory",
            "LLM_BASE_URL": f"{gateway_base}/{session_id}",
            "LLM_MAX_OUTPUT_TOKENS": str(max_output_tokens),
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
    _log(f"run_openhands: cwd={work_dir} gateway={gateway_base} model={route_model}")
    _log(f"run_openhands: cmd={' '.join(command)}")
    _log(f"run_openhands: LLM_BASE_URL={env['LLM_BASE_URL']}")
    _block_github_cdn()
    try:
        proc = subprocess.Popen(
            command,
            cwd=str(work_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        buffer_parts: list[str] = []
        deadline = time.perf_counter() + timeout_s
        timed_out = False
        assert proc.stdout is not None
        while True:
            line = proc.stdout.readline()
            if line == "":
                if proc.poll() is not None:
                    break
                if time.perf_counter() > deadline:
                    timed_out = True
                    break
                continue
            buffer_parts.append(line)
            _log(f"OH| {line.rstrip()}")
        if timed_out and proc.poll() is None:
            _log(f"run_openhands: TIMEOUT after {timeout_s}s, killing pid={proc.pid}")
            proc.kill()
        try:
            remaining = proc.communicate(timeout=10)[0] or ""
        except Exception:
            remaining = ""
        if remaining:
            buffer_parts.append(remaining)
            _log(f"OH| (tail) {remaining.rstrip()}")
        exit_code = proc.returncode
        _log(f"run_openhands: exit_code={exit_code} timed_out={timed_out}")
        return {
            "exit_code": exit_code,
            "output": "".join(buffer_parts),
            "timed_out": timed_out,
        }
    except Exception as exc:
        _log(f"run_openhands: exception {exc!r}")
        return {"exit_code": None, "output": str(exc), "timed_out": False}


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
    text = json.dumps(value, ensure_ascii=False)
    print(text, flush=True)
    # RJob mode: logs_rjob sometimes returns empty stdout, so also persist the
    # result to the shared artifact path (SAFACTORY_RESULT_PATH) on gpfs. The
    # launcher's parse_result_artifact fallback reads this same file.
    result_path = os.environ.get("SAFACTORY_RESULT_PATH")
    if result_path:
        try:
            Path(result_path).parent.mkdir(parents=True, exist_ok=True)
            Path(result_path).write_text(text)
        except Exception:
            # stdout print above is the primary path; never let a file-write
            # failure change the process exit status.
            pass


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


def _run_streamed(
    args: list[str],
    timeout_s: float,
    *,
    cwd: Path = Path("/workspace"),
    env: dict[str, str] | None = None,
) -> tuple[int, str]:
    """Run a command, streaming merged stdout/stderr to the live log file.

    Used for the OpenHands install steps (pip / curl / bash installer) so a
    hang during download is visible in the shared log instead of silent.
    Returns (returncode, combined_output). Never raises on timeout/non-zero.
    """
    try:
        proc = subprocess.Popen(
            args,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
    except Exception as exc:
        _log(f"run_streamed: spawn failed {args[0]}: {exc!r}")
        return -1, str(exc)
    parts: list[str] = []
    deadline = time.perf_counter() + timeout_s
    timed_out = False
    assert proc.stdout is not None
    while True:
        line = proc.stdout.readline()
        if line == "":
            if proc.poll() is not None:
                break
            if time.perf_counter() > deadline:
                timed_out = True
                break
            continue
        parts.append(line)
        _log(f"INSTALL| {line.rstrip()}")
    if timed_out and proc.poll() is None:
        _log(f"run_streamed: TIMEOUT after {timeout_s}s, killing {args[0]}")
        proc.kill()
    try:
        rem = proc.communicate(timeout=10)[0] or ""
    except Exception:
        rem = ""
    if rem:
        parts.append(rem)
        _log(f"INSTALL| (tail) {rem.rstrip()}")
    _log(f"run_streamed: {args[0]} exit={proc.returncode} timed_out={timed_out}")
    return proc.returncode, "".join(parts)


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


def _positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _trim_log(value: str) -> str:
    return str(value or "")[-MAX_LOG_CHARS:]


if __name__ == "__main__":
    raise SystemExit(main())
