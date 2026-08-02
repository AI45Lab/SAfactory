#!/usr/bin/env python3
"""SAfactory adapter for PatchEval's official Claude Code Exp1 baseline."""
from __future__ import annotations

import json
import os
import pwd
import queue
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit


DEFAULT_TIMEOUT_S = 2700.0
DEFAULT_INSTALL_TIMEOUT_S = 900.0
DEFAULT_TOOL_LIMIT = 100
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
        experiment = str(dataset.get("agent_experiment") or "").strip().lower()
        if experiment != "exp1":
            raise ValueError(f"unsupported Claude Code PatchEval experiment: {experiment}")
        work_dir = Path(_required_text(dataset.get("work_dir"), "work_dir"))
        if not work_dir.is_dir():
            raise RuntimeError(f"PatchEval work directory does not exist: {work_dir}")
        problem_statement = _required_text(dataset.get("problem_statement"), "problem_statement")
        gateway_url = _gateway_session_url(session_id)
        timeout_s = _positive_float(request.get("agent_start_timeout_s"), DEFAULT_TIMEOUT_S)
        install_timeout_s = _positive_float(
            os.environ.get("PATCHEVAL_CLAUDE_INSTALL_TIMEOUT_S"),
            DEFAULT_INSTALL_TIMEOUT_S,
        )
        tool_limit = _positive_int(
            os.environ.get("PATCHEVAL_CLAUDE_TOOL_LIMIT"),
            DEFAULT_TOOL_LIMIT,
        )

        _hide_evaluation_artifacts()
        _prepare_repository(work_dir)
        claude_path = _ensure_claude_code(install_timeout_s)
        prompt = _build_official_prompt(cve_id, work_dir, problem_statement)
        command_name = _install_official_command(work_dir, prompt)

        execution = _run_claude(
            command_name=command_name,
            claude_path=claude_path,
            work_dir=work_dir,
            gateway_url=gateway_url,
            timeout_s=timeout_s,
            tool_limit=tool_limit,
        )

        patch, patch_source = _extract_patch(work_dir)
        metrics = {
            "bench": "patcheval",
            "protocol": "official_claudecode_exp1",
            "setting": "agent-exp1",
            "agent_framework": "claude-code",
            "agent_experiment": experiment,
            "cve_id": cve_id,
            "patch": patch,
            "patch_generated": bool(patch.strip()),
            "patch_source": patch_source,
            "tool_calls": execution["tool_calls"],
            "tool_limit": tool_limit,
            "tool_limit_reached": execution["tool_limit_reached"],
            "claude_exit_code": execution["exit_code"],
            "claude_timed_out": execution["timed_out"],
            "claude_log": _trim_log(execution["output"]),
            "duration_ms": round((time.perf_counter() - started_at) * 1000, 3),
        }
        _write_result(
            {
                "session_id": session_id,
                "status": "succeeded",
                "total_reward": 0.0,
                "step_count": max(1, execution["tool_calls"]),
                "terminated": True,
                "truncated": execution["timed_out"] or execution["tool_limit_reached"],
                "error_text": None if patch.strip() else "Claude Code did not generate a patch",
                "metrics": metrics,
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
                    "protocol": "official_claudecode_exp1",
                    "setting": "agent-exp1",
                    "agent_framework": "claude-code",
                    "agent_experiment": "exp1",
                    "cve_id": cve_id or None,
                    "patch": "",
                    "infrastructure_error": True,
                    "duration_ms": round((time.perf_counter() - started_at) * 1000, 3),
                },
            }
        )
    return 0


def _build_official_prompt(cve_id: str, work_dir: Path, problem_statement: str) -> str:
    official_root = Path(
        os.environ.get("PATCHEVAL_OFFICIAL_ROOT")
        or Path(__file__).resolve().parent / "PatchEval" / "patcheval"
    )
    template_path = official_root / "exp_agent" / "claudecode" / "templates" / "default.md"
    if not template_path.is_file():
        raise FileNotFoundError(f"official Claude Code template is unavailable: {template_path}")
    content = template_path.read_text(encoding="utf-8")
    replacements = {
        "{{CVE_ID}}": cve_id,
        "{{WORK_DIR}}": str(work_dir),
        "{{REPO_NAME}}": work_dir.name,
        "{{PROBLEM_STATEMENT}}": problem_statement,
    }
    for placeholder, value in replacements.items():
        content = content.replace(placeholder, value)
    return content


def _hide_evaluation_artifacts() -> None:
    secret_dir = Path("/tmp/patcheval-secret")
    secret_dir.mkdir(parents=True, exist_ok=True)
    for patch_path in Path("/workspace").glob("*.patch"):
        if patch_path.name == "test.patch":
            shutil.move(str(patch_path), str(secret_dir / patch_path.name))
        else:
            patch_path.unlink(missing_ok=True)


def _install_official_command(work_dir: Path, prompt: str) -> str:
    command_name = "patcheval-exp1"
    command_dir = work_dir / ".claude" / "commands"
    command_dir.mkdir(parents=True, exist_ok=True)
    command_file = command_dir / f"{command_name}.md"
    command_file.write_text(prompt, encoding="utf-8")
    account = pwd.getpwnam("claude_user")
    os.chown(command_dir.parent, account.pw_uid, account.pw_gid)
    os.chown(command_dir, account.pw_uid, account.pw_gid)
    os.chown(command_file, account.pw_uid, account.pw_gid)
    return command_name


def _prepare_repository(work_dir: Path) -> None:
    status = _run(["git", "status", "--porcelain"], cwd=work_dir, timeout_s=60, check=False)
    if status.returncode != 0:
        raise RuntimeError(f"PatchEval work directory is not a git repository: {work_dir}")
    if status.stdout.strip():
        _run(["git", "config", "user.email", "patcheval@example.invalid"], cwd=work_dir, timeout_s=30)
        _run(["git", "config", "user.name", "PatchEval Baseline"], cwd=work_dir, timeout_s=30)
        _run(["git", "add", "-A"], cwd=work_dir, timeout_s=60)
        _run(
            ["git", "commit", "--no-verify", "-m", "PatchEval Claude Code baseline"],
            cwd=work_dir,
            timeout_s=120,
            check=False,
        )
    exclude_file = work_dir / ".git" / "info" / "exclude"
    exclude_file.parent.mkdir(parents=True, exist_ok=True)
    existing = exclude_file.read_text(encoding="utf-8") if exclude_file.exists() else ""
    additions = [pattern for pattern in (".claude/", "final-cve-fix.patch") if pattern not in existing]
    if additions:
        with exclude_file.open("a", encoding="utf-8") as handle:
            if existing and not existing.endswith("\n"):
                handle.write("\n")
            handle.write("\n".join(additions) + "\n")


def _ensure_claude_code(timeout_s: float) -> str:
    node_major = 0
    if shutil.which("node"):
        version = _run(["node", "--version"], cwd=Path("/workspace"), timeout_s=30, check=False)
        try:
            node_major = int(version.stdout.strip().lstrip("v").split(".", 1)[0])
        except (ValueError, IndexError):
            node_major = 0
    if node_major < 18 or not shutil.which("npm"):
        if not shutil.which("apt-get"):
            raise RuntimeError("Claude Code requires node/npm, and apt-get is unavailable")
        _run(["apt-get", "update"], cwd=Path("/workspace"), timeout_s=timeout_s)
        _run(
            ["apt-get", "install", "-y", "ca-certificates", "curl"],
            cwd=Path("/workspace"),
            timeout_s=timeout_s,
        )
        _run(
            ["curl", "-fsSL", "https://deb.nodesource.com/setup_22.x", "-o", "/tmp/nodesource.sh"],
            cwd=Path("/workspace"),
            timeout_s=timeout_s,
        )
        _run(["bash", "/tmp/nodesource.sh"], cwd=Path("/workspace"), timeout_s=timeout_s)
        _run(["apt-get", "install", "-y", "nodejs"], cwd=Path("/workspace"), timeout_s=timeout_s)
    claude_path = _find_claude_code()
    if not claude_path:
        install_prefix = Path("/opt/claude-code")
        _run(
            [
                "npm",
                "install",
                "--prefix",
                str(install_prefix),
                "@anthropic-ai/claude-code",
            ],
            cwd=Path("/workspace"),
            timeout_s=timeout_s,
        )
        claude_path = _find_claude_code(install_prefix)
    if not claude_path:
        raise RuntimeError(
            "Claude Code installation completed but no executable was found "
            "under /opt/claude-code or the npm global prefix"
        )
    try:
        account = pwd.getpwnam("claude_user")
    except KeyError:
        _run(
            ["useradd", "--create-home", "--shell", "/bin/bash", "claude_user"],
            cwd=Path("/workspace"),
            timeout_s=60,
        )
        account = pwd.getpwnam("claude_user")
    _run(
        ["chown", "-R", f"{account.pw_uid}:{account.pw_gid}", "/workspace"],
        cwd=Path("/workspace"),
        timeout_s=300,
    )
    return claude_path


def _find_claude_code(install_prefix: Path | None = None) -> str:
    candidates: list[Path] = []
    if install_prefix is not None:
        candidates.append(install_prefix / "node_modules" / ".bin" / "claude")
    path_command = shutil.which("claude")
    if path_command:
        candidates.append(Path(path_command))
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate.resolve())
    if shutil.which("npm"):
        prefix = _run(
            ["npm", "prefix", "-g"],
            cwd=Path("/workspace") if Path("/workspace").is_dir() else Path.cwd(),
            timeout_s=30,
            check=False,
        )
        if prefix.returncode == 0 and prefix.stdout.strip():
            candidates.append(Path(prefix.stdout.strip()) / "bin" / "claude")
    candidates.extend(
        [
            Path("/usr/local/bin/claude"),
            Path("/usr/bin/claude"),
            Path("/root/.local/bin/claude"),
            Path("/root/.npm-global/bin/claude"),
        ]
    )
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate.resolve())
    return ""


def _run_claude(
    *,
    command_name: str,
    claude_path: str,
    work_dir: Path,
    gateway_url: str,
    timeout_s: float,
    tool_limit: int,
) -> dict[str, Any]:
    env = os.environ.copy()
    account = pwd.getpwnam("claude_user")
    gateway_host = urlsplit(gateway_url).hostname or ""
    route_model = _required_text(
        os.environ.get("PATCHEVAL_CLAUDE_MODEL"),
        "PATCHEVAL_CLAUDE_MODEL",
    )
    env.update(
        {
            "ANTHROPIC_BASE_URL": gateway_url,
            "ANTHROPIC_API_KEY": "safactory",
            "ANTHROPIC_AUTH_TOKEN": "safactory",
            "ANTHROPIC_MODEL": route_model,
            "ANTHROPIC_DEFAULT_OPUS_MODEL": route_model,
            "ANTHROPIC_DEFAULT_SONNET_MODEL": route_model,
            "ANTHROPIC_DEFAULT_HAIKU_MODEL": route_model,
            "CLAUDE_CODE_AUTO_CONNECT_IDE": "false",
            "HOME": account.pw_dir,
            "USER": account.pw_name,
            "LOGNAME": account.pw_name,
            "NO_PROXY": _append_no_proxy(env.get("NO_PROXY", ""), gateway_host),
            "no_proxy": _append_no_proxy(env.get("no_proxy", ""), gateway_host),
        }
    )
    if str(os.environ.get("PATCHEVAL_CLAUDE_DISABLE_INTERLEAVED_THINKING") or "").lower() in {
        "1",
        "true",
        "yes",
    }:
        env["DISABLE_INTERLEAVED_THINKING"] = "true"
    else:
        env.pop("DISABLE_INTERLEAVED_THINKING", None)
    max_thinking_tokens = str(
        os.environ.get("PATCHEVAL_CLAUDE_MAX_THINKING_TOKENS") or ""
    ).strip()
    if max_thinking_tokens:
        env["MAX_THINKING_TOKENS"] = max_thinking_tokens
    else:
        env.pop("MAX_THINKING_TOKENS", None)
    command = [
        claude_path,
        "--print",
        "--dangerously-skip-permissions",
        "--permission-mode",
        "bypassPermissions",
        "--output-format",
        "stream-json",
        "--verbose",
        f"/{command_name}",
    ]
    process = subprocess.Popen(
        command,
        cwd=str(work_dir),
        env=env,
        preexec_fn=lambda: (os.setgid(account.pw_gid), os.setuid(account.pw_uid)),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    output_queue: queue.Queue[str | None] = queue.Queue()

    def read_output() -> None:
        try:
            for line in process.stdout:
                output_queue.put(line)
        finally:
            output_queue.put(None)

    reader = threading.Thread(target=read_output, daemon=True)
    reader.start()
    deadline = time.monotonic() + timeout_s
    output: list[str] = []
    tool_calls = 0
    tool_limit_reached = False
    timed_out = False
    stream_closed = False

    while not stream_closed:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            timed_out = True
            process.kill()
            break
        try:
            line = output_queue.get(timeout=min(1.0, remaining))
        except queue.Empty:
            if process.poll() is not None:
                break
            continue
        if line is None:
            stream_closed = True
            continue
        output.append(line)
        tool_calls += _count_tool_uses(line)
        if tool_calls >= tool_limit:
            tool_limit_reached = True
            process.kill()
            break

    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)
    reader.join(timeout=2)
    while True:
        try:
            line = output_queue.get_nowait()
        except queue.Empty:
            break
        if line is not None:
            output.append(line)
    return {
        "exit_code": process.returncode,
        "output": "".join(output),
        "tool_calls": tool_calls,
        "tool_limit_reached": tool_limit_reached,
        "timed_out": timed_out,
    }


def _count_tool_uses(line: str) -> int:
    try:
        event = json.loads(line)
    except json.JSONDecodeError:
        return 0
    if event.get("type") != "assistant":
        return 0
    message = event.get("message")
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, list):
        return 0
    return sum(1 for block in content if isinstance(block, dict) and block.get("type") == "tool_use")


def _extract_patch(work_dir: Path) -> tuple[str, str]:
    for candidate in (
        Path("/workspace/final-cve-fix.patch"),
        work_dir / "final-cve-fix.patch",
        work_dir / ".claude" / "outputs" / "patch.diff",
    ):
        if candidate.is_file():
            patch = candidate.read_text(encoding="utf-8", errors="replace")
            if patch.strip():
                return patch, str(candidate)
    diff = _run(["git", "diff", "HEAD", "--", "."], cwd=work_dir, timeout_s=120, check=False)
    if diff.stdout.strip():
        return diff.stdout, "git_diff_fallback"
    return "", ""


def _read_request() -> dict[str, Any]:
    raw = os.environ.get("SAFACTORY_START_REQUEST_JSON")
    if raw is None:
        raw = sys.stdin.read()
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("start request must be a JSON object")
    return value


def _write_result(value: dict[str, Any]) -> None:
    print(json.dumps(value, ensure_ascii=False))


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


def _gateway_session_url(session_id: str) -> str:
    base_url = _required_text(
        os.environ.get("PATCHEVAL_CLAUDE_GATEWAY_BASE_URL"),
        "PATCHEVAL_CLAUDE_GATEWAY_BASE_URL",
    )
    return f"{base_url.rstrip('/')}/{session_id}"


def _run(
    args: list[str],
    *,
    cwd: Path,
    timeout_s: float,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=max(1.0, timeout_s),
        check=check,
    )


def _append_no_proxy(current: str, *values: str) -> str:
    parts = [part.strip() for part in current.split(",") if part.strip()]
    for value in values:
        if value not in parts:
            parts.append(value)
    return ",".join(parts)


def _trim_log(value: str) -> str:
    return str(value or "")[-MAX_LOG_CHARS:]


if __name__ == "__main__":
    raise SystemExit(main())
