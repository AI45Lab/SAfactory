#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlsplit, urlunsplit

DEFAULT_PROVIDER_ID = "safactory"
DEFAULT_MODEL_REF = "safactory/dsv4pro"
DEFAULT_ROUTE_MODEL = "dsv4pro"
DEFAULT_API_KEY = "hello"
DEFAULT_TIMEOUT_S = 1800.0
DEFAULT_REPO_ROOT = "/Users/bin-mac/CodeX/WildClawBench"
DEFAULT_OUTPUT_SUBDIR = "output/safactory"
DEFAULT_TMP_WORKSPACE = "/tmp_workspace"
TRANSCRIPT_PATH = Path("/root/.openclaw/agents/main/sessions/chat.jsonl")
RESULT_JSON_PREFIX = "SAFACTORY_RESULT_JSON "
RESULT_PATH_ENV = "SAFACTORY_RESULT_PATH"


def main() -> int:
    started_at = time.perf_counter()
    request = _read_request()
    session_id = _required_str(request.get("session_id"), "session_id")
    job_id = str(request.get("job_id") or "").strip()
    env_params = request.get("env_params") if isinstance(request.get("env_params"), dict) else {}
    dataset = env_params.get("dataset") if isinstance(env_params.get("dataset"), dict) else {}

    repo_root = Path(
        dataset.get("wildclawbench_root")
        or env_params.get("wildclawbench_root")
        or os.environ.get("WILDCLAWBENCH_ROOT")
        or DEFAULT_REPO_ROOT
    ).expanduser()
    _load_dotenv(repo_root / ".env")
    if not repo_root.is_dir():
        return _write_failure(session_id, f"WildClawBench root does not exist: {repo_root}", started_at)

    task_path = _resolve_task_path(repo_root, dataset)
    if not task_path.is_file():
        return _write_failure(session_id, f"WildClawBench task file does not exist: {task_path}", started_at)

    task = _parse_task_md(repo_root, task_path)
    route_model = str(dataset.get("route_model") or env_params.get("route_model") or request.get("model") or DEFAULT_ROUTE_MODEL).strip()
    model_ref = str(dataset.get("model_ref") or env_params.get("model_ref") or _model_ref(route_model)).strip()
    if "/" not in model_ref:
        model_ref = _model_ref(model_ref)
    provider_id = str(dataset.get("provider_id") or env_params.get("provider_id") or DEFAULT_PROVIDER_ID).strip()
    api_key = str(
        dataset.get("api_key")
        or env_params.get("api_key")
        or os.environ.get("SAFACTORY_GATEWAY_API_KEY")
        or os.environ.get("MY_PROXY_API_KEY")
        or os.environ.get("OPENROUTER_API_KEY")
        or DEFAULT_API_KEY
    )
    output_subdir = str(dataset.get("output_subdir") or env_params.get("output_subdir") or DEFAULT_OUTPUT_SUBDIR).strip().strip("/")
    if job_id:
        output_subdir = f"{output_subdir}/{_safe_name(job_id)}"
    tmp_workspace = Path(os.environ.get("TMP_WORKSPACE") or DEFAULT_TMP_WORKSPACE)
    timeout_s = max(
        1.0,
        float(dataset.get("timeout_s") or env_params.get("timeout_s") or request.get("agent_start_timeout_s") or task["timeout_seconds"] or DEFAULT_TIMEOUT_S),
    )
    gateway_session_base_url = _gateway_session_base_url(str(request.get("gateway_base_url") or ""), session_id=session_id)

    env = _build_env(gateway_session_base_url=gateway_session_base_url, route_model=route_model, api_key=api_key)
    output_dir = _build_output_dir(
        repo_root=repo_root,
        output_subdir=output_subdir,
        category=str(dataset.get("category") or task["category"]),
        task_id=str(dataset.get("task_id") or task["task_id"]),
        model_ref=model_ref,
    )

    gateway_proc: subprocess.Popen[str] | None = None
    gateway_log = None
    agent_log = None
    error_text: str | None = None
    truncated = False
    elapsed_time = timeout_s
    score: dict[str, Any] = {}
    usage: dict[str, Any] = _empty_usage()

    try:
        _write_models_config(provider_id=provider_id, route_model=route_model, base_url=gateway_session_base_url, api_key=api_key)
        _prepare_workspace(task=task, tmp_workspace=tmp_workspace)
        _setup_skills(task=task)
        _run_warmup(task.get("warmup", ""), cwd=tmp_workspace, timeout_s=min(timeout_s, 300.0))
        _set_openclaw_model(model_ref=model_ref, image_model=str(dataset.get("image_model") or env_params.get("image_model") or model_ref))

        gateway_log = (output_dir / "gateway.log").open("w", encoding="utf-8")
        gateway_proc = subprocess.Popen(
            [
                "bash",
                "-lc",
                f"cd {sh_quote(str(tmp_workspace))} && "
                f"export OPENROUTER_API_KEY={sh_quote(api_key)} && "
                f"export OPENROUTER_BASE_URL={sh_quote(gateway_session_base_url)} && "
                f"openclaw gateway --port {sh_quote(str(os.environ.get('GATEWAY_PORT') or '18789'))}",
            ],
            env=env,
            stdout=gateway_log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        time.sleep(2.0)

        prompt = _system_prompt(task["timeout_seconds"]) + str(task["prompt"])
        agent_log = (output_dir / "agent.log").open("w", encoding="utf-8")
        started_agent = time.perf_counter()
        try:
            agent_run = subprocess.run(
                [
                    "bash",
                    "-lc",
                    f"cd {sh_quote(str(tmp_workspace))} && "
                    f"openclaw agent --session-id chat --timeout {sh_quote(str(task['timeout_seconds']))} "
                    f"--message {sh_quote(prompt)}",
                ],
                env=env,
                stdout=agent_log,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout_s,
                check=False,
            )
            elapsed_time = time.perf_counter() - started_agent
            if agent_run.returncode != 0:
                error_text = f"OpenClaw agent exited with code {agent_run.returncode}"
        except subprocess.TimeoutExpired:
            elapsed_time = timeout_s
            truncated = True
            error_text = f"OpenClaw agent timed out after {timeout_s:.1f}s"

        score = _grade_task(task=task, output_dir=output_dir, tmp_workspace=tmp_workspace)
        usage = _write_usage(output_dir=output_dir, elapsed_time=elapsed_time)
        _collect_task_output(output_dir=output_dir, tmp_workspace=tmp_workspace)
    except Exception as exc:
        error_text = str(exc)
    finally:
        if gateway_proc is not None:
            try:
                gateway_proc.terminate()
                gateway_proc.wait(timeout=5)
            except Exception:
                try:
                    gateway_proc.kill()
                except Exception:
                    pass
        for handle in (gateway_log, agent_log):
            if handle is not None and not handle.closed:
                handle.close()

    raw_score_1 = _extract_raw_score(score)
    score_10 = _score_to_10(raw_score_1)
    status = "failed" if error_text else "succeeded"
    _write_result(
        {
            "session_id": session_id,
            "status": status,
            "total_reward": score_10,
            "step_count": max(1, int(usage.get("request_count") or 1)),
            "terminated": True,
            "truncated": truncated,
            "error_text": error_text,
            "metrics": {
                "bench": "wildclawbench",
                "task_id": dataset.get("task_id") or task["task_id"],
                "category": dataset.get("category") or task["category"],
                "task_path": str(task_path),
                "repo_root": str(repo_root),
                "output_dir": str(output_dir),
                "score": score,
                "raw_score_1": raw_score_1,
                "score_10": score_10,
                "usage": usage,
                "model_ref": model_ref,
                "route_model": route_model,
                "provider_id": provider_id,
                "gateway_session_base_url": gateway_session_base_url,
                "duration_ms": round((time.perf_counter() - started_at) * 1000, 3),
            },
        }
    )
    return 0


def _read_request() -> dict[str, Any]:
    raw = sys.stdin.read().strip() or os.environ.get("SAFACTORY_START_REQUEST_JSON", "").strip()
    if not raw:
        raise RuntimeError("SimulationStartRequest JSON was not provided on stdin")
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise RuntimeError("SimulationStartRequest must be a JSON object")
    return data


def _required_str(value: Any, name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise RuntimeError(f"SimulationStartRequest missing {name}")
    return text


def _load_dotenv(path: Path) -> None:
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def _resolve_task_path(repo_root: Path, dataset: dict[str, Any]) -> Path:
    raw = str(dataset.get("task_path") or dataset.get("task") or "").strip()
    if not raw:
        task_id = str(dataset.get("task_id") or "").strip()
        category = str(dataset.get("category") or "").strip()
        raw = f"tasks/{category}/{task_id}.md" if task_id and category else ""
    if not raw:
        raise RuntimeError("WildClawBench dataset row requires task_path or task_id/category")
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve(strict=False)


def _parse_task_md(repo_root: Path, task_file: Path) -> dict[str, Any]:
    content = task_file.read_text(encoding="utf-8")
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)", content, re.DOTALL)
    if not match:
        raise RuntimeError(f"YAML frontmatter not found: {task_file}")
    metadata = _parse_simple_frontmatter(match.group(1))
    body = match.group(2)
    sections: dict[str, str] = {}
    current = None
    lines: list[str] = []
    for line in body.splitlines():
        header = re.match(r"^##\s+(.+)$", line)
        if header:
            if current is not None:
                sections[current] = "\n".join(lines).strip()
            current = header.group(1)
            lines = []
        else:
            lines.append(line)
    if current is not None:
        sections[current] = "\n".join(lines).strip()

    workspace_path = _strip_codeblock(sections.get("Workspace Path", ""))
    if not workspace_path:
        raise RuntimeError(f"Missing ## Workspace Path in task.md: {task_file}")
    wp = Path(workspace_path)
    if not wp.is_absolute():
        wp = (repo_root / wp).resolve(strict=False)
    skills_path = (repo_root / "skills").resolve(strict=False)
    return {
        "task_id": metadata.get("id", task_file.stem),
        "prompt": sections.get("Prompt", "").strip(),
        "workspace_path": str(wp),
        "skills_path": str(skills_path),
        "automated_checks": _strip_codeblock(sections.get("Automated Checks", "")),
        "env": _strip_codeblock(sections.get("Env", "")),
        "skills": _strip_codeblock(sections.get("Skills", "")),
        "warmup": _strip_codeblock(sections.get("Warmup", "")),
        "timeout_seconds": int(metadata.get("timeout_seconds", 120)),
        "category": task_file.parent.name,
    }


def _strip_codeblock(raw: str) -> str:
    text = re.sub(r"^```[^\n]*\n?", "", str(raw or "").strip())
    text = re.sub(r"\n?```$", "", text).strip()
    return text


def _parse_simple_frontmatter(raw: str) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for raw_line in str(raw or "").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
            value = value[1:-1]
        if re.fullmatch(r"-?\d+", value):
            metadata[key] = int(value)
        else:
            metadata[key] = value
    return metadata


def _model_ref(route_model: str) -> str:
    return f"{DEFAULT_PROVIDER_ID}/{str(route_model or DEFAULT_ROUTE_MODEL).strip()}"


def _gateway_session_base_url(raw_base_url: str, *, session_id: str) -> str:
    raw_base_url = _required_str(raw_base_url, "gateway_base_url")
    parts = urlsplit(raw_base_url)
    if not parts.scheme or not parts.netloc:
        raise RuntimeError(f"gateway_base_url must be absolute: {raw_base_url!r}")
    hostname = (parts.hostname or "").lower()
    netloc = parts.netloc
    if hostname in {"127.0.0.1", "localhost", "0.0.0.0", "::1"}:
        port = f":{parts.port}" if parts.port else ""
        netloc = f"host.docker.internal{port}"
    path = f"{parts.path.rstrip('/')}/{session_id}"
    return urlunsplit((parts.scheme, netloc, path, "", "")).rstrip("/")


def _build_env(*, gateway_session_base_url: str, route_model: str, api_key: str) -> dict[str, str]:
    env = dict(os.environ)
    env.update(
        {
            "OPENROUTER_BASE_URL": gateway_session_base_url,
            "OPENROUTER_API_KEY": api_key,
            "MY_PROXY_API_KEY": api_key,
            "JUDGE_MODEL": route_model,
            "DEFAULT_MODEL": _model_ref(route_model),
            "HTTP_PROXY": "",
            "HTTPS_PROXY": "",
            "ALL_PROXY": "",
            "http_proxy": "",
            "https_proxy": "",
            "all_proxy": "",
            "HTTP_PROXY_INNER": "",
            "HTTPS_PROXY_INNER": "",
            "NO_PROXY": "host.docker.internal,localhost,127.0.0.1,::1",
            "no_proxy": "host.docker.internal,localhost,127.0.0.1,::1",
            "NO_PROXY_INNER": "host.docker.internal,localhost,127.0.0.1,::1",
        }
    )
    return env


def _write_models_config(*, provider_id: str, route_model: str, base_url: str, api_key: str) -> None:
    config_path = Path("/root/.openclaw/openclaw.json")
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config = _read_json(config_path)
    if not isinstance(config, dict):
        config = {}
    config["models"] = {
        "providers": {
            provider_id: {
                "baseUrl": base_url,
                "api": "openai-completions",
                "apiKey": api_key,
                "models": [{"id": route_model, "name": route_model}],
            }
        }
    }
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _prepare_workspace(*, task: dict[str, Any], tmp_workspace: Path) -> None:
    workspace = Path(str(task["workspace_path"]))
    exec_path = workspace / "exec"
    tmp_path = workspace / "tmp"
    if not exec_path.exists():
        exec_path.mkdir(parents=True, exist_ok=True)
    elif not exec_path.is_dir():
        raise RuntimeError(f"WildClawBench task exec workspace not found: {exec_path}")
    if tmp_workspace.exists():
        shutil.rmtree(tmp_workspace)
    tmp_workspace.mkdir(parents=True, exist_ok=True)
    _copy_tree_contents(exec_path, tmp_workspace)
    if tmp_path.exists():
        dest_tmp = tmp_workspace / "tmp"
        dest_tmp.mkdir(parents=True, exist_ok=True)
        _copy_tree_contents(tmp_path, dest_tmp)
    os.chmod(tmp_workspace, 0o755)
    openclaw_workspace = Path("/root/.openclaw/workspace")
    if openclaw_workspace.exists() or openclaw_workspace.is_symlink():
        if openclaw_workspace.is_dir() and not openclaw_workspace.is_symlink():
            shutil.rmtree(openclaw_workspace)
        else:
            openclaw_workspace.unlink()
    openclaw_workspace.parent.mkdir(parents=True, exist_ok=True)
    openclaw_workspace.symlink_to(tmp_workspace)
    gt_host = workspace / "gt"
    if gt_host.is_dir():
        gt_dest = tmp_workspace / "gt"
        if gt_dest.exists():
            shutil.rmtree(gt_dest)
        shutil.copytree(gt_host, gt_dest)


def _copy_tree_contents(src: Path, dest: Path) -> None:
    for item in src.iterdir():
        target = dest / item.name
        if item.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(item, target, symlinks=True)
        else:
            if target.exists():
                target.unlink()
            shutil.copy2(item, target, follow_symlinks=False)


def _setup_skills(*, task: dict[str, Any]) -> None:
    skills = str(task.get("skills") or "").strip()
    if not skills:
        return
    skills_root = Path(str(task.get("skills_path") or "skills"))
    container_root = Path("/root/skills")
    container_root.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    for raw_line in skills.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        src_rel = line.replace("\\", "/").strip("/")
        dest_name = PurePosixPath(src_rel).name
        if not dest_name or dest_name in seen:
            continue
        seen.add(dest_name)
        src = skills_root / src_rel
        dest = container_root / dest_name
        if dest.exists():
            shutil.rmtree(dest)
        if src.is_dir():
            shutil.copytree(src, dest)


def _run_warmup(warmup: str, *, cwd: Path, timeout_s: float) -> None:
    commands = [line.strip() for line in str(warmup or "").splitlines() if line.strip() and not line.strip().startswith("#")]
    for command in commands:
        subprocess.run(["bash", "-lc", command], cwd=str(cwd), text=True, timeout=timeout_s, check=True)


def _set_openclaw_model(*, model_ref: str, image_model: str) -> None:
    subprocess.run(["openclaw", "models", "set", model_ref], text=True, capture_output=True, check=True)
    subprocess.run(
        ["openclaw", "config", "set", "agents.defaults.imageModel.primary", image_model],
        text=True,
        capture_output=True,
        check=False,
    )


def _system_prompt(timeout_seconds: int) -> str:
    return (
        "You are an expert in a restricted, non-interactive environment. "
        f"Solve the task efficiently before the timeout ({timeout_seconds}s). "
        "Run all processes in the foreground without user input or background services. "
        "Provide a complete, functional solution in a single pass with no placeholders. \n"
    )


def _grade_task(*, task: dict[str, Any], output_dir: Path, tmp_workspace: Path) -> dict[str, Any]:
    checks = str(task.get("automated_checks") or "").strip()
    if not checks:
        score = {"overall_score": 0.0, "error": "no automated checks"}
        _write_json(output_dir / "score.json", score)
        return score
    namespace: dict[str, Any] = {}
    exec(checks, namespace)
    grade = namespace.get("grade")
    if not callable(grade):
        score = {"overall_score": 0.0, "error": "automated checks did not define grade()"}
        _write_json(output_dir / "score.json", score)
        return score
    transcript = _load_transcript(TRANSCRIPT_PATH)
    try:
        score = grade(transcript=transcript, workspace_path=str(tmp_workspace))
    except Exception as exc:
        score = {"overall_score": 0.0, "error": str(exc)}
    if not isinstance(score, dict):
        score = {"overall_score": 0.0, "error": f"grade() returned {type(score).__name__}"}
    _write_json(output_dir / "score.json", score)
    return score


def _load_transcript(path: Path) -> list[Any]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            rows.append({"raw": line})
    return rows


def _write_usage(*, output_dir: Path, elapsed_time: float) -> dict[str, Any]:
    transcript_host = output_dir / "chat.jsonl"
    if TRANSCRIPT_PATH.is_file():
        shutil.copy2(TRANSCRIPT_PATH, transcript_host)
    usage = _extract_usage_from_jsonl(transcript_host)
    usage["elapsed_time"] = round(elapsed_time, 2)
    _write_json(output_dir / "usage.json", usage)
    return usage


def _extract_usage_from_jsonl(path: Path) -> dict[str, Any]:
    totals = _empty_usage()
    if not path.is_file():
        return totals
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if entry.get("type") != "message":
            continue
        msg = entry.get("message", {})
        if msg.get("role") != "assistant":
            continue
        totals["request_count"] += 1
        usage = msg.get("usage", {})
        totals["input_tokens"] += usage.get("input", 0)
        totals["output_tokens"] += usage.get("output", 0)
        totals["cache_read_tokens"] += usage.get("cacheRead", 0)
        totals["cache_write_tokens"] += usage.get("cacheWrite", 0)
        totals["total_tokens"] += usage.get("totalTokens", 0)
        cost = usage.get("cost", {})
        totals["cost_usd"] += cost.get("total", 0.0)
    totals["cost_usd"] = round(totals["cost_usd"], 6)
    return totals


def _empty_usage() -> dict[str, Any]:
    return {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
        "total_tokens": 0,
        "cost_usd": 0.0,
        "request_count": 0,
    }


def _collect_task_output(*, output_dir: Path, tmp_workspace: Path) -> None:
    task_output_dir = output_dir / "task_output"
    task_output_dir.mkdir(parents=True, exist_ok=True)
    openclaw_tmp = Path("/tmp/openclaw")
    if openclaw_tmp.exists():
        dest = task_output_dir / "openclaw"
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(openclaw_tmp, dest, symlinks=True)
    results = tmp_workspace / "results"
    if results.exists():
        dest = task_output_dir / "workspace" / "results"
        if dest.exists():
            shutil.rmtree(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(results, dest, symlinks=True)


def _build_output_dir(*, repo_root: Path, output_subdir: str, category: str, task_id: str, model_ref: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    run_id = uuid.uuid4().hex[:6]
    short_model = re.sub(r"[^a-zA-Z0-9.\-_]", "_", model_ref.rsplit("/", 1)[-1])
    output_dir = repo_root / output_subdir / "openclaw" / category / task_id / f"{short_model}_{timestamp}_{run_id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _read_json(path: Path) -> Any:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _extract_raw_score(score: Any) -> float | None:
    if not isinstance(score, dict):
        return None
    value = score.get("overall_score")
    if isinstance(value, (int, float)):
        return float(value)
    numeric = [float(item) for item in score.values() if isinstance(item, (int, float))]
    if not numeric:
        return None
    return sum(numeric) / len(numeric)


def _score_to_10(raw_score: float | None) -> float:
    if raw_score is None:
        return 0.0
    if raw_score <= 1.0:
        return max(0.0, min(10.0, raw_score * 10.0))
    return max(0.0, min(10.0, raw_score))


def _safe_name(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value or "").strip()).strip("-._")
    return text or "run"


def sh_quote(value: str) -> str:
    return "'" + str(value).replace("'", "'\\''") + "'"


def _write_failure(session_id: str, error: str, started_at: float) -> int:
    _write_result(
        {
            "session_id": session_id,
            "status": "failed",
            "total_reward": 0.0,
            "step_count": 0,
            "terminated": True,
            "truncated": False,
            "error_text": error,
            "metrics": {"bench": "wildclawbench", "duration_ms": round((time.perf_counter() - started_at) * 1000, 3)},
        }
    )
    return 0


def _write_result(result: dict[str, Any]) -> None:
    _persist_result_artifact(result)
    sys.stdout.write(RESULT_JSON_PREFIX + json.dumps(result, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _persist_result_artifact(result: dict[str, Any]) -> None:
    raw_path = str(os.environ.get(RESULT_PATH_ENV) or "").strip()
    if not raw_path:
        return
    try:
        path = Path(raw_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(path.name + ".tmp")
        tmp_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        tmp_path.replace(path)
    except Exception as exc:
        print(f"SAFACTORY_RUNNER_DIAGNOSTIC result_artifact_write_failed: {exc}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        _write_result(
            {
                "session_id": os.environ.get("SAFACTORY_SESSION_ID", ""),
                "status": "failed",
                "total_reward": 0.0,
                "step_count": 0,
                "terminated": True,
                "truncated": False,
                "error_text": str(exc),
                "metrics": {"bench": "wildclawbench"},
            }
        )
        raise SystemExit(0)
