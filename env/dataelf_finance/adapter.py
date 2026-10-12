"""DataElf finance (DDR-Bench 10-K) adapter for the fixed SAfactory runner.

The runner owns the SAfactory protocol; this module maps exactly one dataset
row (one ``<cik>__entity`` case) onto one native ``dataelf run`` invocation
and collects the native output.  DataElf's own exploration/scoring logic is
never reimplemented here.

Tail-stage scoring (2026-10-12 migration, benchmark_notes/
safactory_skill_incontainer_eval_migration.md): after the native run
succeeds, the adapter scores the episode IN-CONTAINER.  It loads the
read-only-mounted dataelf_eval scoring subset (``<scoring_root>/benchmarks/
<benchmark>/reward_calculator.py`` + ``<scoring_root>/eval/config.yaml``),
calls ``evaluate_case`` with a case dict built from the metrics plus the
dataset row (whose ``reward_rubric`` carries this entity's checklist), and
writes the unified ``dataelf_eval.reward.v1`` document into
``metrics["reward"]``.  The host-side rule evaluator only normalizes it.
Scoring runs under a daemon-thread budget (``scoring_timeout_s``, default
1800 s): judge calls share the episode's failure domain and budget, and a
blown budget fails the episode instead of hanging the runner — daemon
threads throughout (the calculator's judge fanout included) so process exit
is never blocked on stuck judge calls.  The full score document is also
persisted to ``reward_debug.json`` beside the episode artifacts.

Model wiring per episode (no credentials are stored in this repo):

* a per-episode DataElf config is generated and selected via
  ``DATAELF_CONFIG_FILE``; its ``env`` block points the pi explorer's
  ``vllm``-style provider at the Gateway session URL
  (``VLLM_BASE_URL``/``VLLM_API_KEY`` env refs) and pins
  ``PI_CODING_AGENT_DIR`` to a per-episode agent dir;
* that agent dir holds a generated ``models.json`` with one provider
  ``safactory`` whose single model id equals the Gateway route key, so pi's
  OpenAI-compatible client posts ``model=<route>`` to
  ``<session_url>/chat/completions``;
* a ``npm`` symlink beside the agent dir re-exposes DataElf's managed pi
  package store (``pi_package_json`` resolves it from the agent dir's
  parent), keeping the finance tool extensions loadable.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time
from typing import Any
from urllib.parse import urlsplit

_PROVIDER = "safactory"
_DEFAULT_TOOLS = {"sql": True, "python": True, "web": False}
_DEFAULT_MAX_TOOL_CALLS = 600
_DEFAULT_MAX_RUNTIME_SECONDS = 3600
_DEFAULT_SCORING_ROOT = "/opt/dataelf_eval-scoring"
_DEFAULT_SCORING_TIMEOUT_S = 1800
_STDERR_TAIL_CHARS = 4000
# Grace for the pipe write-ends to close after the process group is SIGKILLed;
# a descendant that escaped the group (pi runs in its own session) can hold
# them open forever, so even the post-kill read must be bounded.
_KILL_GRACE_SECONDS = 10
# Keys DataElf resolves from the *process* env over the generated per-episode
# config (DATAELF_* via _env(), and dataelf/config.py copies _CHILD_ENV_KEYS
# — PI_CODING_AGENT_DIR etc. — into the child env after the config env block).
# VLLM_* feeds the generated models.json $-refs. Stripping all of them from
# the inherited environment keeps the per-episode pins (agent dir, workspaces,
# model endpoint) authoritative even on a polluted runner env.
_ENV_OVERRIDE_KEYS = {
    "PI_CODING_AGENT_DIR",
    "PI_CODING_AGENT_SESSION_DIR",
    "PI_PACKAGE_DIR",
    "VLLM_BASE_URL",
    "VLLM_API_KEY",
}


def run_case(
    request: dict[str, Any],
    task: dict[str, Any],
    session_url: str,
) -> tuple[dict[str, Any], int]:
    """Run one DataElf finance job for the row in ``task``."""
    env_params = request.get("env_params")
    if not isinstance(env_params, dict):
        raise TypeError("env_params must be a JSON object")
    native = env_params.get("dataelf")
    if not isinstance(native, dict):
        raise ValueError("env_params.dataelf must be a JSON object (see dataelf_finance_config.yaml)")

    session_id = _required_text(request.get("session_id"), "session_id")
    job_id = _required_text(request.get("job_id"), "job_id")
    route_model = _first_text(os.environ.get("SAFACTORY_ROUTE_MODEL"), request.get("model"))
    if not route_model:
        raise ValueError("cannot resolve Gateway route model (request.model)")

    cik, objective = _case_identity(task)
    benchmark = _first_text(native.get("benchmark"), "ddr_10k")
    db_path = _required_text(native.get("db_path"), "env_params.dataelf.db_path")
    repo_root = Path(_first_text(native.get("repo_root"), "/opt/DataElf"))
    tools = dict(_DEFAULT_TOOLS)
    tools.update(native.get("tools") or {})
    # Presence checks, not `or`: an explicit 0 (e.g. max_runtime_seconds: 0
    # meaning "no cap") is a meaningful value an `or` default would swallow.
    max_tool_calls = _int_option(native, "max_tool_calls", _DEFAULT_MAX_TOOL_CALLS)
    max_runtime_seconds = _int_option(native, "max_runtime_seconds", _DEFAULT_MAX_RUNTIME_SECONDS)
    run_timeout = _int_option(native, "run_timeout_seconds", max_runtime_seconds + 600)
    compact_stream_events = _optional_bool(
        native.get("compact_stream_events"), "env_params.dataelf.compact_stream_events"
    )

    results_root = Path(_first_text(env_params.get("results_root"), "/tmp/safactory-dataelf-finance-results"))
    episode_dir = results_root / _safe_path_part(f"{job_id}-{session_id}")
    episode_dir.mkdir(parents=True, exist_ok=True)

    config_path = _write_dataelf_config(
        episode_dir,
        session_url=session_url,
        route_model=route_model,
        benchmark=benchmark,
        db_path=db_path,
        tools=tools,
        max_tool_calls=max_tool_calls,
        max_runtime_seconds=max_runtime_seconds,
        api_key=_first_text(native.get("api_key"), "EMPTY"),
        proxies=native.get("proxies"),
        compact_stream_events=compact_stream_events,
    )
    _write_pi_agent_dir(episode_dir, route_model, repo_root, native.get("pi_model"), native.get("pi_settings"))

    command = _native_command(native, repo_root, benchmark, db_path, tools, objective)
    process_env = _native_process_env(session_url)
    process_env.update({
        "DATAELF_CONFIG_FILE": str(config_path),
        "DATAELF_PI_MODEL": f"{_PROVIDER}/{route_model}",
        "PYTHONDONTWRITEBYTECODE": "1",
    })
    started_at = time.time()
    # dataelf spawns pi with start_new_session=True, so only a process-group
    # kill reaches the pi tree; killing the dataelf child alone would leave
    # orphaned pi processes posting to the already-finalized Gateway session.
    process = subprocess.Popen(
        command,
        cwd=str(repo_root) if repo_root.is_dir() else None,
        env=process_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    timed_out = False
    try:
        stdout, stderr = process.communicate(timeout=run_timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            os.killpg(process.pid, signal.SIGKILL)  # pgid == pid via start_new_session
        except ProcessLookupError:
            pass
        try:
            stdout, stderr = process.communicate(timeout=_KILL_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = "", ""
    # Native stdout/stderr stay out of the runner's stdout; keep a copy on
    # disk on every path (including the timeout kill) for post-mortem.
    _write_text(episode_dir / "dataelf_stdout.log", stdout or "")
    _write_text(episode_dir / "dataelf_stderr.log", stderr or "")
    if timed_out:
        raise RuntimeError(
            f"dataelf run timed out after {run_timeout}s (process group killed); "
            f"stderr tail: {(stderr or '').strip()[-_STDERR_TAIL_CHARS:]}"
        )
    if process.returncode != 0:
        raise RuntimeError(
            "dataelf run failed (exit "
            f"{process.returncode}): {(stderr or '').strip()[-_STDERR_TAIL_CHARS:]}"
        )

    workspace = _find_workspace(episode_dir, started_at)
    result_path = workspace / "results" / "results.json"
    if not result_path.is_file():
        raise RuntimeError(
            f"dataelf reported success but {result_path} is missing; "
            f"workspace: {workspace}; stderr tail: {(stderr or '').strip()[-_STDERR_TAIL_CHARS:]}"
        )
    try:
        native_result = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot parse native results.json ({result_path}): {exc}") from exc
    summary = native_result.get("summary")
    key_findings = native_result.get("key_findings")
    if not isinstance(summary, str) or not summary.strip():
        raise RuntimeError(f"native results.json ({result_path}) is missing the required 'summary'")
    if not isinstance(key_findings, list):
        raise RuntimeError(f"native results.json ({result_path}) is missing the required 'key_findings'")

    metrics: dict[str, Any] = {
        "bench": "dataelf_finance",
        "benchmark": benchmark,
        "case_id": task.get("id") or task.get("task_id"),
        "entity_id": cik,
        "objective": objective,
        "native_status": "completed",
        "summary": summary,
        "key_findings": key_findings,
        "native_output_path": str(result_path),
        "native_workspace": str(workspace),
        "pi_events_path": str(workspace / "logs" / "pi_events.jsonl"),
        "dataelf_config": str(config_path),
    }
    for key in ("evidence_refs", "confidence", "limitations"):
        if key in native_result:
            metrics[key] = native_result[key]
    metrics["reward"] = _score_episode(
        env_params, task, metrics,
        session_id=session_id, job_id=job_id, episode_dir=episode_dir,
    )
    step_count = _pi_event_count(workspace)
    return metrics, step_count


def _score_episode(
    env_params: dict[str, Any],
    task: dict[str, Any],
    metrics: dict[str, Any],
    *,
    session_id: str,
    job_id: str,
    episode_dir: Path,
) -> dict[str, Any]:
    """Score this episode in-container via the mounted dataelf_eval module.

    The case dict follows ``dataelf_eval/eval/REWARD_SCHEMA.md`` §1: native
    deliverable + artifact paths + the dataset row (whose ``reward_rubric``
    carries this entity's checklist).  The returned unified reward document
    (``dataelf_eval.reward.v1``) is written into ``metrics["reward"]`` and
    persisted to ``reward_debug.json``.  Any scoring failure — module not
    mounted, case rejected, judge crash, blown budget, non-finite score —
    fails the episode (scoring shares the rollout's failure domain by
    design).
    """
    scoring = env_params.get("dataelf_eval")
    if not isinstance(scoring, dict):
        raise ValueError(
            "env_params.dataelf_eval must be a JSON object "
            "(scoring_root/scoring_timeout_s; see dataelf_finance_config.yaml)"
        )
    scoring_root = Path(_first_text(scoring.get("scoring_root"), _DEFAULT_SCORING_ROOT))
    scoring_timeout_s = _int_option(scoring, "scoring_timeout_s", _DEFAULT_SCORING_TIMEOUT_S)
    benchmark = str(metrics.get("benchmark") or "")
    module_path = scoring_root / "benchmarks" / benchmark / "reward_calculator.py"
    if not module_path.is_file():
        raise RuntimeError(
            f"in-container scoring module not mounted at {module_path} (expected the "
            "read-only dataelf_eval scoring subset; see dataelf_finance_start.yaml)"
        )

    case = {
        "benchmark": benchmark,
        "question_id": metrics.get("case_id"),
        "entity_id": metrics.get("entity_id"),
        "objective": metrics.get("objective"),
        "session_id": session_id,
        "job_id": job_id,
        "summary": metrics.get("summary"),
        "key_findings": metrics.get("key_findings"),
        "artifacts": {
            "result_file": metrics.get("native_output_path"),
            "pi_events": metrics.get("pi_events_path"),
            "workspace": metrics.get("native_workspace"),
        },
        # The row is passed verbatim: reward_calculator._checklist reads
        # row["reward_rubric"] (digest-verified single-case checklist).
        "row": task,
        # Points the calculator's _config at the mounted eval/config.yaml
        # (judge authority; the qa_file fallback string simply won't resolve
        # in-container and is never reached when the row carries a rubric).
        "eval_params": {"root": str(scoring_root)},
    }

    outcome: dict[str, Any] = {}

    def _score() -> None:
        try:
            spec = importlib.util.spec_from_file_location(
                f"dataelf_scoring_{benchmark or 'benchmark'}", module_path
            )
            if spec is None or spec.loader is None:
                raise RuntimeError(f"cannot load scoring module {module_path}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            calculator = getattr(module, "REWARD_CALCULATOR", None)
            if calculator is None or not hasattr(calculator, "evaluate_case"):
                raise RuntimeError(f"{module_path} does not export REWARD_CALCULATOR.evaluate_case")
            outcome["score"] = calculator.evaluate_case(case)
        except BaseException as exc:  # reported into the episode error below
            outcome["error"] = f"{type(exc).__name__}: {exc}"

    # Daemon thread + bounded join (NOT ThreadPoolExecutor: its workers are
    # non-daemon and their atexit join would hang the runner's exit when this
    # budget fires on stuck judge calls).
    worker = threading.Thread(target=_score, name="dataelf-reward-scoring", daemon=True)
    worker.start()
    worker.join(max(1, scoring_timeout_s))

    debug_path = episode_dir / "reward_debug.json"
    if worker.is_alive():
        _write_text(
            debug_path,
            json.dumps(
                {"phase": "timeout", "benchmark": benchmark, "scoring_timeout_s": scoring_timeout_s},
                ensure_ascii=False, indent=2,
            ),
        )
        raise RuntimeError(
            f"in-container reward scoring exceeded its {scoring_timeout_s}s budget "
            "(judge endpoint stuck?); scoring thread abandoned, episode failed"
        )
    if "error" in outcome:
        _write_text(
            debug_path,
            json.dumps({"phase": "error", "benchmark": benchmark, "error": outcome["error"]}, ensure_ascii=False, indent=2),
        )
        raise RuntimeError(f"in-container reward scoring failed: {outcome['error']}")
    score = outcome["score"]
    if not isinstance(score, dict) or score.get("status") != "succeeded":
        raise RuntimeError(
            "in-container reward scoring returned a non-succeeded document: "
            f"{json.dumps(score, default=str)[:500]}"
        )
    value = score.get("normalized_score_10")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise RuntimeError(f"in-container reward score normalized_score_10 is not finite: {value!r}")
    _write_text(debug_path, json.dumps(score, ensure_ascii=False, indent=2, allow_nan=False))
    return score


def _case_identity(task: dict[str, Any]) -> tuple[str, str]:
    """Rows carry only the case identity: id/task/entity_id.

    The prompt and tool surface is generated natively by DataElf's ddr_10k
    benchmark config; ``task`` is passed verbatim as the native objective.
    """
    cik = _first_text(task.get("entity_id"))
    row_id = _first_text(task.get("id"))
    if not cik and row_id and "__" in row_id:
        cik = row_id.split("__", 1)[0]
    if not cik:
        raise ValueError("cannot derive CIK from dataset row (entity_id or id '<cik>__entity')")
    objective = _first_text(task.get("task")) or f"Analyze company with CIK {cik}"
    return cik, objective


def _native_command(
    native: dict[str, Any],
    repo_root: Path,
    benchmark: str,
    db_path: str,
    tools: dict[str, Any],
    objective: str,
) -> list[str]:
    binary = _first_text(native.get("bin"))
    if not binary:
        candidate = repo_root / ".venv" / "bin" / "dataelf"
        binary = str(candidate) if candidate.is_file() else "dataelf"
    binary = _resolve_binary(binary)
    finance_source = json.dumps({"sqlite": {"db_path": db_path, "link": True}}, separators=(",", ":"))
    finance_tools = json.dumps(tools, separators=(",", ":"))
    return [
        binary, "run",
        "--domain", "finance",
        "--param", f"benchmark={benchmark}",
        "--param", f"finance_source={finance_source}",
        "--param", f"finance_tools={finance_tools}",
        objective,
    ]


def _native_process_env(session_url: str) -> dict[str, str]:
    """Inherited env minus the keys that would override the per-episode pins.

    See ``_ENV_OVERRIDE_KEYS``: a stray ``PI_CODING_AGENT_DIR``/``DATAELF_*``
    from the runner's environment would silently repin the agent dir,
    workspace or model wiring that the generated config owns. The Gateway
    session host is additionally exempted from any inherited proxy so model
    calls cannot be routed through it.
    """
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("DATAELF_") and key not in _ENV_OVERRIDE_KEYS
    }
    host = (urlsplit(session_url).hostname or "").strip()
    if host:
        for var in ("NO_PROXY", "no_proxy"):
            entries = [entry.strip() for entry in env.get(var, "").split(",") if entry.strip()]
            if host not in entries:
                entries.append(host)
            env[var] = ",".join(entries)
    return env


def _resolve_binary(binary: str) -> str:
    """Make a path-like binary absolute before the subprocess cwd changes."""
    if "/" not in binary and "\\" not in binary:
        resolved = shutil.which(binary)
        return resolved or binary
    path = Path(binary)
    return str(path if path.is_absolute() else path.resolve())


def _write_dataelf_config(
    episode_dir: Path,
    *,
    session_url: str,
    route_model: str,
    benchmark: str,
    db_path: str,
    tools: dict[str, Any],
    max_tool_calls: int,
    max_runtime_seconds: int,
    api_key: str,
    proxies: dict[str, Any] | None,
    compact_stream_events: bool | None,
) -> Path:
    """Write the per-episode DataElf config (selected via DATAELF_CONFIG_FILE).

    Replaces the developer's ``dataelf.local.yaml`` for this episode: the
    model endpoint becomes the Gateway session URL and the job workspace
    moves under the episode's results directory.  The file carries the
    session URL and API key in plain text (models.json consumes them via
    $-refs), so it is written 0600 and lives only for this episode on the
    results volume.
    """
    config = {
        "runtime": {
            "workspaces_dir": str(episode_dir / "workspaces"),
            "enable_sqlite": False,
        },
        "explorer": {
            "type": "pi",
            "pi": _pi_section(route_model, compact_stream_events),
        },
        "domains": {
            "finance": {
                "benchmark": benchmark,
                "source": {"sqlite": {"db_path": db_path, "link": True}},
                "tools": tools,
                "analysis": {
                    "max_tool_calls": max_tool_calls,
                    "max_runtime_seconds": max_runtime_seconds,
                },
            },
        },
        "env": {
            # Resolved by the generated pi models.json ($VLLM_* refs below).
            "VLLM_BASE_URL": session_url.rstrip("/"),
            "VLLM_API_KEY": api_key,
            "PI_CODING_AGENT_DIR": str(episode_dir / "pi-agent"),
        },
    }
    if isinstance(proxies, dict):
        config["env"].update({str(k): str(v) for k, v in proxies.items()})
    path = episode_dir / "dataelf.config.yaml"
    _write_text(path, json.dumps(config, ensure_ascii=False, indent=2))  # JSON is valid YAML.
    path.chmod(0o600)
    return path


def _pi_section(route_model: str, compact_stream_events: bool | None) -> dict[str, Any]:
    """explorer.pi config for the generated DataElf config.

    ``compact_stream_events`` is forwarded only when explicitly set: DataElf
    defaults it to False, and True strips the cumulative ``message_update``
    snapshots from ``pi_events.jsonl`` (deltas + ``message_end`` remain), which
    keeps long explorations from writing hundreds of MB of redundant frames.
    """
    section: dict[str, Any] = {"model": f"{_PROVIDER}/{route_model}", "mode": "json"}
    if compact_stream_events is not None:
        section["compact_stream_events"] = compact_stream_events
    return section


def _write_pi_agent_dir(
    episode_dir: Path,
    route_model: str,
    repo_root: Path,
    pi_model: dict[str, Any] | None,
    pi_settings: dict[str, Any] | None,
) -> None:
    """Generate the per-episode pi agent dir (models.json) plus npm symlink.

    ``models.json`` defines one provider whose model id equals the Gateway
    route key, so pi sends ``model=<route>`` on every chat completion.  The
    ``npm`` symlink re-exposes DataElf's managed pi packages: pi_package_json
    resolves the package store from ``<agent_dir>/../npm``.  ``settings.json``
    raises pi's HTTP idle timeout: the 300s default cancels and retries long
    turns whose time-to-first-token grows with the dialogue context.
    """
    settings = {"httpIdleTimeoutMs": 900_000}
    if isinstance(pi_settings, dict):
        settings.update(pi_settings)
    model_entry = {
        "id": route_model,
        "name": f"SAfactory Gateway route {route_model}",
        "reasoning": False,
        "input": ["text"],
        "contextWindow": 262144,
        "maxTokens": 16384,
        "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
    }
    if isinstance(pi_model, dict):
        model_entry.update(pi_model)
    models = {
        "providers": {
            _PROVIDER: {
                "baseUrl": "$VLLM_BASE_URL",
                "api": "openai-completions",
                "apiKey": "$VLLM_API_KEY",
                # Mirrors DataElf's vllm provider entry: conservative OpenAI-
                # compatible behavior expected from Gateway routes.
                "compat": {
                    "supportsDeveloperRole": False,
                    "supportsReasoningEffort": False,
                    "supportsUsageInStreaming": False,
                    "maxTokensField": "max_tokens",
                },
                "models": [model_entry],
            }
        }
    }
    agent_dir = episode_dir / "pi-agent"
    agent_dir.mkdir(parents=True, exist_ok=True)
    _write_text(agent_dir / "models.json", json.dumps(models, ensure_ascii=False, indent=2))
    _write_text(agent_dir / "settings.json", json.dumps(settings, ensure_ascii=False, indent=2))

    npm_source = repo_root / ".pi" / "npm"
    npm_link = episode_dir / "npm"
    if npm_source.is_dir():
        if npm_link.is_symlink() or npm_link.exists():
            npm_link.unlink() if npm_link.is_symlink() else shutil.rmtree(npm_link)
        npm_link.symlink_to(npm_source, target_is_directory=True)
    else:
        print(
            f"dataelf_finance: managed pi package store not found at {npm_source}; "
            "native run may lack the finance tool extensions",
            file=sys.stderr,
        )


def _find_workspace(episode_dir: Path, started_at: float) -> Path:
    """Newest workspace created by *this* native invocation.

    ``episode_dir`` survives retries (``mkdir(exist_ok=True)``), so without
    the mtime filter a rerun whose dataelf invocation exited 0 without
    creating a fresh workspace would silently score the previous attempt's
    results.json.  The 2s slack absorbs coarse filesystem timestamps.
    """
    workspaces = episode_dir / "workspaces"
    candidates = (
        [p for p in workspaces.iterdir() if p.is_dir()]
        if workspaces.is_dir()
        else []
    )
    candidates = [p for p in candidates if p.stat().st_mtime >= started_at - 2.0]
    if not candidates:
        raise RuntimeError(f"no DataElf job workspace from this run under {workspaces}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _int_option(native: dict[str, Any], key: str, default: int) -> int:
    """Explicit-presence int option: 0 is a meaningful value, ``or`` is not."""
    if native.get(key) is None:
        return default
    return int(native[key])


def _pi_event_count(workspace: Path) -> int:
    events = workspace / "logs" / "pi_events.jsonl"
    try:
        with events.open(encoding="utf-8") as handle:
            return max(1, sum(1 for line in handle if line.strip()))
    except OSError:
        return 1


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


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


def _optional_bool(value: Any, name: str) -> bool | None:
    if value is None:
        return None
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean when set (got {type(value).__name__})")
    return value


def _safe_path_part(value: Any) -> str:
    text = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(value))
    return text.strip("._") or "episode"
