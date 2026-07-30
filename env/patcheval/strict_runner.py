#!/usr/bin/env python3
"""Thin SAfactory adapter around the official PatchEval S1.x components."""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
import types
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen


DEFAULT_TIMEOUT_S = 900.0
MAX_LOG_CHARS = 16_000
SETTING_EPOCHS = {"s1.1": 1, "s1.2": 1, "s1.3": 1, "s1.4": 5}
LANGUAGE_COMMENT_MAP = {
    "py": ("Python", "#"),
    "js": ("JavaScript", "//"),
    "go": ("Go", "//"),
}
_OFFICIAL_COMPONENTS: dict[str, Any] | None = None


class _LogManager:
    """Minimal logger contract required by PatchEval helper classes."""

    def __init__(self) -> None:
        self.logger = logging.getLogger("safactory.patcheval.official")

    def bind_current_task(self, _task_id: str) -> None:
        return None

    def get_current_logger(self) -> logging.Logger:
        return self.logger


def _load_official_components() -> dict[str, Any]:
    global _OFFICIAL_COMPONENTS
    if _OFFICIAL_COMPONENTS is not None:
        return _OFFICIAL_COMPONENTS

    official_root = Path(
        os.environ.get("PATCHEVAL_OFFICIAL_ROOT")
        or Path(__file__).resolve().parent / "PatchEval" / "patcheval"
    ).resolve()
    if not (official_root / "exp_llm" / "helper" / "llm_suite.py").is_file():
        raise RuntimeError(f"official PatchEval source is unavailable at {official_root}")
    if str(official_root) not in sys.path:
        sys.path.insert(0, str(official_root))

    # CVE images need not install the OpenAI SDK: the official LLMClient is
    # reused for prompt construction only; SAfactory owns the gateway call.
    try:
        import openai  # noqa: F401
    except ImportError:
        openai_stub = types.ModuleType("openai")

        class _UnavailableOpenAI:
            def __init__(self, *_args: Any, **_kwargs: Any) -> None:
                raise RuntimeError("OpenAI SDK calls must pass through the SAfactory gateway adapter")

        openai_stub.OpenAI = _UnavailableOpenAI
        sys.modules["openai"] = openai_stub
    try:
        import requests  # noqa: F401
    except ImportError:
        # requests is another import-time-only dependency for the helpers used
        # here; network traffic is intentionally handled by _call_gateway.
        sys.modules["requests"] = types.ModuleType("requests")

    from exp_llm.helper.func_replacer import FuncReplacer
    from exp_llm.helper.llm_suite import CodeApplier, FeedbackHelper, LLMClient, PatchParser, Validators

    _OFFICIAL_COMPONENTS = {
        "CodeApplier": CodeApplier,
        "FeedbackHelper": FeedbackHelper,
        "FuncReplacer": FuncReplacer,
        "LLMClient": LLMClient,
        "PatchParser": PatchParser,
        "Validators": Validators,
    }
    return _OFFICIAL_COMPONENTS


def main() -> int:
    started_at = time.perf_counter()
    request = _read_request()
    session_id = _required_text(request.get("session_id"), "session_id")
    cve_id = ""
    setting = ""

    try:
        env_params = request.get("env_params") if isinstance(request.get("env_params"), dict) else {}
        dataset = env_params.get("dataset") if isinstance(env_params.get("dataset"), dict) else {}
        cve_id = _required_text(dataset.get("cve_id"), "cve_id").upper()
        setting = _required_text(dataset.get("setting"), "setting").lower()
        if setting not in SETTING_EPOCHS:
            raise ValueError(f"unsupported PatchEval setting: {setting}")
        record = dataset.get("official_record")
        if not isinstance(record, dict):
            raise ValueError("official_record must be an object")
        if str(record.get("cve_id") or "").upper() != cve_id:
            raise ValueError(f"official_record CVE does not match {cve_id}")
        prompt_template = _required_text(dataset.get("prompt_template"), "prompt_template")
        work_dir = Path(_required_text(dataset.get("work_dir"), "work_dir"))
        if not work_dir.is_dir():
            raise RuntimeError(f"PatchEval work directory does not exist: {work_dir}")

        timeout_s = _float(request.get("agent_start_timeout_s"), DEFAULT_TIMEOUT_S)
        model = _required_text(
            request.get("model") or env_params.get("route_model") or os.environ.get("SAFACTORY_ROUTE_MODEL"),
            "model",
        )
        base_url = _resolve_base_url(request, session_id)
        result = _run_setting(
            cve_id=cve_id,
            setting=setting,
            record=record,
            prompt_template=prompt_template,
            work_dir=work_dir,
            base_url=base_url,
            model=model,
            timeout_s=timeout_s,
        )
        result["duration_ms"] = round((time.perf_counter() - started_at) * 1000, 3)
        _write_result(
            {
                "session_id": session_id,
                "status": "succeeded",
                # The official per-case rule evaluator owns the final reward.
                "total_reward": 0.0,
                "step_count": result["rounds_executed"],
                "terminated": True,
                "truncated": False,
                "error_text": None,
                "metrics": result,
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
                    "cve_id": cve_id or None,
                    "setting": setting or None,
                    "patch": "",
                    "infrastructure_error": True,
                    "duration_ms": round((time.perf_counter() - started_at) * 1000, 3),
                },
            }
        )
    return 0


def _run_setting(
    *,
    cve_id: str,
    setting: str,
    record: dict[str, Any],
    prompt_template: str,
    work_dir: Path,
    base_url: str,
    model: str,
    timeout_s: float,
) -> dict[str, Any]:
    vul_functions = record.get("vul_func")
    if not isinstance(vul_functions, list) or not vul_functions:
        raise ValueError(f"{cve_id} has no official vul_func entries")

    last_feedbacks: dict[str, str] = {}
    last_generated: dict[str, str] = {}
    round_metrics: list[dict[str, Any]] = []
    final_patch = ""
    strict_success = False

    _remove_answer_leaks()
    for epoch in range(SETTING_EPOCHS[setting]):
        _run(["bash", "/workspace/prepare.sh"], cwd=Path("/workspace"), timeout_s=timeout_s)
        prompt = _build_prompt(
            record=record,
            vul_functions=vul_functions,
            feedbacks=last_feedbacks,
            prompt_template=prompt_template,
            use_cot=setting != "s1.2",
        )
        response = _call_gateway(
            base_url=base_url,
            model=model,
            prompt=prompt,
            timeout_s=timeout_s,
        )
        generated = _parse_response(response, cve_id)
        apply_error = ""
        try:
            final_patch = _apply_function_replacements(work_dir, vul_functions, generated)
        except Exception as exc:
            final_patch = ""
            apply_error = str(exc)

        evaluation = _evaluate_official_scripts(final_patch, timeout_s)
        strict_success = bool(evaluation["strict_success"])
        round_metrics.append(
            {
                "round": epoch + 1,
                "response_parsed": bool(generated),
                "function_count": len(generated),
                "patch_generated": bool(final_patch),
                "apply_error": apply_error,
                **evaluation,
            }
        )
        if strict_success:
            break

        last_generated = generated
        if setting == "s1.4":
            test_message = str(evaluation.get("poc_log") or apply_error or "Patch generation failed.")
            feedback_helper = _load_official_components()["FeedbackHelper"](_LogManager())
            feedback_helper.update_feedback(last_feedbacks, last_generated, test_message)

    final_eval = round_metrics[-1]
    return {
        "bench": "patcheval",
        "protocol": "official_components_s1x",
        "setting": setting,
        "cve_id": cve_id,
        "score": 1.0 if strict_success else 0.0,
        "strict_success": strict_success,
        "rounds_executed": len(round_metrics),
        "patch": final_patch,
        "patch_applied": final_eval["patch_applied"],
        "poc_passed": final_eval["poc_passed"],
        "unit_test_present": final_eval["unit_test_present"],
        "unit_tests_passed": final_eval["unit_tests_passed"],
        "failure_stage": final_eval["failure_stage"],
        "poc_log": final_eval["poc_log"],
        "unit_test_log": final_eval["unit_test_log"],
        "round_metrics": round_metrics,
    }


def _build_prompt(
    *,
    record: dict[str, Any],
    vul_functions: list[dict[str, Any]],
    feedbacks: dict[str, str],
    prompt_template: str,
    use_cot: bool,
) -> str:
    functions: list[dict[str, str]] = []
    for function in vul_functions:
        func_id = _required_text(function.get("id"), "vul_func.id")
        original_code = _process_original_code(function.get("snippet"))
        functions.append({"id": func_id, "original_code": original_code})

    official_client = _load_official_components()["LLMClient"]
    return official_client.build_prompt(
        None,
        functions,
        record,
        feedbacks,
        prompt_template,
        use_cot,
        str(record.get("cve_id") or ""),
        [],
    )


def _process_original_code(value: Any) -> str:
    validators = _load_official_components()["Validators"]()
    return validators.process_original_code(value)


def _parse_response(raw_response: str, cve_id: str = "SAFACTORY") -> dict[str, str]:
    parser = _load_official_components()["PatchParser"](_LogManager())
    return parser.parse(str(raw_response or ""), cve_id)


def _apply_function_replacements(
    work_dir: Path,
    vul_functions: list[dict[str, Any]],
    generated: dict[str, str],
) -> str:
    if not generated:
        return ""

    components = _load_official_components()
    log_manager = _LogManager()
    validators = components["Validators"]()
    replacer = components["FuncReplacer"](log_manager)
    code_applier = components["CodeApplier"](log_manager)
    changed_paths: set[tuple[str, str]] = set()

    # This ordering is the one used by VulFixer._process_single_cve.
    for function in sorted(vul_functions, key=lambda item: int(item["start_line"]), reverse=True):
        func_id = str(function["id"])
        replacement = generated.get(func_id, generated.get(func_id.replace("vul", "fix")))
        if replacement is None:
            raise ValueError(f"response omitted function {func_id}")
        relative_path = str(function["file_path"])
        path = (work_dir / relative_path).resolve()
        path.relative_to(work_dir.resolve())
        language, _comment = validators.get_language_info(func_id, LANGUAGE_COMMENT_MAP)
        code_applier.apply_change(replacer, str(work_dir), function, replacement, language)
        changed_paths.add((str(work_dir), relative_path))

    return code_applier.generate_cve_diff(replacer, changed_paths, "SAFACTORY")


def _evaluate_official_scripts(patch: str, timeout_s: float) -> dict[str, Any]:
    result = {
        "strict_success": False,
        "patch_applied": False,
        "poc_passed": False,
        "unit_test_present": Path("/workspace/unit_test.sh").is_file(),
        "unit_tests_passed": False,
        "failure_stage": "patch_generation",
        "poc_log": "",
        "unit_test_log": "",
    }
    if not patch:
        return result

    Path("/workspace/fix.patch").write_text(patch, encoding="utf-8")
    _run(["bash", "/workspace/prepare.sh"], cwd=Path("/workspace"), timeout_s=timeout_s)
    result["patch_applied"] = True
    result["failure_stage"] = "poc"
    poc = _run(
        ["bash", "/workspace/fix-run.sh"],
        cwd=Path("/workspace"),
        timeout_s=timeout_s,
        check=False,
    )
    result["poc_log"] = _trim_log(poc.stdout + poc.stderr)
    if poc.returncode != 0:
        return result
    result["poc_passed"] = True

    if not result["unit_test_present"]:
        result.update(strict_success=True, unit_tests_passed=True, failure_stage=None)
        return result

    _run(["bash", "/workspace/prepare.sh"], cwd=Path("/workspace"), timeout_s=timeout_s)
    result["failure_stage"] = "unit_tests"
    unit = _run(
        ["bash", "/workspace/unit_test.sh"],
        cwd=Path("/workspace"),
        timeout_s=timeout_s,
        check=False,
    )
    result["unit_test_log"] = _trim_log(unit.stdout + unit.stderr)
    if unit.returncode == 0:
        result.update(strict_success=True, unit_tests_passed=True, failure_stage=None)
    return result


def _call_gateway(*, base_url: str, model: str, prompt: str, timeout_s: float) -> str:
    payload = json.dumps(
        {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
            "max_tokens": 16384,
            "stream": False,
        }
    ).encode("utf-8")
    request = Request(
        f"{base_url}/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=timeout_s) as response:
        body = json.loads(response.read().decode("utf-8"))
    return str(body["choices"][0]["message"]["content"])


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


def _float(value: Any, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _resolve_base_url(request: dict[str, Any], session_id: str) -> str:
    explicit = str(os.environ.get("SAFACTORY_GATEWAY_SESSION_URL_CONTAINER") or "").strip()
    if explicit:
        return explicit.rstrip("/")
    raw = _required_text(request.get("gateway_base_url"), "gateway_base_url").rstrip("/")
    parts = urlsplit(raw)
    hostname = parts.hostname
    if hostname in {"127.0.0.1", "localhost", "::1"}:
        hostname = "host.docker.internal"
    netloc = hostname or ""
    if parts.port is not None:
        netloc = f"{netloc}:{parts.port}"
    base = urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment)).rstrip("/")
    return f"{base}/{session_id}"


def _remove_answer_leaks() -> None:
    path = Path("/workspace/llm.patch")
    if path.exists():
        path.unlink()


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


def _trim_log(value: str) -> str:
    return str(value or "")[-MAX_LOG_CHARS:]


if __name__ == "__main__":
    raise SystemExit(main())
