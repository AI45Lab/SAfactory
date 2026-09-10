from __future__ import annotations

import asyncio
import importlib.util
import logging
import os
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any

from evaluator.eval_types import EvalRequest, EvalResult, EvalSpec, EvalStatus, Trajectory


_OFFICIAL_EVALUATION: type[Any] | None = None
_OFFICIAL_LOCK = threading.Lock()
_ADAPTER_INSTALLED = False
_MAX_LOG_CHARS = 16_000


async def evaluate_rule(
    *,
    request: EvalRequest,
    spec: EvalSpec,
    trajectory: Trajectory,
) -> EvalResult:
    """Evaluate one generated patch with PatchEval's official evaluator."""
    metrics = _start_metrics(request)
    dataset = request.env_params.get("dataset")
    dataset = dataset if isinstance(dataset, dict) else {}
    cve_id = str(metrics.get("cve_id") or dataset.get("cve_id") or "").strip().upper()
    patch = str(metrics.get("patch") or "")
    official_record = dataset.get("official_record")
    official_record = official_record if isinstance(official_record, dict) else {}
    language = str(official_record.get("programming_language") or "").strip()

    # Pre-gate (relaxed): align with PatchEval's native scoring semantics.
    # The official bench has NO upfront rejection of empty/missing patches -- an
    # empty patch flows through the Docker oracle and scores 0 (validation_fail),
    # it is a legitimate negative sample, NOT "no data". Returning FAILED here
    # made RewardCommitter refuse to write a reward (reward stays NULL), which
    # silently dropped these trajectories and starved RL of negative samples.
    # So instead of FAILED/null, return SUCCEEDED with score 0 so a reward of 0.0
    # is committed and the trajectory can join a GRPO group.
    if not cve_id or not patch or not language:
        missing = []
        if not cve_id:
            missing.append("cve_id")
        if not patch:
            missing.append("patch")
        if not language:
            missing.append("language")
        return EvalResult(
            session_id=request.session_id,
            eval_id=spec.eval_id,
            method=spec.method.value,
            status=EvalStatus.SUCCEEDED.value,
            raw_score=0.0,
            normalized_score_10=0.0,
            reason=(
                "PatchEval pre-gate: missing "
                + ", ".join(missing)
                + " -- scored 0 (no valid patch produced)"
            ),
            artifacts={
                "bench": "patcheval",
                "cve_id": cve_id or None,
                "patch_generated": bool(patch),
                "language": language or None,
                "validation_type": "validation_fail",
                "metrics": metrics,
            },
        )

    try:
        evaluation_class = _load_official_evaluation(request.env_params)
        evaluation = evaluation_class(
            logger=logging.getLogger(f"safactory.patcheval.official.{cve_id.lower()}"),
            cve=cve_id,
        )
        poc_passed, poc_log, unit_tests_passed, unit_test_log, validation_type = await asyncio.to_thread(
            evaluation.run_evaluation,
            cve_id,
            patch,
            language,
            f"safactory_{request.session_id.replace('-', '_')}",
            [],
        )
    except Exception as exc:
        # In rjob mode the runner already runs the official PatchEval evaluation
        # (PoC + unit tests) inside the RJob pod using the real CVE image, and
        # records strict_success/poc_passed/unit_tests_passed in its metrics.
        # The launcher-side re-evaluation here needs a local Docker daemon,
        # which rjob coordinator hosts typically lack (k8s worker nodes run
        # containerd, not dockerd). When Docker is unavailable, fall back to the
        # runner's authoritative in-pod result instead of failing the episode.
        if _is_docker_unavailable(exc):
            fallback = _fallback_from_runner_metrics(request, spec, metrics, exc)
            if fallback is not None:
                return fallback
        # Align with PatchEval native semantics: an exception during official
        # evaluation (typically Docker unavailable on the rjob coordinator) is
        # a validation failure, NOT "no data". Score 0 so a reward of 0.0 is
        # committed and the trajectory joins a GRPO group instead of being
        # silently dropped (FAILED -> RewardCommitter refuses -> reward NULL).
        return EvalResult(
            session_id=request.session_id,
            eval_id=spec.eval_id,
            method=spec.method.value,
            status=EvalStatus.SUCCEEDED.value,
            raw_score=0.0,
            normalized_score_10=0.0,
            reason="official PatchEval evaluation raised an exception (Docker unavailable?) -- scored 0",
            error_text=str(exc),
            artifacts={"bench": "patcheval", "cve_id": cve_id, "patch": patch, "validation_type": "validation_fail"},
        )

    strict_success = validation_type == "Repair Success"
    if poc_passed is None:
        # Align with PatchEval native semantics: the bench swallows the
        # container-start exception (run_evaluation.py) and treats an
        # unstartable CVE container as validation_fail -> 0, a legitimate
        # negative sample. Returning FAILED here made RewardCommitter refuse
        # to write a reward, silently dropping these patch-producing
        # trajectories and starving RL of negative samples.
        return EvalResult(
            session_id=request.session_id,
            eval_id=spec.eval_id,
            method=spec.method.value,
            status=EvalStatus.SUCCEEDED.value,
            raw_score=0.0,
            normalized_score_10=0.0,
            reason="official PatchEval evaluator could not start the CVE container -- scored 0 (validation_fail)",
            error_text=_trim_log(poc_log or "unknown Docker evaluation error"),
            artifacts={
                "bench": "patcheval",
                "cve_id": cve_id,
                "patch": patch,
                "validation_type": "validation_fail",
                "poc_log": _trim_log(poc_log),
                "unit_test_log": _trim_log(unit_test_log),
            },
        )

    return EvalResult(
        session_id=request.session_id,
        eval_id=spec.eval_id,
        method=spec.method.value,
        status=EvalStatus.SUCCEEDED.value,
        raw_score=1.0 if strict_success else 0.0,
        normalized_score_10=10.0 if strict_success else 0.0,
        reason=(
            "Official PatchEval strict evaluation passed"
            if strict_success
            else f"Official PatchEval strict evaluation failed: {validation_type or 'unknown'}"
        ),
        artifacts={
            "bench": "patcheval",
            "official_evaluator": "evaluation/run_evaluation.py:Evaluation",
            "cve_id": cve_id,
            "setting": metrics.get("setting"),
            "strict_success": strict_success,
            "poc_passed": poc_passed is True,
            "unit_tests_passed": unit_tests_passed is True,
            "validation_type": validation_type,
            "poc_log": _trim_log(poc_log),
            "unit_test_log": _trim_log(unit_test_log),
            "protocol": metrics.get("protocol"),
            "patch": patch,
        },
    )


def _start_metrics(request: EvalRequest) -> dict[str, Any]:
    start_result = getattr(request, "start_result", None)
    metrics = getattr(start_result, "metrics", None)
    return dict(metrics) if isinstance(metrics, dict) else {}


def _trim_log(value: Any) -> str:
    return str(value or "")[-_MAX_LOG_CHARS:]


def _is_docker_unavailable(exc: BaseException) -> bool:
    """True when the exception indicates the Docker daemon is not reachable."""
    text = str(exc).lower()
    if "fetching server api version" in text:
        return True
    if "no such file or directory" in text and "docker" in text:
        return True
    walked = exc
    while walked is not None:
        cls = type(walked)
        if cls.__module__ == "docker.errors" or cls.__name__ == "DockerException":
            return True
        walked = walked.__cause__ or walked.__context__
    return False


def _fallback_from_runner_metrics(
    request: EvalRequest,
    spec: EvalSpec,
    metrics: dict[str, Any],
    exc: BaseException,
) -> EvalResult | None:
    """Build an EvalResult from the runner's in-pod official evaluation.

    The runner runs the official CVE evaluation (PoC + unit tests) inside the
    RJob pod with the real CVE image; its metrics carry strict_success,
    poc_passed, unit_tests_passed, etc. When the launcher cannot re-run that
    evaluation (no local Docker daemon), those metrics are the authoritative
    result.
    """
    if not isinstance(metrics, dict):
        return None
    if "strict_success" not in metrics or "poc_passed" not in metrics:
        return None
    strict_success = bool(metrics.get("strict_success"))
    score = 1.0 if strict_success else 0.0
    failure_stage = metrics.get("failure_stage")
    return EvalResult(
        session_id=request.session_id,
        eval_id=spec.eval_id,
        method=spec.method.value,
        status=EvalStatus.SUCCEEDED.value,
        raw_score=score,
        normalized_score_10=10.0 if strict_success else 0.0,
        reason=(
            "runner in-pod evaluation passed (launcher Docker unavailable)"
            if strict_success
            else f"runner in-pod evaluation did not pass: {failure_stage or 'patch not fixed'}"
        ),
        error_text=str(exc),
        artifacts={
            "bench": "patcheval",
            "cve_id": metrics.get("cve_id") or None,
            "setting": metrics.get("setting"),
            "eval_source": "runner_in_pod_fallback",
            "strict_success": strict_success,
            "poc_passed": metrics.get("poc_passed") is True,
            "unit_test_present": metrics.get("unit_test_present") is True,
            "unit_tests_passed": metrics.get("unit_tests_passed") is True,
            "failure_stage": failure_stage,
            "poc_log": _trim_log(metrics.get("poc_log")),
            "unit_test_log": _trim_log(metrics.get("unit_test_log")),
            "patch": metrics.get("patch"),
            "launcher_docker_error": str(exc),
        },
    )


def _load_official_evaluation(env_params: dict[str, Any]) -> type[Any]:
    global _ADAPTER_INSTALLED, _OFFICIAL_EVALUATION
    with _OFFICIAL_LOCK:
        shared_tmp = str(env_params.get("patcheval_shared_tmp") or "").strip()
        if shared_tmp:
            shared_tmp_path = Path(shared_tmp).expanduser().resolve()
            shared_tmp_path.mkdir(parents=True, exist_ok=True)
            os.environ["TMPDIR"] = str(shared_tmp_path)
            tempfile.tempdir = str(shared_tmp_path)
        official_root = Path(
            str(env_params.get("patcheval_official_root") or Path(__file__).parent / "PatchEval" / "patcheval")
        ).expanduser().resolve()
        evaluation_file = official_root / "evaluation" / "run_evaluation.py"
        if not evaluation_file.is_file():
            raise FileNotFoundError(f"official PatchEval evaluator not found: {evaluation_file}")

        if not _ADAPTER_INSTALLED:
            archive_dir = str(env_params.get("patcheval_image_archive_dir") or "").strip()
            adapter_file = Path(
                str(
                    env_params.get("patcheval_docker_adapter")
                    or Path(__file__).parent / "docker_archive_adapter" / "sitecustomize.py"
                )
            ).expanduser().resolve()
            if archive_dir:
                os.environ["PATCHEVAL_IMAGE_ARCHIVE_DIR"] = archive_dir
            os.environ["PATCHEVAL_CLEANUP_IMAGE"] = "0"
            for source, target in (
                ("patcheval_http_proxy", "PATCHEVAL_HTTP_PROXY"),
                ("patcheval_http_proxy", "PATCHEVAL_HTTPS_PROXY"),
                ("patcheval_no_proxy", "PATCHEVAL_NO_PROXY"),
            ):
                value = str(env_params.get(source) or "").strip()
                if value:
                    os.environ[target] = value
            if adapter_file.is_file() and archive_dir:
                adapter_spec = importlib.util.spec_from_file_location(
                    "safactory_patcheval_docker_adapter", adapter_file
                )
                if adapter_spec is None or adapter_spec.loader is None:
                    raise ImportError(f"cannot load PatchEval Docker adapter: {adapter_file}")
                adapter_module = importlib.util.module_from_spec(adapter_spec)
                adapter_spec.loader.exec_module(adapter_module)
            _ADAPTER_INSTALLED = True

        if _OFFICIAL_EVALUATION is None:
            evaluation_dir = str(evaluation_file.parent)
            if evaluation_dir not in sys.path:
                sys.path.insert(0, evaluation_dir)
            module_spec = importlib.util.spec_from_file_location(
                "safactory_official_patcheval_run_evaluation", evaluation_file
            )
            if module_spec is None or module_spec.loader is None:
                raise ImportError(f"cannot load official PatchEval evaluator: {evaluation_file}")
            module = importlib.util.module_from_spec(module_spec)
            module_spec.loader.exec_module(module)
            _OFFICIAL_EVALUATION = module.Evaluation
        return _OFFICIAL_EVALUATION
