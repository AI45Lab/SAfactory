from __future__ import annotations

import asyncio
import json
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List, Sequence

from clusters.rjob_cluster import RJobClusterBackend
from clusters.sandbox_cluster import SandboxClusterBackend
from core.result_artifact import _finite_float, _reject_constant, _unique_object
from evaluator.gateway_client import GatewayClient, is_durably_closed

from .episode_common import safe_path_part

log = logging.getLogger("manager.resume_cleanup")


async def prepare_sandbox_resume(
    *, cfg: Any, manager_cfg: Dict[str, Any],
    environment_rows: Sequence[Dict[str, Any]], data_manager: Any,
) -> tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """Reconcile writers before deleting any trajectory; preserve recoverable sessions."""
    backend = SandboxClusterBackend(cluster_cfg=dict(manager_cfg.get("cluster") or {}))
    client = GatewayClient(
        gateway_base_url=cfg.gateway_base_url,
        close_timeout_s=cfg.gateway_close_timeout_s,
        retry_backoff_s=cfg.gateway_close_retry_backoff_s,
    )
    try:
        results_root = Path(cfg.sandbox_config["results"]["local_root"])
        gateway_states: Dict[str, Any] = {}
        for row in environment_rows:
            session_id = str(row.get("env_id") or row.get("session_id") or "")
            path = results_root / safe_path_part(cfg.job_id) / safe_path_part(session_id) / "runtime.json"
            if not path.is_file():
                continue
            try:
                reference = json.loads(
                    path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object,
                    parse_constant=_reject_constant, parse_float=_finite_float,
                )
            except (OSError, ValueError) as exc:
                raise RuntimeError(f"Cannot safely resume {session_id}: invalid runtime reference JSON") from exc
            if not isinstance(reference, dict) or any(
                reference.get(key) != expected
                for key, expected in (("job_id", cfg.job_id), ("session_id", session_id))
            ):
                raise RuntimeError(f"Cannot safely resume {session_id}: runtime reference identity mismatch")
            attempt = reference.get("attempt_id")
            if not isinstance(attempt, str) or not attempt or safe_path_part(attempt) != attempt:
                raise RuntimeError(f"Cannot safely resume {session_id}: invalid runtime reference attempt_id")
            receipt = reference.get("gateway_finalized")
            if receipt is not None and not isinstance(receipt, bool):
                raise RuntimeError(f"Cannot safely resume {session_id}: invalid Gateway finalization receipt")
            if receipt is False:
                raise RuntimeError(f"Cannot safely resume {session_id}: Gateway finalization previously failed")
            if receipt is not True:
                expected_epoch = (manager_cfg.get("cluster") or {}).get("sandbox", {}).get("gateway_instance_id")
                epoch = reference.get("gateway_instance_id")
                if not isinstance(epoch, str) or not epoch or epoch != expected_epoch:
                    raise RuntimeError(f"Cannot safely resume {session_id}: Gateway instance changed without a durable receipt")
                gateway_states[session_id] = await client.get_session_status(session_id)
                if gateway_states[session_id] is None:
                    raise RuntimeError(
                        f"Cannot safely resume {session_id}: no durable Gateway finalization receipt or live binding"
                    )
                if gateway_states[session_id].get("instance_id") != epoch:
                    raise RuntimeError(f"Cannot safely resume {session_id}: live Gateway instance changed without a durable receipt")
        rerun, recovery = await backend.prepare_resume(environment_rows)
        for row in rerun:
            session_id = str(row.get("env_id") or row.get("session_id") or "")
            reference = results_root / safe_path_part(cfg.job_id) / safe_path_part(session_id) / "runtime.json"
            if not reference.is_file() and await data_manager.list_session_steps(
                session_id, job_id=cfg.job_id, limit=1,
            ):
                raise RuntimeError(f"Cannot safely resume {session_id}: trajectory exists without a runtime reference")
            # The backend has stopped the writer; drain accepted Gateway writes too.
            if session_id not in gateway_states:
                gateway_states[session_id] = await client.get_session_status(session_id)
            if gateway_states[session_id] is not None:
                closed = await client.close_session(session_id, reason="system_error", completion_mode="abort")
                if not is_durably_closed(closed, session_id=session_id):
                    raise RuntimeError(f"Cannot reset undrained Sandbox trajectory: {session_id}")
                epoch = gateway_states[session_id].get("instance_id")
                if epoch is not None and closed.get("instance_id") != epoch:
                    raise RuntimeError(f"Cannot reset Sandbox trajectory after Gateway instance changed: {session_id}")
        return rerun, recovery
    finally:
        for outcome in await asyncio.gather(client.aclose(), backend.close(), return_exceptions=True):
            if isinstance(outcome, Exception):
                log.warning("Sandbox resume connection cleanup failed: %s", outcome)


def select_generated_resume_environments(
    *,
    job_id: str,
    results_root: Path,
    environment_rows: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Return unique unfinished environments that have generated result paths."""
    root = _resolve_results_root(results_root)
    targets: Dict[Path, Dict[str, Any]] = {}
    for row in environment_rows:
        session_id = str(row.get("env_id") or "").strip()
        if not session_id:
            continue
        result_path = root / safe_path_part(job_id) / safe_path_part(session_id)
        if not (result_path.exists() or result_path.is_symlink()):
            continue
        existing = targets.get(result_path)
        if existing is not None:
            existing_session_id = str(existing.get("env_id") or "").strip()
            if existing_session_id != session_id:
                raise RuntimeError(
                    "resume result path collision: "
                    f"{existing_session_id!r} and {session_id!r} map to {result_path}"
                )
            continue
        target = dict(row)
        target["job_id"] = job_id
        target["env_id"] = session_id
        targets[result_path] = target
    return list(targets.values())


async def cleanup_resume_artifacts(
    *,
    job_id: str,
    model: str,
    data_manager: Any,
    manager_cfg: Dict[str, Any],
    results_root: Path,
    environment_rows: Sequence[Dict[str, Any]] | None = None,
    rjob_backend: RJobClusterBackend | None = None,
) -> List[Path]:
    """Remove stale RJobs and result paths for unfinished resume sessions."""
    rows = (
        list(environment_rows)
        if environment_rows is not None
        else await _load_environment_refs(data_manager, job_id=job_id)
    )
    root = _resolve_results_root(results_root)
    targets = select_generated_resume_environments(
        job_id=job_id,
        results_root=root,
        environment_rows=rows,
    )
    if not targets:
        log.info(
            "resume cleanup skipped: job_id=%s unfinished=%d generated=0",
            job_id,
            len(rows),
        )
        return []

    owned_backend = rjob_backend is None
    backend = rjob_backend or RJobClusterBackend(
        cluster_cfg=dict(manager_cfg.get("cluster") or {})
    )
    removed: List[Path] = []

    try:
        for row in targets:
            session_id = str(row.get("env_id") or "").strip()
            agent_name = str(row.get("env_name") or "").strip()
            if not agent_name:
                raise RuntimeError(
                    f"cannot clean resume artifacts for session {session_id}: env_name is missing"
                )

            cleaned_jobs = await backend.cleanup_resume_session(
                agent_name=agent_name,
                model=model,
                job_id=job_id,
                session_id=session_id,
            )
            result_path = (
                root
                / safe_path_part(job_id)
                / safe_path_part(session_id)
            )
            if result_path.exists() or result_path.is_symlink():
                await asyncio.to_thread(_remove_result_path, result_path)
                removed.append(result_path)
            log.info(
                "resume cleanup completed: job_id=%s session_id=%s rjobs=%s result_path=%s",
                job_id,
                session_id,
                cleaned_jobs,
                str(result_path),
            )
    finally:
        if owned_backend:
            await backend.close()

    log.info(
        "resume artifact cleanup completed: job_id=%s root=%s unfinished=%d "
        "generated=%d removed_paths=%d",
        job_id,
        root,
        len(rows),
        len(targets),
        len(removed),
    )
    return removed


async def _load_environment_refs(data_manager: Any, *, job_id: str) -> List[Dict[str, Any]]:
    """Lightweight path for direct callers; the launcher reuses rows it already read."""
    return await data_manager.list_environment_refs(
        job_id=job_id,
        finished=False,
        is_deleted=False,
    )


def _resolve_results_root(results_root: Path) -> Path:
    root = Path(results_root).expanduser().resolve(strict=False)
    if root.parent == root or not root.is_dir():
        raise ValueError(f"invalid resume results root: {root}")
    return root


def _remove_result_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
        return
    if path.is_dir():
        shutil.rmtree(path)
