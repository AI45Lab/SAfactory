from __future__ import annotations

import asyncio
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List, Sequence

from clusters.rjob_cluster import RJobClusterBackend

from .episode_common import safe_path_part

log = logging.getLogger("manager.resume_cleanup")


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
