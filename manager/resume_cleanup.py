from __future__ import annotations

import asyncio
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List, Sequence

from clusters.rjob_cluster import RJobClusterBackend

from .episode_common import safe_path_part

log = logging.getLogger("manager.resume_cleanup")


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
    if not rows:
        log.info("resume cleanup skipped: job_id=%s unfinished=0", job_id)
        return []

    owned_backend = rjob_backend is None
    backend = rjob_backend or RJobClusterBackend(
        cluster_cfg=dict(manager_cfg.get("cluster") or {})
    )
    root = Path(results_root).expanduser().resolve(strict=False)
    if root.parent == root or not root.is_dir():
        raise ValueError(f"invalid resume results root: {root}")
    removed: List[Path] = []

    try:
        for row in rows:
            session_id = str(row.get("env_id") or "").strip()
            if not session_id:
                continue
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
        "resume artifact cleanup completed: job_id=%s root=%s unfinished=%d removed_paths=%d",
        job_id,
        root,
        len(rows),
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


def _remove_result_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
        return
    if path.is_dir():
        shutil.rmtree(path)
