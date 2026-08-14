from __future__ import annotations

import asyncio
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List

from clusters.rjob_cluster import RJobClusterBackend

from .episode_common import result_session_dir_candidates

log = logging.getLogger("manager.resume_cleanup")


async def cleanup_resume_artifacts(
    *,
    job_id: str,
    model: str,
    data_manager: Any,
    manager_cfg: Dict[str, Any],
    rjob_backend: RJobClusterBackend | None = None,
) -> List[Path]:
    """Remove stale RJobs and result paths for unfinished resume sessions."""
    rows = await data_manager.get_all_environments(job_id)
    owned_backend = rjob_backend is None
    backend = rjob_backend or RJobClusterBackend(
        cluster_cfg=dict(manager_cfg.get("cluster") or {})
    )
    removed: List[Path] = []

    try:
        for row in rows:
            if _truthy(row.get("finished")) or _truthy(row.get("is_deleted")):
                continue

            session_id = str(row.get("env_id") or "").strip()
            if not session_id:
                continue
            env_params = row.get("env_params") if isinstance(row.get("env_params"), dict) else {}
            result_paths = [
                path
                for path in result_session_dir_candidates(
                    job_id=job_id,
                    session_id=session_id,
                    env_params=env_params,
                )
                if path.exists() or path.is_symlink()
            ]
            if not result_paths:
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
            for path in result_paths:
                await asyncio.to_thread(_remove_result_path, path)
                removed.append(path)
            log.info(
                "resume cleanup completed: job_id=%s session_id=%s rjobs=%s result_paths=%s",
                job_id,
                session_id,
                cleaned_jobs,
                [str(path) for path in result_paths],
            )
    finally:
        if owned_backend:
            await backend.close()

    log.info(
        "resume result preflight completed: job_id=%s unfinished=%d removed_paths=%d",
        job_id,
        sum(
            1
            for row in rows
            if not _truthy(row.get("finished")) and not _truthy(row.get("is_deleted"))
        ),
        len(removed),
    )
    return removed


def _remove_result_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
        return
    if path.is_dir():
        shutil.rmtree(path)


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)
