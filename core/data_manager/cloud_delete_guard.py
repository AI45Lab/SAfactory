"""Fail-closed safeguards for destructive Cloud landing-table operations."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


log = logging.getLogger("core.data_manager.cloud_delete_guard")


class CloudDestructiveOperationError(RuntimeError):
    """Raised when a Cloud delete has not passed every required safety check."""


class CloudDeleteGuard:
    """Target preflight and verified archive for Cloud landing deletes."""

    def __init__(
        self,
        *,
        client: Any,
        db_uri: str,
        landing_table: str,
        confirmed_job_id: str = "",
        confirm_production: bool = False,
        archive_dir: str = "",
    ) -> None:
        self.client = client
        self.db_uri = str(db_uri or "").strip()
        self.landing_table = str(landing_table or "").strip()
        self.confirmed_job_id = str(confirmed_job_id or "").strip()
        self.confirm_production = bool(confirm_production)
        self.archive_dir = str(archive_dir or "").strip()

    async def preflight(
        self,
        *,
        operation: str,
        job_id: str,
        landing_filter: str,
        environment_rows: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """Verify the exact landing target and archive before deletion."""
        normalized_job_id = str(job_id or "").strip()
        if not normalized_job_id:
            raise CloudDestructiveOperationError(
                f"{operation} requires an exact job_id for a partition-scoped cloud delete"
            )
        if not self.db_uri or not self.landing_table:
            raise CloudDestructiveOperationError(
                f"{operation} refused because the Cloud DB URI and landing table "
                "could not both be resolved explicitly"
            )

        landing_rows = _rows_as_dicts(await asyncio.to_thread(
            self.client.query_data,
            filter_query=landing_filter,
            limit=None,
            partition=normalized_job_id,
            checkout_latest=True,
            deserialize_json=False,
            table=self.landing_table,
        ))
        profile = str(os.environ.get("WT_SDK_PROFILE") or "").strip().lower()
        production = (
            profile in {"prod", "production"}
            or self.landing_table == "wind_tunnel_landing"
        )
        log.warning(
            "Cloud delete preflight: operation=%s profile=%s db_uri=%s "
            "landing_table=%s job_id=%s landing_rows=%d filter=%s",
            operation,
            profile or "<default>",
            self.db_uri,
            self.landing_table,
            normalized_job_id,
            len(landing_rows),
            landing_filter,
        )

        if self.confirmed_job_id != normalized_job_id:
            raise CloudDestructiveOperationError(
                f"{operation} refused for cloud job_id={normalized_job_id!r}; pass "
                f"--confirm-cloud-delete-job-id {normalized_job_id} after reviewing "
                f"db_uri={self.db_uri!r}, landing_table={self.landing_table!r}, "
                "and the preflight counts"
            )
        if production and not self.confirm_production:
            raise CloudDestructiveOperationError(
                f"{operation} refused for production target "
                f"{self.landing_table!r}; --confirm-production is required"
            )
        if production and not self.archive_dir:
            raise CloudDestructiveOperationError(
                f"{operation} refused for production: --cloud-delete-archive-dir "
                "is required and must point to durable storage"
            )
        if self.archive_dir:
            archive_path = self._write_verified_archive(
                operation=operation,
                job_id=normalized_job_id,
                landing_filter=landing_filter,
                landing_rows=landing_rows,
                environment_rows=environment_rows or [],
                profile=profile,
            )
            log.warning("Cloud delete archive verified: %s", archive_path)
        return landing_rows

    def _write_verified_archive(
        self,
        *,
        operation: str,
        job_id: str,
        landing_filter: str,
        landing_rows: List[Dict[str, Any]],
        environment_rows: List[Dict[str, Any]],
        profile: str,
    ) -> Path:
        archive_dir = Path(self.archive_dir).expanduser()
        archive_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        job_digest = hashlib.sha256(job_id.encode("utf-8")).hexdigest()[:16]
        archive_path = archive_dir / f"{timestamp}-{operation}-{job_digest}.json"
        archive_data = {
            "operation": operation,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "profile": profile or None,
            "db_uri": self.db_uri,
            "landing_table": self.landing_table,
            "job_id": job_id,
            "landing_filter": landing_filter,
            "landing_rows": landing_rows,
            "environment_rows": environment_rows,
        }
        canonical = json.dumps(
            archive_data,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        checksum = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        archive_fd = os.open(
            archive_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        with os.fdopen(archive_fd, "w", encoding="utf-8") as handle:
            json.dump(
                {"sha256": checksum, "data": archive_data},
                handle,
                ensure_ascii=False,
                default=str,
            )
            handle.flush()
            os.fsync(handle.fileno())
        directory_fd = os.open(archive_dir, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)

        with archive_path.open("r", encoding="utf-8") as handle:
            verified = json.load(handle)
        verified_canonical = json.dumps(
            verified.get("data"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        verified_checksum = hashlib.sha256(verified_canonical.encode("utf-8")).hexdigest()
        if verified.get("sha256") != checksum or verified_checksum != checksum:
            raise CloudDestructiveOperationError(
                f"cloud delete archive verification failed: {archive_path}"
            )
        if len(verified["data"].get("landing_rows") or []) != len(landing_rows):
            raise CloudDestructiveOperationError(
                f"cloud delete archive row-count verification failed: {archive_path}"
            )
        return archive_path


def _rows_as_dicts(value: Any) -> List[Dict[str, Any]]:
    if value is None:
        return []
    if hasattr(value, "to_dict"):
        try:
            value = value.to_dict(orient="records")
        except TypeError:
            value = value.to_dict()
    if isinstance(value, dict):
        return [dict(value)]
    return [dict(row) for row in value]
