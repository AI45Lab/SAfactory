"""Fail-closed safeguards for destructive Cloud landing-table operations."""

from __future__ import annotations

import asyncio
import logging
import os
import re
from typing import Any


log = logging.getLogger("core.data_manager.cloud_delete_guard")

# Accept the predicates emitted by CloudStrategy's delete methods, rather than
# trying to prove the safety of arbitrary SQL. Quoted values may contain AND,
# OR, or apostrophes escaped as two single quotes.
_SQL_STRING = r"'(?:''|[^'])*'"
_DELETE_FILTER = re.compile(
    rf"\s*job_id\s*=\s*(?P<job>{_SQL_STRING})"
    rf"(?:\s+AND\s+(?:id|session_id)\s+IN\s*\(\s*{_SQL_STRING}"
    rf"(?:\s*,\s*{_SQL_STRING})*\s*\))*\s*",
    re.IGNORECASE,
)


class CloudDestructiveOperationError(RuntimeError):
    """Raised when a Cloud delete has not passed every required safety check."""


class CloudDeleteGuard:
    """Target preflight for Cloud landing deletes."""

    def __init__(
        self,
        *,
        client: Any,
        db_uri: str,
        landing_table: str,
        profile: str | None = None,
        confirmed_job_id: str = "",
        confirm_production: bool = False,
    ) -> None:
        self.client = client
        self.db_uri = str(db_uri or "").strip()
        self.landing_table = str(landing_table or "").strip()
        self.profile = profile
        self.confirmed_job_id = str(confirmed_job_id or "").strip()
        self.confirm_production = bool(confirm_production)

    async def preflight(
        self,
        *,
        operation: str,
        job_id: str,
        landing_filter: str,
    ) -> int:
        """Verify the exact target and count matching rows using only their ids."""
        normalized_job_id = str(job_id or "").strip()
        if not normalized_job_id:
            raise CloudDestructiveOperationError(
                f"{operation} requires an exact job_id for a job-scoped cloud delete"
            )
        matched = _DELETE_FILTER.fullmatch(str(landing_filter or ""))
        if matched is None or matched.group("job")[1:-1].replace("''", "'") != normalized_job_id:
            raise CloudDestructiveOperationError(
                f"{operation} requires an exact job_id equality filter matching "
                f"{normalized_job_id!r}, optionally narrowed by id/session_id IN predicates"
            )
        if not self.db_uri or not self.landing_table:
            raise CloudDestructiveOperationError(
                f"{operation} refused because the Cloud DB URI and landing table "
                "could not both be resolved explicitly"
            )

        profile = str(
            self.profile if self.profile is not None else os.environ.get("WT_SDK_PROFILE") or ""
        ).strip().lower()
        production = (
            profile in {"prod", "production"}
            or self.landing_table in {"wind_tunnel_landing", "wt_landing"}
        )
        landing_row_count = _row_count(await asyncio.to_thread(
            self.client.query_data,
            filter_query=landing_filter,
            limit=None,
            columns=["id"],
            checkout_latest=True,
            deserialize_json=False,
            table=self.landing_table,
        ))
        log.warning(
            "Cloud delete preflight: operation=%s profile=%s db_uri=%s "
            "landing_table=%s job_id=%s landing_rows=%d filter=%s",
            operation,
            profile or "<default>",
            self.db_uri,
            self.landing_table,
            normalized_job_id,
            landing_row_count,
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
        return landing_row_count


def _row_count(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, dict):
        return 1
    try:
        return len(value)
    except TypeError:
        return sum(1 for _ in value)
