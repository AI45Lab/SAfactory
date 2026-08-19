"""Cross-process claim held while one launcher initializes environment rows."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import socket
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class JobInitializationClaim:
    path: Path
    file_descriptor: int
    owner_token: str
    job_id: str
    _released: bool = False

    def release(self) -> None:
        if self._released:
            return
        try:
            _write_claim_state(
                self.file_descriptor,
                {
                    "state": "released",
                    "job_id": self.job_id,
                    "owner_token": self.owner_token,
                    "pid": os.getpid(),
                    "host": socket.gethostname(),
                    "released_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            fcntl.flock(self.file_descriptor, fcntl.LOCK_UN)
        finally:
            os.close(self.file_descriptor)
            self._released = True


def acquire_job_initialization_claim(
    *,
    job_id: str,
    storage_type: str,
    storage_identity: str,
    claim_dir: str = "",
) -> JobInitializationClaim:
    """Acquire a non-blocking OS lease visible to every launcher using the path."""
    normalized_job_id = str(job_id or "").strip()
    if not normalized_job_id:
        raise ValueError("job initialization claim requires a non-empty job_id")
    normalized_storage = str(storage_type or "").strip().lower()
    configured_dir = str(claim_dir or "").strip()
    if normalized_storage == "cloud":
        configured_dir = configured_dir or str(
            os.environ.get("SAFACTORY_CLOUD_JOB_CLAIM_DIR") or ""
        ).strip()
        if not configured_dir:
            raise RuntimeError(
                "Cloud environment initialization requires --cloud-job-claim-dir "
                "or SAFACTORY_CLOUD_JOB_CLAIM_DIR pointing to a durable shared "
                "filesystem visible to every launcher"
            )
        # EnvConfigManager allocates physical IDs from the whole config table,
        # so every writer for the same store must share one claim even when the
        # logical job IDs differ.
        identity_digest = hashlib.sha256(
            str(storage_identity or normalized_storage).encode("utf-8")
        ).hexdigest()[:16]
        scope = f"cloud-environment-config-{identity_digest}"
    else:
        configured_dir = configured_dir or str(
            os.environ.get("SAFACTORY_JOB_CLAIM_DIR") or ".safactory-locks"
        ).strip()
        identity_digest = hashlib.sha256(
            str(storage_identity or normalized_storage).encode("utf-8")
        ).hexdigest()[:16]
        job_digest = hashlib.sha256(normalized_job_id.encode("utf-8")).hexdigest()[:16]
        scope = f"sqlite-{identity_digest}-{job_digest}"

    root = Path(configured_dir).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{scope}.lock"
    file_descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(file_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        owner = _read_claim_state(file_descriptor)
        os.close(file_descriptor)
        raise RuntimeError(
            "environment initialization is already claimed by another launcher: "
            f"job_id={normalized_job_id!r} claim={path} owner={owner}"
        ) from exc

    owner_token = uuid.uuid4().hex
    _write_claim_state(
        file_descriptor,
        {
            "state": "held",
            "job_id": normalized_job_id,
            "owner_token": owner_token,
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "storage_type": normalized_storage,
            "storage_identity": storage_identity,
            "acquired_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    return JobInitializationClaim(
        path=path,
        file_descriptor=file_descriptor,
        owner_token=owner_token,
        job_id=normalized_job_id,
    )


def _write_claim_state(file_descriptor: int, payload: dict) -> None:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    os.lseek(file_descriptor, 0, os.SEEK_SET)
    os.ftruncate(file_descriptor, 0)
    written = 0
    while written < len(encoded):
        written += os.write(file_descriptor, encoded[written:])
    os.fsync(file_descriptor)


def _read_claim_state(file_descriptor: int) -> dict:
    try:
        os.lseek(file_descriptor, 0, os.SEEK_SET)
        raw = os.read(file_descriptor, 16 * 1024)
        value = json.loads(raw.decode("utf-8")) if raw else {}
        return value if isinstance(value, dict) else {"raw": value}
    except Exception:
        return {"state": "unknown"}
