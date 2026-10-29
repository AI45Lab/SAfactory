"""Durable scan progress for one rollout buffer run."""

from __future__ import annotations

import json
import os
import tempfile
import hashlib
from pathlib import Path


class RolloutProgress:
    def __init__(self, path: Path, *, db_url: str, job_session: str) -> None:
        self.path = path
        self.db_url = db_url
        self.db_identity = database_identity(db_url)
        self.job_session = job_session
        self.cursor = 0
        self.pending: set[str] = set()
        self.last_acked_batch: str | None = None
        self.loaded = False

    @classmethod
    def load(cls, path: Path, *, db_url: str, job_session: str) -> RolloutProgress:
        state = cls(path, db_url=db_url, job_session=job_session)
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            if (
                data.get("db_url") == db_url
                and data.get("db_identity") == state.db_identity
                and data.get("job_session") == job_session
            ):
                state.cursor = int(data["cursor"])
                state.pending = set(data["pending"])
                state.last_acked_batch = data.get("last_acked_batch")
                state.loaded = True
        return state

    async def skip_existing_finished(self, manager, *, page_size: int = 500) -> None:
        """Preserve restart_training behavior for a run without a checkpoint."""
        while True:
            page = await manager.list_environment_rows(
                after_id=self.cursor, limit=page_size
            )
            if not page:
                break
            for env in page:
                if not env.get("finished") and env.get("env_id"):
                    self.pending.add(str(env["env_id"]))
                self.cursor = max(self.cursor, int(env["id"]))
            self.save()
            if len(page) < page_size:
                break

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "db_url": self.db_url,
            "db_identity": self.db_identity,
            "job_session": self.job_session,
            "cursor": self.cursor,
            "pending": sorted(self.pending),
            "last_acked_batch": self.last_acked_batch,
        }
        fd, temp_path = tempfile.mkstemp(
            prefix=".rollout-progress-", dir=self.path.parent
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, self.path)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def acknowledge(self, batch_id: str, env_ids: set[str]) -> None:
        previous_pending = self.pending.copy()
        previous_batch = self.last_acked_batch
        self.pending.difference_update(env_ids)
        self.last_acked_batch = batch_id
        try:
            self.save()
        except Exception:
            self.pending = previous_pending
            self.last_acked_batch = previous_batch
            raise

    async def terminal_rows(self, manager, *, page_size: int = 500) -> list[dict]:
        """Fetch terminal rows for completed, unacknowledged environments."""
        while True:
            page = await manager.list_environment_rows(
                after_id=self.cursor, limit=page_size
            )
            if not page:
                break
            for env in page:
                env_id = str(env.get("env_id") or "")
                if env_id:
                    self.pending.add(env_id)
                self.cursor = max(self.cursor, int(env["id"]))
            self.save()
            if len(page) < page_size:
                break

        finished_ids: list[str] = []
        pending_ids = sorted(self.pending)
        for offset in range(0, len(pending_ids), page_size):
            envs = await manager.list_environment_rows(
                env_ids=pending_ids[offset : offset + page_size], finished=True
            )
            finished_ids.extend(str(env["env_id"]) for env in envs if env.get("env_id"))

        rows: list[dict] = []
        for offset in range(0, len(finished_ids), page_size):
            rows.extend(
                await manager.list_terminal_steps_for_sessions(
                    finished_ids[offset : offset + page_size]
                )
            )
        return rows


def database_identity(db_url: str) -> str | None:
    if not db_url.startswith("sqlite:///"):
        return None
    try:
        stat = os.stat(db_url[len("sqlite:///") :])
    except OSError:
        return None
    return f"{stat.st_dev}:{stat.st_ino}"


def checkpoint_path(run_dir: str, db_url: str) -> Path:
    key = hashlib.sha256(db_url.encode("utf-8")).hexdigest()[:16]
    return Path(run_dir) / f"rollout-progress-{key}.json"
