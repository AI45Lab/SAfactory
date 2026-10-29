from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from rl.rollout_progress import RolloutProgress, checkpoint_path


class FakeManager:
    def __init__(self) -> None:
        self.envs = [
            {"id": 1, "env_id": "slow", "finished": False},
            {"id": 2, "env_id": "fast", "finished": True},
        ]
        self.scanned_after: list[int] = []

    async def list_environment_rows(
        self,
        *,
        after_id: int = 0,
        limit: int | None = None,
        env_ids: list[str] | None = None,
        finished: bool | None = None,
    ) -> list[dict]:
        if env_ids is None:
            self.scanned_after.append(after_id)
        rows = [row for row in self.envs if row["id"] > after_id]
        if env_ids is not None:
            rows = [row for row in rows if row["env_id"] in env_ids]
        if finished is not None:
            rows = [row for row in rows if row["finished"] == finished]
        return rows[:limit] if limit is not None else rows

    async def list_terminal_steps_for_sessions(self, env_ids: list[str]) -> list[dict]:
        return [{"env_id": env_id} for env_id in env_ids]


def test_late_finishing_env_survives_checkpoint_restart(tmp_path: Path) -> None:
    db = tmp_path / "training.db"
    db.touch()
    url = f"sqlite:///{db}"
    path = checkpoint_path(str(tmp_path), url)
    manager = FakeManager()
    state = RolloutProgress.load(path, db_url=url, job_session="job")

    assert asyncio.run(state.terminal_rows(manager, page_size=2)) == [
        {"env_id": "fast"}
    ]
    state.acknowledge("batch-1", {"fast"})

    manager.envs[0]["finished"] = True
    restored = RolloutProgress.load(path, db_url=url, job_session="job")
    assert restored.last_acked_batch == "batch-1"
    assert asyncio.run(restored.terminal_rows(manager, page_size=2)) == [
        {"env_id": "slow"}
    ]
    assert manager.scanned_after[-1] == 2


def test_recreated_database_discards_old_cursor(tmp_path: Path) -> None:
    db = tmp_path / "training.db"
    db.touch()
    url = f"sqlite:///{db}"
    path = checkpoint_path(str(tmp_path), url)
    state = RolloutProgress.load(path, db_url=url, job_session="job")
    state.cursor = 100
    state.pending.add("old")
    state.save()

    db.unlink()
    db.touch()
    restored = RolloutProgress.load(path, db_url=url, job_session="job")
    assert restored.cursor == 0
    assert restored.pending == set()


def test_restart_skips_only_preexisting_finished_envs(tmp_path: Path) -> None:
    db = tmp_path / "training.db"
    db.touch()
    url = f"sqlite:///{db}"
    state = RolloutProgress.load(
        checkpoint_path(str(tmp_path), url), db_url=url, job_session="job"
    )
    manager = FakeManager()

    asyncio.run(state.skip_existing_finished(manager, page_size=2))
    assert state.cursor == 2
    assert state.pending == {"slow"}

    manager.envs[0]["finished"] = True
    assert asyncio.run(state.terminal_rows(manager, page_size=2)) == [
        {"env_id": "slow"}
    ]


def test_terminal_read_failure_keeps_env_pending(tmp_path: Path) -> None:
    db = tmp_path / "training.db"
    db.touch()
    url = f"sqlite:///{db}"
    path = checkpoint_path(str(tmp_path), url)
    manager = FakeManager()

    async def fail_once(env_ids: list[str]) -> list[dict]:
        raise RuntimeError("temporary read failure")

    manager.list_terminal_steps_for_sessions = fail_once
    state = RolloutProgress.load(path, db_url=url, job_session="job")
    with pytest.raises(RuntimeError, match="temporary read failure"):
        asyncio.run(state.terminal_rows(manager, page_size=2))

    restored = RolloutProgress.load(path, db_url=url, job_session="job")
    assert restored.pending == {"slow", "fast"}
    manager.list_terminal_steps_for_sessions = lambda env_ids: asyncio.sleep(
        0, result=[{"env_id": env_id} for env_id in env_ids]
    )
    assert asyncio.run(restored.terminal_rows(manager, page_size=2)) == [
        {"env_id": "fast"}
    ]
