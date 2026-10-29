from __future__ import annotations

from rl.rollout_groups import accumulate_group_batch


def _items(instance_id: str, count: int) -> list[dict]:
    return [{"instance_id": instance_id, "sample": index} for index in range(count)]


def test_oversampled_instance_counts_as_only_one_ready_group() -> None:
    pending: dict[str, list[dict]] = {}

    groups, finished = accumulate_group_batch(
        pending,
        _items("task-a", 24),
        group_size=12,
        required_groups=2,
    )

    assert groups == []
    assert finished == []
    assert len(pending["task-a"]) == 24


def test_full_batch_pops_one_group_per_instance_and_keeps_surplus() -> None:
    pending = {"task-a": _items("task-a", 24)}

    groups, finished = accumulate_group_batch(
        pending,
        _items("task-b", 12),
        group_size=12,
        required_groups=2,
    )

    assert [len(group) for _, group in groups] == [12, 12]
    assert finished == ["task-a", "task-b"]
    assert len(pending["task-a"]) == 12
    assert "task-b" not in pending
