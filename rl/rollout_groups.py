from __future__ import annotations

from typing import Any, Dict, List, MutableMapping, Sequence, Tuple


RolloutItem = Dict[str, Any]
ReadyGroup = Tuple[str, List[RolloutItem]]


def accumulate_group_batch(
    pending: MutableMapping[str, List[RolloutItem]],
    new_items: Sequence[RolloutItem],
    *,
    group_size: int,
    required_groups: int,
) -> tuple[List[ReadyGroup], List[str]]:
    """Pop one group per instance only after a full group batch is ready."""
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    if required_groups <= 0:
        return [], []

    for item in new_items:
        instance_id = str(item.get("instance_id", ""))
        if instance_id:
            pending.setdefault(instance_id, []).append(item)

    ready_instance_ids = [
        instance_id
        for instance_id, bucket in pending.items()
        if len(bucket) >= group_size
    ]
    if len(ready_instance_ids) < required_groups:
        return [], []

    ready_groups: List[ReadyGroup] = []
    selected_ids = ready_instance_ids[:required_groups]
    for instance_id in selected_ids:
        bucket = pending[instance_id]
        ready_groups.append((instance_id, list(bucket[:group_size])))
        del bucket[:group_size]
        if not bucket:
            pending.pop(instance_id, None)

    return ready_groups, selected_ids
