from collections import OrderedDict
import math
from typing import Any, Dict, List, TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    from slime.utils.types import Sample


def _group_key(sample: "Sample", fallback_index: int) -> Any:
    group_index = getattr(sample, "group_index", None)
    if group_index is not None:
        return group_index

    metadata = getattr(sample, "metadata", None) or {}
    return metadata.get("group_id", fallback_index)


def post_process_rewards(args, samples: List["Sample"]) -> Tuple[List[float], List[float]]:
    """Normalize rewards by rollout group for message_cut variable-size groups."""
    raw_rewards = [sample.get_reward_value(args) for sample in samples]
    if not (
        args.advantage_estimator in ["grpo", "gspo", "reinforce_plus_plus_baseline"]
        and args.rewards_normalization
    ):
        return raw_rewards, raw_rewards

    groups: Dict[Any, List[int]] = OrderedDict()
    for i, sample in enumerate(samples):
        groups.setdefault(_group_key(sample, i), []).append(i)

    rewards = [0.0] * len(samples)
    for indices in groups.values():
        group_rewards = [float(raw_rewards[i]) for i in indices]
        mean = sum(group_rewards) / len(group_rewards)
        normalized = [reward - mean for reward in group_rewards]

        if args.advantage_estimator in ["grpo", "gspo"] and args.grpo_std_normalization and len(indices) > 1:
            variance = sum((reward - mean) ** 2 for reward in group_rewards) / (len(group_rewards) - 1)
            std = math.sqrt(variance)
            normalized = [reward / (std + 1e-6) for reward in normalized]

        for i, reward in zip(indices, normalized, strict=False):
            rewards[i] = reward

    return raw_rewards, rewards
