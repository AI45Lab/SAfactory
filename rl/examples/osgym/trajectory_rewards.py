"""OSGym trajectory-level reward processing for Slime."""

from collections import OrderedDict
from typing import Any, Dict, List, Tuple


def _task_key(sample, fallback_index: int) -> Any:
    group_index = getattr(sample, "group_index", None)
    if group_index is not None:
        return group_index

    metadata = getattr(sample, "metadata", None) or {}
    return metadata.get("group_id", fallback_index)


def _session_key(sample) -> Any:
    metadata = getattr(sample, "metadata", None) or {}
    session_id = metadata.get("session_id")
    if session_id:
        return session_id

    sample_index = getattr(sample, "index", None)
    if isinstance(sample_index, str) and ":" in sample_index:
        return sample_index.rsplit(":", 1)[0]

    raise ValueError(
        "OSGym trajectory-level GRPO requires session_id in sample.metadata "
        "or a '<session_id>:<step_id>' sample index"
    )


def _has_invalid_action(sample) -> bool:
    metadata = getattr(sample, "metadata", None) or {}
    if bool(metadata.get("invalid_action", False)):
        return True

    validation = metadata.get("action_validation")
    if not isinstance(validation, dict):
        return False
    return bool(
        validation.get("valid") is False
        or validation.get("syntax_invalid")
        or validation.get("semantic_invalid")
        or validation.get("parser_failed")
        or validation.get("invalid_reasons")
    )


def _mask_invalid_action_advantages(
    samples,
    advantages: List[float],
) -> List[float]:
    masked = list(advantages)
    for index, sample in enumerate(samples):
        if _has_invalid_action(sample):
            masked[index] = 0.0
    return masked


def _trajectory_reward(
    samples,
    indices: List[int],
    sample_rewards: List[float],
    task_key: Any,
    session_id: Any,
) -> float:
    terminal_rewards = []
    for index in indices:
        metadata = getattr(samples[index], "metadata", None) or {}
        if "terminal_reward" in metadata:
            terminal_rewards.append(float(metadata["terminal_reward"]))

    rewards = terminal_rewards or [sample_rewards[index] for index in indices]
    reference_reward = rewards[0]
    if any(abs(reward - reference_reward) > 1e-8 for reward in rewards[1:]):
        reward_kind = "terminal_reward metadata" if terminal_rewards else "Sample rewards"
        raise ValueError(
            f"Expected one trajectory reward in {reward_kind}, but "
            f"task={task_key!r}, session={session_id!r} has rewards={rewards}"
        )
    return reference_reward


def post_process_rewards(args, samples) -> Tuple[List[float], List[float]]:
    """Return trajectory-relative rewards and length-balanced step advantages.

    OSGym emits one Slime ``Sample`` per action step. For each task, this
    processor computes one advantage per trajectory, divides it equally among
    that trajectory's action steps, and broadcasts the result back to Slime.
    """
    sample_rewards = [float(sample.get_reward_value(args)) for sample in samples]
    if not (
        args.advantage_estimator
        in ["grpo", "gspo", "reinforce_plus_plus_baseline"]
        and args.rewards_normalization
    ):
        return sample_rewards, _mask_invalid_action_advantages(samples, sample_rewards)

    task_groups: Dict[Any, List[int]] = OrderedDict()
    for index, sample in enumerate(samples):
        if getattr(sample, "remove_sample", False):
            continue
        task_groups.setdefault(_task_key(sample, index), []).append(index)

    raw_rewards = list(sample_rewards)
    step_advantages = [0.0] * len(samples)
    expected_trajectories = getattr(args, "n_samples_per_prompt", None)

    for task_key, task_indices in task_groups.items():
        trajectories: Dict[Any, List[int]] = OrderedDict()
        for index in task_indices:
            trajectories.setdefault(_session_key(samples[index]), []).append(index)

        if (
            expected_trajectories is not None
            and len(trajectories) != expected_trajectories
        ):
            raise ValueError(
                "Incomplete OSGym GRPO task group before reward normalization: "
                f"task={task_key!r}, trajectories={len(trajectories)}, "
                f"expected={expected_trajectories}."
            )

        trajectory_rewards: Dict[Any, float] = OrderedDict()
        for session_id, indices in trajectories.items():
            completion_markers = [
                bool(
                    (getattr(samples[index], "metadata", None) or {}).get(
                        "is_session_completed", False
                    )
                )
                for index in indices
                if "is_session_completed"
                in (getattr(samples[index], "metadata", None) or {})
            ]
            if completion_markers and sum(completion_markers) != 1:
                raise ValueError(
                    "Incomplete OSGym trajectory before reward normalization: "
                    f"task={task_key!r}, session={session_id!r}, "
                    f"terminal_rows={sum(completion_markers)}."
                )

            trajectory_reward = _trajectory_reward(
                samples,
                indices,
                sample_rewards,
                task_key,
                session_id,
            )
            trajectory_rewards[session_id] = trajectory_reward
            for index in indices:
                raw_rewards[index] = trajectory_reward

        task_mean = sum(trajectory_rewards.values()) / len(trajectory_rewards)
        for session_id, indices in trajectories.items():
            trajectory_advantage = trajectory_rewards[session_id] - task_mean
            step_advantage = trajectory_advantage / len(indices)
            for index in indices:
                step_advantages[index] = step_advantage

    return raw_rewards, _mask_invalid_action_advantages(samples, step_advantages)
