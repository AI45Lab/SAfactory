"""Monkey-patch slime's process_rollout_data to truncate long trajectories.

Long agent trajectories (e.g. 40-step CVE patcheval) can exceed 50k tokens,
which OOMs the training GPU. This patch truncates each trajectory to the
last ``TRAJ_TRUNCATION_MAX_SEQ_LEN`` tokens *for training only* — the full
trajectory is still used for reward/advantage computation during rollout.

The truncation keeps the most recent context + response tokens, preserving
the final actions that are most relevant for learning. The GRPO advantage
is group-relative and computed on the full trajectory, so it remains correct
even after truncation.

Configure via env var ``TRAJ_TRUNCATION_MAX_SEQ_LEN`` (default  8192).
Set to 0 or empty to disable.
"""
import os

# Read config at import time so it's picked up from the Ray runtime env.
TRAJ_TRUNCATION_MAX_SEQ_LEN = int(os.environ.get("TRAJ_TRUNCATION_MAX_SEQ_LEN", "8192"))


def _apply_truncation(rollout_data):
    """Truncate tokens / loss_masks / log_probs in-place for samples that exceed the limit."""
    if TRAJ_TRUNCATION_MAX_SEQ_LEN <= 0:
        return rollout_data

    max_len = TRAJ_TRUNCATION_MAX_SEQ_LEN
    tokens_list = rollout_data.get("tokens")
    loss_masks_list = rollout_data.get("loss_masks")
    total_lengths = rollout_data.get("total_lengths")
    response_lengths = rollout_data.get("response_lengths")

    if tokens_list is None or total_lengths is None:
        return rollout_data

    truncated_count = 0
    for i in range(len(total_lengths)):
        tl = total_lengths[i]
        if tl <= max_len:
            continue

        truncated_count += 1
        keep = max_len

        # Truncate tokens and loss masks to the last ``keep`` tokens.
        if tokens_list is not None and i < len(tokens_list):
            t = tokens_list[i]
            if hasattr(t, "__len__") and len(t) > keep:
                tokens_list[i] = t[-keep:]

        if loss_masks_list is not None and i < len(loss_masks_list):
            lm = loss_masks_list[i]
            if hasattr(lm, "__len__") and len(lm) > keep:
                loss_masks_list[i] = lm[-keep:]
                new_resp_len = int(sum(loss_masks_list[i]))
            else:
                new_resp_len = response_lengths[i] if response_lengths else 0
        else:
            new_resp_len = response_lengths[i] if response_lengths else 0

        total_lengths[i] = keep
        if response_lengths is not None:
            response_lengths[i] = new_resp_len

        # Truncate rollout_log_probs (only response tokens are stored).
        for key in ("rollout_log_probs", "teacher_log_probs"):
            lp_list = rollout_data.get(key)
            if lp_list is None or i >= len(lp_list):
                continue
            lp = lp_list[i]
            if hasattr(lp, "__len__") and len(lp) > new_resp_len:
                lp_list[i] = lp[-new_resp_len:]

    if truncated_count > 0:
        import sys
        print(
            f"[traj_truncation] Truncated {truncated_count}/{len(total_lengths)} "
            f"trajectories to last {max_len} tokens",
            file=sys.stderr,
        )

    return rollout_data


def _install_patch():
    """Patch slime.utils.data.process_rollout_data to add truncation."""
    try:
        import slime.utils.data as _slime_data
    except ImportError:
        return

    _orig = _slime_data.process_rollout_data

    def _patched(args, rollout_data_ref, dp_rank, dp_size):
        rollout_data = _orig(args, rollout_data_ref, dp_rank, dp_size)
        return _apply_truncation(rollout_data)

    _patched.__name__ = "process_rollout_data"
    _slime_data.process_rollout_data = _patched

    # Also patch the import in megatron_utils.actor (already imported).
    try:
        import slime.backends.megatron_utils.actor as _actor_mod
        _actor_mod.process_rollout_data = _patched
    except ImportError:
        pass

    # Also patch the import in fsdp_utils.actor.
    try:
        import slime.backends.fsdp_utils.actor as _fsdp_mod
        _fsdp_mod.process_rollout_data = _patched
    except ImportError:
        pass


_install_patch()
