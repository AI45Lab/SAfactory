"""OSGym-specific wrapper around the shared Slime rollout generator."""

import copy
import logging
import random

from env.osgym.core.action_validator import validate_qwen35_action_response
from rl import slime_generator as shared_slime_generator


logger = logging.getLogger(__name__)


def _decode_current_assistant_response(sample, tokenizer) -> str:
    """Decode the final trainable span, which is the current OSGym action."""
    tokens = list(getattr(sample, "tokens", None) or [])
    loss_mask = list(getattr(sample, "loss_mask", None) or [])
    if len(loss_mask) > len(tokens):
        raise ValueError(
            "OSGym sample loss_mask cannot be longer than tokens: "
            f"index={getattr(sample, 'index', None)!r}, "
            f"loss_mask={len(loss_mask)}, tokens={len(tokens)}"
        )

    try:
        span_end = max(index for index, value in enumerate(loss_mask) if value == 1)
    except ValueError:
        return ""

    span_start = span_end
    while span_start > 0 and loss_mask[span_start - 1] == 1:
        span_start -= 1

    token_offset = len(tokens) - len(loss_mask)
    response_tokens = tokens[
        token_offset + span_start : token_offset + span_end + 1
    ]
    return tokenizer.decode(
        response_tokens,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


def _annotate_action_validation(
    sample_groups,
    tokenizer,
) -> None:
    """Attach per-step action validity before samples are padded and shuffled."""
    if tokenizer is None:
        raise RuntimeError("OSGym action validation requires an initialized tokenizer")

    for sample_group in sample_groups:
        for sample in sample_group:
            if getattr(sample, "remove_sample", False):
                continue

            raw_response = _decode_current_assistant_response(sample, tokenizer)
            validation = validate_qwen35_action_response(raw_response)
            metadata = dict(getattr(sample, "metadata", None) or {})
            metadata["action_validation"] = validation
            metadata["invalid_action"] = not bool(validation["valid"])
            sample.metadata = metadata


def _pad_and_shuffle_training_samples(
    args,
    rollout_id: int,
    sample_groups,
):
    """Flatten complete trajectories and pad to a static optimizer batch."""
    samples = [sample for group in sample_groups for sample in group]
    if not samples:
        return samples

    global_batch_size = int(args.global_batch_size)
    if global_batch_size <= 0:
        raise ValueError(
            f"global_batch_size must be positive, got {global_batch_size}"
        )

    padding_count = (-len(samples)) % global_batch_size
    if padding_count:
        template = min(samples, key=lambda sample: len(sample.tokens))
        padding_reward = sum(
            float(sample.get_reward_value(args)) for sample in samples
        ) / len(samples)

        for padding_index in range(padding_count):
            padding_sample = copy.copy(template)
            padding_sample.index = f"padding:{rollout_id}:{padding_index}"
            padding_sample.reward = padding_reward
            padding_sample.loss_mask = (
                list(template.loss_mask)
                if template.loss_mask is not None
                else None
            )
            padding_sample.metadata = dict(template.metadata or {})
            padding_sample.metadata["osgym_padding"] = True
            padding_sample.remove_sample = True
            samples.append(padding_sample)

        logger.info(
            "Padded OSGym rollout for static batching: rollout_id=%d "
            "real_samples=%d padding_samples=%d global_batch_size=%d",
            rollout_id,
            len(samples) - padding_count,
            padding_count,
            global_batch_size,
        )

    random.Random(rollout_id).shuffle(samples)
    return samples


def generate_rollout(args, rollout_id, data_buffer, evaluation=False):
    """Run the shared generator, preserving complete OSGym trajectories."""
    sample_groups = shared_slime_generator.generate_rollout(
        args,
        rollout_id,
        data_buffer,
        evaluation=evaluation,
    )
    if evaluation:
        return sample_groups
    _annotate_action_validation(sample_groups, shared_slime_generator.TOKENIZER)
    return _pad_and_shuffle_training_samples(args, rollout_id, sample_groups)


__all__ = ["generate_rollout"]
