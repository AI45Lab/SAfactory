"""Registry for OSGym model protocols."""

import logging
from typing import Type

from .base import ModelProtocol
from .kimi import KimiProtocol
from .qwen import QwenProtocol

logger = logging.getLogger("osgym.model_protocols")


PROTOCOLS: dict[str, Type[ModelProtocol]] = {
    "kimi": KimiProtocol,
    "qwen": QwenProtocol,
}


def get_model_protocol(
    prompt_format: str,
    prompt_observation_type: str,
    screen_width: int = 1920,
    screen_height: int = 1080,
) -> ModelProtocol:
    """Create the model protocol selected by prompt_format."""
    protocol_name = (prompt_format or "kimi").strip().lower()
    protocol_cls = PROTOCOLS.get(protocol_name)
    if protocol_cls is None:
        logger.warning("Unknown prompt_format: %s. Defaulting to 'kimi'.", prompt_format)
        protocol_cls = KimiProtocol
    return protocol_cls(
        prompt_observation_type=prompt_observation_type,
        screen_width=screen_width,
        screen_height=screen_height,
    )
