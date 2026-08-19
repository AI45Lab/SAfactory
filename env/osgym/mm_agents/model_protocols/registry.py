"""Registry for OSGym model protocols."""

from typing import Type

from .base import ModelProtocol
from .kimi import KimiProtocol
from .qwen3vl import Qwen3VLProtocol
from .qwen35 import Qwen35Protocol


PROTOCOLS: dict[str, Type[ModelProtocol]] = {
    "kimi": KimiProtocol,
    "qwen3vl": Qwen3VLProtocol,
    "qwen35": Qwen35Protocol,
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
        supported = ", ".join(sorted(PROTOCOLS))
        raise ValueError(
            f"Unsupported prompt_format {prompt_format!r}; expected one of: {supported}"
        )
    return protocol_cls(
        prompt_observation_type=prompt_observation_type,
        screen_width=screen_width,
        screen_height=screen_height,
    )
