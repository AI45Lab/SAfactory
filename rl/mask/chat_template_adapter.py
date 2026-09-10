"""Chat template adapter abstraction.

Different models (Qwen, Llama, DeepSeek, ...) have different chat template
quirks that the RL pipeline must accommodate:
  - Message normalization (OpenAI format → model template format)
  - Tool call parsing (model text output → OpenAI structured tool_calls)
  - Message delta rendering (for training mask alignment)

This module provides the base ``ChatTemplateAdapter`` interface and a factory
``create_adapter`` that selects the right adapter by name. New models only need
to add a subclass and register it in the factory.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Shared base chat history used for delta rendering. Most adapters use this
# to compute a stable prefix that can be stripped when rendering a single
# message's template fragment.
BASE_CHAT_HISTORY = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "I am a user."},
]


class ChatTemplateAdapter:
    """Base chat template adapter.

    Subclasses override the methods that need model-specific logic.
    The defaults work for models whose ``apply_chat_template`` accepts
    standard OpenAI-format messages without extra guards.
    """

    def __init__(self, tokenizer: Any, processor: Any = None) -> None:
        self.tokenizer = tokenizer
        self.processor = processor
        self.base_messages_str: str = self.tokenizer.apply_chat_template(
            BASE_CHAT_HISTORY,
            add_generation_prompt=False,
            tokenize=False,
        )

    # -- message normalization ------------------------------------------------

    def normalize_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Normalize OpenAI-format messages so the model's chat template can
        render them. Default: no-op (standard models accept OpenAI format as-is).
        """
        return messages

    # -- tool call parsing -----------------------------------------------------

    def parse_tool_calls(
        self, assistant_text: str
    ) -> Tuple[Optional[str], Optional[List[Dict[str, Any]]], Optional[str]]:
        """Parse the model's raw text output into OpenAI ``tool_calls``.

        Returns ``(content, tool_calls, finish_reason)``:
          - content: reasoning text with tool-call markup removed (None if empty)
          - tool_calls: list of OpenAI tool_call dicts, or None if none found
          - finish_reason: "tool_calls" if any, else None (caller decides)

        Default: no parsing — the model already returns structured tool_calls
        via the API, so the raw text is the content.
        """
        return assistant_text, None, None

    # -- message delta rendering (for training mask) --------------------------

    def render_message_delta(self, message: Dict[str, Any]) -> str:
        """Render a single message's template fragment for training mask.

        Default: render ``[BASE_CHAT_HISTORY, message]`` and strip the
        ``BASE_CHAT_HISTORY`` prefix. This works for models whose template
        has no restrictions on system message position or user message presence.
        """
        full = self.tokenizer.apply_chat_template(
            BASE_CHAT_HISTORY + [message],
            add_generation_prompt=False,
            tokenize=False,
        )
        if not full.startswith(self.base_messages_str):
            raise ValueError("failed to extract single-message template fragment")
        return full[len(self.base_messages_str):]

    def render_first_system_delta(
        self,
        message: Dict[str, Any],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Render the session's first system message, optionally with tools.

        Default: same as ``render_message_delta`` (most models don't need
        special first-system handling). Models that only inject ``<tools>``
        into the first system message (e.g. Qwen) override this.
        """
        return self.render_message_delta(message)

    def needs_tools_on_first_system_only(self) -> bool:
        """Whether tools should only be rendered on the first system message.

        Default: False. Qwen overrides to True because its template only
        injects the ``<tools>`` block into the first system message.
        """
        return False


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_REGISTRY: Dict[str, type[ChatTemplateAdapter]] = {}


def register_adapter(name: str, cls: type[ChatTemplateAdapter]) -> None:
    """Register a chat template adapter class under ``name``."""
    _REGISTRY[name] = cls


def create_adapter(
    name_or_type: Optional[str],
    tokenizer: Any,
    processor: Any = None,
) -> ChatTemplateAdapter:
    """Create a chat template adapter by name.

    ``name_or_type`` is matched case-insensitively against registered adapters.
    Falls back to the base ``ChatTemplateAdapter`` (no-op) if not found, so
    unknown models still work for standard cases.
    """
    if name_or_type is None:
        name_or_type = ""
    key = name_or_type.strip().lower()

    # Map common aliases
    alias_map = {
        "qwen": "qwen",
        "qwen3": "qwen",
        "qwen3_5": "qwen",
        "qwen3.5": "qwen",
        "qwen3_8": "qwen",
        "qwen3.8": "qwen",
        "qwen3_6": "qwen",
        "qwen3.6": "qwen",
    }
    key = alias_map.get(key, key)

    cls = _REGISTRY.get(key)
    if cls is not None:
        logger.info("Chat template adapter: %s -> %s", name_or_type, cls.__name__)
        return cls(tokenizer, processor)

    logger.warning(
        "Unknown chat template adapter %r, falling back to base (no-op). "
        "Register it via register_adapter() if the model needs special handling.",
        name_or_type,
    )
    return ChatTemplateAdapter(tokenizer, processor)


# ---------------------------------------------------------------------------
# Auto-register built-in adapters
# ---------------------------------------------------------------------------

def _autoregister() -> None:
    """Import and register built-in adapters. Called once at module load."""
    try:
        # Try relative import first (when used as a package, e.g. rl.mask.chat_template_adapter)
        from .qwen_chat_template_adapter import QwenChatTemplateAdapter
    except ImportError:
        # Fall back to absolute import (when rl/mask/ is on sys.path and this
        # module is imported as a top-level module, e.g. `from chat_template_adapter import ...`)
        from qwen_chat_template_adapter import QwenChatTemplateAdapter
    register_adapter("qwen", QwenChatTemplateAdapter)


_autoregister()
