"""Qwen3.5/3.6/3.8 chat template adapter.

Handles three Qwen-specific quirks that the base adapter cannot:

1. **Message normalization**: Qwen's chat template iterates ``tool_call.arguments``
   via the Jinja ``items`` filter, so arguments must be a dict (OpenAI sends a
   JSON string). The template also accesses ``content`` directly (not ``.get``),
   so None content raises ``KeyError`` and non-string content raises
   ``AttributeError`` on ``.startswith``.

2. **Tool call parsing**: Qwen emits tool calls as *text* using
   ``<function=NAME>...<parameter=KEY>VALUE</parameter>...</function>``
   wrapped in delimiter tokens. OpenHands only executes structured
   ``tool_calls``, so we parse the text format back into OpenAI dicts.

3. **System message rendering**: Qwen's template has two hard checks:
     - system message must be at index 0
     - a user message must exist
   Rendering a standalone system message triggers the second check. We work
   around it by rendering ``[system, user_base]`` and stripping ``user_base``.

4. **Tools block injection**: Qwen only injects ``<tools>...</tools>`` into
   the FIRST system message. We render the first system message WITH tools
   so the training mask aligns with what sglang rendered.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

try:
    from .chat_template_adapter import BASE_CHAT_HISTORY, ChatTemplateAdapter
except ImportError:
    from chat_template_adapter import BASE_CHAT_HISTORY, ChatTemplateAdapter

# User-only base used for the system-message rendering trick.
_USER_ONLY_BASE = [{"role": "user", "content": "I am a user."}]

# Regexes for parsing Qwen text-format tool calls.
_FUNCTION_BLOCK_RE = re.compile(r"<function=(\w+)>(.*?)</function>", re.DOTALL)
_PARAM_BLOCK_RE = re.compile(r"<parameter=(\w+)>(.*?)</parameter>", re.DOTALL)
# Trailing tool-call delimiter token (a `<...>` tag) right before the first
# `<function=` — stripped from the reasoning content we return.
_TRAILING_TAG_RE = re.compile(r"\s*<[^>]*>\s*$")


class QwenChatTemplateAdapter(ChatTemplateAdapter):
    """Chat template adapter for Qwen3.5 / 3.6 / 3.8 models."""

    def __init__(self, tokenizer: Any, processor: Any = None) -> None:
        super().__init__(tokenizer, processor)
        self._user_suffix_str: Optional[str] = None

    # -- message normalization ------------------------------------------------

    def normalize_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Normalize OpenAI-format messages so the Qwen chat template can
        render them.

        - ``tool_calls.arguments``: JSON string → dict (template uses ``items``)
        - ``content``: None → "" (template accesses content directly)
        - ``content``: non-string → string (template calls ``.startswith``)
        """
        for msg in messages or []:
            if not isinstance(msg, dict):
                continue
            tool_calls = msg.get("tool_calls")
            if isinstance(tool_calls, list):
                for tc in tool_calls:
                    if not isinstance(tc, dict):
                        continue
                    fn = tc.get("function")
                    if not isinstance(fn, dict):
                        continue
                    args = fn.get("arguments")
                    if isinstance(args, str):
                        try:
                            fn["arguments"] = json.loads(args) if args else {}
                        except (json.JSONDecodeError, ValueError):
                            fn["arguments"] = {"_raw": args}
            content = msg.get("content")
            if content is None:
                msg["content"] = ""
            elif not isinstance(content, str):
                try:
                    msg["content"] = json.dumps(content, ensure_ascii=False)
                except Exception:
                    msg["content"] = str(content)
        return messages

    # -- tool call parsing -----------------------------------------------------

    def parse_tool_calls(
        self, assistant_text: str
    ) -> Tuple[Optional[str], Optional[List[Dict[str, Any]]], Optional[str]]:
        """Convert Qwen text-format tool calls into OpenAI ``tool_calls``.

        Qwen emits ``<function=NAME>...<parameter=KEY>VALUE</parameter>...
        </function>`` as text. OpenHands only executes structured
        ``tool_calls``, so we parse the text and convert. The raw
        ``assistant_text`` is still what gets recorded into the training
        trajectory — this conversion only shapes the response handed back
        to the agent.
        """
        blocks = list(_FUNCTION_BLOCK_RE.finditer(assistant_text))
        if not blocks:
            return assistant_text, None, None

        tool_calls: List[Dict[str, Any]] = []
        for idx, blk in enumerate(blocks):
            name = blk.group(1)
            args: Dict[str, Any] = {}
            for p in _PARAM_BLOCK_RE.finditer(blk.group(2)):
                args[p.group(1)] = p.group(2).strip("\n")
            tool_calls.append({
                "id": f"call_{idx}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(args, ensure_ascii=False),
                },
            })

        # Content = text before the first tool-call block, with the trailing
        # tool-call delimiter tag stripped. Text between/after blocks is just
        # delimiter noise, discard it.
        content = assistant_text[: blocks[0].start()]
        content = _TRAILING_TAG_RE.sub("", content).strip()
        return (content or None), tool_calls, "tool_calls"

    # -- message delta rendering (for training mask) --------------------------

    def _get_user_suffix_str(self) -> str:
        """The rendered form of a single user message as it appears AFTER a
        system message. Used to strip the trailing user message when rendering
        a standalone system message (Qwen's template guards require a user
        message to be present, so we render [system, user] and strip the user
        suffix).

        NB: render([user]) alone injects a synthetic default system block
        (with reasoning instructions) before the user, so it is NOT a clean
        suffix — compute it from BASE_CHAT_HISTORY + [user] instead.
        """
        if self._user_suffix_str is None:
            full = self.tokenizer.apply_chat_template(
                BASE_CHAT_HISTORY + [{"role": "user", "content": "I am a user."}],
                add_generation_prompt=False,
                tokenize=False,
            )
            if not full.startswith(self.base_messages_str):
                raise ValueError("failed to extract user-suffix template fragment")
            self._user_suffix_str = full[len(self.base_messages_str):]
        return self._user_suffix_str

    def render_message_delta(self, message: Dict[str, Any]) -> str:
        """Render a single message's template fragment.

        Qwen3.5/3.6 chat template has two hard checks:
          1) system message must be at index 0
          2) a user message must exist

        For system messages, rendering [system] alone triggers check (2).
        We render [system, user_base] and strip user_base to get the clean
        system fragment. For non-system messages, the default approach
        (render [BASE, msg] and strip BASE prefix) works fine.
        """
        if message.get("role") == "system":
            user_suffix = self._get_user_suffix_str()
            with_msg = self.tokenizer.apply_chat_template(
                [message] + _USER_ONLY_BASE,
                add_generation_prompt=False,
                tokenize=False,
            )
            if not with_msg.endswith(user_suffix):
                raise ValueError("failed to extract system-message template fragment")
            return with_msg[: len(with_msg) - len(user_suffix)]

        # Non-system: use default base-prefix-strip approach.
        return super().render_message_delta(message)

    def render_first_system_delta(
        self,
        message: Dict[str, Any],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Render the session's first system message WITH tools so the
        template's ``<tools>...</tools>`` system block lands in the recorded
        input_ids, matching what sglang renders for the rollout prompt.

        Qwen's template guards require a user message, so we render
        [system_msg, user_base] with tools and strip the clean user suffix.
        """
        user_suffix = self._get_user_suffix_str()
        with_msg = self.tokenizer.apply_chat_template(
            [message] + _USER_ONLY_BASE,
            tools=tools,
            add_generation_prompt=False,
            tokenize=False,
        )
        if not with_msg.endswith(user_suffix):
            raise ValueError("failed to extract first-system-message template fragment")
        return with_msg[: len(with_msg) - len(user_suffix)]

    def needs_tools_on_first_system_only(self) -> bool:
        """Qwen only injects ``<tools>`` into the first system message."""
        return True
