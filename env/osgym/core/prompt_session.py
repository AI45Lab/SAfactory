"""Prompt history management for OSGym."""

import base64
import io
import math
import re
from typing import Any, Dict, List, Optional

from ..mm_agents.model_protocols.base import screenshot_to_png_bytes


class PromptSession:
    COLLAPSED_SCREENSHOT_TEXT = "This screenshot has been collapsed."

    def __init__(
        self,
        model_protocol,
        message_cut: int = -1,
        history_n: Optional[int] = None,
        image_max: Optional[int] = None,
        fold_size: Optional[int] = None,
        collapse_text: Optional[str] = None,
    ):
        self.model_protocol = model_protocol
        self.message_cut = message_cut
        self.messages: List[Dict[str, Any]] = []
        self.action_history: List[Dict[str, Any]] = []
        self.history_truncated = False
        self.qwen35vl_history_enabled = bool(
            getattr(model_protocol, "uses_qwen35vl_history", True)
        ) and any(value is not None for value in (history_n, image_max, fold_size, collapse_text))
        self.history_n = int(100 if history_n is None else history_n)
        self.image_max = int(20 if image_max is None else image_max)
        self.fold_size = int(10 if fold_size is None else fold_size)
        self.collapse_text = collapse_text or self.COLLAPSED_SCREENSHOT_TEXT
        self.turns: List[Dict[str, Any]] = []
        self.folded_prefix_k = 0

        if self.qwen35vl_history_enabled:
            if self.history_n < 0:
                raise ValueError("history_n must be >= 0")
            if self.image_max < 1:
                raise ValueError("image_max must be >= 1")
            if self.fold_size < 1:
                raise ValueError("fold_size must be >= 1")

    def reset(self) -> None:
        self.model_protocol.reset()
        system_prompt = self.model_protocol.build_system_prompt()
        if self.qwen35vl_history_enabled:
            system_content = [{"type": "text", "text": system_prompt}]
        else:
            system_content = system_prompt
        self.messages[:] = [{"role": "system", "content": system_content}]
        self.action_history.clear()
        self.history_truncated = False
        self.turns.clear()
        self.folded_prefix_k = 0
        self.model_protocol.postprocess_messages(self.messages)

    def add_assistant_message(self, content: str, parsed_actions: List[str]) -> None:
        entry = {
            "description": self._extract_action_description(content),
            "low_level_action": self._extract_low_level_action_description(content, parsed_actions),
            "actions": list(parsed_actions),
            "raw_content": content,
        }
        self.action_history.append(entry)

        if self.qwen35vl_history_enabled:
            if self.turns:
                self.turns[-1].update(
                    {
                        "assistant_content": content,
                        "description": entry["description"],
                        "low_level_action": entry["low_level_action"],
                        "actions": entry["actions"],
                        "raw_content": content,
                    }
                )
                self._rebuild_qwen35vl_messages()
            else:
                self.messages.append(
                    self.model_protocol.build_assistant_message(content, parsed_actions)
                )
            return

        self.messages.append(
            self.model_protocol.build_assistant_message(content, parsed_actions)
        )
        self.model_protocol.postprocess_messages(self.messages)

    def add_user_observation(
        self,
        *,
        processed_obs: Dict[str, Any],
        instruction: str,
    ) -> List[Dict[str, Any]]:
        if self.qwen35vl_history_enabled:
            self.turns.append(
                {
                    "processed_obs": processed_obs,
                    "instruction": instruction,
                    "assistant_content": None,
                    "description": "",
                    "low_level_action": "",
                    "actions": [],
                    "raw_content": "",
                    "image_url": None,
                }
            )
            self._rebuild_qwen35vl_messages()
            return self.messages

        observation_messages = self.model_protocol.build_observation_messages(
            current_obs=processed_obs,
            instruction=instruction,
            previous_actions=self.format_action_history(),
            history_truncated=self.history_truncated,
        )
        self.messages.extend(observation_messages)
        self._trim()
        self.model_protocol.postprocess_messages(self.messages)
        return self.messages

    def _rebuild_qwen35vl_messages(self) -> None:
        system_prompt = self.model_protocol.build_system_prompt()
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]}
        ]

        total_steps = len(self.turns)
        if total_steps == 0:
            self.messages[:] = messages
            return

        self._update_folding_state(total_steps)
        start_step = max(1, total_steps - self.history_n)
        instruction = self.turns[-1].get("instruction") or ""
        previous_actions = [
            f"Step {idx + 1}: {self._turn_action_text(self.turns[idx])}"
            for idx in range(0, min(total_steps - 1, len(self.turns)))
            if self.turns[idx].get("assistant_content") is not None
        ]
        previous_actions_str = "\n".join(previous_actions) if previous_actions else "None"
        instruction_prompt = (
            "\nPlease generate the next move according to the UI screenshot, instruction and previous actions.\n\n"
            f"Instruction: {instruction}\n\n"
            "Previous actions:\n"
            f"{previous_actions_str}"
        )

        for step_num in range(start_step, total_steps + 1):
            turn = self.turns[step_num - 1]
            is_first_turn = step_num == start_step
            is_collapsed = self._should_collapse_step(step_num)

            if is_collapsed:
                parts = [{"type": "text", "text": self.collapse_text}]
                user_content = (
                    [{"type": "text", "text": instruction_prompt}]
                    if is_first_turn
                    else self._wrap_tool_response(parts)
                )
            else:
                image_url = self._turn_image_url(turn)
                if image_url:
                    image_part = {"type": "image_url", "image_url": {"url": image_url}}
                    user_content = (
                        [image_part, {"type": "text", "text": instruction_prompt}]
                        if is_first_turn
                        else self._wrap_tool_response([image_part])
                    )
                else:
                    user_content = (
                        [{"type": "text", "text": instruction_prompt}]
                        if is_first_turn
                        else self._wrap_tool_response(
                            [{"type": "text", "text": self.collapse_text}]
                        )
                    )

            messages.append({"role": "user", "content": user_content})

            assistant_content = self._compact_assistant_history_content(step_num, turn)
            if step_num <= total_steps - 1 and assistant_content:
                messages.append(
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": assistant_content}],
                    }
                )

        self.messages[:] = messages

    def _update_folding_state(self, total_screenshots: int) -> None:
        while (total_screenshots - self.folded_prefix_k) > self.image_max:
            self.folded_prefix_k += self.fold_size
        if self.folded_prefix_k > total_screenshots:
            self.folded_prefix_k = total_screenshots

    def _should_collapse_step(self, step_num_1based: int) -> bool:
        return step_num_1based <= self.folded_prefix_k

    @staticmethod
    def _wrap_tool_response(parts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return (
            [{"type": "text", "text": "<tool_response>\n"}]
            + parts
            + [{"type": "text", "text": "\n</tool_response>"}]
        )

    def _turn_image_url(self, turn: Dict[str, Any]) -> Optional[str]:
        cached_url = turn.get("image_url")
        if cached_url:
            return cached_url

        processed_obs = turn.get("processed_obs") or {}
        screenshot = processed_obs.get("screenshot")
        screenshot_bytes = screenshot_to_png_bytes(screenshot)
        if not screenshot_bytes:
            return None

        encoded = self._encode_qwen35vl_image(screenshot_bytes)
        if encoded:
            turn["image_url"] = encoded
        return encoded

    @staticmethod
    def _encode_qwen35vl_image(image_bytes: bytes) -> Optional[str]:
        try:
            from PIL import Image

            image = Image.open(io.BytesIO(image_bytes))
            width, height = image.size
            try:
                from mm_agents.utils.qwen_vl_utils import smart_resize
            except Exception:
                smart_resize = PromptSession._smart_resize

            resized_height, resized_width = smart_resize(
                height=height,
                width=width,
                factor=32,
                max_pixels=16 * 16 * 4 * 12800,
            )
            image = image.resize((resized_width, resized_height))

            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
            return f"data:image/png;base64,{encoded}"
        except Exception:
            return None

    @staticmethod
    def _smart_resize(
        height: int,
        width: int,
        factor: int = 28,
        min_pixels: int = 56 * 56,
        max_pixels: int = 14 * 14 * 4 * 1280,
        max_long_side: int = 8192,
    ) -> tuple[int, int]:
        if height < 2 or width < 2:
            raise ValueError(f"height:{height} or width:{width} must be larger than factor:{factor}")
        if max(height, width) / min(height, width) > 200:
            raise ValueError(f"absolute aspect ratio must be smaller than 100, got {height} / {width}")

        if max(height, width) > max_long_side:
            beta = max(height, width) / max_long_side
            height, width = int(height / beta), int(width / beta)

        h_bar = round(height / factor) * factor
        w_bar = round(width / factor) * factor
        if h_bar * w_bar > max_pixels:
            beta = math.sqrt((height * width) / max_pixels)
            h_bar = math.floor(height / beta / factor) * factor
            w_bar = math.floor(width / beta / factor) * factor
        elif h_bar * w_bar < min_pixels:
            beta = math.sqrt(min_pixels / (height * width))
            h_bar = math.ceil(height * beta / factor) * factor
            w_bar = math.ceil(width * beta / factor) * factor
        return h_bar, w_bar

    @staticmethod
    def _turn_action_text(turn: Dict[str, Any]) -> str:
        return turn.get("low_level_action") or "(no action description)"

    def _compact_assistant_history_content(self, step_num: int, turn: Dict[str, Any]) -> str:
        if turn.get("assistant_content") is None:
            return ""
        tool_call = self._extract_tool_call(turn.get("raw_content") or "")
        if not tool_call:
            return ""
        return f"Action: {self._turn_action_text(turn)}\n\n{tool_call}"

    @staticmethod
    def _extract_tool_call(content: str) -> str:
        if not content:
            return ""
        match = re.search(r"<tool_call>.*?</tool_call>", content, re.DOTALL)
        return match.group(0).strip() if match else ""

    def format_action_history(self) -> str:
        if not self.action_history:
            return ""

        entries = []
        for idx, item in enumerate(self.action_history, start=1):
            description = item.get("description") or "(no action description)"
            actions = item.get("actions") or []
            entries.append(
                self.model_protocol.format_action_history_entry(
                    idx=idx,
                    description=description,
                    actions=actions,
                    raw_content=item.get("raw_content") or "",
                )
            )
        return "\n\n".join(entries)

    @staticmethod
    def _extract_action_description(content: str) -> str:
        if not content:
            return ""

        explicit_action = PromptSession._extract_explicit_action_description(content)
        if explicit_action:
            return explicit_action

        first_line = next((line.strip() for line in content.splitlines() if line.strip()), "")
        return first_line[:300]

    @staticmethod
    def _extract_low_level_action_description(content: str, parsed_actions: List[str]) -> str:
        explicit_action = PromptSession._extract_explicit_action_description(content)
        if explicit_action:
            return explicit_action

        first_action = next((action for action in parsed_actions if action), "")
        if first_action == "DONE":
            return "Task completed"
        if first_action == "WAIT":
            return "Waiting"
        if "." in first_action:
            return f"Performing {first_action.split('.', 1)[1].split('(', 1)[0]} action"
        if first_action:
            return "Performing action"
        return ""

    @staticmethod
    def _extract_explicit_action_description(content: str) -> str:
        if not content:
            return ""

        action_match = re.search(
            r"##\s*Action\s*:?\s*(.*?)(?:\n\s*##\s*Code\s*:|\n\s*```|$)",
            content,
            re.DOTALL | re.IGNORECASE,
        )
        if action_match:
            return " ".join(action_match.group(1).strip().split())

        plain_match = re.search(
            r"^\s*Action\s*:?\s*(.*?)(?:\n\s*<tool_call>|\n\s*```|$)",
            content,
            re.DOTALL | re.IGNORECASE | re.MULTILINE,
        )
        if plain_match:
            return " ".join(plain_match.group(1).strip().split())
        return ""

    def _trim(self) -> None:
        if self.message_cut <= 0 or not self.messages:
            return

        system_messages = []
        tail_start = 0
        if self.messages[0].get("role") == "system":
            system_messages = [self.messages[0]]
            tail_start = 1

        tail = self.messages[tail_start:]
        keep_count = max(1, self.message_cut * 2 - 1)
        trimmed_tail = tail[-keep_count:]
        if len(trimmed_tail) < len(tail):
            self.history_truncated = True
        self.messages[:] = system_messages + trimmed_tail
