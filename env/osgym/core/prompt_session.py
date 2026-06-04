"""Prompt history management for OSGym."""

import re
from typing import Any, Dict, List


class PromptSession:
    def __init__(self, model_protocol, message_cut: int = -1):
        self.model_protocol = model_protocol
        self.message_cut = message_cut
        self.messages: List[Dict[str, Any]] = []
        self.action_history: List[Dict[str, Any]] = []
        self.history_truncated = False

    def reset(self) -> None:
        system_prompt = self.model_protocol.build_system_prompt()
        self.messages[:] = [{"role": "system", "content": system_prompt}]
        self.action_history.clear()
        self.history_truncated = False

    def add_assistant_message(self, content: str, parsed_actions: List[str]) -> None:
        self.messages.append({"role": "assistant", "content": content})
        self.action_history.append({
            "description": self._extract_action_description(content),
            "actions": list(parsed_actions),
            "raw_content": content,
        })

    def add_user_observation(
        self,
        *,
        processed_obs: Dict[str, Any],
        instruction: str,
    ) -> List[Dict[str, Any]]:
        user_content = self.model_protocol.build_user_content(
            current_obs=processed_obs,
            instruction=instruction,
            previous_actions=self.format_action_history(),
            history_truncated=self.history_truncated,
        )
        self.messages.append({"role": "user", "content": user_content})
        self._trim()
        return self.messages

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

        first_line = next((line.strip() for line in content.splitlines() if line.strip()), "")
        return first_line[:300]

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
