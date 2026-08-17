"""Kimi-style OSGym prompt and action protocol."""

import ast
import re
import textwrap
from typing import List, Optional

from .base import ModelProtocol, PyAutoGuiCoordinateNormalizer


class KimiProtocol(ModelProtocol):
    """Protocol using `## Action` + Python code blocks."""

    def __init__(self, prompt_observation_type: str, screen_width: int = 1920, screen_height: int = 1080):
        super().__init__(prompt_observation_type, screen_width, screen_height)
        self._normalizer = PyAutoGuiCoordinateNormalizer(screen_width, screen_height)

    def build_system_prompt(self) -> str:
        return textwrap.dedent(
            f"""
            You are a GUI agent operating a desktop computer.
            {self.observation_description()}
            The computer password is "osworld-public-evaluation". Use it when sudo rights are required.

            Your goal is to finish the task exactly as instructed. If the task is still running, a page is still loading,
            or a command / installation has not finished yet, use `computer.wait()`. If the task is fully completed, use
            `computer.terminate(status="success")`. If the task is impossible, blocked, or not fully completed, use
            `computer.terminate(status="failure")`.

            For each step, respond in exactly this format and do not add any extra sections:

            ## Action:
            <one concise sentence describing the next step>
            ## Code:
            ```python
            <pyautogui code or a single computer.* call>
            ```

            Requirements:
            - Do not output `Thought`, `Observation`, `Reflection`, or any other section.
            - The `Action` must be concise and grounded in visible UI elements or the accessibility tree when provided.
            - The `Code` must be either valid `pyautogui` code, `computer.wait()`, or `computer.terminate(...)`.
            - Do not call `pyautogui.screenshot()` or `pyautogui.locateCenterOnScreen(...)`.
            - Each step must be self-contained. Do not rely on variables or helper functions from previous steps.
            - Prefer normalized coordinates in the range [0, 1] when using `pyautogui` mouse actions.
            - When typing text, include the exact target text in the `Action` and the corresponding code in `Code`.
            """
        ).strip()

    def user_instruction_hint(self, instruction: str) -> str:
        return (
            f"Task Instruction:\n{instruction}\n\n"
            "Please generate the next move according to the screenshot, task instruction and previous steps (if provided)."
        )

    def parse_actions(self, action_str: str) -> List[str]:
        if not action_str or not action_str.strip():
            return []
        if "<tool_call>" in action_str:
            from .qwen35 import Qwen35Protocol

            return Qwen35Protocol(
                self.prompt_observation_type,
                self.screen_width,
                self.screen_height,
            ).parse_actions(action_str)

        commands = []
        code_blocks = re.findall(r"```python\n(.*?)\n```", action_str, re.DOTALL)
        if not code_blocks:
            code_match = re.search(r"## Code:\s*(.*)", action_str, re.DOTALL)
            if code_match:
                code_blocks = [code_match.group(1).split("##")[0].strip()]

        for block in code_blocks:
            for line in [line.strip() for line in block.split("\n") if line.strip()]:
                sanitized = self._sanitize_command(line)
                if sanitized:
                    commands.append(sanitized)
        return commands

    def _sanitize_command(self, command: str) -> Optional[str]:
        if not command:
            return None
        special = self._try_get_special_command(command)
        if special:
            return special
        if command.startswith("computer."):
            return self._map_computer_helper(command)
        if command.startswith("pyautogui."):
            try:
                tree = ast.parse(command)
                normalized_tree = self._normalizer.visit(tree)
                return ast.unparse(normalized_tree).strip()
            except Exception:
                return command
        return command

    @staticmethod
    def _map_computer_helper(command: str) -> str:
        if "wait()" in command:
            return "WAIT"
        if "terminate" in command:
            if 'status="success"' in command or "status='success'" in command:
                return "DONE"
            return "FAIL"
        return command
