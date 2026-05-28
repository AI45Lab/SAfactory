"""Shared model protocol primitives for OSGym agents."""

import ast
import base64
import io
from typing import Any, Dict, List, Optional, Tuple, Union


MessageContent = Union[str, List[Dict[str, Any]]]


class PyAutoGuiCoordinateNormalizer(ast.NodeTransformer):
    """Convert normalized [0, 1] pyautogui coordinates into absolute pixels."""

    COORDINATE_ACTIONS = {"click", "moveTo", "dragTo", "rightClick", "doubleClick", "middleClick"}

    def __init__(self, screen_width: int, screen_height: int):
        self.width = screen_width
        self.height = screen_height

    def visit_Call(self, node):
        self.generic_visit(node)
        if not self._is_pyautogui_call(node):
            return node

        new_args = []
        for idx, arg in enumerate(node.args):
            if self._is_normalized_float(arg) and node.func.attr in self.COORDINATE_ACTIONS:
                size = self.width if idx == 0 else self.height if idx == 1 else None
                if size is not None:
                    new_args.append(ast.Constant(value=int(arg.value * size)))
                    continue
            new_args.append(arg)
        node.args = new_args
        return node

    @staticmethod
    def _is_pyautogui_call(node) -> bool:
        return (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "pyautogui"
        )

    @staticmethod
    def _is_normalized_float(node) -> bool:
        return isinstance(node, ast.Constant) and isinstance(node.value, float) and 0.0 <= node.value <= 1.0


class ModelProtocol:
    """Base class for model-specific prompt and action parsing logic."""

    SPECIAL_COMMANDS = {"WAIT", "DONE", "FAIL"}

    def __init__(
        self,
        prompt_observation_type: str,
        screen_width: int = 1920,
        screen_height: int = 1080,
    ):
        self.prompt_observation_type = prompt_observation_type
        self.screen_width = screen_width
        self.screen_height = screen_height

    def build_system_prompt(self) -> str:
        raise NotImplementedError

    def build_user_content(
        self,
        current_obs: Dict[str, Any],
        instruction: Optional[str] = None,
        previous_actions: Optional[str] = None,
        history_truncated: bool = False,
    ) -> MessageContent:
        user_text = self.user_instruction_hint(instruction or "")
        history_text = self._build_history_text(previous_actions, history_truncated)
        if history_text:
            user_text = f"{user_text}\n\n{history_text}"
        screenshot = current_obs.get("screenshot") if current_obs else None
        if screenshot is None:
            return user_text

        screenshot_bytes = screenshot_to_png_bytes(screenshot)
        screenshot_url = encode_image_bytes(screenshot_bytes)
        if not screenshot_url:
            return user_text

        return [
            {"type": "text", "text": f"{user_text}\n\nThe latest screenshot is attached."},
            {"type": "image_url", "image_url": {"url": screenshot_url, "detail": "high"}},
        ]

    def user_instruction_hint(self, instruction: str) -> str:
        raise NotImplementedError

    @staticmethod
    def _build_history_text(
        previous_actions: Optional[str],
        history_truncated: bool,
    ) -> str:
        sections = []
        if history_truncated:
            sections.append(
                "Context note: some older chat messages were truncated to save memory. "
                "The Previous Actions section below remains complete."
            )
        if previous_actions:
            sections.append(f"Previous Actions:\n{previous_actions}")
        return "\n\n".join(sections)

    def parse_actions(self, action_str: str) -> List[str]:
        raise NotImplementedError

    def strip_special_command(self, actions: List[str]) -> Tuple[List[str], Optional[str]]:
        special_cmd = None
        remaining_actions = []
        for action in actions:
            cmd = self._try_get_special_command(action)
            if cmd:
                special_cmd = cmd
            else:
                remaining_actions.append(action)
        return remaining_actions, special_cmd

    @classmethod
    def is_special_command(cls, action: str) -> bool:
        return action.strip().upper() in cls.SPECIAL_COMMANDS

    def observation_description(self) -> str:
        return {
            "screenshot": "You will receive the latest screenshot of the desktop.",
            "a11y_tree": "You will receive the latest accessibility tree of the desktop.",
            "screenshot_a11y_tree": (
                "You will receive the latest screenshot of the desktop and a summarized accessibility tree."
            ),
        }[self.prompt_observation_type]

    def _try_get_special_command(self, command: str) -> Optional[str]:
        cleaned = command.strip().upper()
        for cmd in self.SPECIAL_COMMANDS:
            if cleaned.startswith(cmd):
                return cmd
        return None


def screenshot_to_png_bytes(screenshot: Any) -> Optional[bytes]:
    """Convert screenshot bytes or ndarray into PNG bytes."""
    if isinstance(screenshot, (bytes, bytearray, memoryview)):
        return bytes(screenshot)
    try:
        import numpy as np
        from PIL import Image

        if isinstance(screenshot, np.ndarray) and screenshot.size > 0:
            image = Image.fromarray(screenshot)
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            return buffered.getvalue()
    except Exception:
        return None
    return None


def encode_image_bytes(image_bytes: Optional[bytes]) -> Optional[str]:
    """Encode PNG bytes as an image data URL."""
    if not image_bytes:
        return None
    try:
        encoded = base64.b64encode(image_bytes).decode("utf-8")
        return f"data:image/png;base64,{encoded}"
    except Exception:
        return None
