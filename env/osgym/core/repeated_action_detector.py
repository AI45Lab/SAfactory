"""Repeated action detection for OSGym."""

import ast
import math
from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple


CLICK_FUNCTIONS = {"click", "rightClick", "doubleClick", "middleClick"}
SIGNATURE_FUNCTIONS = {
    "moveTo",
    "dragTo",
    "scroll",
    "hscroll",
    "typewrite",
    "write",
    "press",
    "hotkey",
    "keyDown",
    "keyUp",
}


@dataclass(frozen=True)
class ClickAction:
    name: str
    x: Optional[float]
    y: Optional[float]
    button: Optional[str]

    @property
    def has_coordinates(self) -> bool:
        return self.x is not None and self.y is not None


@dataclass(frozen=True)
class ActionSignature:
    name: str
    args: Tuple[Any, ...]


@dataclass(frozen=True)
class RepeatedActionResult:
    repeated: bool
    repeat_count: int = 0
    action: Optional[str] = None
    reason: str = ""


class RepeatedActionDetector:
    """Detect repeated or non-progressing actions."""

    def __init__(
        self,
        click_distance_threshold: float = 10.0,
        click_repeat_limit: int = 3,
        wait_repeat_limit: int = 3,
        action_signature_repeat_limit: int = 3,
    ):
        self.click_distance_threshold = click_distance_threshold
        self.click_repeat_limit = click_repeat_limit
        self.wait_repeat_limit = wait_repeat_limit
        self.action_signature_repeat_limit = action_signature_repeat_limit
        self._last_click: Optional[ClickAction] = None
        self._click_repeat_count = 0
        self._wait_repeat_count = 0
        self._last_action_signature: Optional[Tuple[ActionSignature, ...]] = None
        self._action_signature_repeat_count = 0

    def reset(self) -> None:
        self._last_click = None
        self._click_repeat_count = 0
        self._wait_repeat_count = 0
        self._last_action_signature = None
        self._action_signature_repeat_count = 0

    def check(
        self,
        actions: Sequence[str],
        special_cmd: Optional[str] = None,
    ) -> RepeatedActionResult:
        if special_cmd == "WAIT":
            self._wait_repeat_count += 1
        else:
            self._wait_repeat_count = 0

        if self.wait_repeat_limit > 0 and self._wait_repeat_count >= self.wait_repeat_limit:
            self._reset_click_and_signature()
            return RepeatedActionResult(
                repeated=True,
                repeat_count=self._wait_repeat_count,
                action="WAIT",
                reason="repeated_wait",
            )

        signature = action_sequence_signature(actions)
        if signature is None:
            self._last_action_signature = None
            self._action_signature_repeat_count = 0
        elif self._last_action_signature == signature:
            self._action_signature_repeat_count += 1
        else:
            self._last_action_signature = signature
            self._action_signature_repeat_count = 1

        if (
            signature is not None
            and self.action_signature_repeat_limit > 0
            and self._action_signature_repeat_count >= self.action_signature_repeat_limit
        ):
            self._reset_click_and_wait()
            return RepeatedActionResult(
                repeated=True,
                repeat_count=self._action_signature_repeat_count,
                action=format_action_sequence_signature(signature),
                reason="repeated_action_signature",
            )

        click = self._first_click(actions)
        if click is None:
            self._last_click = None
            self._click_repeat_count = 0
            return RepeatedActionResult(repeated=False)

        if self._last_click is not None and self._is_same_click(self._last_click, click):
            self._click_repeat_count += 1
        else:
            self._last_click = click
            self._click_repeat_count = 1

        if self.click_repeat_limit > 0 and self._click_repeat_count >= self.click_repeat_limit:
            return RepeatedActionResult(
                repeated=True,
                repeat_count=self._click_repeat_count,
                action=self._format_click(click),
                reason="repeated_click",
            )
        return RepeatedActionResult(repeated=False, repeat_count=self._click_repeat_count)

    def _reset_click_and_wait(self) -> None:
        self._last_click = None
        self._click_repeat_count = 0
        self._wait_repeat_count = 0

    def _reset_click_and_signature(self) -> None:
        self._last_click = None
        self._click_repeat_count = 0
        self._last_action_signature = None
        self._action_signature_repeat_count = 0

    def _is_same_click(self, previous: ClickAction, current: ClickAction) -> bool:
        if previous.name != current.name or previous.button != current.button:
            return False

        if previous.has_coordinates and current.has_coordinates:
            distance = math.hypot(
                (previous.x or 0) - (current.x or 0),
                (previous.y or 0) - (current.y or 0),
            )
            return distance <= self.click_distance_threshold

        return not previous.has_coordinates and not current.has_coordinates

    @staticmethod
    def _first_click(actions: Sequence[str]) -> Optional[ClickAction]:
        for action in actions:
            click = parse_click_action(action)
            if click is not None:
                return click
        return None

    @staticmethod
    def _format_click(click: ClickAction) -> str:
        if click.has_coordinates:
            return f"{click.name}({click.x}, {click.y})"
        return f"{click.name}()"


def parse_click_action(action: str) -> Optional[ClickAction]:
    try:
        tree = ast.parse(action)
    except SyntaxError:
        return None

    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Expr):
        return None

    call = tree.body[0].value
    if not _is_pyautogui_click_call(call):
        return None

    name = call.func.attr
    x, y = _extract_coordinates(call)
    button = _extract_string_keyword(call, "button")
    return ClickAction(name=name, x=x, y=y, button=button)


def action_sequence_signature(actions: Sequence[str]) -> Optional[Tuple[ActionSignature, ...]]:
    if any(parse_click_action(action) is not None for action in actions):
        return None

    signatures = []
    for action in actions:
        signature = parse_action_signature(action)
        if signature is not None:
            signatures.append(signature)
    return tuple(signatures) if signatures else None


def parse_action_signature(action: str) -> Optional[ActionSignature]:
    try:
        tree = ast.parse(action)
    except SyntaxError:
        return None

    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Expr):
        return None

    call = tree.body[0].value
    if not _is_pyautogui_signature_call(call):
        return None

    name = call.func.attr
    return ActionSignature(
        name=name,
        args=tuple(_normalized_literal(arg) for arg in call.args),
    )


def format_action_signature(signature: ActionSignature) -> str:
    args = ", ".join(repr(arg) for arg in signature.args)
    return f"pyautogui.{signature.name}({args})"


def format_action_sequence_signature(signatures: Sequence[ActionSignature]) -> str:
    return "; ".join(format_action_signature(signature) for signature in signatures)


def _is_pyautogui_click_call(node) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "pyautogui"
        and node.func.attr in CLICK_FUNCTIONS
    )


def _is_pyautogui_signature_call(node) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "pyautogui"
        and node.func.attr in SIGNATURE_FUNCTIONS
    )


def _extract_coordinates(call: ast.Call) -> Tuple[Optional[float], Optional[float]]:
    x = _number_from_arg(call.args[0]) if len(call.args) >= 1 else None
    y = _number_from_arg(call.args[1]) if len(call.args) >= 2 else None

    for keyword in call.keywords:
        if keyword.arg == "x":
            x = _number_from_arg(keyword.value)
        elif keyword.arg == "y":
            y = _number_from_arg(keyword.value)
    return x, y


def _extract_string_keyword(call: ast.Call, name: str) -> Optional[str]:
    for keyword in call.keywords:
        if keyword.arg == name and isinstance(keyword.value, ast.Constant):
            if isinstance(keyword.value.value, str):
                return keyword.value.value
    return None


def _number_from_arg(node) -> Optional[float]:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        value = _number_from_arg(node.operand)
        return -value if value is not None else None
    return None


def _normalized_literal(node) -> Any:
    if isinstance(node, ast.Constant):
        value = node.value
        if isinstance(value, float):
            return round(value, 2)
        return value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        value = _normalized_literal(node.operand)
        if isinstance(value, (int, float)):
            return -value
    return ast.dump(node)
