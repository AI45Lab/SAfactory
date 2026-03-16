"""
Action Parser Module

Parses agent response strings into executable action commands.
Handles special commands (WAIT, DONE, FAIL) and code block extraction.
"""

import ast
import re
from typing import List, Tuple, Optional


class _PyAutoGuiCoordinateNormalizer(ast.NodeTransformer):
    """Convert normalized pyautogui coordinates into absolute pixel positions."""

    TARGET_FUNCTIONS = {
        "click",
        "doubleClick",
        "dragTo",
        "leftClick",
        "middleClick",
        "mouseDown",
        "mouseUp",
        "moveTo",
        "rightClick",
        "tripleClick",
    }

    def __init__(self, screen_width: int, screen_height: int):
        self.screen_width = screen_width
        self.screen_height = screen_height

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)

        if not self._is_supported_pyautogui_call(node):
            return node

        self._normalize_positional_coordinates(node)
        self._normalize_keyword_coordinates(node)
        return node

    @classmethod
    def _is_supported_pyautogui_call(cls, node: ast.Call) -> bool:
        func = node.func
        return (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Name)
            and func.value.id == "pyautogui"
            and func.attr in cls.TARGET_FUNCTIONS
        )

    def _normalize_positional_coordinates(self, node: ast.Call) -> None:
        if len(node.args) < 2:
            return

        x_node = node.args[0]
        y_node = node.args[1]
        x_val = self._numeric_literal(x_node)
        y_val = self._numeric_literal(y_node)

        if not self._should_normalize(x_node, y_node, x_val, y_val):
            return

        node.args[0] = ast.Constant(value=self._scale_coordinate(x_val, self.screen_width))
        node.args[1] = ast.Constant(value=self._scale_coordinate(y_val, self.screen_height))

    def _normalize_keyword_coordinates(self, node: ast.Call) -> None:
        x_kw = next((kw for kw in node.keywords if kw.arg == "x"), None)
        y_kw = next((kw for kw in node.keywords if kw.arg == "y"), None)
        if x_kw is None or y_kw is None:
            return

        x_val = self._numeric_literal(x_kw.value)
        y_val = self._numeric_literal(y_kw.value)

        if not self._should_normalize(x_kw.value, y_kw.value, x_val, y_val):
            return

        x_kw.value = ast.Constant(value=self._scale_coordinate(x_val, self.screen_width))
        y_kw.value = ast.Constant(value=self._scale_coordinate(y_val, self.screen_height))

    @staticmethod
    def _numeric_literal(node: ast.AST) -> Optional[float]:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return float(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            inner = _PyAutoGuiCoordinateNormalizer._numeric_literal(node.operand)
            if inner is not None:
                return -inner
        return None

    @staticmethod
    def _is_float_literal(node: ast.AST) -> bool:
        return isinstance(node, ast.Constant) and isinstance(node.value, float)

    @classmethod
    def _should_normalize(
        cls,
        x_node: ast.AST,
        y_node: ast.AST,
        x_val: Optional[float],
        y_val: Optional[float],
    ) -> bool:
        if x_val is None or y_val is None:
            return False
        if not (0.0 <= x_val <= 1.0 and 0.0 <= y_val <= 1.0):
            return False
        # Only treat decimal literals as normalized coordinates. Integer 0/1 can be valid pixels.
        return cls._is_float_literal(x_node) or cls._is_float_literal(y_node)

    @staticmethod
    def _scale_coordinate(value: float, size: int) -> int:
        if size <= 1:
            return 0
        scaled = int(round(value * (size - 1)))
        return min(max(scaled, 0), size - 1)


class ActionParser:
    """
    Parses agent response strings into executable action commands.

    Supports:
    - Special commands: WAIT, DONE, FAIL
    - Code blocks with optional language identifiers
    - Multiple actions separated by semicolons
    """

    SPECIAL_COMMANDS = {"WAIT", "DONE", "FAIL"}

    def __init__(self, screen_width: int = 1920, screen_height: int = 1080):
        if screen_width <= 0 or screen_height <= 0:
            raise ValueError("screen_width and screen_height must be positive integers")
        self.screen_width = screen_width
        self.screen_height = screen_height

    def parse_actions(self, action_str: str) -> List[str]:
        """
        Parse action string into list of executable commands.

        Args:
            action_str: Raw action string from agent response

        Returns:
            List of parsed action commands
        """
        if not action_str:
            return []

        normalized = "\n".join([line.strip() for line in action_str.split(';') if line.strip()])
        special_cmds = ActionParser.SPECIAL_COMMANDS

        if normalized in special_cmds:
            return [normalized]

        # Check for ```DONE```, ```FAIL```, ```WAIT``` format special commands
        special_pattern = r"```\s*(DONE|FAIL|WAIT)\s*```"
        special_matches = re.findall(special_pattern, action_str, re.IGNORECASE)
        if special_matches:
            return [m.upper() for m in special_matches]

        # Regex to match code blocks, requiring whitespace/newline after language identifier
        pattern = r"```(?:(\w+)\s+)?(.*?)```"
        matches = re.findall(pattern, action_str, re.DOTALL)
        commands: List[str] = []

        if matches:
            for lang, match in matches:
                snippet = match.strip()
                if not snippet:
                    continue
                last_line = snippet.splitlines()[-1].strip() if snippet.splitlines() else ""
                if snippet in special_cmds:
                    commands.append(snippet)
                    continue
                if last_line in special_cmds:
                    body = "\n".join(snippet.splitlines()[:-1]).strip()
                    if body:
                        commands.append(body)
                    commands.append(last_line)
                else:
                    commands.append(snippet)
        else:
            upper_text = action_str.upper()
            if re.search(r'\bDONE\b', upper_text):
                commands.append("DONE")
            elif re.search(r'\bFAIL\b', upper_text):
                commands.append("FAIL")
            elif re.search(r'\bWAIT\b', upper_text):
                commands.append("WAIT")
            elif normalized:
                commands.append(normalized)

        if not commands:
            stripped = action_str.strip()
            triple_match = re.fullmatch(r"`{3}(?:\w+)?\s*(DONE|FAIL|WAIT)\s*`{3}", stripped, re.IGNORECASE)
            if triple_match:
                commands.append(triple_match.group(1).upper())

        return [self._normalize_pyautogui_coordinates(cmd) for cmd in commands if cmd]

    def _normalize_pyautogui_coordinates(self, command: str) -> str:
        """
        Convert normalized pyautogui coordinates such as click(0.5, 0.25)
        into absolute pixel coordinates for the configured screen size.
        """
        try:
            tree = ast.parse(command)
        except SyntaxError:
            return command

        normalizer = _PyAutoGuiCoordinateNormalizer(
            screen_width=self.screen_width,
            screen_height=self.screen_height,
        )
        normalized_tree = normalizer.visit(tree)
        ast.fix_missing_locations(normalized_tree)
        try:
            return ast.unparse(normalized_tree)
        except Exception:
            return command

    @staticmethod
    def strip_special_command(actions: List[str]) -> Tuple[List[str], Optional[str]]:
        """
        Separate special commands from regular actions.

        Args:
            actions: List of action commands

        Returns:
            Tuple of (cleaned_actions, special_command)
            where special_command is DONE/FAIL/WAIT if present, None otherwise
        """
        special_cmd: Optional[str] = None
        cleaned: List[str] = []

        for cmd in actions:
            normalized = cmd.strip().upper()
            if normalized in {"DONE", "FAIL"}:
                special_cmd = normalized
                continue
            if normalized == "WAIT" and len(actions) == 1:
                special_cmd = normalized
                continue
            cleaned.append(cmd)

        return cleaned, special_cmd

    @staticmethod
    def is_special_command(action: str) -> bool:
        """Check if action is a special command."""
        return action.strip().upper() in ActionParser.SPECIAL_COMMANDS
