"""
Action Parser Module

Parses agent response strings into executable action commands.
Handles special commands (WAIT, DONE, FAIL), `computer.*` helpers,
and the local `## Action` + `## Code` response protocol.
"""

import ast
import json
import logging
import re
from typing import List, Tuple, Optional, Dict

logger = logging.getLogger("osgym.action_parser")


class _PyAutoGuiCoordinateNormalizer(ast.NodeTransformer):
    """Convert normalized [0, 1] coordinates in AST into absolute pixels."""

    def __init__(self, screen_width: int, screen_height: int):
        self.width = screen_width
        self.height = screen_height

    def visit_Call(self, node):
        self.generic_visit(node)
        if isinstance(node.func, ast.Attribute) and node.func.value.id == "pyautogui":
            new_args = []
            for arg in node.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, float) and 0.0 <= arg.value <= 1.0:
                    # Heuristic: assume float in [0,1] is a normalized coordinate
                    if node.func.attr in ["click", "moveTo", "dragTo", "rightClick", "doubleClick", "middleClick"]:
                        arg_idx = node.args.index(arg)
                        if arg_idx == 0:
                            new_args.append(ast.Constant(value=int(arg.value * self.width)))
                            continue
                        elif arg_idx == 1:
                            new_args.append(ast.Constant(value=int(arg.value * self.height)))
                            continue
                new_args.append(arg)
            node.args = new_args
        return node


class ActionParser:
    """
    Parses and sanitizes agent actions.
    """

    SPECIAL_COMMANDS = {"WAIT", "DONE", "FAIL"}

    def __init__(self, screen_width: int = 1920, screen_height: int = 1080, prompt_format: str = "kimi"):
        self.width = screen_width
        self.height = screen_height
        self.prompt_format = prompt_format
        self._normalizer = _PyAutoGuiCoordinateNormalizer(screen_width, screen_height)

    def parse_actions(self, action_str: str) -> List[str]:
        """
        Parse agent response string into a list of executable commands.
        """
        if not action_str or not action_str.strip():
            return []

        if "<tool_call>" in action_str:
            return self._parse_xml_actions(action_str)

        return self._parse_legacy_actions(action_str)

    def _parse_xml_actions(self, action_str: str) -> List[str]:
        """Parse Qwen-style XML tool calls."""
        pyautogui_commands = []
        function_pattern = re.compile(r"<function=(?P<name>.*?)>(?P<body>.*?)</function>", re.DOTALL)
        parameter_pattern = re.compile(r"<parameter=(?P<name>.*?)>(?P<value>.*?)</parameter>", re.DOTALL)

        for func_match in function_pattern.finditer(action_str):
            params = {}
            for param_match in parameter_pattern.finditer(func_match.group("body")):
                params[param_match.group("name").strip()] = param_match.group("value").strip()
            
            if params:
                cmds = self._process_xml_params_to_pyautogui(params)
                pyautogui_commands.extend(cmds)

        return pyautogui_commands

    def _process_xml_params_to_pyautogui(self, params: Dict) -> List[str]:
        """Convert XML parameters into standard pyautogui strings."""
        action = params.get("action")
        if not action:
            return []

        cmds = []
        raw_coord = params.get("coordinate")
        coordinate = None
        if isinstance(raw_coord, str):
            try:
                # Handle cases like "[100, 200]" or "100, 200"
                cleaned_coord = raw_coord.strip("[]() ")
                parts = [p.strip() for p in cleaned_coord.split(",")]
                if len(parts) >= 2:
                    x, y = float(parts[0]), float(parts[1])
                    # If we are parsing XML, we ALWAYS assume 1000x1000 normalization (Qwen-style)
                    x_scale = self.width / 999.0
                    y_scale = self.height / 999.0
                    coordinate = (int(x * x_scale), int(y * y_scale))
            except (ValueError, TypeError):
                pass

        text = params.get("text", "")
        keys = params.get("keys", [])
        if isinstance(keys, str):
            try: 
                keys = json.loads(keys) if (keys.startswith("[") or keys.startswith("{")) else [keys]
            except Exception: 
                keys = [keys]

        def _py_str(s): return json.dumps(str(s), ensure_ascii=False)

        # Mapping of XML actions to pyautogui commands
        action_map = {
            "left_click": ("click", True),
            "right_click": ("rightClick", True),
            "middle_click": ("middleClick", True),
            "double_click": ("doubleClick", True),
            "triple_click": ("doubleClick", True), # Map triple_click to doubleClick
            "mouse_move": ("moveTo", True),
            "left_click_drag": ("dragTo", True),
        }

        if action in action_map:
            func_name, needs_coord = action_map[action]
            if needs_coord and coordinate:
                cmds.append(f"pyautogui.{func_name}({coordinate[0]}, {coordinate[1]})")
            else:
                cmds.append(f"pyautogui.{func_name}()")
        elif action == "type":
            cmds.append(f"pyautogui.typewrite({_py_str(text)})")
        elif action == "key":
            if len(keys) > 1:
                cmds.append(f"pyautogui.hotkey({', '.join(_py_str(k) for k in keys)})")
            elif keys:
                cmds.append(f"pyautogui.press({_py_str(keys[0])})")
        elif action == "wait":
            cmds.append("WAIT")
        elif action in {"terminate", "answer"}:
            status = params.get("status", "success")
            cmds.append("DONE" if status == "success" else "FAIL")
        elif action in {"scroll", "hscroll"}:
            try:
                pixels = int(float(params.get("pixels", 0)))
            except (ValueError, TypeError):
                pixels = 0
            cmds.append(f"pyautogui.scroll({pixels})")
        
        return cmds

    def _parse_legacy_actions(self, action_str: str) -> List[str]:
        """Parse legacy ## Action / ## Code block format."""
        commands = []
        code_blocks = re.findall(r"```python\n(.*?)\n```", action_str, re.DOTALL)
        
        if not code_blocks:
            code_match = re.search(r"## Code:\s*(.*)", action_str, re.DOTALL)
            if code_match:
                content = code_match.group(1).split("##")[0].strip()
                code_blocks = [content]

        for block in code_blocks:
            lines = [line.strip() for line in block.split("\n") if line.strip()]
            for line in lines:
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

    def _map_computer_helper(self, command: str) -> str:
        if "wait()" in command:
            return "WAIT"
        if "terminate" in command:
            if 'status="success"' in command or "status='success'" in command:
                return "DONE"
            return "FAIL"
        return command

    def _try_get_special_command(self, command: str) -> Optional[str]:
        cleaned = command.strip().upper()
        for cmd in self.SPECIAL_COMMANDS:
            if cleaned.startswith(cmd):
                return cmd
        return None

    @staticmethod
    def is_special_command(action: str) -> bool:
        return action.strip().upper() in ActionParser.SPECIAL_COMMANDS

    def strip_special_command(self, actions: List[str]) -> Tuple[List[str], Optional[str]]:
        special_cmd = None
        remaining_actions = []
        for act in actions:
            cmd = self._try_get_special_command(act)
            if cmd:
                special_cmd = cmd
            else:
                remaining_actions.append(act)
        return remaining_actions, special_cmd
