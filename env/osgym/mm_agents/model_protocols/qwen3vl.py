"""Qwen3-VL JSON-tool-call OSGym prompt and action protocol."""

import json
import re
from typing import Any, Dict, List, Optional

from .base import ModelProtocol


class Qwen3VLProtocol(ModelProtocol):
    """Protocol matching qwen3vl_agent.py JSON `<tool_call>` format."""

    TOOL_CALL_RE = re.compile(r"<tool_call>\s*(?P<payload>.*?)\s*</tool_call>", re.DOTALL)

    def build_system_prompt(self) -> str:
        description_prompt_lines = [
            "Use a mouse and keyboard to interact with a computer, and take screenshots.",
            "* This is an interface to a desktop GUI. You do not have access to a terminal or applications menu. You must click on desktop icons to start applications.",
            "* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions. E.g. if you click on Firefox and a window doesn't open, try wait and taking another screenshot.",
            "* The screen's resolution is 1000x1000.",
            "* Whenever you intend to move the cursor to click on an element like an icon, you should consult a screenshot to determine the coordinates of the element before moving the cursor.",
            "* If you tried clicking on a program or link but it failed to load even after waiting, try adjusting your cursor position so that the tip of the cursor visually falls on the element that you want to click.",
            "* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked.",
        ]
        description_prompt = "\n".join(description_prompt_lines)

        action_description_prompt = """
* `key`: Performs key down presses on the arguments passed in order, then performs key releases in reverse order.
* `type`: Type a string of text on the keyboard.
* `mouse_move`: Move the cursor to a specified (x, y) pixel coordinate on the screen.
* `left_click`: Click the left mouse button at a specified (x, y) pixel coordinate on the screen.
* `left_click_drag`: Click and drag the cursor to a specified (x, y) pixel coordinate on the screen.
* `right_click`: Click the right mouse button at a specified (x, y) pixel coordinate on the screen.
* `middle_click`: Click the middle mouse button at a specified (x, y) pixel coordinate on the screen.
* `double_click`: Double-click the left mouse button at a specified (x, y) pixel coordinate on the screen.
* `triple_click`: Triple-click the left mouse button at a specified (x, y) pixel coordinate on the screen (simulated as double-click since it is the closest action).
* `scroll`: Performs a scroll of the mouse scroll wheel.
* `hscroll`: Performs a horizontal scroll (mapped to regular scroll).
* `wait`: Wait specified seconds for the change to happen.
* `terminate`: Terminate the current task and report its completion status.
* `answer`: Answer a question.
        """

        tools_def = {
            "type": "function",
            "function": {
                "name_for_human": "computer_use",
                "name": "computer_use",
                "description": description_prompt,
                "parameters": {
                    "properties": {
                        "action": {
                            "description": action_description_prompt,
                            "enum": [
                                "key",
                                "type",
                                "mouse_move",
                                "left_click",
                                "left_click_drag",
                                "right_click",
                                "middle_click",
                                "double_click",
                                "triple_click",
                                "scroll",
                                "hscroll",
                                "wait",
                                "terminate",
                                "answer",
                            ],
                            "type": "string",
                        },
                        "keys": {
                            "description": "Required only by `action=key`.",
                            "type": "array",
                        },
                        "text": {
                            "description": "Required by `action=type` and `action=answer`.",
                            "type": "string",
                        },
                        "coordinate": {
                            "description": "The x,y coordinates for mouse actions in 1000x1000 scale (0-999).",
                            "type": "array",
                        },
                        "pixels": {
                            "description": "The amount of scrolling.",
                            "type": "number",
                        },
                        "time": {
                            "description": "The seconds to wait.",
                            "type": "number",
                        },
                        "status": {
                            "description": "The status of the task.",
                            "type": "string",
                            "enum": ["success", "failure"],
                        },
                    },
                    "required": ["action"],
                    "type": "object",
                },
                "args_format": "Format the arguments as a JSON object.",
            },
        }

        return (
            "# Tools\n\n"
            "You may call one or more functions to assist with the user query.\n\n"
            "You are provided with function signatures within <tools></tools> XML tags:\n"
            "<tools>\n"
            + json.dumps(tools_def)
            + "\n</tools>\n\n"
            "For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n"
            "<tool_call>\n"
            '{"name": <function-name>, "arguments": <args-json-object>}\n'
            "</tool_call>\n\n"
            "# Response format\n\n"
            "Response format for every step:\n"
            "1) Action: a short imperative describing what to do in the UI.\n"
            '2) A single <tool_call>...</tool_call> block containing only the JSON: {"name": <function-name>, "arguments": <args-json-object>}.\n\n'
            "Rules:\n"
            "- Output exactly in the order: Action, <tool_call>.\n"
            "- Be brief: one sentence for Action.\n"
            "- Do not output anything else outside those parts.\n"
            "- If finishing, use action=terminate in the tool call."
        )

    def user_instruction_hint(self, instruction: str) -> str:
        return f"""
Please generate the next move according to the UI screenshot, instruction and previous actions.

Instruction: {instruction}"""

    def format_action_history_entry(
        self,
        idx: int,
        description: str,
        actions: List[str],
        raw_content: str = "",
    ) -> str:
        action_text = description or self._extract_tool_action(raw_content) or "(no action description)"
        return f"Step {idx}: {action_text}"

    def parse_actions(self, action_str: str) -> List[str]:
        if not action_str or not action_str.strip():
            return []

        commands = []
        for payload in self._extract_tool_payloads(action_str):
            parsed = self._loads_json_object(payload)
            if not parsed:
                continue

            name = parsed.get("name")
            arguments = parsed if "action" in parsed else parsed.get("arguments", {})
            if name and name != "computer_use":
                continue
            if isinstance(arguments, str):
                arguments = self._loads_json_object(arguments) or {}
            if not isinstance(arguments, dict):
                continue
            commands.extend(self._process_tool_arguments(arguments))
        return commands

    def _extract_tool_payloads(self, action_str: str) -> List[str]:
        payloads = [match.group("payload").strip() for match in self.TOOL_CALL_RE.finditer(action_str)]
        if payloads:
            return payloads

        standalone_payloads = []
        for line in action_str.splitlines():
            stripped = line.strip()
            if stripped.startswith("{") and stripped.endswith("}"):
                standalone_payloads.append(stripped)
        return standalone_payloads

    def _process_tool_arguments(self, arguments: Dict[str, Any]) -> List[str]:
        action = arguments.get("action")
        if not action:
            return []

        action = str(action).strip()
        coordinate = self._parse_coordinate(arguments.get("coordinate"))
        keys = self._parse_keys(arguments.get("keys", []))
        text = arguments.get("text", "")

        action_map = {
            "left_click": "click",
            "right_click": "rightClick",
            "middle_click": "middleClick",
            "double_click": "doubleClick",
            "triple_click": "doubleClick",
            "mouse_move": "moveTo",
            "left_click_drag": "dragTo",
        }

        if action in action_map:
            return [self._mouse_command(action_map[action], coordinate)]
        if action == "type":
            return self._type_commands(str(text))
        if action == "key":
            return self._key_commands(keys)
        if action in {"scroll", "hscroll"}:
            return [f"pyautogui.scroll({self._parse_pixels(arguments.get('pixels', 0))})"]
        if action == "wait":
            return ["WAIT"]
        if action in {"terminate", "answer"}:
            status = str(arguments.get("status", "success")).lower()
            return ["DONE" if status == "success" else "FAIL"]
        return []

    @staticmethod
    def _loads_json_object(raw_value: Any) -> Optional[Dict[str, Any]]:
        if isinstance(raw_value, dict):
            return raw_value
        if not isinstance(raw_value, str):
            return None
        try:
            parsed = json.loads(raw_value)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None

    def _mouse_command(self, func_name: str, coordinate: Optional[tuple]) -> str:
        if coordinate:
            return f"pyautogui.{func_name}({coordinate[0]}, {coordinate[1]})"
        return f"pyautogui.{func_name}()"

    def _parse_coordinate(self, raw_coord: Any) -> Optional[tuple]:
        if raw_coord is None:
            return None
        try:
            if isinstance(raw_coord, str):
                cleaned = raw_coord.strip("[]() ")
                parts = [part.strip() for part in cleaned.split(",")]
            else:
                parts = list(raw_coord)
            if len(parts) < 2:
                return None
            x = int(float(parts[0]) * (self.screen_width / 999.0))
            y = int(float(parts[1]) * (self.screen_height / 999.0))
            return x, y
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _parse_keys(keys: Any) -> List[str]:
        if keys is None:
            return []
        if isinstance(keys, str):
            try:
                parsed = json.loads(keys) if keys.startswith(("[", "{")) else [keys]
            except json.JSONDecodeError:
                parsed = [keys]
        else:
            parsed = keys
        if isinstance(parsed, (list, tuple)):
            return [str(key) for key in parsed if str(key)]
        return [str(parsed)] if parsed else []

    @staticmethod
    def _parse_pixels(pixels: Any) -> int:
        try:
            return int(float(pixels))
        except (ValueError, TypeError):
            return 0

    @staticmethod
    def _py_str(value: Any) -> str:
        return json.dumps(str(value), ensure_ascii=False)

    def _type_commands(self, text: str) -> List[str]:
        commands = []
        parts = text.split("\n")
        for idx, part in enumerate(parts):
            if part:
                commands.append(f"pyautogui.typewrite({self._py_str(part)}, interval=0.03)")
            if idx < len(parts) - 1:
                commands.append("pyautogui.press('enter')")
        return commands

    def _key_commands(self, keys: List[str]) -> List[str]:
        if len(keys) > 1:
            return [f"pyautogui.hotkey({', '.join(self._py_str(key) for key in keys)})"]
        if keys:
            return [f"pyautogui.press({self._py_str(keys[0])})"]
        return []

    def _extract_tool_action(self, content: str) -> str:
        payloads = self._extract_tool_payloads(content or "")
        if not payloads:
            return ""
        parsed = self._loads_json_object(payloads[0])
        if not parsed:
            return ""
        arguments = parsed if "action" in parsed else parsed.get("arguments", {})
        if isinstance(arguments, str):
            arguments = self._loads_json_object(arguments) or {}
        if not isinstance(arguments, dict):
            return ""
        action = arguments.get("action")
        return str(action) if action else ""
