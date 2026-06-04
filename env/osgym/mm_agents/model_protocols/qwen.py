"""Qwen XML-tool-call OSGym prompt and action protocol."""

import json
import re
from typing import Dict, List

from .base import ModelProtocol


class QwenProtocol(ModelProtocol):
    """Protocol using `<tool_call>` XML blocks with 1000x1000 coordinates."""

    def build_system_prompt(self) -> str:
        description_prompt_lines = [
            "Use a mouse and keyboard to interact with a computer, and take screenshots.",
            "* This is an interface to a desktop GUI. You do not have access to a terminal or applications menu. You must click on desktop icons to start applications.",
            "* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions.",
            "* The screen's resolution is 1000x1000.",
            "* Whenever you intend to move the cursor to click on an element like an icon, you should consult a screenshot to determine the coordinates of the element before moving the cursor.",
            "* If you tried clicking on a program or link but it failed to load, even after waiting, try adjusting your cursor position so that the tip of the cursor visually falls on the element that you want to click.",
            "* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked.",
        ]
        description_prompt = "\n".join(description_prompt_lines)

        action_description_prompt = """
* `key`: Performs key down presses on the arguments passed in order, then performs key releases in reverse order.
* `type`: Type a string of text on the keyboard.
* `mouse_move`: Move the cursor to a specified (x, y) pixel coordinate on the screen.
* `left_click`: Click the left mouse button at a specified (x, y) pixel coordinate on the screen. Optional `text` parameter can specify modifier keys (e.g., "ctrl", "shift", "ctrl+shift") that will be held during the click.
* `left_click_drag`: Click and drag the cursor to a specified (x, y) coordinate.
* `right_click`: Click the right mouse button at a specified (x, y) pixel coordinate on the screen. Optional `text` parameter can specify modifier keys that will be held during the click.
* `middle_click`: Click the middle mouse button at a specified (x, y) pixel coordinate on the screen. Optional `text` parameter can specify modifier keys that will be held during the click.
* `double_click`: Double-click the left mouse button at a specified (x, y) pixel coordinate on the screen. Optional `text` parameter can specify modifier keys that will be held during the click.
* `triple_click`: Triple-click the left mouse button at a specified (x, y) pixel coordinate on the screen. Optional `text` parameter can specify modifier keys that will be held during the click.
* `scroll`: Performs a scroll of the mouse scroll wheel. Optional `text` parameter can specify a modifier key (e.g., "shift", "ctrl") that will be held during scrolling.
* `hscroll`: Performs a horizontal scroll (mapped to regular scroll). Optional `text` parameter can specify a modifier key that will be held during scrolling.
* `wait`: Wait specified seconds for the change to happen.
* `terminate`: Terminate the current task and report its completion status.
* `answer`: Answer a question."""

        tools_def = {
            "type": "function",
            "function": {
                "name": "computer_use",
                "description": description_prompt,
                "parameters": {
                    "type": "object",
                    "required": ["action"],
                    "properties": {
                        "action": {
                            "type": "string",
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
                        },
                        "keys": {"type": "array", "description": "Required only by `action=key`."},
                        "text": {
                            "type": "string",
                            "description": "Required by `action=type` and `action=answer`. Optional for click actions (left_click, right_click, middle_click, double_click, triple_click) to specify modifier keys.",
                        },
                        "coordinate": {"type": "array", "description": "Pixel (x, y) coordinates in 1000x1000 scale (0-999)."},
                        "pixels": {"type": "number", "description": "Scroll amount."},
                        "time": {"type": "number", "description": "Seconds to wait."},
                        "status": {
                            "type": "string",
                            "description": "Task status for terminate.",
                            "enum": ["success", "failure"],
                        },
                    },
                },
            },
        }

        return (
            "You are a multi-purpose intelligent assistant. Based on my requests, you can use tools to help me complete various tasks.\n\n"
            "# Tools\n\n"
            "You have access to the following functions:\n\n"
            "<tools>\n"
            + json.dumps(tools_def)
            + "\n</tools>\n\n"
            "If you choose to call a function ONLY reply in the following format with NO suffix:\n\n"
            "<tool_call>\n"
            "<function=example_function_name>\n"
            "<parameter=example_parameter_1>\n"
            "value_1\n"
            "</parameter>\n"
            "<parameter=example_parameter_2>\n"
            "This is the value for the second parameter\n"
            "that can span\n"
            "multiple lines\n"
            "</parameter>\n"
            "</function>\n"
            "</tool_call>\n\n"
            "<IMPORTANT>\n"
            "Reminder:\n"
            "- Function calls MUST follow the specified format.\n"
            "- Required parameters MUST be specified.\n"
            "- Collapsed screenshots appear as text: <history_image_removed_for_memory_saving>\n"
            "</IMPORTANT>\n\n"
            "# Response format\n\n"
            "Response format for every step:\n"
            "1) Action: a short imperative describing what to do in the UI.\n"
            "2) A single <tool_call>...</tool_call> block.\n\n"
            "Rules:\n"
            "- Output exactly in the order: Action, <tool_call>.\n"
            "- Be brief: one sentence for Action.\n"
            "- Do not output Code, Python, pyautogui, markdown code fences, or any extra text after </tool_call>.\n"
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
        tool_call = self._extract_tool_call(raw_content)
        code = tool_call or "(no tool_call)"
        return f"{idx}. Description: {description}\nCode:\n{code}"

    def parse_actions(self, action_str: str) -> List[str]:
        if not action_str or not action_str.strip():
            return []
        if "<tool_call>" not in action_str:
            return []

        pyautogui_commands = []
        function_pattern = re.compile(r"<function=(?P<name>.*?)>(?P<body>.*?)</function>", re.DOTALL)
        parameter_pattern = re.compile(r"<parameter=(?P<name>.*?)>(?P<value>.*?)</parameter>", re.DOTALL)

        for func_match in function_pattern.finditer(action_str):
            params = {}
            for param_match in parameter_pattern.finditer(func_match.group("body")):
                params[param_match.group("name").strip()] = param_match.group("value").strip()
            pyautogui_commands.extend(self._process_xml_params_to_pyautogui(params))

        return pyautogui_commands

    @staticmethod
    def _extract_tool_call(content: str) -> str:
        if not content:
            return ""
        match = re.search(r"<tool_call>.*?</tool_call>", content, re.DOTALL)
        return match.group(0).strip() if match else ""

    def _process_xml_params_to_pyautogui(self, params: Dict) -> List[str]:
        action = params.get("action")
        if not action:
            return []

        coordinate = self._parse_coordinate(params.get("coordinate"))
        text = params.get("text", "")
        keys = self._parse_keys(params.get("keys", []))

        def py_str(value):
            return json.dumps(str(value), ensure_ascii=False)

        commands = []
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
            func_name = action_map[action]
            if coordinate:
                commands.append(f"pyautogui.{func_name}({coordinate[0]}, {coordinate[1]})")
            else:
                commands.append(f"pyautogui.{func_name}()")
        elif action == "type":
            commands.append(f"pyautogui.typewrite({py_str(text)})")
        elif action == "key":
            if len(keys) > 1:
                commands.append(f"pyautogui.hotkey({', '.join(py_str(k) for k in keys)})")
            elif keys:
                commands.append(f"pyautogui.press({py_str(keys[0])})")
        elif action == "wait":
            commands.append("WAIT")
        elif action in {"terminate", "answer"}:
            status = params.get("status", "success")
            commands.append("DONE" if status == "success" else "FAIL")
        elif action in {"scroll", "hscroll"}:
            commands.append(f"pyautogui.scroll({self._parse_pixels(params.get('pixels', 0))})")

        return commands

    def _parse_coordinate(self, raw_coord):
        if not isinstance(raw_coord, str):
            return None
        try:
            cleaned_coord = raw_coord.strip("[]() ")
            parts = [part.strip() for part in cleaned_coord.split(",")]
            if len(parts) < 2:
                return None
            x = int(float(parts[0]) * (self.screen_width / 999.0))
            y = int(float(parts[1]) * (self.screen_height / 999.0))
            return x, y
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _parse_keys(keys):
        if not isinstance(keys, str):
            return keys
        try:
            return json.loads(keys) if keys.startswith(("[", "{")) else [keys]
        except Exception:
            return [keys]

    @staticmethod
    def _parse_pixels(pixels) -> int:
        try:
            return int(float(pixels))
        except (ValueError, TypeError):
            return 0
