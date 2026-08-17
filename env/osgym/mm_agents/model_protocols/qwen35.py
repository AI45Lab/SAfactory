"""Qwen3.5 XML-tool-call OSGym prompt and action protocol."""

import json
import re
from typing import Dict, List

from .base import ModelProtocol


class Qwen35Protocol(ModelProtocol):
    """Qwen3.5 protocol using XML tool calls and 1000x1000 coordinates."""

    allow_multiple_tool_calls = True
    step_limit_signal = None

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
                            "description": "Required by `action=type` and `action=answer`. Optional for click actions (left_click, right_click, middle_click, double_click, triple_click) to specify modifier keys (e.g., 'ctrl', 'shift', 'ctrl+shift'). Optional for scroll actions (scroll, hscroll) to specify a modifier key (e.g., 'shift', 'ctrl') to hold during scrolling.",
                        },
                        "coordinate": {"type": "array", "description": "(x, y) coordinates."},
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
            "If you choose to call one or more functions, reply using one block per call in the following format:\n\n"
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
            "- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n"
            "- Required parameters MUST be specified\n"
            "- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n"
            "- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n"
            "- Collapsed screenshots appear as text: This screenshot has been collapsed.\n"
            "</IMPORTANT>\n\n"
            "# Response format\n\n"
            "Response format for every step:\n"
            "1) Action: a short imperative describing what to do in the UI.\n"
            "2) One or more consecutive <tool_call>...</tool_call> blocks.\n\n"
            "Rules:\n"
            "- Output exactly in the order: Action, then every <tool_call> block.\n"
            "- Calls are executed in the order shown, and Action must briefly describe the complete sequence.\n"
            "- Do not output anything else outside those parts.\n"
            "- If finishing, the final tool call must use action=terminate with an explicit status."
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
        matches = re.findall(r"<tool_call>.*?</tool_call>", content, re.DOTALL)
        return "\n".join(match.strip() for match in matches)

    def _process_xml_params_to_pyautogui(self, params: Dict) -> List[str]:
        action = params.get("action")
        if not action:
            return []

        coordinate = self._parse_coordinate(params.get("coordinate"))
        text = params.get("text", "")
        keys = self._parse_keys(params.get("keys", []))

        def py_str(value):
            return json.dumps(str(value), ensure_ascii=False)

        def adjust_coordinates(x: float, y: float):
            return int(float(x) * self.screen_width / 999), int(float(y) * self.screen_height / 999)

        def press_modifier_keys() -> None:
            if text:
                for key in str(text).split("+"):
                    key = key.strip().lower()
                    if key:
                        commands.append(f"pyautogui.keyDown({py_str(key)})")

        def release_modifier_keys() -> None:
            if text:
                modifier_keys = [key.strip().lower() for key in str(text).split("+") if key.strip()]
                for key in reversed(modifier_keys):
                    commands.append(f"pyautogui.keyUp({py_str(key)})")

        commands = []

        if action == "left_click":
            press_modifier_keys()
            if coordinate:
                x, y = adjust_coordinates(*coordinate)
                commands.append(f"pyautogui.click({x}, {y})")
            else:
                commands.append("pyautogui.click()")
            release_modifier_keys()
        elif action == "right_click":
            press_modifier_keys()
            if coordinate:
                x, y = adjust_coordinates(*coordinate)
                commands.append(f"pyautogui.rightClick({x}, {y})")
            else:
                commands.append("pyautogui.rightClick()")
            release_modifier_keys()
        elif action == "middle_click":
            press_modifier_keys()
            if coordinate:
                x, y = adjust_coordinates(*coordinate)
                commands.append(f"pyautogui.middleClick({x}, {y})")
            else:
                commands.append("pyautogui.middleClick()")
            release_modifier_keys()
        elif action == "double_click":
            press_modifier_keys()
            if coordinate:
                x, y = adjust_coordinates(*coordinate)
                commands.append(f"pyautogui.doubleClick({x}, {y})")
            else:
                commands.append("pyautogui.doubleClick()")
            release_modifier_keys()
        elif action == "triple_click":
            press_modifier_keys()
            if coordinate:
                x, y = adjust_coordinates(*coordinate)
                commands.append(f"pyautogui.tripleClick({x}, {y})")
            else:
                commands.append("pyautogui.tripleClick()")
            release_modifier_keys()
        elif action == "type":
            commands.append(f"pyautogui.typewrite({py_str(text)})")
        elif action == "key":
            keys_str = ", ".join(py_str(key) for key in keys)
            if len(keys) > 1:
                commands.append(f"pyautogui.hotkey({keys_str})")
            else:
                commands.append(f"pyautogui.press({keys_str})")
        elif action in {"scroll", "hscroll"}:
            press_modifier_keys()
            commands.append(f"pyautogui.scroll({self._parse_pixels(params.get('pixels', 0))})")
            release_modifier_keys()
        elif action == "wait":
            commands.append("WAIT")
        elif action == "terminate":
            status = str(params.get("status", "success")).lower()
            commands.append("FAIL" if status == "failure" else "DONE")
        elif action == "answer":
            commands.append("DONE")
        elif action == "mouse_move":
            if coordinate:
                x, y = adjust_coordinates(*coordinate)
                commands.append(f"pyautogui.moveTo({x}, {y})")
            else:
                commands.append("pyautogui.moveTo(0, 0)")
        elif action == "left_click_drag":
            if coordinate:
                x, y = adjust_coordinates(*coordinate)
                duration = 0.5
                if "duration" in params:
                    try:
                        duration = float(params["duration"])
                    except Exception:
                        duration = 0.5
                commands.append(f"pyautogui.dragTo({x}, {y}, duration={duration})")
            else:
                commands.append("pyautogui.dragTo(0, 0)")

        return commands

    @staticmethod
    def _parse_coordinate(raw_coord):
        if isinstance(raw_coord, str):
            try:
                raw_coord = json.loads(raw_coord)
            except Exception:
                return None
        if isinstance(raw_coord, list) and len(raw_coord) >= 2:
            return raw_coord[0], raw_coord[1]
        return None

    @staticmethod
    def _parse_keys(raw_keys):
        if isinstance(raw_keys, str):
            try:
                raw_keys = json.loads(raw_keys)
            except Exception:
                raw_keys = [raw_keys]
        if isinstance(raw_keys, list):
            return [str(key).strip() for key in raw_keys]
        return [str(raw_keys).strip()]

    @staticmethod
    def _parse_pixels(pixels) -> int:
        try:
            return int(float(pixels))
        except (ValueError, TypeError):
            return 0
