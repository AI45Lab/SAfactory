import logging
import textwrap
import json
import xml.etree.ElementTree as ET
from typing import Optional, Dict, Any, List, Union

logger = logging.getLogger("osgym.prompt_builder")


class PromptBuilder:
    """
    Builds system and user prompts for environment interaction.
    Supports Kimi-style and Qwen-XML-style formatting.
    """

    SUPPORTED_OBSERVATION_TYPES = {"screenshot", "a11y_tree", "screenshot_a11y_tree"}
    SUPPORTED_PROMPT_FORMATS = {"kimi", "qwen"}

    def __init__(
        self,
        prompt_observation_type: str,
        action_space_type: str,
        a11y_tree_max_nodes: int = 250,
        a11y_tree_max_chars: int = 12000,
        prompt_format: str = "kimi",
        screen_width: int = 1920,
        screen_height: int = 1080,
    ):
        """
        Initialize PromptBuilder.
        """
        if prompt_observation_type not in self.SUPPORTED_OBSERVATION_TYPES:
            raise ValueError(
                f"Unsupported prompt_observation_type: {prompt_observation_type}. "
                f"Expected one of {sorted(self.SUPPORTED_OBSERVATION_TYPES)}."
            )
        
        if prompt_format not in self.SUPPORTED_PROMPT_FORMATS:
            logger.warning(f"Unknown prompt_format: {prompt_format}. Defaulting to 'kimi'.")
            prompt_format = "kimi"

        self.prompt_observation_type = prompt_observation_type
        self.action_space_type = action_space_type
        self.a11y_tree_max_nodes = max(1, int(a11y_tree_max_nodes))
        self.a11y_tree_max_chars = max(200, int(a11y_tree_max_chars))
        self.prompt_format = prompt_format
        self.screen_width = screen_width
        self.screen_height = screen_height

    def build_system_prompt(self, instruction: str) -> str:
        """Build system prompt based on format."""
        observation_description = {
            "screenshot": "You will receive the latest screenshot of the desktop.",
            "a11y_tree": "You will receive the latest accessibility tree of the desktop.",
            "screenshot_a11y_tree": (
                "You will receive the latest screenshot of the desktop and a summarized accessibility tree."
            ),
        }[self.prompt_observation_type]

        if self.prompt_format == "qwen":
            return self._build_qwen_system_prompt(instruction, observation_description)
        
        return self._build_kimi_system_prompt(instruction, observation_description)

    def _build_kimi_system_prompt(self, instruction: str, observation_description: str) -> str:
        return textwrap.dedent(
            f"""
            You are a GUI agent operating a desktop computer.
            {observation_description}
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

            You are asked to complete the following task:
            {instruction}
            """
        ).strip()

    def _build_qwen_system_prompt(self, instruction: str, observation_description: str) -> str:
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

        system_prompt = (
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
            "- If finishing, use action=terminate in the tool call.\n"
            f"\nTask: {instruction}"
        )
        return system_prompt

    def build_user_content(
        self,
        current_obs: Dict[str, Any],
        task_id: str = "",
        screenshot_to_bytes_func=None,
        encode_image_func=None,
    ) -> Union[str, List[Dict[str, Any]]]:
        if self.prompt_format == "qwen":
            instruction_hint = "Review the latest desktop state and return the next action using the `<tool_call>` format."
        else:
            instruction_hint = "Review the latest desktop state and return only the next step in the required `## Action` and `## Code` format."
            
        user_sections: List[str] = [instruction_hint]

        screenshot_bytes = None
        if current_obs.get("screenshot") is not None and screenshot_to_bytes_func:
            screenshot_bytes = screenshot_to_bytes_func(current_obs.get("screenshot"))
            user_sections.append("The latest screenshot is attached.")

        user_text = "\n\n".join(section.strip() for section in user_sections if section).strip()

        if screenshot_bytes and encode_image_func:
            screenshot_url = encode_image_func(screenshot_bytes)
            if screenshot_url:
                return [
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {"url": screenshot_url, "detail": "high"}},
                ]

        return user_text
