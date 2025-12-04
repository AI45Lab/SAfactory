"""Adapters for reusing RiOSWorld prompt templates inside OSGym."""

from typing import Optional, Tuple

from .prompts import (
    SYS_PROMPT_IN_SCREENSHOT_OUT_CODE,
    SYS_PROMPT_IN_SCREENSHOT_OUT_ACTION,
    SYS_PROMPT_IN_SCREENSHOT_OUT_CODE_FEW_SHOT,
    SYS_PROMPT_IN_SCREENSHOT_OUT_ACTION_FEW_SHOT,
    SYS_PROMPT_IN_A11Y_OUT_CODE,
    SYS_PROMPT_IN_A11Y_OUT_ACTION,
    SYS_PROMPT_IN_BOTH_OUT_CODE,
    SYS_PROMPT_IN_BOTH_OUT_ACTION,
    SYS_PROMPT_IN_SOM_OUT_TAG,
)
from .agent import (
    linearize_accessibility_tree,
    trim_accessibility_tree,
    tag_screenshot,
)

PROMPT_MATRIX = {
    ("screenshot", "pyautogui"): SYS_PROMPT_IN_SCREENSHOT_OUT_CODE,
    ("screenshot", "computer_13"): SYS_PROMPT_IN_SCREENSHOT_OUT_ACTION,
    ("a11y_tree", "pyautogui"): SYS_PROMPT_IN_A11Y_OUT_CODE,
    ("a11y_tree", "computer_13"): SYS_PROMPT_IN_A11Y_OUT_ACTION,
    ("screenshot_a11y_tree", "pyautogui"): SYS_PROMPT_IN_BOTH_OUT_CODE,
    ("screenshot_a11y_tree", "computer_13"): SYS_PROMPT_IN_BOTH_OUT_ACTION,
    ("som", "pyautogui"): SYS_PROMPT_IN_SOM_OUT_TAG,
}

PROMPT_MATRIX_FEW_SHOT = {
    ("screenshot", "pyautogui"): SYS_PROMPT_IN_SCREENSHOT_OUT_CODE_FEW_SHOT,
    ("screenshot", "computer_13"): SYS_PROMPT_IN_SCREENSHOT_OUT_ACTION_FEW_SHOT,
    ("a11y_tree", "pyautogui"): SYS_PROMPT_IN_A11Y_OUT_CODE,
    ("a11y_tree", "computer_13"): SYS_PROMPT_IN_A11Y_OUT_ACTION,
    ("screenshot_a11y_tree", "pyautogui"): SYS_PROMPT_IN_BOTH_OUT_CODE,
    ("screenshot_a11y_tree", "computer_13"): SYS_PROMPT_IN_BOTH_OUT_ACTION,
    ("som", "pyautogui"): SYS_PROMPT_IN_SOM_OUT_TAG,
}


def get_system_prompt(observation_type: str, action_space: str, few_shot: bool = False) -> str:
    """Return the canonical RiOSWorld system prompt for the given configuration."""
    matrix = PROMPT_MATRIX_FEW_SHOT if few_shot else PROMPT_MATRIX
    key = (observation_type, action_space)
    if key not in matrix:
        raise ValueError(
            f"Unsupported (observation_type={observation_type}, action_space={action_space}) combination"
        )
    return matrix[key]


def build_observation_prompt(
    observation_type: str,
    screenshot_bytes: Optional[bytes],
    accessibility_tree: Optional[str],
    platform: str = "ubuntu",
    max_tokens: int = 10000,
) -> Tuple[str, Optional[bytes]]:
    """Format user-facing prompt text and pick the right image payload."""
    tree_text = None
    if accessibility_tree:
        tree_text = linearize_accessibility_tree(accessibility_tree, platform=platform)
        tree_text = trim_accessibility_tree(tree_text, max_tokens)

    if observation_type == "screenshot":
        description = "Given the screenshot as below. What's the next step that you will do to help with the task?"
        return description, screenshot_bytes

    if observation_type == "a11y_tree":
        if tree_text:
            description = (
                "Given the info from accessibility tree as below:\n"
                f"{tree_text}\n"
                "What's the next step that you will do to help with the task?"
            )
        else:
            description = "Accessibility tree missing for the current observation."
        return description, None

    if observation_type == "screenshot_a11y_tree":
        if tree_text:
            description = (
                "Given the screenshot and info from accessibility tree as below:\n"
                f"{tree_text}\n"
                "What's the next step that you will do to help with the task?"
            )
        else:
            description = (
                "Given the screenshot as below (accessibility tree unavailable). What's the next step that you will do "
                "to help with the task?"
            )
        return description, screenshot_bytes

    if observation_type == "som":
        if screenshot_bytes and accessibility_tree:
            _, _, tagged_screenshot, element_list = tag_screenshot(
                screenshot_bytes, accessibility_tree, platform=platform
            )
            element_text = trim_accessibility_tree(element_list, max_tokens)
            description = (
                "Given the tagged screenshot and info from accessibility tree as below:\n"
                f"{element_text}\n"
                "What's the next step that you will do to help with the task?"
            )
            return description, tagged_screenshot

        fallback = (
            "Tagged screenshot or accessibility tree missing; describe your next action based on the limited context."
        )
        return fallback, screenshot_bytes

    raise ValueError(f"Unsupported observation_type: {observation_type}")
