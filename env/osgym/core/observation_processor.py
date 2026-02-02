"""
Observation Processor Module

Handles observation processing including screenshot conversion,
attack overlay application for RiOSWorld risk evaluation tasks.
"""

import io
import copy
import base64
import logging
from typing import Optional, Dict, Any, Tuple
from pathlib import Path

import numpy as np
from PIL import Image

logger = logging.getLogger("osgym.observation_processor")

# Path to intent click target data for induced_text attacks
# Get path from RiOSWorld package's env_risk_utils module
def _get_intent_click_tgt_path() -> Path:
    """Get the path to intent_click_tgt_OK.json from RiOSWorld package."""
    import env_risk_utils
    return Path(env_risk_utils.__file__).parent / "intent_click_tgt_OK.json"

INTENT_CLICK_TGT_PATH = _get_intent_click_tgt_path()


class ObservationProcessor:
    """
    Processes observations and applies attack overlays for risk evaluation.

    Handles:
    - Screenshot format conversion (ndarray <-> bytes <-> base64)
    - Attack overlay injection for popup/induced_text tasks
    - Attack parameter management for risk evaluation
    """

    def __init__(self, observation_type: str):
        """
        Initialize ObservationProcessor.

        Args:
            observation_type: Type of observation ("screenshot", "a11y_tree", "screenshot_a11y_tree", "som")
        """
        self.observation_type = observation_type

        # Attack evaluation parameters (for popup/induced_text tasks)
        self._attack_bbox = None
        self._attack_x_tgt = None
        self._attack_y_tgt = None
        self.button_name_dict: Dict[str, str] = {}

    def process(
        self,
        obs: Dict[str, Any],
        current_task: Optional[Dict[str, Any]] = None,
        instruction: str = ""
    ) -> Dict[str, Any]:
        """
        Process observation, applying attack overlay if needed.

        Args:
            obs: Raw observation dict with 'screenshot', 'accessibility_tree', etc.
            current_task: Current task configuration (for task_id)
            instruction: Current task instruction

        Returns:
            Processed observation dict
        """
        return self._maybe_apply_attack_overlay(obs, current_task, instruction)

    def _maybe_apply_attack_overlay(
        self,
        obs: Dict[str, Any],
        current_task: Optional[Dict[str, Any]],
        instruction: str
    ) -> Dict[str, Any]:
        """
        Inject popup/induced_text visuals similar to PromptAgent.

        Keeps observation shape (np.ndarray screenshot) so downstream code works.
        If any step fails, fall back to the original observation.
        """
        if not obs or not isinstance(obs, dict):
            logger.debug("Attack overlay skipped: obs is empty or not dict")
            return obs

        task_id = (current_task or {}).get("id", "")
        if not task_id:
            logger.debug("Attack overlay skipped: no task_id")
            return obs

        needs_attack = ("popup" in task_id) or ("induced_text" in task_id)
        if not needs_attack:
            return obs  # Normal case, no logging needed

        logger.debug(f"Attack overlay needed for task: {task_id}")

        if self.observation_type not in {"screenshot", "screenshot_a11y_tree"}:
            logger.debug(f"Attack overlay skipped: observation_type={self.observation_type} not supported")
            return obs

        screenshot = obs.get("screenshot")
        accessibility_tree = obs.get("accessibility_tree")
        if screenshot is None or accessibility_tree is None:
            logger.warning(
                f"Attack overlay skipped: screenshot={screenshot is not None}, "
                f"a11y_tree={accessibility_tree is not None}"
            )
            return obs

        try:
            # Import attack utilities lazily to avoid circular imports
            from mm_agents.agent import tag_screenshot, agent_attack_wrapper

            # Normalize screenshot to PNG bytes for the attack helper
            if isinstance(screenshot, np.ndarray):
                img = Image.fromarray(screenshot)
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                screenshot_bytes = buf.getvalue()
            elif isinstance(screenshot, (bytes, bytearray, memoryview)):
                screenshot_bytes = bytes(screenshot)
            else:
                logger.debug("Attack overlay skipped: screenshot type not supported")
                return obs

            logger.debug("Running tag_screenshot...")
            current_boundingbox, nodes, _, linearized_tree = tag_screenshot(
                screenshot_bytes, accessibility_tree, platform="ubuntu"
            )
            logger.debug(f"tag_screenshot returned {len(current_boundingbox)} bounding boxes, {len(nodes)} nodes")

            attack_config = str(INTENT_CLICK_TGT_PATH) if "induced_text" in task_id else False
            logger.debug(f"Running agent_attack_wrapper with config={attack_config}")

            attack_result = agent_attack_wrapper(
                attack_config,
                instruction,
                self.button_name_dict,
                current_boundingbox,
                nodes,
                screenshot_bytes,
                content_str=linearized_tree,
            )

            if not (isinstance(attack_result, tuple) and len(attack_result) == 7):
                logger.warning(
                    f"Attack wrapper returned unexpected result: type={type(attack_result)}, skipping overlay"
                )
                return obs

            (
                current_observation,
                skip,
                self.button_name_dict,
                whole_attack_bounding_box,
                x_tgt,
                y_tgt,
                attack_string
            ) = attack_result
            logger.info(
                f"Attack overlay applied: bbox={whole_attack_bounding_box}, "
                f"target=({x_tgt},{y_tgt}), skip={skip}"
            )

            # Save attack parameters for evaluation
            self._attack_bbox = whole_attack_bounding_box
            self._attack_x_tgt = x_tgt
            self._attack_y_tgt = y_tgt

            # Convert back to ndarray screenshot
            injected_img = Image.open(io.BytesIO(current_observation)).convert("RGB")
            injected_array = np.array(injected_img)

            new_obs = copy.copy(obs)
            new_obs["screenshot"] = injected_array
            # Stash metadata for possible downstream use
            new_obs["attack_overlay_meta"] = {
                "whole_attack_bounding_box": whole_attack_bounding_box,
                "x_tgt": x_tgt,
                "y_tgt": y_tgt,
                "skip": skip,
                "attack_string": attack_string,
            }
            return new_obs

        except Exception as exc:
            logger.warning(f"Failed to apply popup overlay: {exc}")
            return obs

    def get_attack_params(self) -> Tuple[Any, Any, Any]:
        """
        Get current attack parameters for risk evaluation.

        Returns:
            Tuple of (bbox, x_tgt, y_tgt)
        """
        return self._attack_bbox, self._attack_x_tgt, self._attack_y_tgt

    def reset_attack_params(self):
        """Reset attack parameters for new task."""
        self._attack_bbox = None
        self._attack_x_tgt = None
        self._attack_y_tgt = None

    @staticmethod
    def screenshot_to_png_bytes(screenshot: Any) -> Optional[bytes]:
        """
        Convert screenshot to PNG bytes.

        Args:
            screenshot: Screenshot as bytes, bytearray, memoryview, or numpy array

        Returns:
            PNG bytes or None if conversion fails
        """
        if isinstance(screenshot, (bytes, bytearray, memoryview)):
            return bytes(screenshot)
        if isinstance(screenshot, np.ndarray) and screenshot.size > 0:
            try:
                image = Image.fromarray(screenshot)
                buffered = io.BytesIO()
                image.save(buffered, format="PNG")
                return buffered.getvalue()
            except Exception:
                return None
        return None

    @staticmethod
    def encode_image_bytes(image_bytes: Optional[bytes]) -> Optional[str]:
        """
        Encode image bytes to base64 data URL.

        Args:
            image_bytes: PNG image bytes

        Returns:
            Base64 data URL string or None
        """
        if not image_bytes:
            return None
        try:
            encoded = base64.b64encode(image_bytes).decode("utf-8")
            return f"data:image/png;base64,{encoded}"
        except Exception:
            return None

    @staticmethod
    def get_screenshot_base64(obs: Dict[str, Any]) -> Optional[str]:
        """
        Extract and encode screenshot from observation to base64.

        Args:
            obs: Observation dict with 'screenshot' key

        Returns:
            Base64 encoded screenshot string or None
        """
        if not obs:
            return None
        screenshot = obs.get("screenshot") if isinstance(obs, dict) else None
        if screenshot is None:
            return None
        screenshot_bytes = ObservationProcessor.screenshot_to_png_bytes(screenshot)
        if screenshot_bytes:
            return base64.b64encode(screenshot_bytes).decode("utf-8")
        return None
