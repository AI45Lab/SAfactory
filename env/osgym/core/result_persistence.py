"""
Result Persistence Module

Handles saving screenshots, trajectories, and task results to disk.
"""

import os
import glob
import json
import datetime
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger("osgym.result_persistence")


class ResultPersistence:
    """
    Handles persistence of step results, screenshots, and task outcomes.

    Creates structured output directories and saves:
    - Per-step screenshots
    - Trajectory JSONL files
    - Task result files
    """

    def __init__(
        self,
        result_dir: Optional[str],
        action_space_type: str,
        observation_type: str,
        save_screenshots: bool = True
    ):
        """
        Initialize ResultPersistence.

        Args:
            result_dir: Base directory for saving results (None to disable).
                        Supports relative paths (resolved relative to env/osgym directory).
            action_space_type: Action space type (for directory structure)
            observation_type: Observation type (for directory structure)
            save_screenshots: Whether to save step screenshots
        """
        # Convert relative paths to absolute paths based on osgym directory
        if result_dir and not os.path.isabs(result_dir):
            osgym_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            result_dir = os.path.join(osgym_dir, result_dir)
        self.result_dir = result_dir
        self.action_space_type = action_space_type
        self.observation_type = observation_type
        self.save_screenshots = save_screenshots
        self._current_result_dir: Optional[str] = None

    def setup_result_dir(self, task_id: str, domain: str) -> Optional[str]:
        """
        Create and initialize result directory for a task.

        Creates directory structure: result_dir/action_space/observation_type/domain/task_id/
        Cleans up old result files (traj.jsonl, screenshots, recordings).

        Args:
            task_id: Task identifier
            domain: Task domain

        Returns:
            Path to task result directory, or None if result_dir not configured
        """
        if not self.result_dir:
            return None

        task_result_dir = os.path.join(
            self.result_dir,
            self.action_space_type,
            self.observation_type,
            domain,
            task_id
        )
        os.makedirs(task_result_dir, exist_ok=True)
        logger.debug(f"Result directory created: {task_result_dir}")

        # Clean old result files (reset for each task execution)
        self._cleanup_old_results(task_result_dir)

        self._current_result_dir = task_result_dir
        return task_result_dir

    def _cleanup_old_results(self, task_result_dir: str):
        """Clean up old result files from previous runs."""
        # 1. Clean traj.jsonl
        traj_path = os.path.join(task_result_dir, "traj.jsonl")
        if os.path.exists(traj_path):
            try:
                os.remove(traj_path)
                logger.debug(f"Removed old trajectory file: {traj_path}")
            except Exception as e:
                logger.warning(f"Failed to remove old trajectory file: {e}")

        # 2. Clean old screenshots (step_*.png)
        old_screenshots = glob.glob(os.path.join(task_result_dir, "step_*.png"))
        for screenshot_path in old_screenshots:
            try:
                os.remove(screenshot_path)
            except Exception as e:
                logger.warning(f"Failed to remove old screenshot {screenshot_path}: {e}")
        if old_screenshots:
            logger.debug(f"Removed {len(old_screenshots)} old screenshot files")

        # 3. Clean old recording
        recording_path = os.path.join(task_result_dir, "recording.mp4")
        if os.path.exists(recording_path):
            try:
                os.remove(recording_path)
                logger.debug(f"Removed old recording file: {recording_path}")
            except Exception as e:
                logger.warning(f"Failed to remove old recording file: {e}")

    def save_step_result(
        self,
        step_idx: int,
        action: str,
        obs: Dict[str, Any],
        reward: float,
        done: bool,
        info: Dict[str, Any],
        screenshot_to_bytes_func
    ):
        """
        Save step result (screenshot and trajectory).

        Args:
            step_idx: Step index
            action: Action taken
            obs: Observation dict
            reward: Step reward
            done: Whether episode is done
            info: Additional info dict
            screenshot_to_bytes_func: Function to convert screenshot to bytes
        """
        if not self._current_result_dir:
            return

        timestamp = datetime.datetime.now().strftime("%Y%m%d@%H%M%S")

        # Save screenshot
        if self.save_screenshots and obs.get("screenshot") is not None:
            screenshot_path = os.path.join(
                self._current_result_dir,
                f"step_{step_idx}_{timestamp}.png"
            )
            screenshot_bytes = screenshot_to_bytes_func(obs["screenshot"])
            if screenshot_bytes:
                try:
                    with open(screenshot_path, "wb") as f:
                        f.write(screenshot_bytes)
                except Exception as e:
                    logger.warning(f"Failed to save screenshot: {e}")

        # Append trajectory to traj.jsonl
        traj_path = os.path.join(self._current_result_dir, "traj.jsonl")
        try:
            with open(traj_path, "a") as f:
                f.write(json.dumps({
                    "step_num": step_idx,
                    "action_timestamp": timestamp,
                    "action": action,
                    "reward": reward,
                    "done": done,
                    "info": str(info)
                }))
                f.write("\n")
        except Exception as e:
            logger.warning(f"Failed to save trajectory: {e}")

    def save_task_result(
        self,
        task_id: str,
        score: float,
        task_completion_score: float = None,
        risk_triggered_score: float = None
    ):
        """
        Save final task result.

        Args:
            task_id: Task identifier
            score: Final task score (combined)
            task_completion_score: Task completion score (1 = completed, 0 = not completed)
            risk_triggered_score: Risk triggered score (1 = risk triggered, 0 = safe)
        """
        if not self._current_result_dir:
            return

        result_path = os.path.join(self._current_result_dir, "result.txt")
        try:
            with open(result_path, "w") as f:
                f.write(f"{score}\n")
            logger.debug(f"Task result saved to {result_path}")
        except Exception as e:
            logger.warning(f"Failed to save task result: {e}")

        # Save detailed scores as JSON
        detailed_result_path = os.path.join(self._current_result_dir, "result_detail.json")
        try:
            detailed_result = {
                "task_id": task_id,
                "final_score": score,
                "task_completion_score": task_completion_score,
                "risk_triggered_score": risk_triggered_score
            }
            with open(detailed_result_path, "w") as f:
                json.dump(detailed_result, f, indent=2)
            logger.debug(f"Detailed task result saved to {detailed_result_path}")
        except Exception as e:
            logger.warning(f"Failed to save detailed task result: {e}")

    def get_current_result_dir(self) -> Optional[str]:
        """Get current result directory path."""
        return self._current_result_dir
