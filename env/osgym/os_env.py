"""
OSGym Environment

Integrated environment for RiOSWorld and OSWorld benchmarks.
This module provides the main OSGym class that coordinates desktop environment
interaction, task management, and evaluation.

Supports two benchmark types:
- "riosworld": Risk assessment benchmark
- "osworld": General desktop task benchmark
"""

import sys
import os
import random
import atexit
import time
import logging
from typing import Optional, Dict, Any, List

import gymnasium as gym
import numpy as np
from pathlib import Path

logger = logging.getLogger("osgym")

# Add current directory to path for local imports
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

try:
    from desktop_env.desktop_env import DesktopEnv
except ImportError:
    print(f"Warning: Could not import DesktopEnv from {CURRENT_DIR}. Please ensure desktop_env is present.")
    DesktopEnv = None

from core.env.base_env import BaseEnv
from core.env.env_register import register_env
from core.types.base import (
    ResetOutput, StepOutput, RenderOutput, PromptOutput
)

# Import refactored components
from .core.task_manager import TaskManager
from .core.action_parser import ActionParser
from .core.observation_processor import ObservationProcessor
from .core.result_persistence import ResultPersistence
from .core.prompt_builder import PromptBuilder
from .evaluation.evaluator import TaskEvaluator


@register_env("os_gym")
class OSGym(BaseEnv):
    """
    Integrated OSGym environment for RiOSWorld/OSWorld benchmarks.

    This class acts as a coordinator that:
    - Manages task loading and iteration via TaskManager
    - Parses agent actions via ActionParser
    - Processes observations via ObservationProcessor
    - Persists results via ResultPersistence
    - Builds prompts via PromptBuilder
    - Evaluates tasks via TaskEvaluator
    """

    def __init__(
        self,
        task_config_path: str = None,
        task_id: str = None,
        benchmark_type: str = "osworld",  # "riosworld" or "osworld"
        provider_name: str = "docker",
        headless: bool = True,
        action_space: str = "pyautogui",
        observation_type: str = "screenshot_a11y_tree",
        screen_width: int = 1920,
        screen_height: int = 1080,
        cache_dir: str = None,
        sleep_after_execution: float = 0.0,
        post_reset_wait: float = 3.0,
        max_steps: int = 15,
        result_dir: str = None,
        save_screenshots: bool = True,
        enable_recording: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.benchmark_type = benchmark_type.lower()

        if DesktopEnv is None:
            raise ImportError("DesktopEnv could not be imported. Check dependencies.")

        # Configuration
        self.provider_name = provider_name
        self.headless = headless
        self.action_space_type = action_space
        self.observation_type = observation_type
        self.screen_size = (screen_width, screen_height)
        self.cache_dir = cache_dir or "/tmp/osgym_cache"
        self.sleep_after_execution = sleep_after_execution
        self.post_reset_wait = post_reset_wait
        self.max_steps = max_steps
        self.enable_recording = enable_recording

        # Initialize TaskManager
        self.task_manager = TaskManager(
            task_config_path=task_config_path,
            benchmark_type=self.benchmark_type,
            current_dir=CURRENT_DIR,
            target_task_id=task_id
        )

        # Initialize other components
        self.action_parser = ActionParser()
        self.obs_processor = ObservationProcessor(observation_type)
        self.result_persistence = ResultPersistence(
            result_dir=result_dir,
            action_space_type=action_space,
            observation_type=observation_type,
            save_screenshots=save_screenshots
        )
        self.prompt_builder = PromptBuilder(
            observation_type=observation_type,
            action_space_type=action_space,
            max_history_len=3
        )

        # State tracking
        self.all_trajs: Dict[str, List] = {}
        self.current_traj: List = []
        self.task_scores: Dict[str, float] = {}
        self.current_step_in_task: int = 0
        self.risk_results: List[Any] = []
        self.history: List[str] = []

        # Expose tasks for compatibility
        self.tasks = self.task_manager.tasks
        self.task_index = 0  # Will be synced with task_manager

        # Create desktop environment
        self.env = self._create_desktop_env()

        # Initialize evaluator after env is created
        self.evaluator = TaskEvaluator(self.env)

        # Current task state
        self.current_task: Optional[Dict] = None
        self.current_instruction: str = ""
        self.current_obs: Dict = {}

        # Define observation and action spaces
        self.action_space = gym.spaces.Text(max_length=10000)
        self.observation_space = gym.spaces.Dict({
            "screenshot": gym.spaces.Box(
                low=0, high=255,
                shape=(screen_height, screen_width, 3),
                dtype=np.uint8
            ),
            "accessibility_tree": gym.spaces.Text(max_length=1000000),
            "instruction": gym.spaces.Text(max_length=1000),
            "terminal": gym.spaces.Text(max_length=100000)
        })

        # Register cleanup on exit
        atexit.register(self.close)

    def _create_desktop_env(self):
        """Create and configure the DesktopEnv instance."""
        try:
            from desktop_env.providers.docker.manager import DockerVMManager
            vm_manager = DockerVMManager()
            vm_path = vm_manager.get_vm_path(os_type="Ubuntu", region=None)
        except Exception as exc:
            vm_path = os.path.join(CURRENT_DIR, "docker_vm_data", "Ubuntu.qcow2")
            logger.warning(f"Auto-download VM failed, falling back to: {vm_path}")
            logger.debug(f"Reason: {exc}")

        logger.info(f"Using VM path: {vm_path}")
        logger.info(f"Using cache dir: {self.cache_dir}")

        return DesktopEnv(
            provider_name=self.provider_name,
            path_to_vm=vm_path,
            action_space=self.action_space_type,
            screen_size=self.screen_size,
            headless=self.headless,
            require_a11y_tree=self.observation_type in ["a11y_tree", "screenshot_a11y_tree", "som"],
            require_terminal=False,
            os_type="Ubuntu",
            cache_dir=self.cache_dir
        )

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> ResetOutput:
        """Reset environment to start or continue with tasks."""
        if seed is not None:
            random.seed(seed)

        if not self.task_manager.tasks:
            raise ValueError("No tasks available to reset.")

        # Reset current task state
        self.current_traj = []
        self.history = []
        self.current_step_in_task = 0
        self.risk_results = []
        self.obs_processor.reset_attack_params()

        # Handle task_index from options
        if options and "task_index" in options:
            self.task_manager.set_index(options["task_index"])

        # Sync task_index
        self.task_index = self.task_manager.task_index

        # Get next task
        task_info = self.task_manager.get_next_task_info()
        if task_info is None:
            logger.info("All tasks have been iterated. Restarting from the first task.")
            self.task_manager.reset_index()
            task_info = self.task_manager.get_next_task_info()

        # Update task_index after getting task
        self.task_index = self.task_manager.task_index

        # Load task configuration
        task_config = self.task_manager.load_task_config(task_info)
        self.current_task = task_config
        self.current_instruction = task_config.get("instruction", "")

        # Setup result directory
        self.result_persistence.setup_result_dir(task_info["id"], task_info["domain"])

        # Reset desktop environment
        obs = self.env.reset(task_config=task_config)
        if self.post_reset_wait and self.post_reset_wait > 0:
            logger.info(f"Waiting {self.post_reset_wait}s for environment stabilization...")
            time.sleep(self.post_reset_wait)
            obs = self.env._get_obs()

        # Execute halfway_setup if present
        if hasattr(self.env, "halfway_config") and self.env.halfway_config:
            logger.info("Running halfway setup...")
            self.env.setup_controller.halfway_setup(self.env.halfway_config)
            obs = self.env._get_obs()

        # Start recording if enabled
        if self.enable_recording and self.result_persistence.get_current_result_dir():
            try:
                self.env.controller.start_recording()
                logger.info("Started recording")
            except Exception as e:
                logger.warning(f"Failed to start recording: {e}")

        self.current_obs = obs

        return ResetOutput(
            observation=self._process_obs(obs),
            info={"task_id": task_info["id"], "domain": task_info["domain"]}
        )

    def step(self, action: str) -> StepOutput:
        """Execute action and return step result."""
        self.current_step_in_task += 1

        # Parse actions
        parsed_actions = self.action_parser.parse_actions(action)
        logger.debug(f"Step {self.current_step_in_task}/{self.max_steps}, Parsed Actions: {parsed_actions}")
        if not parsed_actions:
            parsed_actions = ["WAIT"]

        parsed_actions, special_cmd = self.action_parser.strip_special_command(parsed_actions)

        # Record trajectory
        step_record = {
            "instruction": self.current_instruction,
            "response": action,
            "actions": parsed_actions if parsed_actions else ([special_cmd] if special_cmd else ["WAIT"]),
            "screenshot_base64": ObservationProcessor.get_screenshot_base64(self.current_obs)
        }
        self.current_traj.append(step_record)

        # Update history
        self.history.append(action)
        if len(self.history) > self.prompt_builder.max_history_len:
            self.history.pop(0)

        # Handle special commands without environment actions
        if special_cmd == "WAIT":
            processed_obs = self._process_obs(self.current_obs)
            self.result_persistence.save_step_result(
                self.current_step_in_task, action, processed_obs, 0.0, False,
                {"agent_signal": "WAIT"},
                ObservationProcessor.screenshot_to_png_bytes
            )
            if self.current_step_in_task >= self.max_steps:
                logger.info(f"Reached max steps ({self.max_steps}) during WAIT, truncating task.")
                return self._finish_task(
                    [], {"executed_actions": [], "truncated_reason": "max_steps_reached"},
                    agent_signal="TRUNCATED", skip_evaluation=False
                )
            return self._handle_wait_signal()

        if special_cmd in {"DONE", "FAIL"} and not parsed_actions:
            processed_obs = self._process_obs(self.current_obs)
            self.result_persistence.save_step_result(
                self.current_step_in_task, action, processed_obs, 0.0, True,
                {"agent_signal": special_cmd},
                ObservationProcessor.screenshot_to_png_bytes
            )
            return self._finish_task(
                [], {"executed_actions": []},
                agent_signal=special_cmd,
                score_override=0.0 if special_cmd == "FAIL" else None,
                skip_evaluation=(special_cmd == "FAIL")
            )

        # Execute actions
        executed_actions: List[str] = []
        info: Dict[str, Any] = {}
        reward = 0.0
        done = False

        for act in parsed_actions:
            obs, reward, done, info = self.env.step(act, pause=self.sleep_after_execution)
            executed_actions.append(act)
            self.current_obs = obs
            if done:
                break

        # Save step result
        processed_obs = self._process_obs(self.current_obs)
        self.result_persistence.save_step_result(
            self.current_step_in_task, action, processed_obs, reward, done, info,
            ObservationProcessor.screenshot_to_png_bytes
        )

        # Evaluate step risk
        risk_result = self.evaluator.evaluate_step_risk(
            executed_actions,
            (self.current_task or {}).get("id", ""),
            self.obs_processor.get_attack_params()
        )
        if risk_result is not None:
            info = dict(info) if info else {}
            info["risk_result"] = risk_result
            self.risk_results.append(risk_result)

        info = info or {}
        info.setdefault("executed_actions", executed_actions)

        if done:
            return self._finish_task(executed_actions, info)

        if special_cmd in {"DONE", "FAIL"}:
            return self._finish_task(
                executed_actions, info,
                agent_signal=special_cmd,
                score_override=0.0 if special_cmd == "FAIL" else None,
                skip_evaluation=(special_cmd == "FAIL")
            )

        # Check max steps
        if self.current_step_in_task >= self.max_steps:
            logger.info(f"Reached max steps ({self.max_steps}), truncating task.")
            info["truncated_reason"] = "max_steps_reached"
            return self._finish_task(
                executed_actions, info,
                agent_signal="TRUNCATED",
                skip_evaluation=False
            )

        return StepOutput(
            observation=self._process_obs(self.current_obs),
            reward=reward,
            terminated=False,
            truncated=False,
            info=info
        )

    def _handle_wait_signal(self) -> StepOutput:
        """Handle WAIT signal by waiting and refreshing observation."""
        wait_time = self.sleep_after_execution if self.sleep_after_execution > 0 else 1.0
        logger.debug(f"WAIT signal received, waiting {wait_time}s...")
        time.sleep(wait_time)

        try:
            self.current_obs = self.env._get_obs()
        except Exception as e:
            logger.warning(f"Failed to get new observation after WAIT: {e}")

        return StepOutput(
            observation=self._process_obs(self.current_obs),
            reward=0.0,
            terminated=False,
            truncated=False,
            info={"agent_signal": "WAIT", "executed_actions": []}
        )

    def _finish_task(
        self,
        executed_actions: List[str],
        info: Optional[Dict[str, Any]] = None,
        agent_signal: Optional[str] = None,
        score_override: Optional[float] = None,
        skip_evaluation: bool = False
    ) -> StepOutput:
        """Finish current task and optionally load next task."""
        info = dict(info) if info else {}
        if agent_signal:
            info["agent_signal"] = agent_signal

        current_task_id = self.current_task.get("id", "unknown")
        logger.info(f"Task {current_task_id} finished.")

        # Stop recording
        if self.enable_recording and self.result_persistence.get_current_result_dir():
            try:
                recording_path = os.path.join(
                    self.result_persistence.get_current_result_dir(), "recording.mp4"
                )
                self.env.controller.end_recording(recording_path)
                logger.info(f"Recording saved to {recording_path}")
            except Exception as e:
                logger.warning(f"Failed to save recording: {e}")

        # Compute task score
        task_score = self.evaluator.compute_task_score(
            current_task_id, executed_actions, self.risk_results,
            score_override, skip_evaluation
        )
        self.task_scores[current_task_id] = task_score
        self.all_trajs[current_task_id] = self.current_traj
        self.current_traj = []

        # Save task result
        self.result_persistence.save_task_result(current_task_id, task_score)

        info["task_id"] = current_task_id
        info["task_score"] = task_score
        reward = task_score

        # Try to load next task
        obs, has_more = self._load_next_task()

        if has_more:
            return StepOutput(
                observation=self._process_obs(obs),
                reward=reward,
                terminated=False,
                truncated=False,
                info=info
            )

        # All tasks completed
        logger.info("All tasks completed.")
        logger.info("Final Scores Summary:")
        for tid, scr in self.task_scores.items():
            logger.info(f"  {tid}: {scr}")

        avg_score = sum(self.task_scores.values()) / len(self.task_scores) if self.task_scores else 0.0
        logger.info(f"Average Score: {avg_score}")

        info["final_scores"] = self.task_scores
        info["average_score"] = avg_score

        return StepOutput(
            observation=self._process_obs(obs),
            reward=reward,
            terminated=True,
            truncated=False,
            info=info
        )

    def _load_next_task(self):
        """Load next task if available."""
        task_info = self.task_manager.get_next_task_info()
        self.task_index = self.task_manager.task_index

        if task_info is None:
            return {}, False

        logger.info(f"Switching to next task: {task_info['id']}")

        # Load task configuration
        task_config = self.task_manager.load_task_config(task_info)
        self.current_task = task_config
        self.current_instruction = task_config.get("instruction", "")
        self.current_step_in_task = 0
        self.risk_results = []
        self.history = []
        self.obs_processor.reset_attack_params()

        # Setup result directory
        self.result_persistence.setup_result_dir(task_info["id"], task_info["domain"])

        # Reset environment for next task
        obs = self.env.reset(task_config=task_config)
        if self.post_reset_wait and self.post_reset_wait > 0:
            logger.info(f"Waiting {self.post_reset_wait}s for environment stabilization (next task)...")
            time.sleep(self.post_reset_wait)
            obs = self.env._get_obs()

        # Execute halfway_setup if present
        if hasattr(self.env, "halfway_config") and self.env.halfway_config:
            logger.info("Running halfway setup for next task...")
            self.env.setup_controller.halfway_setup(self.env.halfway_config)
            obs = self.env._get_obs()

        # Start recording for next task
        if self.enable_recording and self.result_persistence.get_current_result_dir():
            try:
                self.env.controller.start_recording()
                logger.info("Started recording for next task")
            except Exception as e:
                logger.warning(f"Failed to start recording: {e}")

        self.current_obs = obs
        return obs, True

    def _process_obs(self, obs):
        """Process observation by applying attack overlay if needed."""
        return self.obs_processor.process(obs, self.current_task, self.current_instruction)

    def get_task_prompt(self) -> PromptOutput:
        """Generate task prompt for agent."""
        return self.prompt_builder.build_task_prompt(
            instruction=self.current_instruction,
            current_obs=self.current_obs,
            history=self.history,
            task_id=(self.current_task or {}).get("id", ""),
            screenshot_to_bytes_func=ObservationProcessor.screenshot_to_png_bytes,
            encode_image_func=ObservationProcessor.encode_image_bytes
        )

    def render(self) -> RenderOutput:
        """Render current environment state."""
        screenshot_data = self.env.controller.get_screenshot()
        return RenderOutput(
            image_data=screenshot_data,
            step=self.env._step_no
        )

    def close(self):
        """Clean up resources."""
        if getattr(self, '_closed', False):
            return

        if self.env:
            try:
                self.env.close()
            except Exception as e:
                logger.error(f"Error closing DesktopEnv: {e}")
            self.env = None

        # Clean up temporary cache directory
        if self.cache_dir and self.cache_dir.startswith("/tmp/osgym_cache"):
            import shutil
            try:
                if os.path.exists(self.cache_dir):
                    shutil.rmtree(self.cache_dir)
                    logger.info(f"Cleaned up cache directory: {self.cache_dir}")
            except Exception as e:
                logger.warning(f"Failed to clean up cache directory {self.cache_dir}: {e}")

        self._closed = True
