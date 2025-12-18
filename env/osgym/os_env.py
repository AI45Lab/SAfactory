"""
OSGym Environment

Integrated environment for RiOSWorld and OSWorld benchmarks.
This module provides the main OSGym class that coordinates desktop environment
interaction, task management, and evaluation.

Single-task mode: Each OSGym instance handles one task from the dataset.
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
# Configure osgym logging output
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

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
    ResetOutput, StepOutput, RenderOutput
)

# Import refactored components
from .core.action_parser import ActionParser
from .core.observation_processor import ObservationProcessor
from .core.result_persistence import ResultPersistence
from .core.prompt_builder import PromptBuilder
from .evaluation.evaluator import TaskEvaluator


@register_env("os_gym")
class OSGym(BaseEnv):
    """
    Integrated OSGym environment for RiOSWorld/OSWorld benchmarks.

    Single-task mode: Each instance handles one task from the dataset.
    Task configuration is provided via the `dataset` parameter from BaseEnv.

    This class acts as a coordinator that:
    - Loads task configuration from dataset
    - Parses agent actions via ActionParser
    - Processes observations via ObservationProcessor
    - Persists results via ResultPersistence
    - Builds prompts via PromptBuilder
    - Evaluates tasks via TaskEvaluator
    """

    def __init__(
        self,
        benchmark_type: str = "riosworld",  # "riosworld" or "osworld"
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

        # Validate dataset is provided
        if not self.dataset or not isinstance(self.dataset, dict):
            raise ValueError(
                "OSGym requires a dataset dict with task configuration. "
                "Please configure 'dataset' in your YAML config file."
            )

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

        # Load and validate task configuration from dataset
        self._validate_and_load_task_config()

        # Initialize components
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
            action_space_type=action_space
        )

        # State tracking
        self.current_traj: List = []
        self.current_step_in_task: int = 0
        self.risk_results: List[Any] = []
        self.messages: List[Dict[str, Any]] = []  # message history for prompts
        self.task_finished: bool = False
        self.task_score: Optional[float] = None

        # Create desktop environment
        self.env = self._create_desktop_env()

        # Initialize evaluator after env is created
        self.evaluator = TaskEvaluator(self.env)

        # Current observation
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

    def _validate_and_load_task_config(self):
        """Validate and load task configuration from dataset."""
        required_fields = ['id', 'instruction']
        for field in required_fields:
            if field not in self.dataset:
                raise ValueError(f"Dataset missing required field: {field}")

        self.task_id = self.dataset['id']
        self.task_domain = self.dataset.get('domain', 'default')
        self.task_config = self.dataset
        self.current_instruction = self.dataset['instruction']

        # Validate evaluator based on benchmark type
        if self.benchmark_type == "riosworld":
            if 'risk_evaluator' not in self.dataset:
                logger.warning(f"RIOSWorld task {self.task_id} missing risk_evaluator")
            if 'evaluator' not in self.dataset:
                logger.warning(f"Task {self.task_id} missing evaluator")
        else:  # osworld
            if 'evaluator' not in self.dataset:
                logger.warning(f"Task {self.task_id} missing evaluator")

        logger.info(f"Loaded task: {self.task_id} (domain: {self.task_domain}, benchmark: {self.benchmark_type})")

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
        """Reset environment for the single task from dataset."""
        if seed is not None:
            random.seed(seed)

        if self.task_finished:
            raise RuntimeError(
                "Task already finished. Create a new OSGym instance for another task."
            )

        # Reset trajectory and state
        self.current_traj = []
        self.current_step_in_task = 0
        self.risk_results = []
        self.task_score = None
        self.obs_processor.reset_attack_params()
        self.evaluator.reset_state()

        # Setup result directory
        self.result_persistence.setup_result_dir(self.task_id, self.task_domain)

        # Reset desktop environment with task config
        obs = self.env.reset(task_config=self.task_config)
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

        # Initialize message list with system prompt
        system_prompt = self.prompt_builder.build_system_prompt(self.current_instruction)
        self.messages = [
            {"role": "system", "content": system_prompt}
        ]

        return ResetOutput(
            observation=self._process_obs(self.current_obs),
            info={"task_id": self.task_id, "domain": self.task_domain}
        )

    def step(self, action: str) -> StepOutput:
        """Execute action and return step result."""
        if self.task_finished:
            raise RuntimeError("Task already finished. Cannot step on a finished task.")

        self.current_step_in_task += 1

        # Append agent action to messages
        self.messages.append({"role": "assistant", "content": action})

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

        # Capture state before action execution for risk evaluation
        self.evaluator.capture_pre_action_state()

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
            self.task_id,
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
        """Finish the task and terminate environment."""
        self.task_finished = True
        info = dict(info) if info else {}
        if agent_signal:
            info["agent_signal"] = agent_signal

        logger.info(f"Task {self.task_id} finished.")

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
        self.task_score = self.evaluator.compute_task_score(
            self.task_id, self.risk_results,
            score_override, skip_evaluation
        )

        # Save task result
        self.result_persistence.save_task_result(self.task_id, self.task_score)

        info["task_id"] = self.task_id
        info["task_score"] = self.task_score
        info["trajectory"] = self.current_traj

        logger.info(f"Task {self.task_id} score: {self.task_score}")

        return StepOutput(
            observation=self._process_obs(self.current_obs),
            reward=self.task_score,
            terminated=True,  # Always terminate after single task
            truncated=(self.current_step_in_task >= self.max_steps),
            info=info
        )

    def _process_obs(self, obs):
        """Process observation by applying attack overlay if needed."""
        return self.obs_processor.process(obs, self.task_config, self.current_instruction)

    def get_task_prompt(self) -> List[Dict[str, Any]]:
        """Generate task prompt for agent."""
        # Build user content for current observation
        user_content = self.prompt_builder.build_user_content(
            current_obs=self.current_obs,
            task_id=self.task_id,
            screenshot_to_bytes_func=ObservationProcessor.screenshot_to_png_bytes,
            encode_image_func=ObservationProcessor.encode_image_bytes
        )

        # Append user content to messages
        self.messages.append({"role": "user", "content": user_content})

        return self.messages

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
