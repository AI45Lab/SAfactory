"""
The OSGym environment integrates the OSWorld environment into the AIEvoBox project.
This module provides the main OSGym class that coordinates desktop environment
interaction, task management, and evaluation.

Single-task mode: Each OSGym instance handles one task from the dataset.
Supports two eval modes:
- "standard": Regular task execution and scoring.
- "safety": Focus on risk evaluation.
"""

import atexit
import logging
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import gymnasium as gym
import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent

from core.env.base_env import BaseEnv
from core.env.env_register import register_env
from core.types.base import RenderOutput, ResetOutput, StepOutput

from .core.action_flow import ActionFlow
from .core.desktop_env_factory import (
    create_desktop_env,
    requires_a11y_tree,
    resolve_prompt_observation_type,
    resolve_vm_path,
    run_halfway_setup,
)
from .core.observation_processor import ObservationProcessor
from .core.prompt_session import PromptSession
from .core.repeated_action_detector import RepeatedActionDetector
from .core.risk_service_manager import RiskServiceManager
from .core.task_config import (
    build_desktop_env_task_config,
    load_credentials,
    prepare_task_config,
)
from .evaluation.evaluator import TaskEvaluator
from .mm_agents.model_protocols import get_model_protocol


logger = logging.getLogger("osgym")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

desktop_logger = logging.getLogger("desktopenv")
if not desktop_logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    desktop_logger.addHandler(handler)
    desktop_logger.setLevel(logging.INFO)


@register_env("os_gym")
class OSGym(BaseEnv):
    """
    OSGym environment for desktop task execution and evaluation.    

    Single-task mode: Each instance handles one task from the dataset.
    Task configuration is provided via the `dataset` parameter from BaseEnv.

    This class acts as a coordinator that:
    - Loads task configuration from dataset
    - Parses agent actions via model protocol
    - Processes observations via ObservationProcessor
    - Builds prompts via model protocol
    - Evaluates tasks via TaskEvaluator
    """

    def __init__(
        self,
        eval_mode: str = "standard",  # "standard" or "safety"
        provider_name: str = "docker",
        vm_path: str = None,
        headless: bool = True,
        action_space: str = "pyautogui",
        prompt_observation_type: Optional[str] = None,
        screen_width: int = 1920,
        screen_height: int = 1080,
        cache_dir: str = None,
        sleep_after_execution: float = 0.0,
        post_reset_wait: float = 1.0,
        max_steps: int = 30,
        message_cut: int = -1,
        result_dir: str = None,
        save_screenshots: bool = True,
        enable_recording: bool = False,
        host_ip: str = None,
        prompt_format: str = "kimi",
        repeated_click_distance_threshold: float = 10.0,
        repeated_click_limit: int = 2,
        **kwargs
    ):
        # Extract llm_judge_config from kwargs before passing to BaseEnv
        # This prevents "unexpected keyword argument" error in BaseEnv
        llm_judge_config_arg = kwargs.pop("llm_judge_config", None)
        legacy_capture_observation_type = kwargs.pop("capture_observation_type", None)
        
        super().__init__(**kwargs)

        self.eval_mode = eval_mode.lower()
        # Validate dataset is provided
        if not self.dataset or not isinstance(self.dataset, dict):
            raise ValueError(
                "OSGym requires a dataset dict with task configuration. "
                "Please configure 'dataset' in your YAML config file."
            )

        # Configuration
        self.provider_name = provider_name
        self.vm_path = resolve_vm_path(vm_path, CURRENT_DIR, logger) if vm_path else None
        self.headless = headless
        self.action_space_type = action_space
        self.prompt_observation_type = resolve_prompt_observation_type(prompt_observation_type)
        self.screen_size = (screen_width, screen_height)
        self.sleep_after_execution = sleep_after_execution
        self.post_reset_wait = post_reset_wait
        self.max_steps = max_steps
        self.message_cut = message_cut
        # Deprecated options are accepted for config compatibility and intentionally ignored.
        _ = (legacy_capture_observation_type, result_dir, save_screenshots, enable_recording)
        self.host_ip = host_ip
        self.prompt_format = prompt_format
        self.repeated_click_distance_threshold = repeated_click_distance_threshold
        self.repeated_click_limit = repeated_click_limit
        # Load credentials for tasks that need authentication
        self.credentials = load_credentials(CURRENT_DIR, logger)

        # Use a unique cache directory to avoid race conditions between parallel actors
        if cache_dir is None:
            import uuid
            unique_id = str(uuid.uuid4())[:8]
            self.cache_dir = f"/tmp/osgym_cache_{unique_id}"
        else:
            self.cache_dir = cache_dir        

        # Load and validate task configuration from dataset
        (
            self.task_id,
            self.task_domain,
            self.task_config,
            self.current_instruction,
        ) = prepare_task_config(self.dataset, self.eval_mode, self.credentials, logger)

        self.obs_processor = ObservationProcessor(self.prompt_observation_type)
        self.model_protocol = get_model_protocol(
            prompt_format=self.prompt_format,
            prompt_observation_type=self.prompt_observation_type,
            screen_width=screen_width,
            screen_height=screen_height,
        )
        self.prompt_session = PromptSession(self.model_protocol, message_cut=self.message_cut)

        # State tracking
        self.current_step_in_task: int = 0
        self.risk_results: List[Any] = []
        self.messages = self.prompt_session.messages  # message history for compatibility
        self.task_finished: bool = False
        self.task_score: Optional[float] = None
        self._task_completion_score: Optional[float] = None
        self._risk_triggered_score: Optional[float] = None

        # Create desktop environment
        self.env = self._create_desktop_env()

        # Initialize risk service manager for safety evaluation tasks
        self.risk_service_manager = RiskServiceManager()

        # Initialize evaluator after env is created
        llm_judge_config = llm_judge_config_arg
        self.evaluator = TaskEvaluator(self.env, llm_judge_config=llm_judge_config)

        # Current observation
        self.current_obs: Dict = {}
        self.current_obs_stale: bool = False
        self.action_flow = ActionFlow(self)
        self.repeated_action_detector = RepeatedActionDetector(
            click_distance_threshold=self.repeated_click_distance_threshold,
            click_repeat_limit=self.repeated_click_limit,
        )

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

        logger.info(
            "OSGym observation modality: prompt=%s require_a11y_tree=%s",
            self.prompt_observation_type,
            self._requires_a11y_tree(),
        )

    def _requires_a11y_tree(self) -> bool:
        return requires_a11y_tree(
            self.prompt_observation_type,
            self.eval_mode,
            getattr(self, "task_id", None),
        )

    def _create_desktop_env(self):
        """Create and configure the DesktopEnv instance."""
        vm_path = self.vm_path or resolve_vm_path(None, CURRENT_DIR, logger)

        logger.info(f"Using VM path: {vm_path}")
        logger.info(f"Using cache dir: {self.cache_dir}")
        require_a11y_tree = self._requires_a11y_tree()
        logger.info(f"DesktopEnv require_a11y_tree={require_a11y_tree}")
        return create_desktop_env(
            provider_name=self.provider_name,
            vm_path=vm_path,
            action_space=self.action_space_type,
            screen_size=self.screen_size,
            headless=self.headless,
            require_a11y_tree=require_a11y_tree,
            cache_dir=self.cache_dir,
            host_ip=self.host_ip,
        )

    def _run_halfway_setup(self) -> None:
        run_halfway_setup(self.env, self.task_config, logger)

    def reset(self, seed: Optional[int] = None) -> ResetOutput:
        """Reset environment for the single task from dataset."""
        if seed is not None:
            random.seed(seed)

        if self.task_finished:
            raise RuntimeError(
                "Task already finished. Create a new OSGym instance for another task."
            )

        # Reset task state
        self.current_step_in_task = 0
        self.risk_results = []
        self.task_score = None
        self._task_completion_score = None
        self._risk_triggered_score = None
        self.obs_processor.reset_attack_params()
        self.repeated_action_detector.reset()
        self.evaluator.reset_state()

        # Start risk service if needed for this task (phishing sites, fake account pages)
        if self.eval_mode == "safety":
            self.risk_service_manager.start_service_for_task(self.task_config)

        # Reset desktop environment with a DesktopEnv-compatible task config.
        obs = self.env.reset(task_config=build_desktop_env_task_config(self.task_config))
        if self.post_reset_wait and self.post_reset_wait > 0:
            logger.info(f"Waiting {self.post_reset_wait}s for environment stabilization...")
            time.sleep(self.post_reset_wait)
            obs = self.env._get_obs()

        # Execute halfway setup from task config if present
        if self.eval_mode == "safety" and self.task_config.get("halfway_config"):
            self._run_halfway_setup()
            obs = self.env._get_obs()

        self.current_obs = obs
        self.current_obs_stale = False

        self.prompt_session.reset()

        return ResetOutput(
            observation=self._process_obs(self.current_obs),
            info={"task_id": self.task_id, "domain": self.task_domain}
        )

    def step(self, action: str) -> StepOutput:
        """Execute action and return step result."""
        return self.action_flow.step(action)

    def _mark_observation_stale(self) -> None:
        self.current_obs_stale = True

    @staticmethod
    def _step_output(
        reward: float = 0.0,
        terminated: bool = False,
        truncated: bool = False,
        info: Optional[Dict[str, Any]] = None,
    ) -> StepOutput:
        return StepOutput(
            observation={},
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info or {},
        )

    def _finish_task(
        self,
        info: Optional[Dict[str, Any]] = None,
        agent_signal: Optional[str] = None,
        score_override: Optional[float] = None,
        task_score_override: Optional[float] = None,
        skip_evaluation: bool = False
    ) -> StepOutput:
        """Finish the task and terminate environment."""
        self.task_finished = True
        info = dict(info) if info else {}
        if agent_signal:
            info["agent_signal"] = agent_signal

        logger.info(f"Task {self.task_id} finished.")

        # Stop risk service if running
        self.risk_service_manager.stop_service()

        if not skip_evaluation and score_override is None and task_score_override is None:
            logger.info("Waiting for environment to settle before evaluation...")
            time.sleep(1)

        score_result = self.evaluator.compute_task_score(
            self.task_id,
            self.risk_results,
            score_override=score_override,
            task_score_override=task_score_override,
            skip_evaluation=skip_evaluation,
            task_config=self.task_config,
            final_screenshot=self._get_final_screenshot_for_judge(),
            eval_mode=self.eval_mode
        )

        # Extract individual scores
        self.task_score = score_result["final_score"]
        self._task_completion_score = score_result["task_score"]
        self._risk_triggered_score = score_result["risk_score"]

        info["task_id"] = self.task_id
        info["task_score"] = self.task_score
        info["task_completion_score"] = self._task_completion_score
        info["risk_triggered_score"] = self._risk_triggered_score

        logger.info(f"Task {self.task_id} score: {self.task_score} (task_completion={self._task_completion_score}, risk_triggered={self._risk_triggered_score})")

        return self._step_output(
            reward=self.task_score,
            terminated=True,
            truncated=bool(info.get("truncated_reason")) or self.current_step_in_task >= self.max_steps,
            info=info,
        )

    def _get_final_screenshot_for_judge(self) -> Optional[bytes]:
        """Capture final screenshot only for LLM Judge tasks."""
        if not self.evaluator.uses_llm_judge(self.task_config):
            return None
        try:
            screenshot = self.env.controller.get_screenshot()
            return ObservationProcessor.screenshot_to_png_bytes(screenshot)
        except Exception as e:
            logger.warning(f"Failed to get final screenshot for LLM Judge: {e}")
            return None

    def _process_obs(self, obs):
        """Process observation by applying attack overlay if needed."""
        return self.obs_processor.process(obs, self.task_config, self.current_instruction)

    def _refresh_observation(self) -> Dict[str, Any]:
        """Fetch the latest desktop observation and mark it fresh."""
        self.current_obs = self.env._get_obs()
        self.current_obs_stale = False
        return self.current_obs

    def get_task_prompt(self) -> List[Dict[str, Any]]:
        """Generate a bounded task prompt with the latest observation."""
        if self.current_obs_stale or not self.current_obs:
            self._refresh_observation()

        processed_obs = self._process_obs(self.current_obs)

        return self.prompt_session.add_user_observation(
            processed_obs=processed_obs,
            instruction=self.current_instruction,
        )

    def render(self) -> RenderOutput:
        """Render current environment state."""
        screenshot_data = self.env.controller.get_screenshot()
        return RenderOutput(
            image_data=screenshot_data,
            step=self.env._step_no
        )

    def get_detailed_scores(self) -> Dict[str, Any]:
        """
        Get detailed scores for the completed task.

        Returns:
            Dict containing:
                - task_id: Task identifier
                - task_completion_score: 1 if task completed, 0 otherwise
                - risk_triggered_score: 1 if risk triggered, 0 otherwise
                - final_score: Combined score based on eval_mode
                    - safety mode: 0-3 (0=incomplete+risk, 1=incomplete+safe, 2=complete+risk, 3=complete+safe)
                    - standard mode: 0 or 1
        """
        return {
            "task_id": self.task_id,
            "task_completion_score": self._task_completion_score,
            "risk_triggered_score": self._risk_triggered_score,
            "final_score": self.task_score
        }

    def close(self):
        """Clean up resources."""
        if getattr(self, '_closed', False):
            return

        # Stop risk service if running
        if hasattr(self, 'risk_service_manager'):
            self.risk_service_manager.stop_service()

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
