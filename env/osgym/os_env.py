"""
The OSGym environment integrates the OSWorld environment into the AIEvoBox project.
This module provides the main OSGym class that coordinates desktop environment
interaction, task management, and evaluation.

Single-task mode: Each OSGym instance handles one task from the dataset.
Supports two eval modes:
- "standard": Regular task execution and scoring.
- "safety": Focus on risk evaluation.
"""

import sys
import os
import random
import atexit
import time
import logging
import copy
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

# Configure desktopenv logging output
desktop_logger = logging.getLogger("desktopenv")
if not desktop_logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    desktop_logger.addHandler(handler)
    desktop_logger.setLevel(logging.INFO)

# Get current directory for relative paths
CURRENT_DIR = Path(__file__).resolve().parent

try:
    from desktop_env.desktop_env import DesktopEnv
except ImportError:
    print("Warning: Could not import DesktopEnv. Please run: pip install -r requirements.txt")
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
from .core.risk_service_manager import RiskServiceManager
from .evaluation.evaluator import TaskEvaluator
from .evaluation.risk_adapter import (
    is_locally_supported_evaluator_config,
    requires_local_evaluator_adapter,
)


@register_env("os_gym")
class OSGym(BaseEnv):
    """
    OSGym environment for desktop task execution and evaluation.    

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
        eval_mode: str = "standard",  # "standard" or "safety"
        provider_name: str = "docker",
        vm_path: str = None,
        headless: bool = True,
        action_space: str = "pyautogui",
        capture_observation_type: Optional[str] = None,
        prompt_observation_type: Optional[str] = None,
        screen_width: int = 1920,
        screen_height: int = 1080,
        cache_dir: str = None,
        sleep_after_execution: float = 0.0,
        post_reset_wait: float = 3.0,
        max_steps: int = 30,
        message_cut: int = -1,
        result_dir: str = None,
        save_screenshots: bool = True,
        enable_recording: bool = False,
        host_ip: str = None,
        prompt_format: str = "kimi",
        **kwargs
    ):
        # Extract llm_judge_config from kwargs before passing to BaseEnv
        # This prevents "unexpected keyword argument" error in BaseEnv
        llm_judge_config_arg = kwargs.pop("llm_judge_config", None)
        
        super().__init__(**kwargs)

        self.eval_mode = eval_mode.lower()

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
        if vm_path:
            expanded_vm_path = os.path.expandvars(os.path.expanduser(vm_path))
            if not os.path.isabs(expanded_vm_path):
                expanded_vm_path = os.path.join(CURRENT_DIR, expanded_vm_path)
            self.vm_path = os.path.abspath(expanded_vm_path)
        else:
            self.vm_path = None
        self.headless = headless
        self.action_space_type = action_space
        self.capture_observation_type, self.prompt_observation_type = self._resolve_observation_modalities(
            capture_observation_type=capture_observation_type,
            prompt_observation_type=prompt_observation_type,
        )
        self.screen_size = (screen_width, screen_height)
        self.cache_dir = cache_dir or "/tmp/osgym_cache"
        self.sleep_after_execution = sleep_after_execution
        self.post_reset_wait = post_reset_wait
        self.max_steps = max_steps
        self.message_cut = message_cut
        self.enable_recording = enable_recording
        self.host_ip = host_ip
        self.prompt_format = prompt_format
        # Load credentials for tasks that need authentication
        self._load_credentials()

        # Load and validate task configuration from dataset
        self._validate_and_load_task_config()

        # Initialize components
        self.action_parser = ActionParser(
            screen_width=screen_width,
            screen_height=screen_height,
            prompt_format=self.prompt_format
        )
        self.obs_processor = ObservationProcessor(self.prompt_observation_type)
        self.result_persistence = ResultPersistence(
            result_dir=result_dir,
            action_space_type=action_space,
            observation_type=self._build_result_observation_label(),
            save_screenshots=save_screenshots
        )
        self.prompt_builder = PromptBuilder(
            prompt_observation_type=self.prompt_observation_type,
            action_space_type=action_space,
            prompt_format=self.prompt_format,
            screen_width=screen_width,
            screen_height=screen_height
        )

        # State tracking
        self.current_traj: List = []
        self.current_step_in_task: int = 0
        self.risk_results: List[Any] = []
        self.messages: List[Dict[str, Any]] = []  # message history for prompts
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
            "OSGym observation modalities: capture=%s prompt=%s",
            self.capture_observation_type,
            self.prompt_observation_type,
        )

    @staticmethod
    def _validate_observation_type(observation_type: Optional[str], field_name: str) -> Optional[str]:
        if observation_type is None:
            return None

        supported = {"screenshot", "a11y_tree", "screenshot_a11y_tree"}
        normalized = observation_type.strip()
        if normalized not in supported:
            raise ValueError(
                f"Invalid {field_name}: {observation_type}. Expected one of {sorted(supported)}."
            )
        return normalized

    @classmethod
    def _resolve_observation_modalities(
        cls,
        capture_observation_type: Optional[str],
        prompt_observation_type: Optional[str],
    ) -> tuple[str, str]:
        """
        Resolve observation modalities from explicit parameters.
        Defaults to screenshot_a11y_tree for capture and screenshot for prompt if not specified.
        """
        capture_type = cls._validate_observation_type(
            capture_observation_type, "capture_observation_type"
        )
        prompt_type = cls._validate_observation_type(
            prompt_observation_type, "prompt_observation_type"
        )

        # Apply defaults
        if capture_type is None:
            capture_type = "screenshot_a11y_tree"
        if prompt_type is None:
            prompt_type = "screenshot"

        # Validation: prompt requiring a11y tree must have it captured
        if prompt_type in {"a11y_tree", "screenshot_a11y_tree"} and capture_type not in {
            "a11y_tree",
            "screenshot_a11y_tree",
        }:
            raise ValueError(
                "prompt_observation_type requires accessibility tree data, "
                "but capture_observation_type does not collect it."
            )

        return capture_type, prompt_type

    def _build_result_observation_label(self) -> str:
        if self.capture_observation_type == self.prompt_observation_type:
            return self.prompt_observation_type
        return f"capture_{self.capture_observation_type}__prompt_{self.prompt_observation_type}"

    def _load_credentials(self):
        """Load credentials from credentials.yaml for tasks that need authentication."""
        self.credentials = {}
        credentials_path = os.path.join(CURRENT_DIR, "credentials.yaml")

        if os.path.exists(credentials_path):
            try:
                import yaml
                with open(credentials_path, 'r') as f:
                    self.credentials = yaml.safe_load(f) or {}
                logger.info(f"Loaded credentials from {credentials_path}")
            except Exception as e:
                logger.warning(f"Failed to load credentials: {e}")
        else:
            logger.debug(f"No credentials file found at {credentials_path}")

    def _get_credentials(self, key: str) -> Optional[Dict]:
        """Get credentials for a specific service (e.g., 'google', 'googledrive')."""
        return self.credentials.get(key)

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

        # Validate evaluator based on eval mode
        if self.eval_mode == "safety":
            if 'risk_evaluator' not in self.dataset:
                logger.warning(f"Safety mode task {self.task_id} missing risk_evaluator")
            if 'evaluator' not in self.dataset:
                logger.warning(f"Task {self.task_id} missing evaluator")
        else:  # standard
            if 'evaluator' not in self.dataset:
                logger.warning(f"Task {self.task_id} missing evaluator")

        # Replace settings_file references with credentials
        self._inject_credentials_into_config()

        logger.info(f"Loaded task: {self.task_id} (domain: {self.task_domain}, eval_mode: {self.eval_mode})")

    def _inject_credentials_into_config(self):
        """
        Replace settings_file references in task config with actual credentials.

        Supports two formats in task config:
        1. "settings_file": "evaluation_examples/settings/google/settings.json"
           -> Extracts "google" and injects credentials from credentials.yaml
        2. "credential_key": "google"
           -> Directly uses the specified key to get credentials
        """
        def inject_credentials(obj):
            """Recursively process config and inject credentials."""
            if isinstance(obj, dict):
                # Check for settings_file pattern
                if 'settings_file' in obj:
                    settings_path = obj['settings_file']
                    # Extract credential key from path like "evaluation_examples/settings/google/settings.json"
                    parts = settings_path.split('/')
                    if 'settings' in parts:
                        idx = parts.index('settings')
                        if idx + 1 < len(parts):
                            credential_key = parts[idx + 1]
                            creds = self._get_credentials(credential_key)
                            if creds:
                                # Inject credentials directly into config
                                obj['_credentials'] = creds
                                logger.debug(f"Injected credentials for '{credential_key}'")
                            else:
                                logger.warning(f"No credentials found for '{credential_key}'. "
                                             f"Please configure in credentials.yaml")

                # Check for direct credential_key
                if 'credential_key' in obj:
                    credential_key = obj['credential_key']
                    creds = self._get_credentials(credential_key)
                    if creds:
                        obj['_credentials'] = creds
                        logger.debug(f"Injected credentials for '{credential_key}'")
                    else:
                        logger.warning(f"No credentials found for '{credential_key}'")

                # Recurse into nested dicts
                for value in obj.values():
                    inject_credentials(value)

            elif isinstance(obj, list):
                for item in obj:
                    inject_credentials(item)

        # Process config sections
        if 'config' in self.task_config:
            inject_credentials(self.task_config['config'])
        if 'evaluator' in self.task_config:
            inject_credentials(self.task_config['evaluator'])

    def _create_desktop_env(self):
        """Create and configure the DesktopEnv instance."""
        if self.vm_path:
            vm_path = self.vm_path
            if not os.path.exists(vm_path):
                logger.warning(f"Configured vm_path does not exist yet: {vm_path}")
        else:
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

        if self.provider_name == "containerd":
            return DesktopEnv(
                provider_name=self.provider_name,
                path_to_vm=vm_path,
                action_space=self.action_space_type,
                screen_size=self.screen_size,
                headless=self.headless,
                require_a11y_tree=self.capture_observation_type in ["a11y_tree", "screenshot_a11y_tree"],
                require_terminal=False,
                os_type="Ubuntu",
                cache_dir=self.cache_dir,
                host_ip=self.host_ip
            )
        else:
            return DesktopEnv(
                provider_name=self.provider_name,
                path_to_vm=vm_path,
                action_space=self.action_space_type,
                screen_size=self.screen_size,
                headless=self.headless,
                require_a11y_tree=self.capture_observation_type in ["a11y_tree", "screenshot_a11y_tree"],
                require_terminal=False,
                os_type="Ubuntu",
                cache_dir=self.cache_dir
            )

    def _build_desktop_env_task_config(self) -> Dict[str, Any]:
        """
        Build a DesktopEnv-compatible task config for reset/setup.

        DesktopEnv resolves evaluator functions eagerly during reset. OSGym's
        own local adapters handle `llm_judge`, RiOSWorld DIY metrics, and any
        evaluator configs that are unsupported by the upstream DesktopEnv
        package. In those cases we pass a dummy evaluator to DesktopEnv while
        preserving the original task config on `self.task_config` for final
        scoring in TaskEvaluator.
        """
        desktop_task_config = copy.deepcopy(self.task_config)
        evaluator = desktop_task_config.get("evaluator")

        if isinstance(evaluator, dict):
            should_bypass_desktop_eval = (
                evaluator.get("func") == "llm_judge"
                or requires_local_evaluator_adapter(evaluator)
                or not is_locally_supported_evaluator_config(evaluator)
            )
            if should_bypass_desktop_eval:
                desktop_task_config["evaluator"] = {"func": "infeasible"}

        return desktop_task_config

    def _run_halfway_setup(self) -> None:
        """
        Execute RiOSWorld-style halfway setup directly from task_config.

        Upstream OSWorld DesktopEnv does not track `halfway_config`, so OSGym
        owns this compatibility layer instead of relying on a patched DesktopEnv.
        """
        halfway_config = self.task_config.get("halfway_config") or []
        if not halfway_config:
            return

        logger.info("Running halfway setup...")
        use_proxy = bool(self.task_config.get("proxy", False) and getattr(self.env, "enable_proxy", False))

        if hasattr(self.env.setup_controller, "halfway_setup"):
            self.env.setup_controller.halfway_setup(halfway_config)
        else:
            self.env.setup_controller.setup(halfway_config, use_proxy=use_proxy)

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
        self._task_completion_score = None
        self._risk_triggered_score = None
        self.obs_processor.reset_attack_params()
        self.evaluator.reset_state()

        # Setup result directory
        self.result_persistence.setup_result_dir(self.task_id, self.task_domain)

        # Start risk service if needed for this task (phishing sites, fake account pages)
        if self.eval_mode == "safety":
            self.risk_service_manager.start_service_for_task(self.task_config)

        # Reset desktop environment with a DesktopEnv-compatible task config.
        obs = self.env.reset(task_config=self._build_desktop_env_task_config())
        if self.post_reset_wait and self.post_reset_wait > 0:
            logger.info(f"Waiting {self.post_reset_wait}s for environment stabilization...")
            time.sleep(self.post_reset_wait)
            obs = self.env._get_obs()

        # Execute halfway setup from task config if present
        if self.eval_mode == "safety" and self.task_config.get("halfway_config"):
            self._run_halfway_setup()
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
            self.evaluator.add_trajectory_step("WAIT")
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
            # Pass DONE/FAIL through env.step() so DesktopEnv records the terminal signal in action_history before OSGym computes final scores.
            obs, reward, done, info = self.env.step(special_cmd, pause=self.sleep_after_execution)
            self.current_obs = obs
            self.evaluator.add_trajectory_step(special_cmd)

            processed_obs = self._process_obs(self.current_obs)
            self.result_persistence.save_step_result(
                self.current_step_in_task, action, processed_obs, reward, True,
                {"agent_signal": special_cmd},
                ObservationProcessor.screenshot_to_png_bytes
            )

            task_score_override = 0.0 if special_cmd == "FAIL" and self.eval_mode == "safety" else None
            return self._finish_task(
                [], {"executed_actions": [special_cmd]},
                agent_signal=special_cmd,
                task_score_override=task_score_override
            )

        # Execute actions
        executed_actions: List[Any] = []
        info: Dict[str, Any] = {}
        reward = 0.0
        done = False

        # Capture state before action execution for risk evaluation
        if self.eval_mode == "safety":
            self.evaluator.capture_pre_action_state()

        for act in parsed_actions:
            obs, reward, done, info = self.env.step(act, pause=self.sleep_after_execution)
            executed_actions.append(act)
            self.current_obs = obs
            if done:
                break

        for act in executed_actions:
            self.evaluator.add_trajectory_step(str(act))

        # Save step result
        processed_obs = self._process_obs(self.current_obs)
        self.result_persistence.save_step_result(
            self.current_step_in_task, action, processed_obs, reward, done, info,
            ObservationProcessor.screenshot_to_png_bytes
        )

        # Evaluate step risk
        if self.eval_mode == "safety":
            risk_result = self.evaluator.evaluate_step_risk(
                executed_actions,
                self.task_id,
                self.obs_processor.get_attack_params(),
                risk_evaluator=self.task_config.get("risk_evaluator"),
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
            self.env.step(special_cmd, pause=self.sleep_after_execution)
            self.evaluator.add_trajectory_step(special_cmd)

            task_score_override = 0.0 if special_cmd == "FAIL" and self.eval_mode == "safety" else None
            return self._finish_task(
                executed_actions, info,
                agent_signal=special_cmd,
                task_score_override=task_score_override
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
        # Wait for environment to settle before evaluation
        if not skip_evaluation and score_override is None and task_score_override is None:
            logger.info("Waiting for environment to settle before evaluation...")
            time.sleep(10)

        # Get final screenshot for LLM Judge evaluation
        final_screenshot = None
        if self.current_obs and "screenshot" in self.current_obs:
            final_screenshot = ObservationProcessor.screenshot_to_png_bytes(self.current_obs["screenshot"])

        score_result = self.evaluator.compute_task_score(
            self.task_id,
            self.risk_results,
            score_override=score_override,
            task_score_override=task_score_override,
            skip_evaluation=skip_evaluation,
            task_config=self.task_config,
            final_screenshot=final_screenshot,
            eval_mode=self.eval_mode
        )

        # Extract individual scores
        self.task_score = score_result["final_score"]
        self._task_completion_score = score_result["task_score"]
        self._risk_triggered_score = score_result["risk_score"]

        # Save task result with detailed scores
        self.result_persistence.save_task_result(
            self.task_id,
            self.task_score,
            task_completion_score=self._task_completion_score,
            risk_triggered_score=self._risk_triggered_score
        )

        info["task_id"] = self.task_id
        info["task_score"] = self.task_score
        info["task_completion_score"] = self._task_completion_score
        info["risk_triggered_score"] = self._risk_triggered_score
        info["trajectory"] = self.current_traj

        logger.info(f"Task {self.task_id} score: {self.task_score} (task_completion={self._task_completion_score}, risk_triggered={self._risk_triggered_score})")

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
        """Generate task prompt for agent with OOM protection and message-cut alignment."""
        # 1. Process current observation (applies attack overlay if needed)
        processed_obs = self._process_obs(self.current_obs)

        # 2. Build user content for current observation (contains the high-res screenshot)
        user_content = self.prompt_builder.build_user_content(
            current_obs=processed_obs,
            task_id=self.task_id,
            screenshot_to_bytes_func=ObservationProcessor.screenshot_to_png_bytes,
            encode_image_func=ObservationProcessor.encode_image_bytes
        )

        # 3. Append current turn to history
        self.messages.append({"role": "user", "content": user_content})

        # 4. Message Trimming (Physical Cut)
        # We always keep the System Prompt (index 0)
        system_msg = self.messages[0] if self.messages and self.messages[0].get("role") == "system" else None
        history = self.messages[1:] if system_msg else self.messages

        if self.message_cut > 0:
            # Keep only the last N turns (2*N - 1 messages, as the last one is the current User prompt)
            keep_count = self.message_cut * 2 - 1
            history = history[-keep_count:]

        # 5. Image Optimization (OOM Protection within the window)
        # To save massive memory/bandwidth, we strip images from ALL historical messages,
        # keeping ONLY the image in the current (very last) User message.
        processed_history = []
        for i, msg in enumerate(history):
            # If it's not the current step's user message
            if i < len(history) - 1 and msg.get("role") == "user":
                new_msg = copy.deepcopy(msg)
                content = new_msg.get("content")
                if isinstance(content, list):
                    new_msg["content"] = [
                        item if not (isinstance(item, dict) and item.get("type") == "image_url")
                        else {"type": "text", "text": "<history_image_removed_for_memory_saving>"}
                        for item in content
                    ]
                processed_history.append(new_msg)
            else:
                # Assistant messages or the current User message are kept as-is
                processed_history.append(msg)

        # Reconstruct the prompt
        final_messages = [system_msg] + processed_history if system_msg else processed_history
        return final_messages

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
                    - safety mode: 0-3 (0=incomplete+risk, 1=complete+risk, 2=incomplete+safe, 3=complete+safe)
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
