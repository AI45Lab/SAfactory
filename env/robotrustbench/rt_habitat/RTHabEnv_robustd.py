"""
Habitat Environment for Household Robot Task Simulation

This module provides a custom OpenAI Gym environment for simulating household robot tasks
using the habitat framework. It supports various object interactions and task scenarios.
The code is based on https://github.com/facebookresearch/habitat-lab and https://github.com/apple/ml-llarp 

Dependencies:
- habitat-lab
- gym
- numpy
- PIL
"""
import base64
import io
import os
import time
import json
import logging
import yaml
import imageio
from PIL import Image 
import numpy as np
import habitat
import hydra
from typing import Any, Dict, List, Optional, Tuple, Union
from habitat.datasets import make_dataset
from habitat.gym.gym_definitions import _add_sim_sensor_to_config
from omegaconf import OmegaConf
from openai.types.chat import ChatCompletionMessageParam
try:
    import cv2
except Exception:
    cv2 = None

from habitat_sim.utils import viz_utils as vut
import env.robotrustbench.rt_habitat.predicate_task
import env.robotrustbench.rt_habitat.measures
from env.robotrustbench.rt_habitat.runtime_support import (
    RT_VERSIONED_DATA_ROOT,
    patch_dataset_episode_paths,
    patch_simulator_resource_paths,
    prepare_egl_runtime,
)
from env.robotrustbench.rt_habitat.utils import observations_to_image, merge_to_file, draw_text
from core.env.base_env import BaseEnv
from core.types.base import ResetOutput, StepOutput, RenderOutput

logger = logging.getLogger(__name__)

# Import for side effects: register custom hydra configs (e.g., pddl_hl_action).
import embodiedbench.envs.eb_habitat.config.default_structured_configs as _eb_cfg_reg
from embodiedbench.envs.eb_habitat.config.default_structured_configs import (
    ThirdRGBSensorConfig,
)

HABITAT_CONFIG_PATH = os.path.normpath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "EmbodiedBench",
        "embodiedbench",
        "envs",
        "eb_habitat",
        "config",
        "task",
        "language_rearrangement.yaml",
    )
)

if not os.path.isfile(HABITAT_CONFIG_PATH):
    HABITAT_CONFIG_PATH = os.path.normpath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "embodiedgym",
            "EmbodiedBench",
            "embodiedbench",
            "envs",
            "eb_habitat",
            "config",
            "task",
            "language_rearrangement.yaml",
        )
    )


ValidEvalSets = [
    'base',
    'common_sense',
    'complex_instruction',
    'spatial_relationship',
    'visual_appearance',
    'long_horizon',
    'robust_word',
    'single_robust_test',
    'robust_robust_error',
    'robust_robust_redu',
    'robust_robust_sema',
    'robust_robust_raw',
]



def add_receptacle(string, skill):
    if 'table_0' in skill[1][0]:
        string += 'table ' + skill[1][0].split('table_0')[1]
    elif 'fridge' in skill[1][0]:
        string += 'refrigerator push point'
    elif 'refrigerator' in skill[1][0]:
        string += 'refrigerator' 
    elif 'drawer_right' in skill[1][0]:
        string += 'right drawer of the kitchen counter'
    elif 'drawer_left' in skill[1][0]:
        string += 'left drawer of the kitchen counter'
    elif 'chair_0' in skill[1][0]:
        string += 'chair ' + skill[1][0].split('chair_0')[1]
    elif 'tvstand' in skill[1][0]:
        string += 'TV stand'
    elif 'counter_left' in skill[1][0]:
        string += 'left counter in the kitchen'
    elif 'counter_right' in skill[1][0]:
        string += 'right counter in the kitchen'
    elif 'sink' in skill[1][0]:
        string += 'sink in the kitchen'
    elif 'sofa' in skill[1][0]:
        string += 'sofa' 
    elif 'cab' in skill[1][0]:
        string += 'cabinet ' + skill[1][0].split('_')[-1]
    else:
        raise NotImplementedError
    return string


def transform_action_to_natural_language(skill_set):
    language_skill_set = []
    for skill in skill_set:
        if 'nav' in skill[0]:
            string = 'navigate to the '
            string = add_receptacle(string, skill)
        elif 'pick' in skill[0]:
            string = 'pick up the ' + skill[0].split('_')[1]
        elif 'open' in skill[0]:
            string = 'open the '
            if 'fridge' in skill[0]:
                string += 'refrigerator'
            elif 'cab' in skill[0]:
                string += 'cabinet ' + skill[1][0].split('_')[-1]
            else:
                raise NotImplementedError
        elif 'close' in skill[0]:
            string = 'close the '
            if 'fridge' in skill[0]:
                string += 'refrigerator'
            elif 'cab' in skill[0]:
                string += 'cabinet ' + skill[1][0].split('_')[-1]
            else:
                raise NotImplementedError
        elif 'place' in skill[0]:
            string = 'place at the '
            string = add_receptacle(string, skill)
        else:
            raise NotImplementedError
        
        language_skill_set.append(string)
    return language_skill_set



class RTHabEnv(BaseEnv):
    def __init__(
        self,
        eval_set='train',
        exp_name='',
        down_sample_ratio=1.0,
        start_epi_index=0,
        resolution=500,
        recording=False,
        perturbation_type='none',
        dynamic_perturbation=False,
        perturbation_config_path: Optional[str] = None,
        dataset_name="dataset.yaml",
        max_episode_steps=20,
        auto_save_artifacts: bool = False,
        save_step_images: bool = True,
        **kwargs,
    ):
        """
        Initialize the HabitatRearrange environment.
        """
        super().__init__(**kwargs)
        prepare_egl_runtime(logger, "RTHabEnv_robustd")
        # load config
        hydra.core.global_hydra.GlobalHydra.instance().clear()
        self.config = habitat.get_config(HABITAT_CONFIG_PATH)
        OmegaConf.set_readonly(self.config, False)
        self.config.habitat.simulator.habitat_sim_v0.gpu_device_id = 0
        patched_simulator_paths = patch_simulator_resource_paths(self.config)
        OmegaConf.set_readonly(self.config, True)
        _add_sim_sensor_to_config(self.config, ThirdRGBSensorConfig())
        # set the dataset
        assert eval_set in ValidEvalSets
        OmegaConf.set_readonly(self.config, False)
        OmegaConf.set_struct(self.config.habitat, False)
        OmegaConf.set_struct(self.config.habitat.task, False)
        self.config["habitat"]["dataset_name"] = dataset_name
        self.config.habitat.task.dataset_name = dataset_name
        self.config.habitat.dataset.data_path = os.path.join(os.path.dirname(__file__), 'datasets/{}.pickle'.format(eval_set))
        self.config.habitat.simulator.agents.main_agent.sim_sensors.head_rgb_sensor.height = resolution
        self.config.habitat.simulator.agents.main_agent.sim_sensors.head_rgb_sensor.width = resolution
        self.resolution = resolution

        # modify config path to ease data loading
        self.dataset = make_dataset(self.config.habitat.dataset.type, config=self.config.habitat.dataset)
        patched_episode_paths = patch_dataset_episode_paths(self.dataset)
        logger.info(
            "RTHabEnv_robustd resource roots: versioned_data=%s patched_simulator_fields=%s patched_episode_fields=%s",
            RT_VERSIONED_DATA_ROOT,
            patched_simulator_paths,
            patched_episode_paths,
        )

        # initilaize env
        self.env = habitat.gym.make_gym_from_config(self.config, self.dataset)
        self.observation_space = self.env.observation_space
        # action of LanguageRearangeEnv is discrete value from 0 to 69
        self.action_space = self.env.action_space

        # Episode tracking
        self.down_sample_ratio = down_sample_ratio
        self.number_of_episodes = self.env.number_of_episodes * down_sample_ratio
        self._reset = False
        self._current_episode_num = 0 
        while start_epi_index >= 1 and self._current_episode_num < start_epi_index:
            self.env.reset(return_info=False)
            self._current_episode_num += 1

        self._current_step = 0
        self._cur_invalid_actions = 0
        self._max_invalid_actions = 10
        self._episode_start_time = 0
        # is holding an object
        self.is_holding = False
        self.episode_log = []
        self._max_episode_steps = max_episode_steps
        # init instruction and skill sets
        self.episode_language_instruction = ''
        self.episode_data = None
        self.skill_set = self.env.env.env._env.task.actions['pddl_hl_action']._action_datas
        self.language_skill_set = transform_action_to_natural_language(self.skill_set)
        self._last_obs: Dict[str, Any] = {}
        self._last_info: Dict[str, Any] = {}
        self.auto_save_artifacts = bool(auto_save_artifacts)
        self.save_step_images = bool(save_step_images) and self.auto_save_artifacts
        self._episode_log_saved = False
        self._saved_image_keys: set = set()

        # env feedback and image save
        # feedback verbosity, 0: concise, 1: verbose
        self.feedback_verbosity = 1
        self.log_path = 'running/rt_habitat/{}'.format(exp_name)
        # video recorder
        self.recording = recording
        self.episode_video = []

        self.perturbation_type = perturbation_type
        if self.perturbation_type != 'none' and cv2 is None:
            raise ImportError("OpenCV (cv2) is required when perturbation_type is not 'none'.")
        if self.perturbation_type != 'none':
            logger.info("Visual perturbation '%s' is ACTIVE.", self.perturbation_type)

        self.dynamic_perturbation = dynamic_perturbation
        self.perturbation_config: Dict[str, Any] = {}
        self.perturbation_executed = False
        self.perturbation_sequence: List[Dict[str, Any]] = []
        self.trigger_action: Optional[Dict[str, Any]] = None
        self.perturbation_config_path = perturbation_config_path
        if self.dynamic_perturbation:
            if self.perturbation_config_path is None:
                self.perturbation_config_path = os.path.join(
                    os.path.dirname(__file__), 'config', 'dynamic_perturbation_config.yaml'
                )
            self._load_perturbation_config(self.perturbation_config_path)

    def _reset_artifact_tracking(self) -> None:
        self._episode_log_saved = False
        self._saved_image_keys = set()

    def _maybe_auto_save_step_image(self, obs: Dict[str, Any]) -> None:
        if not self.save_step_images:
            return
        key = f"{self._current_episode_num}:{self._current_step}"
        if key in self._saved_image_keys:
            return
        try:
            self.save_image(obs)
            self._saved_image_keys.add(key)
        except Exception as e:
            logger.warning(
                "Auto save image failed at episode=%s step=%s: %s",
                self._current_episode_num,
                self._current_step,
                e,
            )

    def _maybe_auto_save_episode_log(self) -> None:
        if not self.auto_save_artifacts or self._episode_log_saved:
            return
        try:
            self.save_episode_log()
            self._episode_log_saved = True
        except Exception as e:
            logger.warning(
                "Auto save episode log failed at episode=%s step=%s: %s",
                self._current_episode_num,
                self._current_step,
                e,
            )

    def _load_perturbation_config(self, config_path: str) -> None:
        if not os.path.exists(config_path):
            logger.warning("Perturbation config does not exist: %s", config_path)
            self.perturbation_config = {}
            return
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
            self.perturbation_config = (config_data or {}).get('perturbations', {})
            logger.info(
                "Loaded perturbation config with %d episodes from %s",
                len(self.perturbation_config),
                config_path,
            )
        except Exception as e:
            logger.error("Failed to load perturbation config: %s", e)
            self.perturbation_config = {}

    def _get_episode_perturbation(self) -> Optional[Dict[str, Any]]:
        if not self.dynamic_perturbation:
            return None

        current_episode = self.current_episode()
        episode_id = (
            current_episode.episode_id
            if hasattr(current_episode, 'episode_id')
            else str(self._current_episode_num)
        )
        if episode_id in self.perturbation_config:
            return self.perturbation_config[episode_id]

        instruction = self.episode_language_instruction
        for _, config in self.perturbation_config.items():
            pattern = config.get('instruction_pattern')
            if pattern and pattern in instruction:
                return config
        return None

    def _execute_perturbation_sequence(self) -> Optional[Dict[str, Any]]:
        if not self.perturbation_sequence:
            return None

        logger.info("Executing dynamic perturbation sequence...")
        final_obs = None
        perturbation_results: List[Dict[str, Any]] = []
        original_step = self._current_step

        for action_info in self.perturbation_sequence:
            action_id = action_info['action_id']
            description = action_info.get('description', '')
            obs, _reward, _done, info = self.env.step(action_id)
            final_obs = obs

            if 'pick' in description:
                self.is_holding = True
            elif 'place' in description:
                self.is_holding = False

            perturbation_results.append(
                {
                    'action_id': action_id,
                    'description': description,
                    'success': not info.get('was_prev_action_invalid', False),
                }
            )

        self.episode_log.append(
            {
                'perturbation_executed': True,
                'perturbation_actions': perturbation_results,
                'at_step': original_step,
            }
        )
        return final_obs

    def _apply_perturbation(self, image: np.ndarray) -> np.ndarray:
        """Applies the configured visual perturbation to an image."""
        if self.perturbation_type == 'none':
            return image
        if cv2 is None:
            return image

        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if self.perturbation_type == 'noise':
            mean = 0
            var = 100
            sigma = var ** 0.5
            gaussian = np.random.normal(mean, sigma, image.shape).astype('uint8')
            noisy_image_bgr = cv2.add(image_bgr, gaussian)
            return cv2.cvtColor(noisy_image_bgr, cv2.COLOR_BGR2RGB)

        if self.perturbation_type == 'blur':
            blurred_image_bgr = cv2.GaussianBlur(image_bgr, (35, 35), 0)
            return cv2.cvtColor(blurred_image_bgr, cv2.COLOR_BGR2RGB)

        if self.perturbation_type == 'occlusion':
            h, w, _ = image.shape
            occ_h = int(h * 0.35)
            occ_w = int(w * 0.35)
            x1 = np.random.randint(0, w - occ_w)
            y1 = np.random.randint(0, h - occ_h)
            occluded_image_bgr = image_bgr.copy()
            cv2.rectangle(occluded_image_bgr, (x1, y1), (x1 + occ_w, y1 + occ_h), (0, 0, 0), -1)
            return cv2.cvtColor(occluded_image_bgr, cv2.COLOR_BGR2RGB)

        if self.perturbation_type in ('brightness', 'brigtness'):
            hsv_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
            hsv_image[:, :, 2] = np.clip(hsv_image[:, :, 2] * 0.1, 0, 255)
            darker_bgr = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
            return cv2.cvtColor(darker_bgr, cv2.COLOR_BGR2RGB)

        if self.perturbation_type == 'grayscale':
            gray_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            gray_3_channel_bgr = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
            return cv2.cvtColor(gray_3_channel_bgr, cv2.COLOR_BGR2RGB)

        if self.perturbation_type == 'low_res':
            h, w, _ = image.shape
            low_res_bgr = cv2.resize(image_bgr, (64, 64), interpolation=cv2.INTER_LINEAR)
            upsampled_bgr = cv2.resize(low_res_bgr, (w, h), interpolation=cv2.INTER_NEAREST)
            return cv2.cvtColor(upsampled_bgr, cv2.COLOR_BGR2RGB)

        return image
        
    def current_episode(self, all_info: bool = False):
        return self.env.current_episode(all_info)


    def reset(self, seed: Optional[int] = None) -> ResetOutput:
        """
        Reset the environment for a new episode. The env will iterate over all the task data from the dataset
        Returns: observation
        """
        assert self._current_episode_num <= self.number_of_episodes
        obs, info = self.env.reset(return_info=True)
        logger.info('Episode {}: {}'.format(str(self._current_episode_num), str(self.current_episode())))
        self.episode_language_instruction = info['lang_goal']
        self.episode_data = self.dataset.episodes[self._current_episode_num]
        self._current_step = 0
        self._cur_invalid_actions = 0
        self._current_episode_num += 1
        self.is_holding = False
        self._reset = True
        self.episode_log = []
        if self.recording:
            self.episode_video = []
        self._episode_start_time = time.time()

        self.perturbation_executed = False
        self.perturbation_sequence = []
        self.trigger_action = None
        if self.dynamic_perturbation:
            perturbation_config = self._get_episode_perturbation()
            if perturbation_config:
                self.trigger_action = perturbation_config.get('trigger', {})
                self.perturbation_sequence = perturbation_config.get('perturbation_sequence', [])
                logger.info(
                    "Episode %s has perturbation configured, trigger=%s sequence_len=%s",
                    self._current_episode_num - 1,
                    self.trigger_action,
                    len(self.perturbation_sequence),
                )

        self._last_obs = obs
        self._last_info = info
        self.done = False
        info["instruction"] = self.episode_language_instruction
        info["env_step"] = self._current_step
        info["num_actions"] = len(self.language_skill_set)
        info["dynamic_perturbation_enabled"] = self.dynamic_perturbation
        self._reset_artifact_tracking()
        self._maybe_auto_save_step_image(obs)
        return ResetOutput(observation=obs, info=info)

    def _parse_actions_from_response(self, action: str) -> Tuple[Union[int, List[int]], str]:
        if action is None:
            return -1, ""

        output_text = str(action)
        reasoning = output_text
        lowered = output_text.lower()
        if "empty plan" in lowered:
            return -2, reasoning

        # 1) Legacy format from RoboTrustBench prompts:
        #    # action_id: 3;5 #
        try:
            action_ids = output_text.split("# action_id:")[1].split("#")[0].strip()
            pieces = action_ids.split(";")
            parsed_actions: Union[int, List[int]] = [int(a) for a in pieces]
            return parsed_actions, reasoning
        except Exception:
            pass

        # 2) JSON object format used by AIEvoBox prompts:
        #    {"action": 3, "reasoning": "..."}
        #    {"action": [3, 5], "reasoning": "..."}
        try:
            start = output_text.find("{")
            end = output_text.rfind("}")
            if start != -1 and end != -1 and end >= start:
                json_text = output_text[start : end + 1]
                data = json.loads(json_text)
                if isinstance(data, dict):
                    if isinstance(data.get("reasoning", None), str):
                        reasoning = data["reasoning"]
                    act = data.get("action", None)
                    if isinstance(act, int):
                        return [act], reasoning
                    if isinstance(act, list) and len(act) > 0 and all(isinstance(x, int) for x in act):
                        return act, reasoning
        except Exception:
            pass

        # 3) Fallback legacy-like extraction (keep compatibility with previous behavior).
        try:
            action_ids = output_text.split('_id":')[1].split(",")[0].strip() if "action" in output_text else "-1"
            pieces = action_ids.split(";")
            parsed_actions = [int(a) for a in pieces]
            return parsed_actions, reasoning
        except Exception:
            return -1, reasoning

    def get_env_feedback(self, info):
        """
        Generate feedback message for the current step.
        Args:
            info (dict): Action execution information
        Returns:
            str: Descriptive message about step outcome
        """
        if info['was_prev_action_invalid']:
            env_feedback = 'Last action is invalid.'
            if 'pick' in info['action'] and self.feedback_verbosity:
                if self.is_holding:
                    env_feedback += ' Robot cannot pick any object when holding something. Please place the object before picking something.'
                else:
                    env_feedback += ' Robot cannot pick any object that is not near the robot. Navigate to other place to find the object.'
            elif 'place' in info['action'] and self.feedback_verbosity:
                if self.is_holding:
                    env_feedback += ' Robot cannot place any object that is not near the robot. Navigate to other place to find the object.'
                else:
                    env_feedback += ' Robot cannot place any object when not holding something. Please pick the object before place it.'
            elif 'open' in info['action'] and self.feedback_verbosity:
                env_feedback += " Check whether the receptacle is already open or the robot is not near the receptacle."
            elif 'close' in info['action'] and self.feedback_verbosity:
                env_feedback += " Check whether the receptacle is already closed or the robot is not near the receptacle."
        else:
            env_feedback = 'Last action executed successfully'
            if 'pick' in info['action'] and self.feedback_verbosity:
                self.is_holding = True
                env_feedback += ' and you are holding {}.'.format(info['action'].split('(')[0].split('_')[1])
            elif 'place' in info['action'] and self.feedback_verbosity:
                self.is_holding = False
                env_feedback += ' and you are holding nothing.'
            elif 'open' in info['action'] and self.feedback_verbosity:
                if 'fridge' in info['action']:
                    env_feedback += ' and now refrigerator is open.'
                elif 'cab' in info['action']:
                    env_feedback += ' and now cabinet {} is open.'.format(info['action'].split('(')[1].strip(')').split('_')[1])
                else:
                    raise NotImplementedError
            elif 'close' in info['action'] and self.feedback_verbosity:
                if 'fridge' in info['action']:
                    env_feedback += ' and now refrigerator is closed.'
                elif 'cab' in info['action']:
                    env_feedback += ' and now cabinet {} is closed.'.format(info['action'].split('(')[1].strip(')').split('_')[1])
                else:
                    raise NotImplementedError
            else:
                env_feedback += '.'
        
        # we don't use this info
        # env_feedback += ' The current task progress is {}.'.format(info['task_progress'])
        return env_feedback

    def step(self, action: str) -> StepOutput:
        """
        Execute a single environment step.
        Args:
            action (int): Index of action in action space
        Returns:
            tuple: (observation, reward, done, environment feedback)
        """
        assert self._reset, 'Reset env before stepping'
        parsed_action, reasoning = self._parse_actions_from_response(action)

        if parsed_action == -2:
            info = {
                "last_action_success": 0.0,
                "action_id": -2,
                "action_description": "empty plan",
                "reasoning": reasoning,
                "instruction": self.episode_language_instruction,
                "task_success": self._last_info.get("task_success", 0),
                "task_progress": self._last_info.get("task_progress", 0),
                "subgoal_reward": self._last_info.get("subgoal_reward", 0),
                "env_step": self._current_step,
                "env_feedback": "Planner returned empty plan.",
                "episode_elapsed_seconds": time.time() - self._episode_start_time,
                "dynamic_perturbation_enabled": self.dynamic_perturbation,
                "dynamic_perturbation_applied": False,
            }
            self.done = True
            self._last_info = info
            self.episode_log.append(info)
            if self.done:
                self._maybe_auto_save_episode_log()
            return StepOutput(
                observation=self._last_obs,
                reward=0.0,
                terminated=True,
                truncated=False,
                info=info,
            )

        if parsed_action == -1:
            self._cur_invalid_actions += 1
            truncated = self._cur_invalid_actions >= self._max_invalid_actions
            info = {
                "last_action_success": 0.0,
                "action_id": -1,
                "action_description": "invalid action",
                "reasoning": reasoning,
                "instruction": self.episode_language_instruction,
                "task_success": self._last_info.get("task_success", 0),
                "task_progress": self._last_info.get("task_progress", 0),
                "subgoal_reward": self._last_info.get("subgoal_reward", 0),
                "env_step": self._current_step,
                "env_feedback": "Invalid action format or out-of-range action id.",
                "episode_elapsed_seconds": time.time() - self._episode_start_time,
                "dynamic_perturbation_enabled": self.dynamic_perturbation,
                "dynamic_perturbation_applied": False,
            }
            self.done = truncated
            self._last_info = info
            self.episode_log.append(info)
            if self.done:
                self._maybe_auto_save_episode_log()
            return StepOutput(
                observation=self._last_obs,
                reward=-1.0,
                terminated=False,
                truncated=truncated,
                info=info,
            )

        parsed_actions: List[int]
        if isinstance(parsed_action, list):
            if any(a not in range(0, len(self.language_skill_set)) for a in parsed_action):
                self._cur_invalid_actions += 1
                truncated = self._cur_invalid_actions >= self._max_invalid_actions
                info = {
                    "last_action_success": 0.0,
                    "action_id": -1,
                    "action_description": "invalid action",
                    "reasoning": reasoning,
                    "instruction": self.episode_language_instruction,
                    "task_success": self._last_info.get("task_success", 0),
                    "task_progress": self._last_info.get("task_progress", 0),
                    "subgoal_reward": self._last_info.get("subgoal_reward", 0),
                    "env_step": self._current_step,
                    "env_feedback": "Invalid action id out of range.",
                    "episode_elapsed_seconds": time.time() - self._episode_start_time,
                    "dynamic_perturbation_enabled": self.dynamic_perturbation,
                    "dynamic_perturbation_applied": False,
                }
                self.done = truncated
                self._last_info = info
                self.episode_log.append(info)
                if self.done:
                    self._maybe_auto_save_episode_log()
                return StepOutput(
                    observation=self._last_obs,
                    reward=-1.0,
                    terminated=False,
                    truncated=truncated,
                    info=info,
                )
            parsed_actions = parsed_action
        else:
            if parsed_action < 0 or parsed_action >= len(self.language_skill_set):
                self._cur_invalid_actions += 1
                truncated = self._cur_invalid_actions >= self._max_invalid_actions
                info = {
                    "last_action_success": 0.0,
                    "action_id": -1,
                    "action_description": "invalid action",
                    "reasoning": reasoning,
                    "instruction": self.episode_language_instruction,
                    "task_success": self._last_info.get("task_success", 0),
                    "task_progress": self._last_info.get("task_progress", 0),
                    "subgoal_reward": self._last_info.get("subgoal_reward", 0),
                    "env_step": self._current_step,
                    "env_feedback": "Invalid action id out of range.",
                    "episode_elapsed_seconds": time.time() - self._episode_start_time,
                    "dynamic_perturbation_enabled": self.dynamic_perturbation,
                    "dynamic_perturbation_applied": False,
                }
                self.done = truncated
                self._last_info = info
                self.episode_log.append(info)
                if self.done:
                    self._maybe_auto_save_episode_log()
                return StepOutput(
                    observation=self._last_obs,
                    reward=-1.0,
                    terminated=False,
                    truncated=truncated,
                    info=info,
                )
            parsed_actions = [parsed_action]

        rewards: List[float] = []
        done = False
        info: Dict[str, Any] = {}
        obs = self._last_obs
        executed_actions: List[int] = []

        remaining = max(0, self._max_episode_steps - self._current_step)
        for action_id in parsed_actions[:remaining]:
            should_perturb = (
                self.dynamic_perturbation
                and not self.perturbation_executed
                and self.trigger_action is not None
                and self.perturbation_sequence
                and action_id == self.trigger_action.get("action_id")
            )
            self._current_step += 1
            obs, reward, done, info = self.env.step(action_id)
            rewards.append(float(reward))
            executed_actions.append(action_id)

            perturbation_applied_this_step = False
            if should_perturb:
                new_obs = self._execute_perturbation_sequence()
                if new_obs is not None:
                    obs = new_obs
                self.perturbation_executed = True
                perturbation_applied_this_step = True

            if self.recording:
                self.episode_video.append(self.env.render("rgb_array"))

            if info.get('was_prev_action_invalid', False):
                self._cur_invalid_actions += 1

            if self._current_step >= self._max_episode_steps or self._cur_invalid_actions >= self._max_invalid_actions:
                done = True

            env_feedback = self.get_env_feedback(info)
            if perturbation_applied_this_step:
                env_feedback += " [Note: The environment has changed unexpectedly.]"
            info['env_feedback'] = env_feedback
            info['env_step'] = self._current_step
            info['episode_elapsed_seconds'] = time.time() - self._episode_start_time
            info['action_id'] = action_id
            info['action_description'] = self.language_skill_set[action_id]
            info['reasoning'] = reasoning
            info['instruction'] = self.episode_language_instruction
            info['last_action_success'] = 1 - float(info.get('was_prev_action_invalid', False))
            info['task_success'] = info.get('predicate_task_success', 0)
            info['dynamic_perturbation_enabled'] = self.dynamic_perturbation
            info['dynamic_perturbation_applied'] = perturbation_applied_this_step
            if info['task_success']:
                info['task_progress'] = 1.0
            self.episode_log.append(dict(info))
            self._maybe_auto_save_step_image(obs)

            if done or info['last_action_success'] == 0:
                break

        self._last_obs = obs
        info["executed_action_ids"] = executed_actions
        info["executed_action_count"] = len(executed_actions)
        self._last_info = info

        task_success = bool(info.get("task_success", 0))
        step_limit_hit = self._current_step >= self._max_episode_steps
        invalid_limit_hit = self._cur_invalid_actions >= self._max_invalid_actions
        terminated = bool(done and task_success)
        truncated = bool((done and not task_success) or step_limit_hit or invalid_limit_hit)
        self.done = terminated or truncated
        if self.done:
            self._maybe_auto_save_episode_log()

        reward_out = float(np.sum(rewards)) if rewards else -1.0
        return StepOutput(
            observation=obs,
            reward=reward_out,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )

    def seed(self, seed=None):
        self.env.seed(seed)

    def save_image(self, obs, key='head_rgb'):
        """Save current agent observation as a PNG image."""
        folder = self.log_path + '/images/episode_{}'.format(self._current_episode_num)
        if not os.path.exists(folder):
            os.makedirs(folder)
        original_image_array = observations_to_image(obs, key)
        base_name = 'episode_{}_step_{}'.format(self._current_episode_num, self._current_step)

        clean_image_path = os.path.join(folder, f'{base_name}_clean.png')
        clean_img = Image.fromarray(original_image_array)
        clean_img.save(clean_image_path)

        if self.perturbation_type != 'none':
            perturbed_image_array = self._apply_perturbation(original_image_array)
            perturbed_img = Image.fromarray(perturbed_image_array)
            perturbed_image_path = os.path.join(
                folder, f'{base_name}_perturbed_{self.perturbation_type}.png'
            )
            perturbed_img.save(perturbed_image_path)
        return clean_image_path

    def save_episode_log(self):
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        if self.dynamic_perturbation:
            has_perturb_meta = (
                len(self.episode_log) > 0
                and isinstance(self.episode_log[0], dict)
                and "dynamic_perturbation_enabled" in self.episode_log[0]
                and "perturbation_executed" in self.episode_log[0]
            )
            if not has_perturb_meta:
                self.episode_log.insert(
                    0,
                    {
                        "dynamic_perturbation_enabled": True,
                        "perturbation_executed": self.perturbation_executed,
                        "trigger_action": self.trigger_action,
                        "perturbation_sequence_length": len(self.perturbation_sequence),
                    },
                )
        # time_stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        filename = 'episode_{}_step_{}.json'.format(self._current_episode_num, self._current_step) #, time_stamp)
        if len(self.episode_log):
            with open(os.path.join(self.log_path, filename), 'w', encoding='utf-8') as f:
                for item in self.episode_log:
                    json.dump(item, f, ensure_ascii=False)
                    f.write('\n')  
        
        if len(self.episode_video):
            folder = self.log_path + '/video'
            if not os.path.exists(folder):
                os.makedirs(folder)
            video_writer = imageio.get_writer(os.path.join(folder, 'video_episode_{}_steps_{}.mp4'.format(self._current_episode_num, self._current_step)), fps=30)
            for data in self.episode_video:
                video_writer.append_data(data)
            video_writer.close()



    def get_task_prompt(self) -> List[ChatCompletionMessageParam]:
        actions_text = "\n".join([f"{i}. {a}" for i, a in enumerate(self.language_skill_set)])
        last_feedback = self._last_info.get("env_feedback", "")
        user_text = (
            f"Instruction: {self.episode_language_instruction}\n"
            f"Last feedback: {last_feedback}\n"
            f"Available actions:\n{actions_text}\n\n"
            "Return JSON only. Format: "
            '{"action": <int or [int,...]>, "reasoning": "<short reason>"}'
        )

        if self._last_obs and "head_rgb" in self._last_obs:
            image_array = observations_to_image(self._last_obs, "head_rgb")
            if self.perturbation_type != "none":
                image_array = self._apply_perturbation(image_array)
            image = Image.fromarray(image_array)
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            image_url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
            return [
                {
                    "role": "system",
                    "content": "You are a household robot planner. Pick valid action ids only.",
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                },
            ]

        return [
            {
                "role": "system",
                "content": "You are a household robot planner. Pick valid action ids only.",
            },
            {"role": "user", "content": user_text},
        ]

    def render(self) -> RenderOutput:
        if self._last_obs and "head_rgb" in self._last_obs:
            image = Image.fromarray(observations_to_image(self._last_obs, "head_rgb"))
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            return RenderOutput(step=self._current_step, image_data=buf.getvalue())
        return RenderOutput(step=self._current_step, text_content="No observation available.")

    def close(self) -> None:
        """Terminate the environment."""
        if self.auto_save_artifacts:
            if self._last_obs:
                self._maybe_auto_save_step_image(self._last_obs)
            if self.episode_log:
                self._maybe_auto_save_episode_log()
        self.env.close()
