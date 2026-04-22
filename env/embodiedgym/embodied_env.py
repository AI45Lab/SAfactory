"""
EmbodiedAlfredGym - Alfred Environment Adapter for AIEvoBox
将 EmbodiedBench 的 Alfred 环境适配到 AIEvoBox 框架
"""

import re
import os
import sys
import io
import json
import base64
import numpy as np
import gymnasium as gym
from PIL import Image
from typing import Optional, Dict, Any, List, Tuple
from enum import Enum

from openai.types.chat import ChatCompletionMessageParam
from core.types.base import ResetOutput, StepOutput, RenderOutput
from core.env.base_env import BaseEnv
from core.env.env_register import register_env

# 导入 prompt 模板
from env.embodiedgym.prompt import (
    SYSTEM_PROMPT,
    TASK_HEADER,
    PREVIOUS_ACTION_HEADER,
    PREVIOUS_ACTION_TEMPLATE,
    SUCCESS_STATUS_YES,
    SUCCESS_STATUS_NO,
    INSTRUCTION_AFTER_FIND_DESTINATION_HOLDING,
    INSTRUCTION_AFTER_FIND_DESTINATION_NOT_HOLDING,
    INSTRUCTION_AFTER_FIND_EQUIPMENT,
    INSTRUCTION_AFTER_FIND_OBJECT,
    INSTRUCTION_AFTER_PICKUP,
    INSTRUCTION_AFTER_NAVIGATE,
    INSTRUCTION_AFTER_TURN_ON,
    INSTRUCTION_AFTER_TURN_OFF,
    INSTRUCTION_AFTER_INTERACTION_DEFAULT,
    INSTRUCTION_SUCCESS_GENERIC,
    INSTRUCTION_FAILURE,
    OBSERVATION_HEADER,
    OBSERVATION_PROMPT,
    ACTIONS_HEADER,
    ACTIONS_SUBHEADER,
    ACTIONS_TRUNCATED,
    OUTPUT_FORMAT_SECTION,
    EQUIPMENT_KEYWORDS,
    DESTINATION_KEYWORDS,
    get_equipment_next_action_hint,
    get_pickup_next_step_hint,
    get_priority_keywords_after_action,
    DEFAULT_PRIORITY_KEYWORDS,
    OTHER_ACTION_KEYWORDS,
)

# ALFRED 标准动作列表（作为延迟初始化的默认值，防止 import 错误）
# 当环境真正启动后，会被环境内部的真实动作列表覆盖
DEFAULT_ALFRED_ACTIONS = [
    "RotateLeft", "RotateRight", "MoveAhead", "LookDown", "LookUp", 
    "PickupObject", "PutObject", "OpenObject", "CloseObject", 
    "ToggleObjectOn", "ToggleObjectOff", "SliceObject", 
    "CleanObject", "HeatObject", "CoolObject"
]

# 动态导入 EmbodiedBench 模块
try:
    # --- 路径动态适配 ---
    project_root = os.getcwd()
    embodied_bench_path = os.path.join(project_root, "env", "embodiedgym", "EmbodiedBench-master")
    
    if not os.path.exists(embodied_bench_path):
        current_file_path = os.path.abspath(__file__)
        d = os.path.dirname(current_file_path)
        while d != "/" and d != os.path.dirname(d):
            if os.path.exists(os.path.join(d, "env", "embodiedgym")):
                embodied_bench_path = os.path.join(d, "env", "embodiedgym", "EmbodiedBench-master")
                break
            d = os.path.dirname(d)

    # 添加 EmbodiedBench 路径到 sys.path
    if embodied_bench_path not in sys.path:
        sys.path.insert(0, embodied_bench_path)
    
    # [修复] 仅导入 EBAlfEnv，不再尝试导入可能不存在的 action_primitives
    from embodiedbench.envs.eb_alfred.EBAlfEnv import EBAlfEnv
    
except ImportError as e:
    print(f"警告: 无法导入 EmbodiedBench Alfred 环境: {e}")
    print(f"尝试加载的路径: {locals().get('embodied_bench_path', 'Not Computed')}")
    print("请确保 EmbodiedBench 已正确安装且路径正确")
    EBAlfEnv = None


@register_env("embodied_alfred")
class EmbodiedAlfredGym(BaseEnv):
    """
    EmbodiedBench Alfred 环境适配器
    """
    
    def __init__(
        self,
        resolution: int = 500,
        detection_box: bool = False,
        max_episode_steps: int = 50,
        max_invalid_actions: int = 10,
        exp_name: str = 'aievobox_alfred',
        eval_set: str = 'base',
        down_sample_ratio: float = 1.0,
        selected_indexes: Optional[List[int]] = None,
        alfred_data_path: Optional[str] = None,
        **kwargs
    ):
        """
        初始化 Alfred 环境适配器
        """
        super().__init__(**kwargs)
        
        if EBAlfEnv is None:
            raise RuntimeError("EBAlfEnv 未成功导入，请检查 EmbodiedBench 安装")
        
        self.resolution = resolution
        self.max_episode_steps = max_episode_steps
        self.max_invalid_actions = max_invalid_actions
        self.exp_name = exp_name
        
        # 检查是否使用 dataset 模式
        self.use_dataset_mode = bool(self.dataset)
        
        # 保存初始化参数供延迟初始化使用
        self._init_params = {
            'detection_box': detection_box,
            'resolution': resolution,
            'exp_name': exp_name,
            'down_sample_ratio': down_sample_ratio,
            'selected_indexes': selected_indexes or [],
        }
        
        if self.use_dataset_mode:
            # ========== Dataset 模式（延迟初始化） ==========
            self.task_id = self.dataset.get("id", 0)
            self.task_path = self.dataset.get("task", "")
            self.repeat_idx = self.dataset.get("repeat_idx", 0)
            self.task_instruction = self.dataset.get("instruction", "")
            self.eval_set = self.dataset.get("eval_set", "base")
            
            if not self.task_path:
                raise ValueError("dataset 中缺少 'task' 字段")
            
            self.single_task = {
                "task": self.task_path,
                "repeat_idx": self.repeat_idx,
                "instruction": self.task_instruction
            }
            
            self.alfred_env = None
            self._env_initialized = False
            
            # [修复] 使用默认动作列表进行初始化，不再依赖外部 import
            self.language_skill_set = DEFAULT_ALFRED_ACTIONS
            self.action_space = gym.spaces.Discrete(len(self.language_skill_set))
            
            print(f"✓ EmbodiedAlfredGym 初始化成功 [Dataset 模式 - 延迟加载]")
            print(f"  - 任务 ID: {self.task_id}")
            print(f"  - 任务路径: {self.task_path[:60]}...")
            
        else:
            # ========== 传统 eval_set 模式（立即初始化） ==========
            self.eval_set = eval_set
            selected_indexes = selected_indexes or []
            
            self.alfred_env = EBAlfEnv(
                eval_set=eval_set,
                exp_name=exp_name,
                down_sample_ratio=down_sample_ratio,
                selected_indexes=selected_indexes,
                detection_box=detection_box,
                resolution=resolution
            )
            self._env_initialized = True
            
            self.alfred_env._max_episode_steps = max_episode_steps
            self.alfred_env._max_invalid_actions = max_invalid_actions
            
            self.action_space = self.alfred_env.action_space
            self.language_skill_set = self.alfred_env.language_skill_set
            
            print(f"✓ EmbodiedAlfredGym 初始化成功 [Eval_set 模式]")
        
        self.current_step = 0
        self.current_action = None
        self.current_reasoning = ""
        self.episode_instruction = ""
        self.current_episode_num = 0
        self.current_task_type = ""
        self.last_step_info = {}
        self.is_holding_object = False
        
        # 行为循环检测相关状态
        self.max_progress_achieved = 0.0
        self.steps_since_progress = 0
    
    def _lazy_init_alfred_env(self):
        """
        延迟初始化 EBAlfEnv（仅在 dataset 模式下使用）
        """
        if self._env_initialized:
            return
        
        print(f"🚀 正在初始化 AI2THOR 环境...")
        
        # 创建 EBAlfEnv
        self.alfred_env = EBAlfEnv(
            eval_set=self.eval_set,
            exp_name=self._init_params['exp_name'],
            down_sample_ratio=1.0,
            selected_indexes=[0],
            detection_box=self._init_params['detection_box'],
            resolution=self._init_params['resolution']
        )
        
        # 覆盖数据集为单任务
        self.alfred_env.dataset = [self.single_task]
        self.alfred_env.number_of_episodes = 1
        self.alfred_env._current_episode_num = 0
        
        self.alfred_env._max_episode_steps = self.max_episode_steps
        self.alfred_env._max_invalid_actions = self.max_invalid_actions
        
        # [关键] 从真实环境实例更新动作空间，确保完全一致
        if hasattr(self.alfred_env, 'language_skill_set'):
            self.language_skill_set = self.alfred_env.language_skill_set
            self.action_space = self.alfred_env.action_space
            # print(f"  - 动作空间已从环境实例同步 (大小: {len(self.language_skill_set)})")
        
        self._env_initialized = True
        print(f"✓ AI2THOR 环境初始化完成")
    
    def reset(self, seed: Optional[int] = None) -> ResetOutput:
        """
        重置环境到初始状态
        """
        # Dataset 模式下的延迟初始化
        if self.use_dataset_mode and not self._env_initialized:
            self._lazy_init_alfred_env()
        
        obs = self.alfred_env.reset()
        
        self.current_step = 0
        self.current_action = None
        self.current_reasoning = ""
        self.episode_instruction = self.alfred_env.episode_language_instruction
        self.current_episode_num = self.alfred_env._current_episode_num - 1
        self.is_holding_object = False
        
        try:
            current_episode = self.alfred_env.current_episode()
            self.current_task_type = current_episode.get('task_type', 'unknown')
        except Exception as e:
            print(f"⚠ 无法加载任务类型: {e}")
            self.current_task_type = 'unknown'
        
        self.last_step_info = {}
        
        # 重置行为循环检测状态
        self.max_progress_achieved = 0.0
        self.steps_since_progress = 0
        
        self.messages = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]
        
        observation = {
            'head_rgb': obs['head_rgb'],
            'instruction': self.episode_instruction,
            'available_actions': self.language_skill_set
        }
        
        info = {
            'episode_num': self.current_episode_num,
            'task_type': self.current_task_type,
            'eval_set': self.eval_set,
            'instruction': self.episode_instruction,
            'num_actions': len(self.language_skill_set)
        }
        
        print(f"\n{'='*80}")
        print(f"Episode {info['episode_num']} 开始 [{self.eval_set}]")
        print(f"任务类型: {self.current_task_type}")
        print(f"任务指令: {self.episode_instruction}")
        print(f"{'='*80}\n")
        
        return ResetOutput(observation=observation, info=info)
    
    def step(self, action: str) -> StepOutput:
        """
        执行一步环境交互
        """
        super().step(action=action)
        
        self.messages.append(
            {"role": "assistant", "content": action}
        )
        
        self.current_step += 1
        
        action_id, reasoning, is_valid = self.parse_llm_response(action)
        self.current_action = action_id
        self.current_reasoning = reasoning
        
        if action_id == -2 and is_valid:
            print(f"🏁 步骤 {self.current_step}: LLM 认为任务已完成")
            print(f"  推理: {reasoning[:150]}...")
            
            task_progress = self.last_step_info.get('task_progress', 0.0) if self.last_step_info else 0.0
            is_task_really_done = task_progress >= 1.0
            obs = {'head_rgb': self.alfred_env.env.last_event.frame}

            if is_task_really_done:
                print("  环境确认: 任务成功完成!")
                reward = 10.0
                info = {
                    'task_success': 1.0, 'task_progress': 1.0, 'last_action_success': 1.0,
                    'env_feedback': 'LLM correctly signaled task completion.',
                    'instruction': self.episode_instruction, 'llm_done_signal': True, 'reasoning': reasoning
                }
                self.last_step_info = {
                    'action_id': -2, 'action_description': 'Task Complete Signal (Correct)',
                    'last_action_success': 1, 'env_feedback': info['env_feedback'],
                    'task_progress': 1.0, 'task_success': 1.0, 'reasoning': reasoning
                }
                return StepOutput(observation=obs, reward=reward, terminated=True, truncated=False, info=info)
            else:
                print(f"  环境纠正: 任务未完成! 当前进度: {task_progress:.1%}")
                reward = -2.0
                corrective_feedback = (
                    f"You incorrectly claimed the task was complete. The task is only {task_progress:.1%} finished. "
                    f"Please review the original instruction carefully: '{self.episode_instruction}'. "
                    f"Identify what part of the task is remaining and execute the next necessary action. "
                    f"Do NOT declare the task complete again until all objectives are met."
                )
                info = {
                    'task_success': 0.0, 'task_progress': task_progress, 'last_action_success': 0.0,
                    'env_feedback': corrective_feedback, 'instruction': self.episode_instruction,
                    'llm_done_signal': True, 'reasoning': reasoning
                }
                self.last_step_info = {
                    'action_id': -2, 'action_description': 'Task Complete Signal (Incorrect)',
                    'last_action_success': 0, 'env_feedback': corrective_feedback,
                    'task_progress': task_progress, 'task_success': 0.0, 'reasoning': reasoning
                }
                return StepOutput(observation=obs, reward=reward, terminated=False, truncated=False, info=info)
        
        if not is_valid:
            print(f"⚠ 步骤 {self.current_step}: LLM 输出解析失败")
            print(f"  原始输出: {action[:500]}...")
            obs = {'head_rgb': self.alfred_env.env.last_event.frame}
            reward = -1.0
            done = self.alfred_env._cur_invalid_actions >= self.max_invalid_actions
            info = {
                'task_success': 0.0, 'task_progress': 0.0, 'last_action_success': 0.0,
                'env_feedback': 'Failed to parse LLM output. Expected JSON format with executable_plan.',
                'instruction': self.episode_instruction, 'invalid_output': True, 'reasoning': reasoning
            }
            self.last_step_info = {
                'action_id': None, 'action_description': 'Invalid Output', 'last_action_success': 0,
                'env_feedback': info['env_feedback'], 'task_progress': 0.0, 'task_success': 0.0, 'reasoning': reasoning
            }
            return StepOutput(observation=obs, reward=reward, terminated=done, truncated=False, info=info)
        
        try:
            obs, reward, done, info = self.alfred_env.step(action_id, reasoning=reasoning)
        except Exception as e:
            print(f"✗ 步骤 {self.current_step}: 环境执行出错 - {str(e)}")
            obs = {'head_rgb': self.alfred_env.env.last_event.frame}
            reward = -1.0
            done = True
            action_str = self.language_skill_set[action_id] if isinstance(action_id, int) and 0 <= action_id < len(self.language_skill_set) else str(action_id)
            info = {
                'task_success': 0.0, 'task_progress': 0.0, 'last_action_success': 0.0,
                'env_feedback': f'Environment error: {str(e)}',
                'instruction': self.episode_instruction, 'error': True
            }
            self.last_step_info = {
                'action_id': action_id, 'action_description': action_str, 'last_action_success': 0,
                'env_feedback': info['env_feedback'], 'task_progress': 0.0, 'task_success': 0.0, 'reasoning': reasoning
            }
        
        # 行为循环/停滞检测
        current_progress = info.get('task_progress', 0.0)
        if current_progress > self.max_progress_achieved:
            self.max_progress_achieved = current_progress
            self.steps_since_progress = 0
            print(f"  📈 进度提升至: {current_progress:.1%}")
        else:
            self.steps_since_progress += 1

        action_str = self.language_skill_set[action_id] if isinstance(action_id, int) and 0 <= action_id < len(self.language_skill_set) else str(action_id)
        success_icon = "✓" if info.get('last_action_success', 0) else "✗"
        print(f"{success_icon} 步骤 {self.current_step}: {action_str}")
        print(f"  奖励: {reward:.3f} | 进度: {info.get('task_progress', 0):.2%} | 动作成功: {bool(info.get('last_action_success', 0))} | 任务完成: {bool(info.get('task_success', 0))}")
        
        env_feedback = info.get('env_feedback', '')
        if env_feedback and len(env_feedback) < 200:
            print(f"  反馈: {env_feedback}")
        
        if reasoning and self.current_step <= 3:
            print(f"  推理: {reasoning[:150]}{'...' if len(reasoning) > 150 else ''}")
        
        self.last_step_info = {
            'action_id': action_id, 'action_description': action_str,
            'last_action_success': info.get('last_action_success', 0),
            'env_feedback': info.get('env_feedback', ''),
            'task_progress': info.get('task_progress', 0.0),
            'task_success': info.get('task_success', 0.0),
            'reasoning': reasoning
        }
        
        if info.get('last_action_success', 0):
            action_lower = action_str.lower()
            if 'pick' in action_lower and 'up' in action_lower:
                self.is_holding_object = True
            elif 'put' in action_lower and 'down' in action_lower:
                self.is_holding_object = False
        
        env_feedback = info.get('env_feedback', '')
        if 'currently holding' in env_feedback.lower():
            self.is_holding_object = True
        
        return StepOutput(
            observation=obs, reward=reward, terminated=done, truncated=False, info=info
        )
    
    def get_task_prompt(self) -> List[ChatCompletionMessageParam]:
        """
        生成任务提示（包含图像的多模态 prompt）
        """
        current_frame = self._get_current_image()
        img_base64 = self._numpy_to_base64(current_frame)
        
        user_text_parts = [
            TASK_HEADER,
            self.episode_instruction,
            ""
        ]
        
        if hasattr(self, 'last_step_info') and self.last_step_info and self.current_step > 0:
            last_action_desc = self.last_step_info.get('action_description', 'N/A')
            progress = self.last_step_info.get('task_progress', 0)
            
            last_feedback = self.last_step_info.get('env_feedback', '').lower()
            last_success = self.last_step_info.get('last_action_success', 0)
            special_feedback_triggered = False

            success_status = SUCCESS_STATUS_YES if last_success else SUCCESS_STATUS_NO
            user_text_parts.extend([
                PREVIOUS_ACTION_HEADER.format(step=self.current_step),
                PREVIOUS_ACTION_TEMPLATE.format(
                    action=last_action_desc, success_status=success_status,
                    progress=progress * 100,
                    feedback=self.last_step_info.get('env_feedback', 'No feedback')[:500]
                ),
                ""
            ])

            # 1. 优先处理最严重的状态错误
            if not last_success and 'currently holding' in last_feedback:
                state_error_feedback = (
                    "CRITICAL ERROR: Your last action failed because you tried to 'pick up' an object while you were already holding one. "
                    "Pay close attention to your state. You are currently HOLDING an object.\n"
                    "Your next action must be something other than 'pick up'."
                )
                user_text_parts.append(state_error_feedback)
                user_text_parts.append("")
                special_feedback_triggered = True

            # 2. 然后处理行为循环/进度停滞
            if not special_feedback_triggered and self.steps_since_progress >= 5:
                stuck_feedback = (
                    f"NOTICE: You have performed several actions, but the task progress has been stuck at {self.max_progress_achieved:.1%}. "
                    f"This indicates you are repeating actions without addressing all parts of the main goal: '{self.episode_instruction}'.\n\n"
                    "HINT: For tasks involving 'wash' or 'clean', simply turning a faucet on and off is NOT enough. "
                    "You must explicitly use a specific action like 'clean the <object>' while near a sink. Re-evaluate your plan."
                )
                user_text_parts.append(stuck_feedback)
                user_text_parts.append("")
                self.steps_since_progress = 0 # 重置计数器
                special_feedback_triggered = True

            # 3. 如果没有触发特殊反馈，且上一步成功，则给出常规指导
            if not special_feedback_triggered and last_success:
                 user_text_parts.extend(
                    self._build_instruction_after_action(last_action_desc, last_success, progress)
                )

        user_text_parts.extend([
            OBSERVATION_HEADER,
            OBSERVATION_PROMPT.format(step=self.current_step + 1),
            "",
            ACTIONS_HEADER.format(count=len(self.language_skill_set)),
            ACTIONS_SUBHEADER
        ])
        
        shown_actions = self._build_action_list(user_text_parts)
        
        if len(shown_actions) < len(self.language_skill_set):
            user_text_parts.append(
                ACTIONS_TRUNCATED.format(total=len(self.language_skill_set), shown=len(shown_actions))
            )
        
        user_text_parts.extend([
            "",
            OUTPUT_FORMAT_SECTION.format(max_action_id=len(self.language_skill_set) - 1)
        ])
        
        user_text = "\n".join(user_text_parts)
        
        self.messages.append({
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}},
                {"type": "text", "text": user_text}
            ]
        })
        
        return self.messages
    
    def _build_instruction_after_action(
        self, last_action_desc: str, last_success: int, progress: float
    ) -> List[str]:
        result = []
        progress_pct = progress * 100
        if last_success and progress < 1.0:
            action_lower = last_action_desc.lower()
            if 'find' in action_lower:
                is_equipment = any(kw in action_lower for kw in EQUIPMENT_KEYWORDS)
                is_destination = any(kw in action_lower for kw in DESTINATION_KEYWORDS)
                if is_destination:
                    result.append(INSTRUCTION_AFTER_FIND_DESTINATION_HOLDING.format(progress=progress_pct) if self.is_holding_object else INSTRUCTION_AFTER_FIND_DESTINATION_NOT_HOLDING.format(progress=progress_pct))
                elif is_equipment:
                    next_action_hint = get_equipment_next_action_hint(action_lower)
                    result.append(INSTRUCTION_AFTER_FIND_EQUIPMENT.format(progress=progress_pct, next_action_hint=next_action_hint))
                else:
                    result.append(INSTRUCTION_AFTER_FIND_OBJECT.format(progress=progress_pct))
            elif 'pick' in action_lower or 'pickup' in action_lower:
                next_step_hint = get_pickup_next_step_hint(self.episode_instruction)
                result.append(INSTRUCTION_AFTER_PICKUP.format(progress=progress_pct, next_step_hint=next_step_hint))
            elif 'go to' in action_lower or 'navigate' in action_lower:
                result.append(INSTRUCTION_AFTER_NAVIGATE.format(progress=progress_pct))
            elif any(kw in action_lower for kw in ['clean', 'heat', 'cool', 'slice', 'toggle', 'open', 'close', 'turn']):
                if 'turn on' in action_lower or 'toggle on' in action_lower:
                    result.append(INSTRUCTION_AFTER_TURN_ON.format(progress=progress_pct))
                elif 'turn off' in action_lower or 'toggle off' in action_lower or 'close' in action_lower:
                    result.append(INSTRUCTION_AFTER_TURN_OFF.format(progress=progress_pct))
                else:
                    result.append(INSTRUCTION_AFTER_INTERACTION_DEFAULT.format(progress=progress_pct))
            else:
                result.append(INSTRUCTION_SUCCESS_GENERIC.format(progress=progress_pct))
        if result: result.append("")
        return result
    
    def _build_action_list(self, user_text_parts: List[str]) -> List[int]:
        """
        构建动作列表，智能选择优先展示的动作
        """
        shown_actions = []
        
        if self.last_step_info and self.last_step_info.get('last_action_success', 0):
            last_action = self.last_step_info.get('action_description', '')
            priority_keywords = get_priority_keywords_after_action(last_action, self.episode_instruction, self.is_holding_object)
        else:
            priority_keywords = DEFAULT_PRIORITY_KEYWORDS

        instruction_lower = self.episode_instruction.lower()
        forced_keywords = []
        if ('wash' in instruction_lower or 'clean' in instruction_lower) and self.is_holding_object:
            forced_keywords.append('clean the')
        if 'heat' in instruction_lower and self.is_holding_object:
            forced_keywords.append('heat up')
        if 'cool' in instruction_lower and self.is_holding_object:
            forced_keywords.append('cool down')
        
        priority_keywords = list(dict.fromkeys(forced_keywords + priority_keywords))

        for keyword in priority_keywords:
            for i, action in enumerate(self.language_skill_set):
                if keyword in action.lower() and i not in shown_actions:
                    shown_actions.append(i)
                    user_text_parts.append(f"  - action_id {i}: {action}")
                    if len(shown_actions) >= 60: break
            if len(shown_actions) >= 60: break
        
        if len(shown_actions) < 80:
            for keyword in OTHER_ACTION_KEYWORDS:
                for i, action in enumerate(self.language_skill_set):
                    if keyword in action.lower() and i not in shown_actions:
                        shown_actions.append(i)
                        user_text_parts.append(f"  - action_id {i}: {action}")
                        if len(shown_actions) >= 80: break
                if len(shown_actions) >= 80: break
        
        return shown_actions
    
    def render(self) -> RenderOutput:
        from PIL import ImageDraw, ImageFont
        current_frame = self._get_current_image()
        img = Image.fromarray(current_frame)
        status_info = {
            "episode_num": self.current_episode_num, "task_type": self.current_task_type,
            "eval_set": self.eval_set, "action_id": self.last_step_info.get('action_id', None),
            "action_description": self.last_step_info.get('action_description', 'N/A'),
            "action_success": bool(self.last_step_info.get('last_action_success', 0)),
            "env_feedback": self.last_step_info.get('env_feedback', ''),
            "task_progress": self.last_step_info.get('task_progress', 0.0),
            "task_success": self.last_step_info.get('task_success', 0.0),
            "reasoning": self.last_step_info.get('reasoning', '')
        }
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        except IOError:
            font = ImageFont.load_default()
        action_desc = status_info['action_description']; success_icon = '✓' if status_info['action_success'] else '✗'; progress = status_info['task_progress'] * 100
        if len(action_desc) > 35: action_desc = action_desc[:32] + '...'
        text_lines = [f"Ep{self.current_episode_num} [{self.eval_set}]", f"Task: {self.current_task_type}", f"Step: {self.current_step}", f"Action: {action_desc}", f"Success: {success_icon}", f"Progress: {progress:.1f}%"]
        line_height=18; text_height=len(text_lines)*line_height+10; text_width=300; x_offset,y_offset=img.width-text_width-10,10
        background = Image.new('RGBA', img.size, (255,255,255,0)); bg_draw=ImageDraw.Draw(background)
        bg_draw.rectangle([x_offset-5,y_offset-5,x_offset+text_width,y_offset+text_height],fill=(0,0,0,180))
        img=Image.alpha_composite(img.convert('RGBA'),background).convert('RGB'); draw=ImageDraw.Draw(img)
        for i,line in enumerate(text_lines):
            y_pos=y_offset+i*line_height; draw.text((x_offset+1,y_pos+1),line,fill=(0,0,0),font=font); draw.text((x_offset,y_pos),line,fill=(255,255,255),font=font)
        buffer = io.BytesIO(); img.save(buffer, format='PNG'); image_data = buffer.getvalue(); buffer.close()
        return RenderOutput(image_data=image_data, step=self.current_step, text_dict=status_info)
    
    def close(self) -> None:
        super().close()
        if hasattr(self, 'alfred_env') and self.alfred_env is not None:
            self.alfred_env.close()
        print("✓ EmbodiedAlfredGym 已关闭")
    
    def parse_llm_response(self, response_text: str) -> Tuple[int, str, bool]:
        response_text=str(response_text).strip(); json_str=None
        json_match=re.search(r'```(?:json)?\s*(\{.*?\})\s*```',response_text,re.DOTALL)
        if json_match: json_str=json_match.group(1)
        if not json_str:
            brace_level,start_index=0,-1
            for i,char in enumerate(response_text):
                if char=='{':
                    if start_index==-1:start_index=i
                    brace_level+=1
                elif char=='}':
                    if start_index!=-1:
                        brace_level-=1
                        if brace_level==0:json_str=response_text[start_index:i+1];break
            if not json_str and start_index!=-1:json_str=response_text[start_index:]
        if not json_str:json_str=response_text
        try:
            data=json.loads(json_str); reasoning=data.get('reasoning','No reasoning provided')
            if isinstance(reasoning,str):reasoning=reasoning[:200]
            executable_plan=data.get('executable_plan',[])
            if not isinstance(executable_plan,list):return -1,"executable_plan must be a list",False
            if len(executable_plan)==0:return -2,reasoning,True
            first_action=executable_plan[0]
            if not isinstance(first_action,dict):return -1,"Invalid action format",False
            action_id=first_action.get('action_id',-1)
            if not isinstance(action_id,int) or not(0<=action_id<len(self.language_skill_set)):return -1,f"Invalid action_id: {action_id}",False
            return action_id,reasoning,True
        except(json.JSONDecodeError,Exception)as e:
            action_match=re.search(r'"action_id"\s*:\s*(\d+)',response_text)
            if action_match:
                try:
                    action_id=int(action_match.group(1))
                    if 0<=action_id<len(self.language_skill_set):return action_id,f"Fallback parsing (Error: {type(e).__name__})",True
                except:pass
            return -1,f"Parse error: {str(e)[:100]}",False
    
    def _numpy_to_base64(self,image_array:np.ndarray)->str:
        img=Image.fromarray(image_array.astype('uint8'),'RGB');buffer=io.BytesIO()
        img.save(buffer,format='PNG');img_base64=base64.b64encode(buffer.getvalue()).decode('utf-8');buffer.close()
        return img_base64
    
    def _get_current_image(self)->np.ndarray:
        return self.alfred_env.env.last_event.frame