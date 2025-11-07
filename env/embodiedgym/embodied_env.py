"""
EmbodiedAlfredGym - Alfred Environment Adapter for AIEvoBox
完全模仿 trading_env.py 的结构，将 EmbodiedBench 的 Alfred 环境适配到 AIEvoBox 框架
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

from core.types.base import ResetOutput, StepOutput, RenderOutput, PromptOutput, TextContent, ImageContent, OpenAIMessage, MessageContent
from core.env.base_env import BaseEnv
from core.env.env_register import register_env

# 动态导入 EmbodiedBench 模块
try:
    # 添加 EmbodiedBench 路径到 sys.path
    embodied_bench_path = "/mnt/shared-storage-user/evobox-share/gaozhenkun/gzk/eval/EmbodiedBench-master"
    if embodied_bench_path not in sys.path:
        sys.path.insert(0, embodied_bench_path)
    
    from embodiedbench.envs.eb_alfred.EBAlfEnv import EBAlfEnv
except ImportError as e:
    print(f"警告: 无法导入 EmbodiedBench Alfred 环境: {e}")
    print("请确保 EmbodiedBench 已正确安装且路径正确")
    EBAlfEnv = None


@register_env("embodied_alfred")
class EmbodiedAlfredGym(BaseEnv):
    """
    EmbodiedBench Alfred 环境适配器
    
    将 EBAlfEnv 适配到 AIEvoBox 框架，完全模仿 TradingGym 的结构
    """
    
    def __init__(
        self,
        eval_set: str = 'base',
        down_sample_ratio: float = 1.0,
        resolution: int = 500,
        detection_box: bool = False,
        selected_indexes: Optional[List[int]] = None,
        max_episode_steps: int = 30,
        max_invalid_actions: int = 10,
        alfred_data_path: Optional[str] = None,
        exp_name: str = 'aievobox_alfred',
        ** kwargs
    ):
        """
        初始化 Alfred 环境适配器
        
        Args:
            eval_set: 评测集名称 ('base', 'long_horizon', 'spatial', 等)
            down_sample_ratio: 数据采样比例
            resolution: 图像分辨率
            detection_box: 是否显示检测框
            selected_indexes: 选定的 episode 索引列表
            max_episode_steps: 每个 episode 最大步数
            max_invalid_actions: 最大无效动作数
            alfred_data_path: Alfred 数据集路径（可选）
            exp_name: 实验名称
        """
        super().__init__(**kwargs)
        
        if EBAlfEnv is None:
            raise RuntimeError("EBAlfEnv 未成功导入，请检查 EmbodiedBench 安装")
        
        # 保存配置参数
        self.eval_set = eval_set
        self.resolution = resolution
        self.max_episode_steps = max_episode_steps
        self.max_invalid_actions = max_invalid_actions
        self.exp_name = exp_name
        
        # 初始化 EmbodiedBench Alfred 环境
        selected_indexes = selected_indexes or []
        self.alfred_env = EBAlfEnv(
            eval_set=eval_set,
            exp_name=exp_name,
            down_sample_ratio=down_sample_ratio,
            selected_indexes=selected_indexes,
            detection_box=detection_box,
            resolution=resolution
        )
        
        # 设置最大步数
        self.alfred_env._max_episode_steps = max_episode_steps
        self.alfred_env._max_invalid_actions = max_invalid_actions
        
        # 动作空间（从 alfred_env 获取）
        self.action_space = self.alfred_env.action_space
        self.language_skill_set = self.alfred_env.language_skill_set
        
        # 状态变量
        self.current_step = 0
        self.current_action = None
        self.current_reasoning = ""
        self.episode_instruction = ""
        
        print(f"✓ EmbodiedAlfredGym 初始化成功")
        print(f"  - 评测集: {eval_set}")
        print(f"  - Episode 总数: {self.alfred_env.number_of_episodes}")
        print(f"  - 动作空间大小: {len(self.language_skill_set)}")
        print(f"  - 图像分辨率: {resolution}x{resolution}")
    
    def reset(self, seed: Optional[int] = None) -> ResetOutput:
        """
        重置环境到初始状态
        
        Returns:
            ResetOutput: 包含初始观测和信息
        """
        # 调用 alfred_env.reset()
        obs = self.alfred_env.reset()
        
        # 重置状态变量
        self.current_step = 0
        self.current_action = None
        self.current_reasoning = ""
        self.episode_instruction = self.alfred_env.episode_language_instruction
        
        # 构建观测字典
        observation = {
            'head_rgb': obs['head_rgb'],  # numpy array
            'instruction': self.episode_instruction,
            'available_actions': self.language_skill_set
        }
        
        # 构建信息字典
        info = {
            'episode_num': self.alfred_env._current_episode_num - 1,
            'instruction': self.episode_instruction,
            'num_actions': len(self.language_skill_set)
        }
        
        print(f"\n{'='*80}")
        print(f"Episode {info['episode_num']} 开始")
        print(f"任务指令: {self.episode_instruction}")
        print(f"{'='*80}\n")
        
        return ResetOutput(observation=observation, info=info)
    
    def step(self, action: str) -> StepOutput:
        """
        执行一步环境交互
        
        Args:
            action: LLM 生成的字符串输出（JSON 格式）
            
        Returns:
            StepOutput: 包含新观测、奖励、终止状态等
        """
        super().step(action=action)
        self.current_step += 1
        
        # 解析 LLM 输出
        action_id, reasoning, is_valid = self.parse_llm_response(action)
        self.current_action = action_id
        self.current_reasoning = reasoning
        
        # 处理解析失败的情况
        if not is_valid:
            print(f"⚠ 步骤 {self.current_step}: LLM 输出解析失败")
            # 返回无效动作的反馈
            obs = {'head_rgb': self.alfred_env.env.last_event.frame}
            reward = -1.0
            done = self.alfred_env._cur_invalid_actions >= self.max_invalid_actions
            info = {
                'task_success': 0.0,
                'task_progress': 0.0,
                'last_action_success': 0.0,
                'env_feedback': 'Failed to parse LLM output. Expected JSON format with executable_plan.',
                'instruction': self.episode_instruction,
                'invalid_output': True,
                'reasoning': reasoning
            }
            
            return StepOutput(
                observation=obs,
                reward=reward,
                terminated=done,
                truncated=False,
                info=info
            )
        
        # 调用 alfred_env.step()
        try:
            obs, reward, done, info = self.alfred_env.step(action_id, reasoning=reasoning)
        except Exception as e:
            print(f"✗ 步骤 {self.current_step}: 环境执行出错 - {str(e)}")
            obs = {'head_rgb': self.alfred_env.env.last_event.frame}
            reward = -1.0
            done = True
            info = {
                'task_success': 0.0,
                'task_progress': 0.0,
                'last_action_success': 0.0,
                'env_feedback': f'Environment error: {str(e)}',
                'instruction': self.episode_instruction,
                'error': True
            }
        
        # 打印步骤信息
        action_str = self.language_skill_set[action_id] if isinstance(action_id, int) and action_id < len(self.language_skill_set) else str(action_id)
        success_icon = "✓" if info.get('last_action_success', 0) else "✗"
        print(f"{success_icon} 步骤 {self.current_step}: {action_str}")
        print(f"  奖励: {reward:.3f} | 任务进度: {info.get('task_progress', 0):.2%} | 成功: {info.get('task_success', 0)}")
        
        # 构建 StepOutput
        return StepOutput(
            observation=obs,
            reward=reward,
            terminated=done,
            truncated=False,
            info=info
        )
    
    def get_task_prompt(self) -> PromptOutput:
        """
        生成任务提示（包含图像的多模态 prompt）
        
        Returns:
            PromptOutput: 包含 system 和 user 消息
        """
        # 获取当前图像
        current_frame = self._get_current_image()
        img_base64 = self._numpy_to_base64(current_frame)
        
        # 构建 system 消息
        system_text = (
            "You are an embodied AI agent in a household environment. "
            "Your task is to complete the given instruction by selecting appropriate actions. "
            "You must output a JSON format with an executable plan containing action IDs."
        )
        system_content = TextContent(text=system_text)
        system_message = OpenAIMessage(
            role="system",
            content=[MessageContent(root=system_content)]
        )
        
        # 构建 user 消息
        user_text_parts = [
            f"## Task Instruction",
            f"{self.episode_instruction}",
            "",
            f"## Current Observation",
            f"Step {self.current_step}: Please analyze the current image and select the next action.",
            "",
            f"## Available Actions ({len(self.language_skill_set)} total)",
            "Here are all available action IDs and their descriptions:"
        ]
        
        # 添加动作列表（只显示前50个，避免太长）
        max_actions_to_show = 50
        for i in range(min(len(self.language_skill_set), max_actions_to_show)):
            user_text_parts.append(f"  - action_id {i}: {self.language_skill_set[i]}")
        
        if len(self.language_skill_set) > max_actions_to_show:
            user_text_parts.append(f"  ... and {len(self.language_skill_set) - max_actions_to_show} more actions")
        
        user_text_parts.extend([
            "",
            "## Output Format (CRITICAL)",
            "You MUST output valid JSON in the following format:",
            "```json",
            "{",
            '  "reasoning": "Your reasoning process here",',
            '  "executable_plan": [',
            '    {"action_id": 123, "description": "find a apple"},',
            '    {"action_id": 456, "description": "pick up the apple"}',
            "  ]",
            "}",
            "```",
            "",
            "IMPORTANT:",
            "- executable_plan must be a list with at least one action",
            "- Each action must have 'action_id' (integer from 0 to {})".format(len(self.language_skill_set) - 1),
            "- You can plan multiple steps, but start with 1-3 actions",
            "- Output ONLY the JSON, no extra text"
        ])
        
        user_text = "\n".join(user_text_parts)
        
        # 构建包含图像和文本的 user 消息
        user_content: List[MessageContent] = [
            MessageContent(root=ImageContent(
                image_url={"url": f"data:image/png;base64,{img_base64}"}
            )),
            MessageContent(root=TextContent(text=user_text))
        ]
        
        user_message = OpenAIMessage(
            role="user",
            content=user_content
        )
        
        return PromptOutput(
            system_message=system_message,
            user_message=user_message
        )
    
    def render(self) -> RenderOutput:
        """
        渲染环境状态，返回 base64 格式图片
        
        Returns:
            RenderOutput: 包含图像数据和步骤信息
        """
        # 获取当前图像
        current_frame = self._get_current_image()
        
        # 转换为 PIL Image（可选添加标注）
        img = Image.fromarray(current_frame)
        
        # 可以在这里添加文本标注（类似 trading_env 的日志）
        # 暂时直接返回原始图像
        
        # 转换为字节流
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        buffer.seek(0)
        image_data = buffer.read()
        buffer.close()
        
        return RenderOutput(
            image_data=image_data,
            step=self.current_step
        )
    
    def close(self) -> None:
        """关闭环境，释放资源"""
        super().close()
        if hasattr(self, 'alfred_env'):
            self.alfred_env.close()
        print("✓ EmbodiedAlfredGym 已关闭")
    
    def parse_llm_response(self, response_text: str) -> Tuple[int, str, bool]:
        """
        解析 LLM 输出，提取动作 ID 和推理过程
        
        Args:
            response_text: LLM 生成的字符串（期望是 JSON 格式）
            
        Returns:
            (action_id, reasoning, is_valid): 动作 ID、推理文本、是否有效
        """
        response_text = str(response_text).strip()
        
        # 尝试提取 JSON 代码块
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # 尝试直接解析整个响应
            json_str = response_text
        
        try:
            # 解析 JSON
            data = json.loads(json_str)
            
            # 提取 reasoning
            reasoning = data.get('reasoning', 'No reasoning provided')
            
            # 提取 executable_plan
            executable_plan = data.get('executable_plan', [])
            
            if not executable_plan or not isinstance(executable_plan, list):
                return -1, reasoning, False
            
            # 获取第一个动作的 action_id
            first_action = executable_plan[0]
            action_id = first_action.get('action_id', -1)
            
            # 验证 action_id 是否有效
            if not isinstance(action_id, int) or action_id < 0 or action_id >= len(self.language_skill_set):
                return -1, reasoning, False
            
            return action_id, reasoning, True
            
        except json.JSONDecodeError as e:
            # JSON 解析失败
            reasoning = f"JSON parse error: {str(e)}"
            return -1, reasoning, False
        except Exception as e:
            # 其他错误
            reasoning = f"Unexpected error: {str(e)}"
            return -1, reasoning, False
    
    def _numpy_to_base64(self, image_array: np.ndarray) -> str:
        """
        将 numpy 图像数组转换为 base64 字符串
        
        Args:
            image_array: RGB 图像数组 (H, W, 3)
            
        Returns:
            base64 编码的字符串
        """
        # 转换为 PIL Image
        img = Image.fromarray(image_array.astype('uint8'), 'RGB')
        
        # 保存到字节流
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        buffer.seek(0)
        
        # 编码为 base64
        img_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        buffer.close()
        
        return img_base64
    
    def _get_current_image(self) -> np.ndarray:
        """
        获取当前环境的图像帧
        
        Returns:
            RGB 图像数组 (H, W, 3)
        """
        return self.alfred_env.env.last_event.frame

