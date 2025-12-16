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

# 动态导入 EmbodiedBench 模块
# try:
#     # 添加 EmbodiedBench 路径到 sys.path
#     embodied_bench_path = "/workspace/AIEvoBox/env/embodiedgym/EmbodiedBench-master"
#     if embodied_bench_path not in sys.path:
#         sys.path.insert(0, embodied_bench_path)
    
#     from embodiedbench.envs.eb_alfred.EBAlfEnv import EBAlfEnv
# except ImportError as e:
#     print(f"警告: 无法导入 EmbodiedBench Alfred 环境: {e}")
#     print("请确保 EmbodiedBench 已正确安装且路径正确")
#     EBAlfEnv = None

try:
    # --- 路径动态适配修改开始 ---
    # 1. 首先尝试基于当前运行目录 (CWD) 查找，适用于在 AIEvoBox 根目录运行的情况
    project_root = os.getcwd()
    embodied_bench_path = os.path.join(project_root, "env", "embodiedgym", "EmbodiedBench-master")
    
    # 2. 如果路径不存在（可能是在子目录运行），则基于当前文件位置向上查找项目根目录
    if not os.path.exists(embodied_bench_path):
        current_file_path = os.path.abspath(__file__)
        d = os.path.dirname(current_file_path)
        # 向上递归查找包含 env/embodiedgym 的目录
        while d != "/" and d != os.path.dirname(d):
            if os.path.exists(os.path.join(d, "env", "embodiedgym")):
                embodied_bench_path = os.path.join(d, "env", "embodiedgym", "EmbodiedBench-master")
                break
            d = os.path.dirname(d)
    # --- 路径动态适配修改结束 ---

    # 添加 EmbodiedBench 路径到 sys.path
    if embodied_bench_path not in sys.path:
        sys.path.insert(0, embodied_bench_path)
    
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
        self.current_episode_num = 0  # 当前 episode 编号
        self.current_task_type = ""   # 当前任务类型
        self.last_step_info = {}  # 保存最后一步的完整信息
        self.is_holding_object = False  # 追踪robot是否正在holding object
        
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
        self.current_episode_num = self.alfred_env._current_episode_num - 1
        self.is_holding_object = False  # 重置holding状态
        
        # 获取当前任务类型（从已加载的episode数据中提取）
        try:
            current_episode = self.alfred_env.current_episode()
            self.current_task_type = current_episode.get('task_type', 'unknown')
        except Exception as e:
            print(f"⚠ 无法加载任务类型: {e}")
            self.current_task_type = 'unknown'
        
        # 清空上一步信息
        self.last_step_info = {}
        
        # 构建观测字典
        observation = {
            'head_rgb': obs['head_rgb'],  # numpy array
            'instruction': self.episode_instruction,
            'available_actions': self.language_skill_set
        }
        
        # 构建信息字典
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
        
        # 🚫 已移除强制纠正逻辑 - 发现这破坏了正确的任务流程
        # Alfred环境中没有单独的"clean"动作,"find Faucet"在holding状态下是正确的!
        
        # 处理解析失败的情况
        if not is_valid:
            print(f"⚠ 步骤 {self.current_step}: LLM 输出解析失败")
            print(f"  原始输出: {action[:500]}...")  # 打印前500字符
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
            
            # 保存信息（供 render 使用）
            self.last_step_info = {
                'action_id': None,
                'action_description': 'Invalid Output',
                'last_action_success': 0,
                'env_feedback': info['env_feedback'],
                'task_progress': 0.0,
                'task_success': 0.0,
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
            action_str = self.language_skill_set[action_id] if isinstance(action_id, int) and action_id < len(self.language_skill_set) else str(action_id)
            info = {
                'task_success': 0.0,
                'task_progress': 0.0,
                'last_action_success': 0.0,
                'env_feedback': f'Environment error: {str(e)}',
                'instruction': self.episode_instruction,
                'error': True
            }
            
            # 保存信息（供 render 使用）
            self.last_step_info = {
                'action_id': action_id,
                'action_description': action_str,
                'last_action_success': 0,
                'env_feedback': info['env_feedback'],
                'task_progress': 0.0,
                'task_success': 0.0,
                'reasoning': reasoning
            }
        
        # 打印步骤信息
        action_str = self.language_skill_set[action_id] if isinstance(action_id, int) and action_id < len(self.language_skill_set) else str(action_id)
        success_icon = "✓" if info.get('last_action_success', 0) else "✗"
        print(f"{success_icon} 步骤 {self.current_step}: {action_str}")
        print(f"  奖励: {reward:.3f} | 进度: {info.get('task_progress', 0):.2%} | 动作成功: {bool(info.get('last_action_success', 0))} | 任务完成: {bool(info.get('task_success', 0))}")
        
        # 如果有环境反馈，打印出来
        env_feedback = info.get('env_feedback', '')
        if env_feedback and len(env_feedback) < 200:
            print(f"  反馈: {env_feedback}")
        
        # 打印推理（如果有）
        if reasoning and self.current_step <= 3:  # 只打印前3步的推理
            print(f"  推理: {reasoning[:150]}{'...' if len(reasoning) > 150 else ''}")
        
        # 保存本步骤的完整信息（供 render 使用）
        self.last_step_info = {
            'action_id': action_id,
            'action_description': action_str,
            'last_action_success': info.get('last_action_success', 0),
            'env_feedback': info.get('env_feedback', ''),
            'task_progress': info.get('task_progress', 0.0),
            'task_success': info.get('task_success', 0.0),
            'reasoning': reasoning
        }
        
        # 更新holding状态
        if info.get('last_action_success', 0):
            action_lower = action_str.lower()
            if 'pick' in action_lower and 'up' in action_lower:
                self.is_holding_object = True
            elif 'put' in action_lower and 'down' in action_lower:
                self.is_holding_object = False
        # 也可以从feedback检测
        env_feedback = info.get('env_feedback', '')
        if 'currently holding' in env_feedback.lower():
            self.is_holding_object = True
        
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
        system_content = TextContent(text=SYSTEM_PROMPT)
        system_message = OpenAIMessage(
            role="system",
            content=[MessageContent(root=system_content)]
        )
        
        # 构建 user 消息
        user_text_parts = [
            TASK_HEADER,
            self.episode_instruction,
            ""
        ]
        
        # 添加执行历史（如果有）
        if hasattr(self, 'last_step_info') and self.last_step_info and self.current_step > 0:
            last_action_desc = self.last_step_info.get('action_description', 'N/A')
            last_success = self.last_step_info.get('last_action_success', 0)
            progress = self.last_step_info.get('task_progress', 0)
            
            # 添加上一步动作信息
            success_status = SUCCESS_STATUS_YES if last_success else SUCCESS_STATUS_NO
            user_text_parts.extend([
                PREVIOUS_ACTION_HEADER.format(step=self.current_step),
                PREVIOUS_ACTION_TEMPLATE.format(
                    action=last_action_desc,
                    success_status=success_status,
                    progress=progress * 100,
                    feedback=self.last_step_info.get('env_feedback', 'No feedback')[:150]
                ),
                ""
            ])
            
            # 根据动作成功和进度给出明确且强制性的指示
            user_text_parts.extend(
                self._build_instruction_after_action(last_action_desc, last_success, progress)
            )
        
        # 添加当前观测部分
        user_text_parts.extend([
            OBSERVATION_HEADER,
            OBSERVATION_PROMPT.format(step=self.current_step + 1),
            "",
            ACTIONS_HEADER.format(count=len(self.language_skill_set)),
            ACTIONS_SUBHEADER
        ])
        
        # 添加关键动作（智能选择 - 优先展示与上一步相关的动作）
        shown_actions = self._build_action_list(user_text_parts)
        
        if len(shown_actions) < len(self.language_skill_set):
            user_text_parts.append(
                ACTIONS_TRUNCATED.format(total=len(self.language_skill_set), shown=len(shown_actions))
            )
        
        # 构建 OUTPUT FORMAT 规则
        user_text_parts.extend([
            "",
            OUTPUT_FORMAT_SECTION.format(max_action_id=len(self.language_skill_set) - 1)
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
    
    def _build_instruction_after_action(
        self, 
        last_action_desc: str, 
        last_success: int, 
        progress: float
    ) -> List[str]:
        """
        根据上一步动作构建指导性提示
        
        Args:
            last_action_desc: 上一步动作描述
            last_success: 是否成功
            progress: 任务进度
            
        Returns:
            提示文本列表
        """
        result = []
        progress_pct = progress * 100
        
        if last_success and progress < 1.0:
            action_lower = last_action_desc.lower()
            
            if 'find' in action_lower:
                is_equipment = any(kw in action_lower for kw in EQUIPMENT_KEYWORDS)
                is_destination = any(kw in action_lower for kw in DESTINATION_KEYWORDS)
                
                if is_destination:
                    if self.is_holding_object:
                        result.append(INSTRUCTION_AFTER_FIND_DESTINATION_HOLDING.format(progress=progress_pct))
                    else:
                        result.append(INSTRUCTION_AFTER_FIND_DESTINATION_NOT_HOLDING.format(progress=progress_pct))
                elif is_equipment:
                    next_action_hint = get_equipment_next_action_hint(action_lower)
                    result.append(INSTRUCTION_AFTER_FIND_EQUIPMENT.format(
                        progress=progress_pct, 
                        next_action_hint=next_action_hint
                    ))
                else:
                    result.append(INSTRUCTION_AFTER_FIND_OBJECT.format(progress=progress_pct))
            
            elif 'pick' in action_lower or 'pickup' in action_lower:
                next_step_hint = get_pickup_next_step_hint(self.episode_instruction)
                result.append(INSTRUCTION_AFTER_PICKUP.format(
                    progress=progress_pct, 
                    next_step_hint=next_step_hint
                ))
            
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
        
        elif not last_success:
            env_feedback = self.last_step_info.get('env_feedback', '')
            result.append(INSTRUCTION_FAILURE.format(feedback=env_feedback[:200]))
        
        if result:
            result.append("")
        
        return result
    
    def _build_action_list(self, user_text_parts: List[str]) -> List[int]:
        """
        构建动作列表，智能选择优先展示的动作
        
        Args:
            user_text_parts: 用于追加动作文本的列表
            
        Returns:
            已展示的动作索引列表
        """
        shown_actions = []
        
        # 根据上一步动作,获取优先展示的动作关键词
        if self.last_step_info and self.last_step_info.get('last_action_success', 0):
            last_action = self.last_step_info.get('action_description', '')
            priority_keywords = get_priority_keywords_after_action(
                last_action, 
                self.episode_instruction, 
                self.is_holding_object
            )
        else:
            priority_keywords = DEFAULT_PRIORITY_KEYWORDS
        
        # 第1轮: 添加优先级高的动作
        for keyword in priority_keywords:
            for i, action in enumerate(self.language_skill_set):
                if keyword in action.lower() and i not in shown_actions:
                    shown_actions.append(i)
                    user_text_parts.append(f"  - action_id {i}: {action}")
                    if len(shown_actions) >= 60:
                        break
            if len(shown_actions) >= 60:
                break
        
        # 第2轮: 补充其他常用动作
        if len(shown_actions) < 80:
            for keyword in OTHER_ACTION_KEYWORDS:
                for i, action in enumerate(self.language_skill_set):
                    if keyword in action.lower() and i not in shown_actions:
                        shown_actions.append(i)
                        user_text_parts.append(f"  - action_id {i}: {action}")
                        if len(shown_actions) >= 80:
                            break
                if len(shown_actions) >= 80:
                    break
        
        return shown_actions
    
    def render(self) -> RenderOutput:
        """
        渲染环境状态，返回图像和状态信息
        
        Returns:
            RenderOutput: 包含图像数据、步骤信息和动作状态
        """
        from PIL import ImageDraw, ImageFont
        
        # 获取当前图像
        current_frame = self._get_current_image()
        img = Image.fromarray(current_frame)
        
        # 直接使用保存的最后一步信息
        status_info = {
            "episode_num": self.current_episode_num,
            "task_type": self.current_task_type,
            "eval_set": self.eval_set,
            "action_id": self.last_step_info.get('action_id', None),
            "action_description": self.last_step_info.get('action_description', 'N/A'),
            "action_success": bool(self.last_step_info.get('last_action_success', 0)),
            "env_feedback": self.last_step_info.get('env_feedback', ''),
            "task_progress": self.last_step_info.get('task_progress', 0.0),
            "task_success": self.last_step_info.get('task_success', 0.0),
            "reasoning": self.last_step_info.get('reasoning', '')
        }
        
        # 在图片右上角添加文本标注
        draw = ImageDraw.Draw(img)
        
        # 尝试加载字体，如果失败则使用默认字体
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        except:
            font = ImageFont.load_default()
        
        # 准备显示的文本
        action_desc = status_info['action_description']
        if len(action_desc) > 35:
            action_desc = action_desc[:32] + '...'
        
        success_icon = '✓' if status_info['action_success'] else '✗'
        progress = status_info['task_progress'] * 100
        
        # 构建显示文本（每行一个信息）
        text_lines = [
            f"Ep{self.current_episode_num} [{self.eval_set}]",
            f"Task: {self.current_task_type}",
            f"Step: {self.current_step}",
            f"Action: {action_desc}",
            f"Success: {success_icon}",
            f"Progress: {progress:.1f}%"
        ]
        
        # 计算文本框大小
        line_height = 18
        text_height = len(text_lines) * line_height + 10
        text_width = 300
        
        # 右上角位置
        x_offset = img.width - text_width - 10
        y_offset = 10
        
        # 绘制半透明背景
        background = Image.new('RGBA', img.size, (255, 255, 255, 0))
        bg_draw = ImageDraw.Draw(background)
        bg_draw.rectangle(
            [x_offset - 5, y_offset - 5, x_offset + text_width, y_offset + text_height],
            fill=(0, 0, 0, 180)  # 黑色半透明背景
        )
        img = img.convert('RGBA')
        img = Image.alpha_composite(img, background)
        img = img.convert('RGB')
        
        # 重新创建 draw 对象
        draw = ImageDraw.Draw(img)
        
        # 绘制文本（每行）
        for i, line in enumerate(text_lines):
            y_pos = y_offset + i * line_height
            # 绘制文本阴影（增强可读性）
            draw.text((x_offset + 1, y_pos + 1), line, fill=(0, 0, 0), font=font)
            # 绘制实际文本
            draw.text((x_offset, y_pos), line, fill=(255, 255, 255), font=font)
        
        # 转换为字节流
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        buffer.seek(0)
        image_data = buffer.read()
        buffer.close()
        
        return RenderOutput(
            image_data=image_data,
            step=self.current_step,
            text_dict=status_info  # 添加结构化状态信息
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
        
        # 多种方式提取 JSON
        json_str = None
        
        # 方法1: 提取 ```json 代码块
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        
        # 方法2: 提取第一个完整的 JSON 对象 
        if not json_str:
            json_match = re.search(r'\{[^{}]*"executable_plan"[^{}]*\[[^\]]*\][^{}]*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
        
        # 方法3: 直接尝试整个响应
        if not json_str:
            json_str = response_text
        
        try:
            # 解析 JSON
            data = json.loads(json_str)
            
            # 提取 reasoning
            reasoning = data.get('reasoning', 'No reasoning provided')
            if isinstance(reasoning, str):
                reasoning = reasoning[:200]  # 限制长度
            
            # 提取 executable_plan
            executable_plan = data.get('executable_plan', [])
            
            if not executable_plan or not isinstance(executable_plan, list):
                return -1, f"No valid executable_plan found", False
            
            # 获取第一个动作的 action_id
            first_action = executable_plan[0]
            if not isinstance(first_action, dict):
                return -1, f"Invalid action format", False
                
            action_id = first_action.get('action_id', -1)
            
            # 验证 action_id 是否有效
            if not isinstance(action_id, int) or action_id < 0 or action_id >= len(self.language_skill_set):
                return -1, f"Invalid action_id: {action_id}", False
            
            return action_id, reasoning, True
            
        except json.JSONDecodeError as e:
            # JSON 解析失败 - 尝试提取任何数字作为 action_id
            action_match = re.search(r'"action_id"\s*:\s*(\d+)', response_text)
            if action_match:
                try:
                    action_id = int(action_match.group(1))
                    if 0 <= action_id < len(self.language_skill_set):
                        return action_id, f"Fallback parsing", True
                except:
                    pass
            return -1, f"JSON parse error: {str(e)[:100]}", False
        except Exception as e:
            # 其他错误
            return -1, f"Parse error: {str(e)[:100]}", False
    
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

