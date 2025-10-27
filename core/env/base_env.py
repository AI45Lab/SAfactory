import gymnasium as gym
from gymnasium.spaces import Space
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional, List
from core.types.base import ResetOutput, StepOutput, RenderOutput, PromptOutput

class BaseEnv(gym.Env, ABC):
    """
    基础环境抽象类，继承自 gymnasium.Env
    
    子类必须实现的核心属性：
    - observation_space: 定义智能体可以从环境中获取信息的格式、范围和类型
    - action_space: 定义有效动作的范围和结构
    
    子类必须实现的核心方法：
    - reset(): 重置环境到初始状态
    - step(action): 执行动作并返回环境反馈
    - get_task_prompt(observation, action): 生成基于当前观测和动作的任务提示（LLM理解的Prompt）
    - render(): 渲染每一步环境的状态，返回一张图
    
    可选实现方法：
    - close(): 释放环境资源
    """
    
    @abstractmethod
    def reset(self, seed: Optional[int] = None) -> ResetOutput:
        """
        重置环境到初始状态
        :param seed: 随机种子
        :return: 包含初始观测和信息的ResetOutput
        """
        pass
    
    @abstractmethod
    def step(self, action: str) -> StepOutput:
        """
        执行动作并返回环境反馈
        :param action: 智能体的回复
        :return: 包含新观测、奖励、终止状态、环境信息等的StepOutput对象
        """
        pass
    
    @abstractmethod
    def get_task_prompt(self) -> PromptOutput:
        """
        生成任务提示信息
        :return: 自然语言提示字符串
        """
        pass
    
    @abstractmethod
    def render(self) -> RenderOutput:
        """
        渲染环境状态
        :return: 包含图像路径和步骤的RenderOutput对象
        """
        pass
    
    def close(self) -> None:
        """可选：关闭环境并释放资源"""
        pass