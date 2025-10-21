import gymnasium as gym
from gymnasium.spaces import Space
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple

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
    
    可选实现方法：
    - render(): 渲染环境状态
    - close(): 释放环境资源
    """
    
    @abstractmethod
    def reset(self, seed = None):
        super().reset(seed=seed)
    
    @abstractmethod
    def step(self, action):
        pass
    
    @abstractmethod
    def get_task_prompt(self) -> str:
        """生成任务提示信息，基于当前观测和动作"""
        pass
    
    def render(self):
        """
        可选：渲染环境（如可视化界面）
        默认对奖励进行可视化
        """
        super().render()
    
    def close(self):
        """可选：关闭环境并释放资源"""
        super().close()