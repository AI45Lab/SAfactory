"""
GPU 版本的 Minecraft Simulator

特性：
- GPU 硬件加速渲染（VirtualGL）
- 支持 LLM action string 格式
- 支持标准 Gym action 格式
- 与 CPU 版本接口兼容
"""

import json
import os
import re
import sys
import time
import uuid
from pathlib import Path
from datetime import datetime

import cv2
import yaml
import numpy as np
from PIL import Image

from .entry import MinecraftSim


class MCSimulator:
    """
    GPU 版本的 Minecraft Simulator

    支持两种 action 格式：
    1. LLM string: '<think>...</think><answer>[{"action": "forward"}]</answer>'
    2. Standard dict/array: {"forward": 1, ...} 或 np.array([...])
    """

    # 标准动作空间定义（与 deepeyes 版本保持一致）
    STANDARD_ACTIONS = [
        'walk_forward',
        'walk_backward',
        'move_left',
        'move_right',
        'sprint',
        'sneak',
        'jump',
        'use',
        'attack',
        'turn_right',
        'turn_left',
        'look_up',
        'look_down',
        'look_down-left',
        'look_up-right',
        'inventory',
    ]

    def __init__(self, config=None, data_path=None, config_path=None, output_dir=None,
                 display_port=None, working_dir=None, mc_root=None):
        """
        初始化 GPU Simulator

        Args:
            config: 配置字典（优先于 config_path）
            output_dir: 输出目录（视频、日志等）
            display_port: DISPLAY 端口号（用于资源隔离）
            env_port: Minecraft 环境端口号（用于资源隔离）
            working_dir: 工作目录（用于资源隔离）
        """
        
        # 配置来源：优先使用传入的 config；否则尝试从文件加载
        self.config_path = config_path
        self.data_path = data_path

        # 读取 YAML 配置（若提供），目前仅保留以备扩展
        self.config = {}
        if self.config_path:
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    self.config = yaml.safe_load(f) or {}
            except FileNotFoundError:
                print(f"[MCSimulator] Warning: config file not found: {self.config_path}")
            except Exception as e:
                print(f"[MCSimulator] Warning: failed to load config file {self.config_path}: {e}")

        # 场景数据：决定 start_pos / goal_pos
        if config is not None:
            self.data = config
        elif self.data_path:
            try:
                with open(self.data_path, "r", encoding="utf-8") as f:
                    self.data = json.load(f)
            except FileNotFoundError:
                print(f"[MCSimulator] Warning: data file not found: {self.data_path}")
                self.data = {}
            except Exception as e:
                print(f"[MCSimulator] Warning: failed to load data file {self.data_path}: {e}")
                self.data = {}
        else:
            self.data = {}

        # 资源管理参数
        if display_port is None:
            display_port = 1
        self.display_port = display_port
        self.working_dir = working_dir
        self.mc_root=mc_root


        # 设置资源隔离环境变量（如果提供）
        if self.display_port is not None:
            print(f"[MCSimulator] Set DISPLAY=:{self.display_port}")


        if self.working_dir is not None:
            # 确保工作目录存在
            Path(self.working_dir).mkdir(parents=True, exist_ok=True)
            # 创建 output 子目录（视频等会保存到这里）
            output_dir = Path(self.working_dir) / "output"
            output_dir.mkdir(parents=True, exist_ok=True)
            self.output_dir = str(output_dir)
            print(f"[MCSimulator] Using working_dir={self.working_dir}")
            print(f"[MCSimulator] Using output_dir={self.output_dir}")
        else:
            # 若未指定 working_dir，则沿用传入的 output_dir（可为空）
            self.output_dir = output_dir
        
        
        # 初始化 MinecraftSim（GPU 版本）
        self.simulator = MinecraftSim(
            working_dir=self.mc_root,
            output_dir=self.output_dir,
            display_port=self.display_port
        )

    def reset(self):
        if not self.data:
            raise ValueError("Simulator data is empty. Provide `config` or valid `data_path` with start_pos/goal_pos.")
        if 'start_pos' not in self.data:
            raise KeyError("`start_pos` missing in simulator data.")

        reward_fn = {}
        if 'goal_pos' in self.data.keys():
            reward_fn['type'] = 'navigation'
            reward_fn['gt_position'] = [self.data['goal_pos'][0], self.data['goal_pos'][1], self.data['goal_pos'][2]]
            reward_fn['gt_rotation'] = [0, 0]
        else:
            reward_fn['type'] = 'None'
            reward_fn['gt_position'] = [0, 0, 0]
            reward_fn['gt_rotation'] = [0, 0]

        

        start_pos = self.data['start_pos']
        tp_command = f"tp {str(start_pos[0])} {str(start_pos[1])} {str(start_pos[2])}"
        
        # 重置环境
        obs, info = self.simulator.reset(command=tp_command, reward_fn=reward_fn)
        
        return obs, info

    def step(self, action):
        print(action)

        if 'wait' in action:
            action = {'wait': 1, 'forward': 0, 'back': 0, 'jump': 0, 'look_left': 0, 'look_right': 0, 'look_up': 0, 'look_down': 0, 'a': 0, 'd': 0, 'sneak': 0, 'sprint': 0, 'inventory': 0, 'attack': 0, 'use': 0}
        elif 'walk_forward' in action:
            action = {'wait': 0, 'forward': 1, 'back': 0, 'jump': 0, 'look_left': 0, 'look_right': 0, 'look_up': 0, 'look_down': 0, 'a': 0, 'd': 0, 'sneak': 0, 'sprint': 0, 'inventory': 0, 'attack': 0, 'use': 0}
        elif 'walk_backward' in action:
            action = {'wait': 0, 'forward': 0, 'back': 1, 'jump': 0, 'look_left': 0, 'look_right': 0, 'look_up': 0, 'look_down': 0, 'a': 0, 'd': 0, 'sneak': 0, 'sprint': 0, 'inventory': 0, 'attack': 0, 'use': 0}
        elif 'jump' in action:
            action = {'wait': 0, 'forward': 0, 'back': 0, 'jump': 1, 'look_left': 0, 'look_right': 0, 'look_up': 0, 'look_down': 0, 'a': 0, 'd': 0, 'sneak': 0, 'sprint': 0, 'inventory': 0, 'attack': 0, 'use': 0}
        elif 'turn_left' in action:
            action = {'wait': 0, 'forward': 0, 'back': 0, 'jump': 0, 'look_left': 1, 'look_right': 0, 'look_up': 0, 'look_down': 0, 'a': 0, 'd': 0, 'sneak': 0, 'sprint': 0, 'inventory': 0, 'attack': 0, 'use': 0}
        elif 'turn_right' in action:
            action = {'wait': 0, 'forward': 0, 'back': 0, 'jump': 0, 'look_left': 0, 'look_right': 1, 'look_up': 0, 'look_down': 0, 'a': 0, 'd': 0, 'sneak': 0, 'sprint': 0, 'inventory': 0, 'attack': 0, 'use': 0}
        elif 'look_up' in action:
            action = {'wait': 0, 'forward': 0, 'back': 0, 'jump': 0, 'look_left': 0, 'look_right': 0, 'look_up': 1, 'look_down': 0, 'a': 0, 'd': 0, 'sneak': 0, 'sprint': 0, 'inventory': 0, 'attack': 0, 'use': 0}
        elif 'look_down' in action:
            action = {'wait': 0, 'forward': 0, 'back': 0, 'jump': 0, 'look_left': 0, 'look_right': 0, 'look_up': 0, 'look_down': 1, 'a': 0, 'd': 0, 'sneak': 0, 'sprint': 0, 'inventory': 0, 'attack': 0, 'use': 0}
        elif 'sprint' in action:
            action = {'wait': 0, 'forward': 0, 'back': 0, 'jump': 0, 'look_left': 0, 'look_right': 0, 'look_up': 0, 'look_down': 0, 'a': 0, 'd': 0, 'sneak': 0, 'sprint': 1, 'inventory': 0, 'attack': 0, 'use': 0}
        elif 'sneak' in action:
            action = {'wait': 0, 'forward': 0, 'back': 0, 'jump': 0, 'look_left': 0, 'look_right': 0, 'look_up': 0, 'look_down': 0, 'a': 0, 'd': 0, 'sneak': 1, 'sprint': 0, 'inventory': 0, 'attack': 0, 'use': 0}
        elif 'attack' in action:
            action = {'wait': 0, 'forward': 0, 'back': 0, 'jump': 0, 'look_left': 0, 'look_right': 0, 'look_up': 0, 'look_down': 0, 'a': 0, 'd': 0, 'sneak': 0, 'sprint': 0, 'inventory': 0, 'attack': 1, 'use': 0}
        elif 'look_up-left' in action:
            action = {'wait': 0, 'forward': 0, 'back': 0, 'jump': 0, 'look_left': 1, 'look_right': 0, 'look_up': 1, 'look_down': 0, 'a': 0, 'd': 0, 'sneak': 0, 'sprint': 0, 'inventory': 0, 'attack': 0, 'use': 0}
        elif 'look_up-right' in action:
            action = {'wait': 0, 'forward': 0, 'back': 0, 'jump': 0, 'look_left': 0, 'look_right': 1, 'look_up': 1, 'look_down': 0, 'a': 0, 'd': 0, 'sneak': 0, 'sprint': 0, 'inventory': 0, 'attack': 0, 'use': 0}
        elif 'look_down-left' in action:
            action = {'wait': 0, 'forward': 0, 'back': 0, 'jump': 0, 'look_left': 1, 'look_right': 0, 'look_up': 0, 'look_down': 1, 'a': 0, 'd': 0, 'sneak': 0, 'sprint': 0, 'inventory': 0, 'attack': 0, 'use': 0}
        elif 'look_down-right' in action:
            action = {'wait': 0, 'forward': 0, 'back': 0, 'jump': 0, 'look_left': 0, 'look_right': 1, 'look_up': 0, 'look_down': 1, 'a': 0, 'd': 0, 'sneak': 0, 'sprint': 0, 'inventory': 0, 'attack': 0, 'use': 0}
        elif 'move_left' in action:
            action = {'wait': 0, 'forward': 0, 'back': 0, 'jump': 0, 'look_left': 0, 'look_right': 0, 'look_up': 0, 'look_down': 0, 'a': 1, 'd': 0, 'sneak': 0, 'sprint': 0, 'inventory': 0, 'attack': 0, 'use': 0}
        elif 'move_right' in action:
            action = {'wait': 0, 'forward': 0, 'back': 0, 'jump': 0, 'look_left': 0, 'look_right': 0, 'look_up': 0, 'look_down': 0, 'a': 0, 'd': 1, 'sneak': 0, 'sprint': 0, 'inventory': 0, 'attack': 0, 'use': 0}
        elif 'use' in action:
            action = {'wait': 0, 'forward': 0, 'back': 0, 'jump': 0, 'look_left': 0, 'look_right': 0, 'look_up': 0, 'look_down': 0, 'a': 0, 'd': 0, 'sneak': 0, 'sprint': 0, 'inventory': 0, 'attack': 0, 'use': 1}
        elif 'inventory' in action:
            action = {'wait': 0, 'forward': 0, 'back': 0, 'jump': 0, 'look_left': 0, 'look_right': 0, 'look_up': 0, 'look_down': 0, 'a': 0, 'd': 0, 'sneak': 0, 'sprint': 0, 'inventory': 1, 'attack': 0, 'use': 0}
        else:
            print("action error")
        # 执行 step
        obs, reward, terminated, truncated, info = self.simulator.step(action)

        # 微小的 step reward
        reward += 0.001

        return obs, reward, terminated, truncated, info

    def close(self):
        """关闭环境"""
        if self.simulator:
            self.simulator.close()


# 向后兼容：保留旧的测试代码结构
if __name__ == "__main__":
    # 切换到项目根目录
    import os
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    print(f"[INFO] 工作目录: {os.getcwd()}")

    # 简单测试
    config_path = "/mnt/shared-storage-user/leishanzhe/repo/deepeyes/verl/workers/agent/envs/mc/config/kill/base.yaml"
    data_path = "/mnt/shared-storage-user/steai_share/luozhihao/mc_test/raycraft/datasets/data.json"
    # working_dir = "/mnt/shared-storage-user/steai_share/luozhihao/mc_test/raycraft/env_tmp"
    # working_dir ="/mnt/shared-storage-user/steai_share/leishanzhe/env/env/0/.minecraft"
    working_dir = "/mnt/shared-storage-user/leishanzhe/env_tmp"
    mc_root = "/mnt/shared-storage-user/steai_share/leishanzhe/env/env/0/.minecraft"

    sim = MCSimulator(config_path=config_path, data_path=data_path, working_dir=working_dir, mc_root=mc_root )
    obs, info = sim.reset()

    print(f"[TEST] Environment initialized")
    print(f"[TEST] Obs shape: {obs['image'].shape if 'image' in obs else 'N/A'}")
    
    # 创建输出目录
    output_dir = Path("./test_images")
    output_dir.mkdir(exist_ok=True)
    
    # 保存初始观察图像
    if 'image' in obs:
        # OpenCV 需要 BGR 格式，obs['image'] 是 RGB 格式
        img_bgr = cv2.cvtColor(obs['image'], cv2.COLOR_RGB2BGR)
        img_path = output_dir / "frame_000_reset.png"
        cv2.imwrite(str(img_path), img_bgr)
        print(f"[TEST] Saved initial frame to: {img_path}")

    # 使用类常量定义的标准动作空间
    test_actions_all = MCSimulator.STANDARD_ACTIONS

    # 测试标准动作空间
    print(f"[TEST] Testing {len(test_actions_all)} standard actions...")
    step_count = 0
    for action_idx, test_action in enumerate(test_actions_all):
        print(f"\n[TEST] Action {action_idx + 1}/{len(test_actions_all)}: {test_action[:50]}...")
        
        # 每个动作重复26次（与参考实现一致）
        for i in range(2):
            obs, reward, terminated, truncated, info = sim.step(test_action)
            step_count += 1
            
            if (step_count % 50 == 0) or reward != 0.001:
                print(f"[TEST] Step {step_count}: reward={reward:.3f}, done={terminated or truncated}")
            
            # 保存部分观察图像（避免生成太多文件）
            if 'image' in obs and (step_count % 50 == 0 or action_idx == 0 and i < 5):
                img_bgr = cv2.cvtColor(obs['image'], cv2.COLOR_RGB2BGR)
                img_path = output_dir / f"frame_{step_count:04d}_action{action_idx}_step{i}.png"
                cv2.imwrite(str(img_path), img_bgr)
                print(f"[TEST] Saved frame to: {img_path}")

            if terminated or truncated:
                print(f"[TEST] Episode finished at step {step_count}")
                break
        
        if terminated or truncated:
            break

    sim.close()
    print(f"\n[TEST] Test completed! Total steps: {step_count}")
    print(f"[TEST] Images saved to: {output_dir.absolute()}")
