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

from entry import MinecraftSim


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

    def __init__(self, config=None, config_path=None, output_dir=None,
                 display_port=None, working_dir=None, mc_root=None, xvfb=False, data_path=None):
        """
        初始化 GPU Simulator

        Args:
            config: 配置字典（优先于 config_path）
            output_dir: 输出目录（视频、日志等）
            display_port: DISPLAY 端口号（用于资源隔离）
            env_port: Minecraft 环境端口号（用于资源隔离）
            working_dir: 工作目录（用于资源隔离）
        """
        
        # 配置来源：优先使用 config，其次 config_path/data_path
        self.data = config
        load_path = config_path or data_path
        if self.data is None and load_path:
            path_obj = Path(load_path)
            try:
                with open(load_path, "r", encoding="utf-8") as f:
                    if path_obj.suffix.lower() in {".yml", ".yaml"}:
                        self.data = yaml.safe_load(f)
                    else:
                        self.data = json.load(f)
            except FileNotFoundError:
                print(f"[MCSimulator] Warning: data file not found: {load_path}")
                self.data = {}
            except Exception as e:
                print(f"[MCSimulator] Warning: failed to load data file {load_path}: {e}")
                self.data = {}

        if isinstance(self.data, list):
            self.data = self.data[0] if self.data else {}

        if not isinstance(self.data, dict):
            raise ValueError("Simulator data must be a dict or list of dicts.")

        required_keys = ("target_type", "scene", "start_pos")
        missing = [k for k in required_keys if k not in self.data]
        if missing:
            raise KeyError(f"Missing keys in simulator data: {missing}")

        # 资源管理参数
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
        
        
        # 初始化 MinecraftSim（GPU 版本）
        self.simulator = MinecraftSim(
            task=self.data['target_type'],
            scene=self.data['scene'],
            working_dir=self.mc_root,
            output_dir=self.output_dir,
            display_port=self.display_port,
            xvfb=xvfb
        )

    def reset(self):
        reward_fn = {}
        command = []
        start_pos = self.data['start_pos']
        if 'start_rotation' in self.data.keys():
            start_rotation = self.data['start_rotation']
        else:
            start_rotation = [-59.8 , 5.8]
        tp_command = f"cmd tp @s {str(start_pos[0])} {str(start_pos[1])} {str(start_pos[2])} {str(start_rotation[0])} {str(start_rotation[1])}"
        command.append(tp_command)
        if 'goal_pos' in self.data.keys():
            reward_fn['type'] = 'navigation'
            reward_fn['gt_position'] = [self.data['goal_pos'][0], self.data['goal_pos'][1], self.data['goal_pos'][2]]
            reward_fn['gt_rotation'] = [0, 0]
        elif 'goal_obj_status' in self.data.keys():
            reward_fn['type'] = 'operation'
            reward_fn['gt_position'] = [self.data['goal_obj_pos'][0], self.data['goal_obj_pos'][1], self.data['goal_obj_pos'][2]]
            reward_fn['gt_obj_id'] = self.data["goal_obj_id"]
            reward_fn['gt_obj_status'] = self.data["goal_obj_status"]
            self.reset_command = f"cmd setblock {self.data['goal_obj_pos'][0]} {self.data['goal_obj_pos'][1]} {self.data['goal_obj_pos'][2]} {self.data['start_obj_status']}"
            self.operatable_list = self.data['interact_list']
            command.append(self.reset_command)
        else:
            reward_fn['type'] = 'None'
            reward_fn['gt_position'] = [0, 0, 0]
            reward_fn['gt_rotation'] = [0, 0]

        
        
        # 重置环境
        obs, info = self.simulator.reset(command=command, reward_fn=reward_fn)
        
        return obs, info
    
    def fast_reset(self, config):
        self.data = config
        reward_fn = {}
        command = []
        start_pos = self.data['start_pos']
        if 'start_rotation' in self.data.keys():
            start_rotation = self.data['start_rotation']
        else:
            start_rotation = [-59.8 , 5.8]
        tp_command = f"cmd tp @s {str(start_pos[0])} {str(start_pos[1])} {str(start_pos[2])} {str(start_rotation[0])} {str(start_rotation[1])}"
        command.append(tp_command)
        command.append(self.reset_command)
        if 'goal_pos' in self.data.keys():
            reward_fn['type'] = 'navigation'
            reward_fn['gt_position'] = [self.data['goal_pos'][0], self.data['goal_pos'][1], self.data['goal_pos'][2]]
            reward_fn['gt_rotation'] = [0, 0]
            reward_fn['task'] = self.data['target_type']
        elif 'goal_obj_status' in self.data.keys():
            reward_fn['type'] = 'operation'
            reward_fn['gt_position'] = [self.data['goal_obj_pos'][0], self.data['goal_obj_pos'][1], self.data['goal_obj_pos'][2]]
            reward_fn['gt_obj_id'] = self.data["goal_obj_id"]
            reward_fn['gt_obj_status'] = self.data["goal_obj_status"]
            self.reset_command = f"cmd setblock {self.data['goal_obj_pos'][0]} {self.data['goal_obj_pos'][1]} {self.data['goal_obj_pos'][2]} {self.data['start_obj_status']}"
            self.operatable_list = self.data['interact_list']
            command.append(self.reset_command)
        else:
            reward_fn['type'] = 'None'
            reward_fn['gt_position'] = [0, 0, 0]
            reward_fn['gt_rotation'] = [0, 0]
            reward_fn['task'] = self.data['target_type']
        
        # 重置环境
        obs, info = self.simulator.fast_reset(command=command, reward_fn=reward_fn)
        
        return obs, info

    def step(self, action):
        print(action)

        if 'wait' in action:
            action = {'init': 0, 'wait': 1, 'forward': 0, 'back': 0, 'jump': 0, 'look_left': 0, 'look_right': 0, 'look_up': 0, 'look_down': 0, 'a': 0, 'd': 0, 'sneak': 0, 'sprint': 0, 'inventory': 0, 'attack': 0, 'use': 0, 'operate': ""}
        elif 'walk_forward' in action:
            action = {'init': 0, 'wait': 0, 'forward': 1, 'back': 0, 'jump': 0, 'look_left': 0, 'look_right': 0, 'look_up': 0, 'look_down': 0, 'a': 0, 'd': 0, 'sneak': 0, 'sprint': 0, 'inventory': 0, 'attack': 0, 'use': 0, 'operate': ""}
        elif 'walk_backward' in action:
            action = {'init': 0, 'wait': 0, 'forward': 0, 'back': 1, 'jump': 0, 'look_left': 0, 'look_right': 0, 'look_up': 0, 'look_down': 0, 'a': 0, 'd': 0, 'sneak': 0, 'sprint': 0, 'inventory': 0, 'attack': 0, 'use': 0, 'operate': ""}
        elif 'jump' in action:
            action = {'init': 0, 'wait': 0, 'forward': 0, 'back': 0, 'jump': 1, 'look_left': 0, 'look_right': 0, 'look_up': 0, 'look_down': 0, 'a': 0, 'd': 0, 'sneak': 0, 'sprint': 0, 'inventory': 0, 'attack': 0, 'use': 0, 'operate': ""}
        elif 'turn_left' in action:
            action = {'init': 0, 'wait': 0, 'forward': 0, 'back': 0, 'jump': 0, 'look_left': 1, 'look_right': 0, 'look_up': 0, 'look_down': 0, 'a': 0, 'd': 0, 'sneak': 0, 'sprint': 0, 'inventory': 0, 'attack': 0, 'use': 0, 'operate': ""}
        elif 'turn_right' in action:
            action = {'init': 0, 'wait': 0, 'forward': 0, 'back': 0, 'jump': 0, 'look_left': 0, 'look_right': 1, 'look_up': 0, 'look_down': 0, 'a': 0, 'd': 0, 'sneak': 0, 'sprint': 0, 'inventory': 0, 'attack': 0, 'use': 0, 'operate': ""}
        elif 'look_left' in action:
            action = {'init': 0, 'wait': 0, 'forward': 0, 'back': 0, 'jump': 0, 'look_left': 1, 'look_right': 0, 'look_up': 0, 'look_down': 0, 'a': 0, 'd': 0, 'sneak': 0, 'sprint': 0, 'inventory': 0, 'attack': 0, 'use': 0, 'operate': ""}
        elif 'look_right' in action:
            action = {'init': 0, 'wait': 0, 'forward': 0, 'back': 0, 'jump': 0, 'look_left': 0, 'look_right': 1, 'look_up': 0, 'look_down': 0, 'a': 0, 'd': 0, 'sneak': 0, 'sprint': 0, 'inventory': 0, 'attack': 0, 'use': 0, 'operate': ""}
        elif 'look_up' in action:
            action = {'init': 0, 'wait': 0, 'forward': 0, 'back': 0, 'jump': 0, 'look_left': 0, 'look_right': 0, 'look_up': 1, 'look_down': 0, 'a': 0, 'd': 0, 'sneak': 0, 'sprint': 0, 'inventory': 0, 'attack': 0, 'use': 0, 'operate': ""}
        elif 'look_down' in action:
            action = {'init': 0, 'wait': 0, 'forward': 0, 'back': 0, 'jump': 0, 'look_left': 0, 'look_right': 0, 'look_up': 0, 'look_down': 1, 'a': 0, 'd': 0, 'sneak': 0, 'sprint': 0, 'inventory': 0, 'attack': 0, 'use': 0, 'operate': ""}
        elif 'sprint' in action:
            action = {'init': 0, 'wait': 0, 'forward': 0, 'back': 0, 'jump': 0, 'look_left': 0, 'look_right': 0, 'look_up': 0, 'look_down': 0, 'a': 0, 'd': 0, 'sneak': 0, 'sprint': 1, 'inventory': 0, 'attack': 0, 'use': 0, 'operate': ""}
        elif 'sneak' in action:
            action = {'init': 0, 'wait': 0, 'forward': 0, 'back': 0, 'jump': 0, 'look_left': 0, 'look_right': 0, 'look_up': 0, 'look_down': 0, 'a': 0, 'd': 0, 'sneak': 1, 'sprint': 0, 'inventory': 0, 'attack': 0, 'use': 0, 'operate': ""}
        elif 'attack' in action:
            action = {'init': 0, 'wait': 0, 'forward': 0, 'back': 0, 'jump': 0, 'look_left': 0, 'look_right': 0, 'look_up': 0, 'look_down': 0, 'a': 0, 'd': 0, 'sneak': 0, 'sprint': 0, 'inventory': 0, 'attack': 1, 'use': 0, 'operate': ""}
        elif 'look_up-left' in action:
            action = {'init': 0, 'wait': 0, 'forward': 0, 'back': 0, 'jump': 0, 'look_left': 1, 'look_right': 0, 'look_up': 1, 'look_down': 0, 'a': 0, 'd': 0, 'sneak': 0, 'sprint': 0, 'inventory': 0, 'attack': 0, 'use': 0, 'operate': ""}
        elif 'look_up-right' in action:
            action = {'init': 0, 'wait': 0, 'forward': 0, 'back': 0, 'jump': 0, 'look_left': 0, 'look_right': 1, 'look_up': 1, 'look_down': 0, 'a': 0, 'd': 0, 'sneak': 0, 'sprint': 0, 'inventory': 0, 'attack': 0, 'use': 0, 'operate': ""}
        elif 'look_down-left' in action:
            action = {'init': 0, 'wait': 0, 'forward': 0, 'back': 0, 'jump': 0, 'look_left': 1, 'look_right': 0, 'look_up': 0, 'look_down': 1, 'a': 0, 'd': 0, 'sneak': 0, 'sprint': 0, 'inventory': 0, 'attack': 0, 'use': 0, 'operate': ""}
        elif 'look_down-right' in action:
            action = {'init': 0, 'wait': 0, 'forward': 0, 'back': 0, 'jump': 0, 'look_left': 0, 'look_right': 1, 'look_up': 0, 'look_down': 1, 'a': 0, 'd': 0, 'sneak': 0, 'sprint': 0, 'inventory': 0, 'attack': 0, 'use': 0, 'operate': ""}
        elif 'move_left' in action:
            action = {'init': 0, 'wait': 0, 'forward': 0, 'back': 0, 'jump': 0, 'look_left': 0, 'look_right': 0, 'look_up': 0, 'look_down': 0, 'a': 1, 'd': 0, 'sneak': 0, 'sprint': 0, 'inventory': 0, 'attack': 0, 'use': 0, 'operate': ""}
        elif 'move_right' in action:
            action = {'init': 0, 'wait': 0, 'forward': 0, 'back': 0, 'jump': 0, 'look_left': 0, 'look_right': 0, 'look_up': 0, 'look_down': 0, 'a': 0, 'd': 1, 'sneak': 0, 'sprint': 0, 'inventory': 0, 'attack': 0, 'use': 0, 'operate': ""}
        elif 'use' in action:
            action = {'init': 0, 'wait': 0, 'forward': 0, 'back': 0, 'jump': 0, 'look_left': 0, 'look_right': 0, 'look_up': 0, 'look_down': 0, 'a': 0, 'd': 0, 'sneak': 0, 'sprint': 0, 'inventory': 0, 'attack': 0, 'use': 1, 'operate': ""}
        elif 'inventory' in action:
            action = {'init': 0, 'wait': 0, 'forward': 0, 'back': 0, 'jump': 0, 'look_left': 0, 'look_right': 0, 'look_up': 0, 'look_down': 0, 'a': 0, 'd': 0, 'sneak': 0, 'sprint': 0, 'inventory': 1, 'attack': 0, 'use': 0, 'operate': ""}
        elif 'operate' in action:
            flag = 0
            print(self.data)
            for i in range(len(self.operatable_list)):
                if self.operatable_list[i].lower() in action:
                    action = {'init': 0, 'wait': 0, 'forward': 0, 'back': 0, 'jump': 0, 'look_left': 0, 'look_right': 0, 'look_up': 0, 'look_down': 0, 'a': 0, 'd': 0, 'sneak': 0, 'sprint': 0, 'inventory': 0, 'attack': 0, 'use': 0, 'operate': f"aim {self.data['interact_id'][i]} 2 60 0.5"}
                    flag=1
            if flag == 0:
                action = {'init': 0, 'wait': 0, 'forward': 0, 'back': 0, 'jump': 0, 'look_left': 0, 'look_right': 0, 'look_up': 0, 'look_down': 0, 'a': 0, 'd': 0, 'sneak': 0, 'sprint': 0, 'inventory': 0, 'attack': 0, 'use': 0, 'operate': ""}
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
    
    def fast_close(self):
        """关闭环境"""
        if self.simulator:
            self.simulator.fast_close()


# 向后兼容：保留旧的测试代码结构
if __name__ == "__main__":
    # 切换到项目根目录
    import os
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    print(f"[INFO] 工作目录: {os.getcwd()}")

    # 简单测试
    config = {
        "id": 1,
        "scene": "001",
        "target_type": "Door",
        "target_id_str": "tmeo_ultra:shafazhuanjiao",
        "task_str": "find and open the door",
        "start_pos": [2306,101,984],
        "start_rotation": [-59.8 , 5.8],
        "goal_obj_pos": [2308, 101, 992],
        "goal_obj_id": "tmeo_ultra:woshimenjijian_2baisezuokai", 
        "start_obj_status": "tmeo_ultra:woshimenjijian_2baisezuo",
        "goal_obj_status": "tmeo_ultra:woshimenjijian_2baisezuokai",
        "interact_list": ["Door", "Button"],
        "interact_pos": [[2308, 101, 993], [2310, 102, 991.5]]
    }
    data_path = "/mnt/shared-storage-user/steai_share/luozhihao/mc_test/raycraft/datasets/data.json"
    working_dir = "/mnt/shared-storage-user/leishanzhe/env_tmp"
    mc_root = "/mnt/shared-storage-user/leishanzhe/repo/AIEvoBox/env/mc_gpu_gym/7/.minecraft"
    
    sim = MCSimulator(config=config, working_dir=working_dir, mc_root=mc_root)
    obs, info = sim.reset()

    # simulator = MCSimulator(
    #         config=config,
    #         display_port=resources.get("display_port"),
    #         working_dir=resources.get("working_dir"),
    #         mc_root=mc_root,
    #     )
    
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
