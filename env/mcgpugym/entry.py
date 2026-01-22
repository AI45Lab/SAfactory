'''
Date: 2024-11-11 05:20:17
LastEditors: caishaofei-mus1 1744260356@qq.com
LastEditTime: 2024-12-23 19:44:25
FilePath: /MineStudio/minestudio/simulator/entry.py
'''

import os
import cv2
import argparse
import numpy as np
import torch
import gymnasium
from gymnasium import spaces
from copy import deepcopy
from typing import Dict, List, Tuple, Union, Sequence, Mapping, Any, Optional, Literal
from dataclasses import asdict, dataclass, field, fields

from ._multiagent import MultiAgentEnv

BASIC_RESOLUTION = (640,360)



class MinecraftSim(gymnasium.Env):
    
    
    def __init__(
        self,  
        render_size: Tuple[int, int] = (640, 360),      # the original resolution of the game is 640x360
        inventory: Dict = {},                           # the initial inventory of the agent
        num_empty_frames: int = 20,                     # the number of empty frames to skip when calling reset
        working_dir: str = "None",
        output_dir: str = "None",
        display_port: int = None,
        **kwargs
    ) -> Any:
        super().__init__()
        self.render_size = render_size
        assert np.abs(render_size[0] / render_size[1] - 640 / 360) < 0.001
        self.num_empty_frames = num_empty_frames
        self.working_dir = working_dir
        
        self.env = MultiAgentEnv(self.working_dir, output_dir, display_port)

        self.already_reset = False
        
    
    def step(self, action: Dict[str, Any]) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
            
        obs, reward, done, info = self.env.step(action) 

        terminated, truncated = done, done
        obs, info = self._wrap_obs_info(obs, info)
        return obs, reward, terminated, truncated, info
    
    def command(self, command:str):
        self.env.command(command)

    def reset(self, command: str, reward_fn: Dict) -> Tuple[np.ndarray, Dict]:
        self.reward_fn = reward_fn
        self.env.reset(self.reward_fn)
        self.already_reset = True
        empty_action = {'wait': 1, 'forward': 0, 'back': 0, 'jump': 0, 'look_left': 0, 'look_right': 0, 'look_up': 0, 'look_down': 0, 'a': 0, 'd': 0, 'sneak': 0, 'sprint': 0, 'inventory': 0, 'attack': 0, 'use': 0}
        
        for _ in range(self.num_empty_frames): # skip the frames to avoid the initial black screen
            obs, reward, done, info = self.env.step(empty_action)
            
        for _ in range(8):
            obs, reward, done, info = self.env.step(empty_action)
            import time
            time.sleep(0.02)
            print("======================")
            print(command)
            # TODO: command -> List
            self.command(command)
        
        obs, reward, done, info = self.env.step(empty_action)
        
        obs, info = self._wrap_obs_info(obs, info)
        return obs, info

    def _wrap_obs_info(self, obs: str, info: Dict) -> Dict:
        _info = info.copy()
        _info.update(obs)
        if getattr(self, 'info', None) is None:
            self.info = {}
        for key, value in _info.items():
            self.info[key] = value
        _info = self.info.copy()
        return obs, _info
    

    def close(self) -> None:
        close_status = self.env.close()
        return close_status
    

# if __name__ == '__main__':
#     # test if the simulator works
#     from minestudio.simulator.callbacks import SpeedTestCallback
#     sim = MinecraftSim(
#         action_type="env", 
#         callbacks=[SpeedTestCallback(50)]
#     )
#     obs, info = sim.reset()
#     for i in range(100):
#         action = sim.action_space.sample()
#         obs, reward, terminated, truncated, info = sim.step(action)
#     sim.close()