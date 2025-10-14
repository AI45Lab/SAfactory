import requests
from typing import Dict, Any, Optional
from .base_env_client import BaseEnvClient, StepOutput

class TradingEnvClient(BaseEnvClient):
    """交易环境客户端，支持多环境管理"""
    
    def __init__(self, env_server_base: str, timeout: int = 300):
        super().__init__(env_server_base, timeout)
        # 初始化时不自动创建环境，让用户显式创建
        
    def create_env(self, env_name: str, data_idx: int = 0,** kwargs) -> int:
        """
        创建新的交易环境
        
        Args:
            env_name: 环境名称（用于标识和管理）
            data_idx: 数据索引，用于选择不同数据集
            
        Returns:
            环境ID
        """
        if env_name in self.env_ids:
            raise ValueError(f"环境名称 {env_name} 已存在")
            
        try:
            # 创建环境
            response = requests.post(
                f"{self.env_server_base}/create",
                json={},  # 符合服务端模型要求的空JSON对象
                timeout=self.timeout
            )
            response.raise_for_status()
            result = response.json()
            
            if "error" in result:
                raise RuntimeError(f"创建环境失败: {result['error']}")
                
            env_id = result["id"]
            self.env_ids[env_name] = env_id
            
            # 如果是第一个环境，自动设为当前环境
            if not self.current_env:
                self.current_env = env_name
                
            return env_id
            
        except Exception as e:
            raise RuntimeError(f"创建环境失败: {str(e)}")

    def step(self, action: str, env_name: Optional[str] = None) -> StepOutput:
        """
        执行一步动作
        
        Args:
            action: 动作指令
            env_name: 环境名称，默认使用当前活跃环境
            
        Returns:
            步骤输出结果
        """
        env_name = env_name or self.current_env
        if not env_name or env_name not in self.env_ids:
            raise ValueError("环境不存在或未指定")
            
        env_id = self.env_ids[env_name]
        
        try:
            response = requests.post(
                f"{self.env_server_base}/step",
                json={"id": env_id, "action": action},
                timeout=self.timeout
            )
            response.raise_for_status()
            result = response.json()
            
            if "error" in result:
                raise RuntimeError(f"执行步骤失败: {result['error']}")
                
            return StepOutput(
                state={"text": result.get("observation", "")},
                reward=result.get("reward", 0.0),
                done=result.get("done", False),
                info=result
            )
            
        except Exception as e:
            raise RuntimeError(f"执行步骤失败: {str(e)}")

    def reset(self, env_name: Optional[str] = None, data_idx: int = 0) -> StepOutput:
        """
        重置环境
        
        Args:
            env_name: 环境名称，默认使用当前活跃环境
            data_idx: 数据索引
            
        Returns:
            重置后的初始状态
        """
        env_name = env_name or self.current_env
        if not env_name or env_name not in self.env_ids:
            raise ValueError("环境不存在或未指定")
            
        env_id = self.env_ids[env_name]
        
        try:
            response = requests.post(
                f"{self.env_server_base}/reset",
                json={"id": env_id, "data_idx": data_idx},
                timeout=self.timeout
            )
            response.raise_for_status()
            result = response.json()
            
            if "error" in result:
                raise RuntimeError(f"重置环境失败: {result['error']}")
                
            return StepOutput(
                state={"text": result.get("observation", "")},
                reward=result.get("reward", 0.0),
                done=result.get("done", False),
                info=result
            )
            
        except Exception as e:
            raise RuntimeError(f"重置环境失败: {str(e)}")