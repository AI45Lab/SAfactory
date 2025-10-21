import requests
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

@dataclass
class StepOutput:
    """统一输出数据结构"""
    state: Dict[str, Any]
    reward: float
    done: bool
    info: Optional[Dict[str, Any]] = None

class BaseEnvClient:
    """环境客户端基类，定义统一接口"""
    def __init__(self, env_server_base: str, timeout: int = 300):
        self.env_server_base = env_server_base
        self.timeout = timeout
        self.env_ids: Dict[str, int] = {}  # 环境名称到ID的映射
        self.current_env: Optional[str] = None  # 当前活跃环境

    def create_env(self, env_name: str, **kwargs) -> int:
        """创建新环境并返回ID"""
        raise NotImplementedError("子类必须实现create_env方法")

    def step(self, action: Any, env_name: Optional[str] = None) -> StepOutput:
        """执行一步动作"""
        raise NotImplementedError("子类必须实现step方法")

    def reset(self, env_name: Optional[str] = None,** kwargs) -> StepOutput:
        """重置环境"""
        raise NotImplementedError("子类必须实现reset方法")
    
    def close_env(self, env_name: str) -> bool:
        """关闭指定环境"""
        if env_name not in self.env_ids:
            return False
        
        env_id = self.env_ids[env_name]
        try:
            response = requests.post(
                f"{self.env_server_base}/close",
                json={"id": env_id},
                timeout=self.timeout
            )
            response.raise_for_status()
            del self.env_ids[env_name]
            if self.current_env == env_name:
                self.current_env = None
            return True
        except Exception as e:
            print(f"关闭环境 {env_name} 失败: {str(e)}")
            return False

    def switch_env(self, env_name: str) -> bool:
        """切换当前活跃环境"""
        if env_name in self.env_ids:
            self.current_env = env_name
            return True
        return False

    def get_observation(self, env_name: Optional[str] = None) -> Dict[str, Any]:
        """获取当前观测"""
        env_name = env_name or self.current_env
        if not env_name or env_name not in self.env_ids:
            raise ValueError("环境不存在或未指定")
        
        try:
            response = requests.get(
                f"{self.env_server_base}/observation",
                params={"id": self.env_ids[env_name]},
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise RuntimeError(f"获取观测失败: {str(e)}")

    def get_env_ids(self) -> Dict[str, int]:
        """获取所有环境ID映射"""
        return self.env_ids.copy()