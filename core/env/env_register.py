# core/env/registry.py
"""环境注册器，用于管理所有可被交互器识别的环境类"""

from typing import Dict, Type, Any

# 环境注册表：键为环境名称（字符串），值为环境类
_env_registry: Dict[str, Type[Any]] = {}

def register_env(env_name: str):
    """
    环境注册装饰器，用于将环境类注册到全局注册表中
    
    Args:
        env_name: 环境的唯一标识名称（如"android_gym"）
    """
    def decorator(env_class: Type[Any]) -> Type[Any]:
        if env_name in _env_registry:
            raise ValueError(f"环境名称 '{env_name}' 已被注册，请使用不同名称")
        _env_registry[env_name] = env_class
        return env_class
    return decorator

def get_env_class(env_name: str) -> Type[Any]:
    """
    根据环境名称从注册表中获取环境类
    
    Args:
        env_name: 已注册的环境名称
    Returns:
        对应的环境类
    Raises:
        ValueError: 如果环境名称未注册
    """
    env_class = _env_registry.get(env_name)
    if env_class is None:
        raise ValueError(f"环境 '{env_name}' 未注册，请先使用 @register_env 装饰器注册")
    return env_class

def list_registered_envs() -> Dict[str, Type[Any]]:
    """返回所有已注册的环境"""
    return _env_registry.copy()