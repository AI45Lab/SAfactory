"""
GPU 全局环境池管理器
确保所有客户端共享同一个环境池实例（跨进程）
集成 GlobalResourceManager 进行资源管理
"""

import ray
from typing import Optional
from .pool import GPUEnvPool
from .resource_manager import GlobalResourceManager

# 全局环境池的命名
GLOBAL_GPU_POOL_NAME = "GlobalGPUEnvPool"
GLOBAL_GPU_RESOURCE_MANAGER_NAME = "GlobalGPUResourceManager"

# 本地缓存（进程级优化）
_global_gpu_env_pool: Optional[ray.ObjectRef] = None
_global_gpu_resource_manager: Optional[ray.ObjectRef] = None

def get_global_resource_manager():
    """获取全局资源管理器实例（跨进程共享）

    Returns:
        Ray Actor Handle: 全局资源管理器的引用
    """
    global _global_gpu_resource_manager

    # 优化：如果本地已缓存，直接返回
    if _global_gpu_resource_manager is not None:
        return _global_gpu_resource_manager

    try:
        # 尝试获取已存在的命名 Actor
        _global_gpu_resource_manager = ray.get_actor(GLOBAL_GPU_RESOURCE_MANAGER_NAME)
    except ValueError:
        # Actor 不存在，创建新的命名 Actor
        _global_gpu_resource_manager = GlobalResourceManager.options(
            name=GLOBAL_GPU_RESOURCE_MANAGER_NAME,
            lifetime="detached",
            max_concurrency=100
        ).remote(
            display_port_range=(9000, 9100),
            env_port_range=(10000, 10100),
            working_dir_base="/mnt/shared-storage-user/steai_share/luozhihao/mc_test/raycraft/record"
        )

    return _global_gpu_resource_manager

def get_global_env_pool():
    """获取全局 GPU 环境池实例（跨进程共享）

    使用 Ray Named Actor 确保所有进程（主进程和 Worker）
    访问同一个 GPUEnvPool 实例

    Returns:
        Ray Actor Handle: 全局环境池的引用
    """
    global _global_gpu_env_pool

    # 优化：如果本地已缓存，直接返回
    if _global_gpu_env_pool is not None:
        return _global_gpu_env_pool

    try:
        # 尝试获取已存在的命名 Actor
        _global_gpu_env_pool = ray.get_actor(GLOBAL_GPU_POOL_NAME)
    except ValueError:
        # Actor 不存在，创建新的命名 Actor
        # 先获取资源管理器
        resource_manager = get_global_resource_manager()

        # 创建环境池，传递资源管理器
        _global_gpu_env_pool = GPUEnvPool.options(
            name=GLOBAL_GPU_POOL_NAME,
            lifetime="detached",  # 即使创建者退出也保持存活
            max_concurrency=100   # 支持高并发
        ).remote(resource_manager=resource_manager)

    return _global_gpu_env_pool

def reset_global_env_pool():
    """重置全局 GPU 环境池（用于测试）

    清理命名 Actor 和本地缓存
    """
    global _global_gpu_env_pool, _global_gpu_resource_manager

    # 清理环境池 Actor
    try:
        actor = ray.get_actor(GLOBAL_GPU_POOL_NAME)
        ray.kill(actor)
    except ValueError:
        pass
    except Exception as e:
        print(f"Warning: Failed to kill global GPU pool: {e}")

    # 清理资源管理器 Actor
    try:
        actor = ray.get_actor(GLOBAL_GPU_RESOURCE_MANAGER_NAME)
        ray.kill(actor)
    except ValueError:
        pass
    except Exception as e:
        print(f"Warning: Failed to kill global resource manager: {e}")

    # 清理本地缓存
    _global_gpu_env_pool = None
    _global_gpu_resource_manager = None

def clear_global_env_pool():
    """清理全局环境池（reset_global_env_pool 的别名，用于测试）"""
    reset_global_env_pool()