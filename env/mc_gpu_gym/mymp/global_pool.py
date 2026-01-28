"""
Singleton accessor for the multiprocess environment pool.
"""

from typing import Optional

from .pool import MPEnvPool
from .resource_manager import GlobalResourceManager


GLOBAL_MP_POOL_NAME = "GlobalMPEnvPool"

_global_mp_env_pool: Optional[MPEnvPool] = None
_global_resource_manager: Optional[GlobalResourceManager] = None


def get_global_env_pool() -> MPEnvPool:
    """Get (or create) the shared multiprocess pool instance."""
    global _global_mp_env_pool, _global_resource_manager

    if _global_mp_env_pool is not None:
        return _global_mp_env_pool

    _global_resource_manager = _global_resource_manager or GlobalResourceManager(
        display_port_range=(1, 100),
        working_dir_base="/mnt/shared-storage-user/steai_share/luozhihao/mc_test/raycraft/record",
    )
    _global_mp_env_pool = MPEnvPool(resource_manager=_global_resource_manager)
    return _global_mp_env_pool


def reset_global_env_pool():
    """Destroy the singleton and its resources (for tests)."""
    global _global_mp_env_pool, _global_resource_manager
    if _global_mp_env_pool:
        _global_mp_env_pool.batch_destroy(list(_global_mp_env_pool.env_registry.keys()))
    _global_mp_env_pool = None
    _global_resource_manager = None

