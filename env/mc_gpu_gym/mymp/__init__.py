"""
Multiprocess backend for RayCraft GPU server.

The module mirrors the public interface of ``myray`` so it can be
swapped in by the HTTP server without touching the callers.
"""

from .global_pool import get_global_env_pool, reset_global_env_pool

__all__ = ["get_global_env_pool", "reset_global_env_pool"]

