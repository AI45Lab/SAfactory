"""
Raycraft GPU - GPU-accelerated Minecraft Gym Environment

GPU 版本特性：
- GPU 硬件加速渲染 (VirtualGL + NVIDIA GPU)
- Ray 分布式并行环境创建
- 与 CPU 版本 100% API 兼容
- 高性能视觉质量（目标 60 FPS）

技术栈：
- Ray for distributed computing
- MineStudio (GPU branch) for Minecraft simulation
- VirtualGL for GPU rendering
- OpenAI Gym standard interface

警告：
- 需要 NVIDIA GPU 和 VirtualGL 环境
- 详见 docs/gpu-setup.md
"""

from .myray.client import MCRayClient
from .myray.actors import MCEnvActor

__version__ = "1.0.0-gpu"
__backend__ = "gpu"
__all__ = ["MCRayClient", "MCEnvActor"]
