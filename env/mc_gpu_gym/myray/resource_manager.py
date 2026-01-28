"""
Ray Actor 资源管理模块

解决多 MC 实例并发的资源冲突问题：
- P0: DISPLAY 端口冲突
- P1: Minecraft 环境端口冲突
- P1: 工作目录冲突
- P2: GPU 资源竞争

设计方案：在 Ray Actor 层面管理所有资源，不修改 launchClient.sh
"""

import os
import shutil
import threading
from pathlib import Path
from typing import Dict, Optional, Set
import shutil
import ray


class PortPool:
    """
    线程安全的端口池管理

    解决问题：
    - P0: DISPLAY 端口冲突
    - P1: Minecraft 环境端口冲突

    用法：
        pool = PortPool(start=9000, end=9100)
        port = pool.allocate()  # 获取一个可用端口
        pool.release(port)      # 释放端口回池
    """

    def __init__(self, start: int, end: int, name: str = "PortPool"):
        """
        初始化端口池

        Args:
            start: 起始端口（包含）
            end: 结束端口（不包含）
            name: 池名称（用于日志）
        """
        if start >= end:
            raise ValueError(f"start ({start}) must be less than end ({end})")
        if start < 0 or end > 65536:
            raise ValueError(f"Port range must be within 0-65535")

        self.name = name
        self.start = start
        self.end = end
        self.available: Set[int] = set(range(start, end))
        self.allocated: Set[int] = set()
        self.lock = threading.Lock()

        print(f"[{self.name}] Initialized with {len(self.available)} ports ({start}-{end-1})")

    def allocate(self) -> int:
        """
        分配一个端口

        Returns:
            int: 分配的端口号

        Raises:
            RuntimeError: 如果没有可用端口
        """
        with self.lock:
            if not self.available:
                raise RuntimeError(
                    f"[{self.name}] No available ports! "
                    f"All {len(self.allocated)} ports are allocated."
                )

            port = self.available.pop()
            self.allocated.add(port)
            print(f"[{self.name}] Allocated port {port} ({len(self.available)} remaining)")
            return port

    def release(self, port: int):
        """
        释放一个端口

        Args:
            port: 要释放的端口号
        """
        with self.lock:
            if port in self.allocated:
                self.allocated.remove(port)
                self.available.add(port)
                print(f"[{self.name}] Released port {port} ({len(self.available)} available)")
            else:
                print(f"[{self.name}] Warning: Port {port} was not allocated")

    def is_available(self, port: int) -> bool:
        """检查端口是否可用"""
        with self.lock:
            return port in self.available

    def get_stats(self) -> Dict:
        """获取统计信息"""
        with self.lock:
            return {
                "name": self.name,
                "total": len(self.available) + len(self.allocated),
                "available": len(self.available),
                "allocated": len(self.allocated),
                "range": f"{self.start}-{self.end-1}",
            }


class WorkingDirManager:
    """
    工作目录管理器

    解决问题：
    - P1: 工作目录冲突（多实例写入相同文件）

    用法：
        manager = WorkingDirManager(base_dir="/tmp")
        working_dir = manager.create(uuid="abc123")
        # 使用 working_dir
        manager.cleanup(uuid="abc123")
    """

    def __init__(self, base_dir: str = "/tmp/raycraft_mc"):
        """
        初始化管理器

        Args:
            base_dir: 基础目录
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        self.directories: Dict[str, Path] = {}  # uuid -> path
        self.lock = threading.Lock()

        print(f"[WorkingDirManager] Initialized with base: {self.base_dir}")

    def create(self, uuid: str) -> Path:
        """
        创建工作目录

        Args:
            uuid: 实例的唯一标识

        Returns:
            Path: 创建的目录路径
        """
        with self.lock:
            if uuid in self.directories:
                print(f"[WorkingDirManager] UUID {uuid[:8]} already has directory")
                return self.directories[uuid]

            # 创建目录：/tmp/raycraft_mc/mc-{uuid}/
            working_dir = self.base_dir / f"mc-{uuid}"
            working_dir.mkdir(parents=True, exist_ok=True)

            self.directories[uuid] = working_dir
            print(f"[WorkingDirManager] Created directory for {uuid[:8]}: {working_dir}")
            return working_dir

    def cleanup(self, uuid: str):
        """
        清理工作目录

        Args:
            uuid: 实例的唯一标识
        """
        with self.lock:
            if uuid not in self.directories:
                print(f"[WorkingDirManager] UUID {uuid[:8]} has no directory to clean")
                return

            working_dir = self.directories[uuid]
            if working_dir.exists():
                try:
                    shutil.rmtree(working_dir)
                    print(f"[WorkingDirManager] Cleaned directory for {uuid[:8]}: {working_dir}")
                except Exception as e:
                    print(f"[WorkingDirManager] Failed to clean {uuid[:8]}: {e}")

            del self.directories[uuid]

    def get_stats(self) -> Dict:
        """获取统计信息"""
        with self.lock:
            total_size = 0
            for working_dir in self.directories.values():
                if working_dir.exists():
                    try:
                        total_size += sum(
                            f.stat().st_size
                            for f in working_dir.rglob('*')
                            if f.is_file()
                        )
                    except Exception:
                        pass

            return {
                "count": len(self.directories),
                "total_size_mb": total_size / 1024 / 1024,
                "base_dir": str(self.base_dir),
            }


@ray.remote
class GlobalResourceManager:
    """
    全局资源管理器（Ray Actor）

    统一管理所有资源：
    - DISPLAY 端口（9000-9100）
    - Minecraft 环境端口（10000-10100）
    - 工作目录

    用法：
        manager = GlobalResourceManager.remote()
        resources = ray.get(manager.allocate_resources.remote(uuid="abc"))
        # 使用资源
        ray.get(manager.release_resources.remote(uuid="abc"))
    """

    def __init__(
        self,
        display_port_range=(9000, 9100),
        env_port_range=(10000, 10100),
        working_dir_base="/tmp/raycraft_mc",
    ):
        """
        初始化资源管理器

        Args:
            display_port_range: DISPLAY 端口范围
            env_port_range: Minecraft 环境端口范围
            working_dir_base: 工作目录基础路径
        """
        self.display_pool = PortPool(*display_port_range, name="DISPLAY")
        self.env_pool = PortPool(*env_port_range, name="EnvPort")
        self.working_dir_manager = WorkingDirManager(working_dir_base)

        # 记录每个 UUID 分配的资源
        self.allocations: Dict[str, Dict] = {}

        print("[GlobalResourceManager] Initialized")
        print(f"  DISPLAY ports: {display_port_range}")
        print(f"  Env ports: {env_port_range}")
        print(f"  Working dir: {working_dir_base}")

    def allocate_resources(self, uuid: str) -> Dict:
        """
        为一个实例分配所有资源

        Args:
            uuid: 实例的唯一标识

        Returns:
            dict: {
                "display_port": int,
                "env_port": int,
                "working_dir": str,
            }

        Raises:
            RuntimeError: 如果资源不足或 UUID 已分配
        """
        if uuid in self.allocations:
            raise RuntimeError(f"UUID {uuid} already has resources allocated")

        print(f"\n[GlobalResourceManager] Allocating resources for {uuid[:8]}...")

        try:
            # 分配端口
            display_port = self.display_pool.allocate()
            env_port = self.env_pool.allocate()

            # 创建工作目录
            working_dir = self.working_dir_manager.create(uuid)

            # 记录分配
            self.allocations[uuid] = {
                "display_port": display_port,
                "env_port": env_port,
                "working_dir": str(working_dir),
            }

            print(f"[GlobalResourceManager] Allocated for {uuid[:8]}:")
            print(f"  DISPLAY: :{display_port}")
            print(f"  Env Port: {env_port}")
            print(f"  Working Dir: {working_dir}")

            return self.allocations[uuid]

        except Exception as e:
            # 回滚已分配的资源
            print(f"[GlobalResourceManager] Allocation failed for {uuid[:8]}, rolling back...")
            self._rollback_allocation(uuid)
            raise RuntimeError(f"Failed to allocate resources for {uuid[:8]}: {e}")

    def release_resources(self, uuid: str):
        """
        释放一个实例的所有资源

        Args:
            uuid: 实例的唯一标识
        """
        if uuid not in self.allocations:
            print(f"[GlobalResourceManager] UUID {uuid[:8]} has no resources to release")
            return

        print(f"\n[GlobalResourceManager] Releasing resources for {uuid[:8]}...")

        resources = self.allocations[uuid]

        # 释放端口
        self.display_pool.release(resources["display_port"])
        self.env_pool.release(resources["env_port"])

        # # 清理工作目录
        # self.working_dir_manager.cleanup(uuid)

        # 移除记录
        del self.allocations[uuid]

        print(f"[GlobalResourceManager] Released resources for {uuid[:8]}")

    def get_stats(self) -> Dict:
        """获取资源使用统计"""
        return {
            "display_ports": self.display_pool.get_stats(),
            "env_ports": self.env_pool.get_stats(),
            "working_dirs": self.working_dir_manager.get_stats(),
            "active_instances": len(self.allocations),
            "allocated_uuids": [uuid[:8] for uuid in self.allocations.keys()],
        }

    def _rollback_allocation(self, uuid: str):
        """回滚失败的资源分配"""
        if uuid in self.allocations:
            self.release_resources(uuid)


# 模块导出
__all__ = [
    "PortPool",
    "WorkingDirManager",
    "GlobalResourceManager",
]
