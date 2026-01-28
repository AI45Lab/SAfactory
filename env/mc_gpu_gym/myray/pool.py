"""
GPU 环境池管理器 - 支持 GPU 资源管理
支持 UUID-环境映射和批量环境管理
"""

import ray
import time
import logging
from typing import Dict, List, Optional
from uuid import uuid4
import os
from pathlib import Path

# 日志配置
LOG_DIR = Path(__file__).parent.parent.parent / "logs" / "ray"
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOG_DIR, "gpu_envpool.log")
LOG_FORMAT = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"

logger = logging.getLogger("GPUEnvPool")
logger.setLevel(logging.INFO)
logger.handlers.clear()

file_handler = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter(LOG_FORMAT)
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(file_formatter)
logger.addHandler(console_handler)

logger.propagate = False
logger.info("GPUEnvPool logger configured successfully")


@ray.remote(max_concurrency=64)
class GPUEnvPool:
    """GPU Ray 环境池管理器

    功能：
    - 管理多个 GPU MCEnvActor 实例
    - 支持 UUID->环境的映射
    - 集成 GlobalResourceManager 进行资源管理
    - 环境状态管理和回收
    """

    def __init__(self, resource_manager=None):
        """初始化 GPU 环境池

        Args:
            resource_manager: GlobalResourceManager 实例（可选）
                如果不提供，EnvPool 内部会创建
        """
        import logging
        import sys

        # 确保项目根目录在 sys.path 中
        project_root = Path(__file__).parent.parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        # 确保 MineStudio_gpu 在 sys.path 中
        minestudio_path = project_root / "MineStudio_gpu"
        if minestudio_path.exists() and str(minestudio_path) not in sys.path:
            sys.path.insert(0, str(minestudio_path))

        # 确保日志目录存在
        log_dir = Path(__file__).parent.parent.parent / "logs" / "ray"
        os.makedirs(log_dir, exist_ok=True)

        # 配置 Actor 进程中的日志
        actor_logger = logging.getLogger("GPUEnvPool")
        actor_logger.setLevel(logging.INFO)
        actor_logger.handlers.clear()

        log_file = os.path.join(log_dir, "gpu_envpool.log")
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s")
        file_handler.setFormatter(formatter)
        actor_logger.addHandler(file_handler)
        actor_logger.propagate = False

        # 初始化环境池状态
        self.env_registry = {}    # uuid -> MCEnvActor
        self.env_configs = {}     # uuid -> config_dict
        self.env_status = {}      # uuid -> "idle"/"busy"/"failed"
        self.created_time = {}    # uuid -> creation_timestamp
        self.logger = actor_logger
        self.step_res = {}

        # 资源管理器
        if resource_manager is None:
            # 如果没有提供，创建默认的资源管理器
            from .resource_manager import GlobalResourceManager
            self.resource_manager = GlobalResourceManager.options(
                name="GPUEnvPool_ResourceManager",
                lifetime="detached"
            ).remote(
                display_port_range=(9000, 9100),
                env_port_range=(10000, 10100),
                working_dir_base="/mnt/shared-storage-user/steai_share/luozhihao/mc_test/raycraft/record"
            )
            self.logger.info("Created internal GlobalResourceManager")
        else:
            self.resource_manager = resource_manager
            self.logger.info("Using provided GlobalResourceManager")

        self.logger.info("GPUEnvPool initialized in Ray Actor process")

    def create_envs(self, uuids: List[str], configs: List[dict]) -> bool:
        """批量创建环境（串行创建，避免 GPU 资源竞争）

        Args:
            uuids: UUID 列表
            configs: 配置列表，与 uuids 一一对应

        Returns:
            bool: 是否全部创建成功
        """
        from .actors import MCEnvActor

        successful_uuids = []
        failed_uuids = []

        # 串行创建环境，避免 GPU 资源竞争
        count = 0
        for uuid, config in zip(uuids, configs):
            self.logger.info(f"Creating environment {uuid[:8]}...")

            # 创建环境 Actor
            env_actor = MCEnvActor.remote(
                uuid=uuid,
                count=count,
                config=config,
                resource_manager=self.resource_manager
            )

            # 注册到池中
            self.env_registry[uuid] = env_actor
            self.env_configs[uuid] = config
            self.env_status[uuid] = "idle"  # 标记为 idle，延迟初始化到第一次 reset
            self.created_time[uuid] = time.time()


            # try:
            #     uuid_test = ray.get(self.env_registry[uuid].uuid.remote())
            # except:
            #     continue

            # 不在这里调用 reset，延迟到第一次使用时初始化
            # 这样可以避免批量创建时的长时间等待和超时问题

            successful_uuids.append(uuid)
            self.logger.info(f"Environment {uuid[:8]} registered (will initialize on first reset)")
            count+=1
            print("------------------------------")
            print(uuid)
            print("------------------------------")

        
        
        
        self.logger.info(
            f"Environment creation finished: {len(successful_uuids)} succeeded, "
            f"{len(failed_uuids)} failed"
        )

        return len(failed_uuids) == 0

    
    
    
    
    
    
    
    
    def env_exists(self, uuid: str) -> bool:
        """检查环境是否存在

        Args:
            uuid: 环境UUID

        Returns:
            bool: 环境是否存在且状态正常
        """
        return uuid in self.env_registry and self.env_status.get(uuid) != "failed"

    def get_env_by_uuid(self, uuid: str):
        """根据UUID获取环境

        Args:
            uuid: 环境UUID

        Returns:
            MCEnvActor: 环境Actor引用

        Raises:
            ValueError: UUID不存在或环境状态异常
        """
        if uuid not in self.env_registry:
            raise ValueError(f"Environment with UUID {uuid} not found")

        if self.env_status[uuid] == "failed":
            raise ValueError(f"Environment {uuid} is in failed state")

        # 标记为忙碌状态
        self.env_status[uuid] = "busy"

        self.logger.debug(f"Environment {uuid} allocated")
        return self.env_registry[uuid]

    def return_env(self, uuid: str) -> bool:
        """归还环境到池中

        Args:
            uuid: 环境UUID

        Returns:
            bool: 是否成功归还
        """
        if uuid not in self.env_registry:
            self.logger.warning(f"Trying to return non-existent environment {uuid}")
            return False

        # 标记为空闲状态
        self.env_status[uuid] = "idle"
        self.logger.debug(f"Environment {uuid} returned to pool")
        return True

    def destroy_env(self, uuid: str) -> bool:
        """销毁指定环境

        Args:
            uuid: 环境 UUID

        Returns:
            bool: 是否成功销毁
        """
        if uuid not in self.env_registry:
            self.logger.warning(f"Trying to destroy non-existent environment {uuid[:8]}")
            return False

        try:
            # 调用 close 释放资源
            ray.get(self.env_registry[uuid].close.remote(), timeout=30)

            # 从注册表中移除
            del self.env_registry[uuid]
            del self.env_configs[uuid]
            del self.env_status[uuid]
            del self.created_time[uuid]

            self.logger.info(f"Environment {uuid[:8]} destroyed")
            return True

        except Exception as e:
            self.logger.error(f"Failed to destroy environment {uuid[:8]}: {e}")
            return False

    def batch_destroy(self, uuids: List[str]) -> int:
        """批量销毁环境

        Args:
            uuids: 要销毁的UUID列表

        Returns:
            int: 成功销毁的环境数量
        """
        success_count = 0
        for uuid in uuids:
            if self.destroy_env(uuid):
                success_count += 1

        self.logger.info(f"Batch destroyed {success_count}/{len(uuids)} environments")
        return success_count

    def get_pool_stats(self) -> Dict:
        """获取环境池统计信息

        Returns:
            Dict: 包含各种统计信息的字典
        """
        total_envs = len(self.env_registry)
        idle_envs = sum(1 for status in self.env_status.values() if status == "idle")
        busy_envs = sum(1 for status in self.env_status.values() if status == "busy")
        failed_envs = sum(1 for status in self.env_status.values() if status == "failed")

        return {
            "total_environments": total_envs,
            "idle_environments": idle_envs,
            "busy_environments": busy_envs,
            "failed_environments": failed_envs,
            "uptime": time.time() - min(self.created_time.values()) if self.created_time else 0,
            "environment_list": list(self.env_registry.keys())
        }

    def health_check(self) -> Dict:
        """环境池健康检查

        Returns:
            Dict: 健康状态信息
        """
        stats = self.get_pool_stats()

        # 简单的健康检查逻辑
        healthy = (
            stats["total_environments"] > 0 and
            stats["failed_environments"] == 0
        )

        return {
            "healthy": healthy,
            "total_envs": stats["total_environments"],
            "failed_envs": stats["failed_environments"],
            "timestamp": time.time()
        }

    def get_step_res(self):
        return self.step_res
    def rm_step_res(self, uuid):
        del self.step_res[uuid]
    def add_step_res(self, uuid, obs_optimized, reward, terminated, info):
        self.step_res[uuid] = (obs_optimized, reward, terminated, info)
    
    
    async def step(self, uuid: str , action: str):
        self.env_status[uuid] = "busy"

        self.env_registry[uuid].step.remote(action)

    
    # ===== HTTP API 支持方法 =====

    def create_env(self, env_id: str, env_name: str, env_kwargs: Dict) -> "ray.ActorHandle":
        """创建单个环境（HTTP API专用）

        Args:
            env_id: 环境UUID
            env_name: 环境名称（暂未使用，预留）
            env_kwargs: 环境配置参数

        Returns:
            ray.ActorHandle: 环境Actor引用
        """
        # 使用现有的create_envs方法
        success = self.create_envs([env_id], [env_kwargs])
        if not success:
            raise RuntimeError(f"Failed to create environment {env_id}")

        return self.env_registry[env_id]

    def get_env(self, env_id: str) -> Optional["ray.ActorHandle"]:
        """获取环境（HTTP API专用）

        Args:
            env_id: 环境UUID

        Returns:
            ray.ActorHandle: 环境Actor引用，如果不存在返回None
        """
        if not self.env_exists(env_id):
            return None

        return self.get_env_by_uuid(env_id)

    def get_env_status(self, env_id: str) -> dict:
        """获取环境状态（HTTP API专用）
        
        Args:
            env_id: 环境UUID
            
        Returns:
            dict: 包含状态信息的字典
        """
        if env_id not in self.env_registry:
            return {"status": "not_found", "exists": False}
        
        return {
            "status": self.env_status.get(env_id, "unknown"),
            "exists": True,
            "created_time": self.created_time.get(env_id, 0)
        }
    
    def is_env_ready(self, env_id: str) -> bool:
        """检查环境是否已初始化并准备好接受step请求
        
        Args:
            env_id: 环境UUID
            
        Returns:
            bool: 环境是否已就绪
        """
        if env_id not in self.env_registry:
            return False
        
        # 检查环境是否已经完成第一次reset
        # 状态为 "idle" 且已被使用过（有reset记录）
        env_actor = self.env_registry.get(env_id)
        if env_actor is None:
            return False
        
        try:
            # 通过检查 actor 的 already_reset 属性来判断
            is_reset = ray.get(env_actor.is_already_reset.remote(), timeout=5)
            return is_reset
        except Exception:
            # 如果查询失败，认为未就绪
            return False

    def close_env(self, env_id: str) -> bool:
        """关闭环境（HTTP API专用）

        Args:
            env_id: 环境UUID

        Returns:
            bool: 是否成功关闭
        """
        return self.destroy_env(env_id)

    def list_envs(self) -> List[str]:
        """列出所有环境ID（HTTP API专用）

        Returns:
            List[str]: 环境UUID列表
        """
        return list(self.env_registry.keys())

    def get_num_envs(self) -> int:
        """获取环境数量（HTTP API专用）

        Returns:
            int: 环境总数
        """
        return len(self.env_registry)