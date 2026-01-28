"""
MCEnvActor - Ray Actor封装MCSimulator
直接复用现有的MCSimulator，零修改
"""

import ray
import sys
import time
import logging
import os
from datetime import datetime
from pathlib import Path
from .global_pool import get_global_env_pool

# 配置日志保存到文件
log_dir = Path(__file__).parent.parent.parent / "logs"
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / f"actor_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.getpid()}.txt"

logger = logging.getLogger("MCEnvActor")
logger.setLevel(logging.DEBUG)

# 添加文件处理器
fh = logging.FileHandler(log_file, encoding="utf-8")
fh.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)

# 同时保留控制台输出
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(formatter)
logger.addHandler(ch)

logger.info(f"MCEnvActor logging to file: {log_file.absolute()}")

@ray.remote(max_restarts=3, num_gpus=0.01)  # 每个实例占用 0.1 GPU
class MCEnvActor:
    """
    Ray Actor 封装的 MC 环境实例（带资源管理）

    新特性：
    - 自动分配唯一的 DISPLAY 端口
    - 自动分配唯一的 Minecraft 环境端口
    - 独立的工作目录
    - GPU 资源限制
    """

    def __init__(self, uuid: str, count: int, config: dict, resource_manager=None):
        """初始化 MC 环境

        Args:
            uuid: 实例唯一标识
            config: 环境配置
            resource_manager: 全局资源管理器（可选）
        """
        project_root = Path(__file__).parent.parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        # 确保 MineStudio_gpu 在 sys.path 中
        minestudio_path = project_root / "MineStudio_gpu"
        if minestudio_path.exists() and str(minestudio_path) not in sys.path:
            sys.path.insert(0, str(minestudio_path))

        # 直接复用现有MCSimulator
        from ..mc_simulator import MCSimulator

        self.uuid = uuid
        self.mc_root = "/mnt/shared-storage-user/steai_share/luozhihao/mc_test/env/"+str(count)+"/.minecraft"
        self.config = config
        self.resource_manager = resource_manager
        self.resources = None
        self.simulator = None
        self.step_count = 0
        self.obs = None
        self._info = None
        self.created_at = time.time()

        logger.info(f"[{uuid[:8]}] MCEnvActor initialized (will allocate resources on first reset)")

        self._allocate_resources()

        # 直接复用现有MCSimulator
        from ..mc_simulator import MCSimulator

        import torch
        gpu_ids = ray.get_gpu_ids()        # Ray 分配给你的 GPU 号
        if gpu_ids:
            torch.cuda.set_device(gpu_ids[0])
            logger.info(f"[{self.uuid[:8]}] Set CUDA device to {gpu_ids[0]}")


        # 创建 simulator，传入资源配置
        self.simulator = MCSimulator(
            config=self.config,
            config_path="/mnt/shared-storage-user/steai_share/luozhihao/mc_test/raycraft/configs/kill/base.yaml",
            display_port=self.resources["display_port"],
            env_port=self.resources["env_port"],
            working_dir=self.resources["working_dir"],
            mc_root=self.mc_root
        )
        logger.info(f"[{self.uuid[:8]}] Simulator created with resources")


    def _allocate_resources(self):
        """申请资源（首次 reset 时调用）"""
        if self.resources is not None:
            return  # 已分配

        if self.resource_manager is None:
            # 无资源管理器，使用默认值
            logger.warning(f"[{self.uuid[:8]}] No resource manager, using defaults")
            self.resources = {
                "display_port": 0,  # 使用默认 DISPLAY
                "env_port": 9000,   # 使用默认端口
                "working_dir": "/tmp",
            }
        else:
            # 从资源管理器申请资源
            logger.info(f"[{self.uuid[:8]}] Allocating resources from manager...")
            self.resources = ray.get(
                self.resource_manager.allocate_resources.remote(self.uuid)
            )

        logger.info(f"[{self.uuid[:8]}] Resources allocated:")
        logger.info(f"  DISPLAY:  {self.resources['display_port']}  Env Port:  {self.resources['env_port']}  Working Dir:  {self.resources['working_dir']}")

    def reset(self):
        """重置环境

        Returns:
            观察数据（与HTTP版本格式一致）
        """
        try:
            # # 首次 reset：分配资源并创建 simulator
            # if self.simulator is None:
                

            # 调用 simulator.reset()
            obs, _info = self.simulator.reset()
            self.obs=obs
            self._info = _info
            self.step_count = 0
            logger.debug(f"[{self.uuid[:8]}] Environment reset successfully")

            # 大对象优化：自动检测并存储到Object Store
            return self._optimize_large_object(obs)

        except Exception as e:
            logger.error(f"[{self.uuid[:8]}] Reset failed: {e}")
            raise

    def is_already_reset(self) -> bool:
        """检查环境是否已经初始化（是否已经完成第一次reset）
        
        Returns:
            bool: True if already reset, False otherwise
        """
        return self.simulator is not None

    def step(self, action: str):
        env_pool = get_global_env_pool()
        start_t = time.time()
        obs, reward, terminated, truncated, info = self.simulator.step(action)
        elpased_t = time.time() - start_t
        self.step_count += 1


        logger.debug(f"Step {self.step_count}: reward={reward}, done={terminated}")

        # 增强info信息
        if info is None:
            info = {}
        info.update({
            "step_count": self.step_count,
            "actor_id": ray.get_runtime_context().actor_id,
            "node_id": ray.get_runtime_context().node_id
        })

        # 大对象优化
        obs_optimized = self._optimize_large_object(obs)
        env_pool.add_step_res.remote(self.uuid, obs_optimized, reward, terminated, info)


    # def step(self, action: str):
    #     """执行动作

    #     Args:
    #         action: JSON格式的动作字符串（与HTTP版本一致）

    #     Returns:
    #         (observation, reward, done, info) - 与HTTP版本格式一致
    #     """
    #     # try:
    #     start_t = time.time()
    #     obs, reward, terminated, truncated, info = self.simulator.step(action)
    #     elpased_t = time.time() - start_t
    #     print(f'[INFO] in actors.py, {}, {elpased_t=}')
    #     self.step_count += 1


    #     logger.debug(f"Step {self.step_count}: reward={reward}, done={terminated}")

    #     # 增强info信息
    #     if info is None:
    #         info = {}
    #     info.update({
    #         "step_count": self.step_count,
    #         "actor_id": ray.get_runtime_context().actor_id,
    #         "node_id": ray.get_runtime_context().node_id
    #     })

    #     # 大对象优化
    #     obs_optimized = self._optimize_large_object(obs)

    #     return obs_optimized, reward, terminated, info

        # except Exception as e:
        #     logger.error(f"Step failed: {e}")
        #     raise

    def get_observation(self):
        """获取当前观察

        Returns:
            当前观察数据
        """
        try:
            obs = self.simulator.get_observation()
            return self._optimize_large_object(obs)

        except Exception as e:
            logger.error(f"Get observation failed: {e}")
            raise

    def get_stats(self):
        """获取环境统计信息

        Returns:
            统计信息字典
        """
        return {
            "step_count": self.step_count,
            "config": self.config,
            "created_at": self.created_at,
            "uptime": time.time() - self.created_at,
            "actor_id": ray.get_runtime_context().actor_id,
            "node_id": ray.get_runtime_context().node_id
        }

    def _optimize_large_object(self, obj):
        """大对象优化：超过1MB自动存储到Object Store

        Args:
            obj: 要优化的对象

        Returns:
            优化后的对象或ObjectRef
        """
        try:
            # 检查对象大小
            if isinstance(obj, (str, bytes)) and sys.getsizeof(obj) > 1024 * 1024:  # 1MB
                logger.debug(f"Large object detected ({sys.getsizeof(obj)} bytes), storing to Object Store")
                return ray.put(obj)
            return obj

        except Exception as e:
            logger.warning(f"Large object optimization failed: {e}")
            return obj
    
    def get_reset_result(self):
        return self.obs, self._info

    def close(self):
        """关闭环境（清理资源）"""
        try:
            # 1. 关闭 simulator
            if self.simulator is not None:
                # 需要调用内层的 MinecraftSim.close() 来触发 RecordCallback 保存 MP4
                if hasattr(self.simulator, 'simulator') and hasattr(self.simulator.simulator, 'close'):
                    logger.info(f"[{self.uuid[:8]}] Calling simulator.simulator.close() to trigger recording save")
                    self.simulator.simulator.close()
                elif hasattr(self.simulator, 'close'):
                    logger.info(f"[{self.uuid[:8]}] Calling simulator.close()")
                    self.simulator.close()

            # 2. 释放资源
            if self.resource_manager is not None and self.resources is not None:
                logger.info(f"[{self.uuid[:8]}] Releasing resources...")
                ray.get(
                    self.resource_manager.release_resources.remote(self.uuid)
                )
                self.resources = None

            logger.info(f"[{self.uuid[:8]}] Environment closed successfully")

        except Exception as e:
            logger.error(f"[{self.uuid[:8]}] Close failed: {e}")
            raise