"""
MCRayClient - Ray版本的环境客户端
与HTTP版本MCEnvClient保持接口兼容性
支持UUID构建和批量创建功能
"""

import ray
import time
import logging
from typing import Union, Optional, Any, List
from pathlib import Path
from uuid import uuid4

logger = logging.getLogger("MCRayClient")

class StepResult:
    """步骤结果封装，与HTTP版本兼容"""
    def __init__(self, observation, reward, done, info):
        self.observation = observation
        self.reward = reward
        self.done = done
        self.info = info

class MCRayClient:
    """Ray版本的MC环境客户端

    接口设计与HTTP版本的MCEnvClient保持完全兼容，
    只需要修改初始化参数即可实现零迁移成本
    """

    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = None,
        ray_address: Optional[str] = None,
        uuid: Optional[str] = None,
        **kwargs
    ):
        """初始化Ray客户端

        支持两种模式：
        1. 传统模式：提供config_path，创建新环境
        2. UUID模式：提供uuid，连接到现有环境

        Args:
            config_path: YAML配置文件路径（传统模式必需）
            ray_address: Ray集群地址，如 'ray://localhost:10001'，None表示本地模式
            uuid: 环境UUID（UUID模式使用）
            **kwargs: 其他配置参数
        """
        self.env_actor = None
        self.created = False

        # 初始化Ray连接
        self._init_ray(ray_address)

        if uuid:
            # UUID模式：连接到现有环境
            self.client_uuid = uuid
            self.is_uuid_mode = True
            self.config = None  # 配置由环境池管理
            logger.info(f"MCRayClient initialized in UUID mode: {uuid}")
        else:
            # 传统模式：创建新环境
            if config_path is None:
                raise ValueError("config_path is required in traditional mode")

            self.client_uuid = str(uuid4())
            self.is_uuid_mode = False
            self.config = self._prepare_config(config_path, **kwargs)
            logger.info(f"MCRayClient initialized in traditional mode with config: {config_path}")

        logger.info(f"MCRayClient initialized with Ray address: {ray_address}")

    @classmethod
    def create_batch(
        cls,
        num_envs: int,
        config_path: Union[str, Path, List[Union[str, Path]]],
        ray_address: Optional[str] = None,
        **kwargs
    ) -> List[str]:
        """批量创建环境，返回UUID列表

        Args:
            num_envs: 环境数量
            config_path: 配置文件路径
                - str/Path: 所有环境使用相同配置
                - List[str/Path]: 每个环境使用对应配置（长度必须等于num_envs）
            ray_address: Ray集群地址
            **kwargs: 其他配置参数

        Returns:
            UUID列表: ["uuid1", "uuid2", "uuid3", "uuid4"]

        Examples:
            # 所有环境使用相同配置
            uuids = MCRayClient.create_batch(4, "config.yaml")

            # 每个环境使用不同配置
            uuids = MCRayClient.create_batch(4, [
                "config1.yaml", "config2.yaml", "config3.yaml", "config4.yaml"
            ])
        """
        try:
            # 初始化Ray连接
            cls._init_ray_static(ray_address)

            # 参数验证和配置处理
            if isinstance(config_path, (str, Path)):
                # 单个配置：复制num_envs次
                configs = [config_path] * num_envs
            elif isinstance(config_path, list):
                # 配置列表：验证长度
                if len(config_path) != num_envs:
                    raise ValueError(
                        f"config_path list length ({len(config_path)}) "
                        f"must equal num_envs ({num_envs})"
                    )
                configs = config_path
            else:
                raise TypeError(
                    f"config_path must be str, Path, or List, got {type(config_path)}"
                )

            # 为每个环境生成UUID和配置
            uuids = []
            env_configs = []

            for i in range(num_envs):
                env_uuid = str(uuid4())
                # 传递UUID到配置准备函数，用于自动设置独立的record_path
                config = cls._prepare_config_static(configs[i], env_uuid=env_uuid, **kwargs)

                uuids.append(env_uuid)
                env_configs.append(config)

            # 在全局环境池中创建环境
            from .global_pool import get_global_env_pool
            env_pool = get_global_env_pool()
            success = ray.get(env_pool.create_envs.remote(uuids, env_configs))

            if not success:
                logger.warning("Some environments failed to create")

            logger.info(f"Created batch of {len(uuids)} environments")
            return uuids

        except Exception as e:
            logger.error(f"Failed to create batch environments: {e}")
            raise

    @staticmethod
    def _init_ray_static(ray_address: Optional[str]):
        """静态Ray初始化方法"""
        try:
            if ray_address:
                if not ray.is_initialized():
                    ray.init(address=ray_address)
                    logger.info(f"Connected to Ray cluster: {ray_address}")
            else:
                if not ray.is_initialized():
                    ray.init()
                    logger.info("Started local Ray instance")

        except Exception as e:
            logger.error(f"Failed to initialize Ray: {e}")
            raise RuntimeError(f"Ray initialization failed: {e}")

    @staticmethod
    def _prepare_config_static(config_path: Union[str, Path], env_uuid: Optional[str] = None, **kwargs):
        """静态配置准备方法

        Args:
            config_path: 配置文件路径
            env_uuid: 环境UUID，如果提供则自动为record_path添加UUID子目录
            **kwargs: 其他配置参数
        """
        try:
            from ..utils import load_simulator_setup_from_yaml
            from pathlib import Path as PathLib

            # load_simulator_setup_from_yaml返回 (callbacks, env_overrides)
            sim_callbacks, env_overrides = load_simulator_setup_from_yaml(config_path)

            # 如果提供了env_uuid，自动为RecordCallback的record_path添加UUID子目录
            # 注意：sim_callbacks已经是实例化的callback对象列表，不是字典
            if env_uuid is not None and sim_callbacks:
                for callback in sim_callbacks:
                    # 检查是否是RecordCallback实例
                    if hasattr(callback, 'record_path'):
                        # 直接修改实例的record_path属性
                        original_path = PathLib(callback.record_path)
                        # 使用UUID前8位作为子目录名
                        uuid_short = env_uuid[:8]
                        new_path = original_path / f"env-{uuid_short}"
                        callback.record_path = new_path
                        # 关键: 创建新的目录
                        callback.record_path.mkdir(parents=True, exist_ok=True)
                        logger.info(f"Modified record_path for env {uuid_short}: {new_path}")

            # 构建完整配置
            config = {
                "sim_callbacks": sim_callbacks,
                **env_overrides,  # 展开env_overrides字典
                **kwargs  # kwargs覆盖
            }

            return config

        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")
            raise

    def _init_ray(self, ray_address: Optional[str]):
        """初始化Ray连接"""
        try:
            if ray_address:
                if not ray.is_initialized():
                    ray.init(address=ray_address)
                    logger.info(f"Connected to Ray cluster: {ray_address}")
            else:
                if not ray.is_initialized():
                    ray.init()
                    logger.info("Started local Ray instance")

        except Exception as e:
            logger.error(f"Failed to initialize Ray: {e}")
            raise RuntimeError(f"Ray initialization failed: {e}")

    def _prepare_config(self, config_path: Union[str, Path], **kwargs):
        """准备环境配置，只从YAML文件加载"""
        try:
            from ..utils import load_simulator_setup_from_yaml

            # load_simulator_setup_from_yaml返回 (callbacks, env_overrides)
            sim_callbacks, env_overrides = load_simulator_setup_from_yaml(config_path)

            # 构建完整配置
            config = {
                "sim_callbacks": sim_callbacks,
                **env_overrides,  # 展开env_overrides字典
                **kwargs  # kwargs覆盖
            }

            return config

        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")
            raise


    def step(self, action: str):
        """执行动作

        Args:
            action: JSON格式的动作字符串

        Returns:
            StepResult对象，包含observation, reward, done, info
        """
        try:
            if not self.created or not self.env_actor:
                raise RuntimeError("Environment not initialized. Call reset() first.")

            result = ray.get(self.env_actor.step.remote(action))
            obs, reward, done, info = result

            return StepResult(
                observation=self._resolve_object(obs),
                reward=reward,
                done=done,
                info=info
            )

        except Exception as e:
            logger.error(f"Step failed: {e}")
            raise

    def reset(self, idx: int = 0):
        """重置环境（符合Gym标准）

        Args:
            idx: 环境索引（兼容性参数）

        Returns:
            重置后的观察
        """
        try:
            # 如果环境还未创建，先获取环境
            if not self.created or not self.env_actor:
                if self.is_uuid_mode:
                    # UUID模式：从全局环境池获取现有环境
                    from .global_pool import get_global_env_pool
                    env_pool = get_global_env_pool()
                    self.env_actor = ray.get(env_pool.get_env_by_uuid.remote(self.client_uuid))
                    logger.info(f"Retrieved environment from pool: {self.client_uuid}")
                else:
                    # 传统模式：创建新环境
                    from .actors import MCEnvActor
                    self.env_actor = MCEnvActor.remote(self.config)
                    logger.info("Created new environment")

                self.created = True

            obs = ray.get(self.env_actor.reset.remote())
            logger.debug("Environment reset successfully")
            return self._resolve_object(obs)

        except Exception as e:
            logger.error(f"Reset failed: {e}")
            raise


    def close(self):
        """关闭环境，清理资源"""
        try:
            if self.env_actor:
                # 无论哪种模式，都需要先调用env_actor.close()来触发RecordCallback保存MP4
                try:
                    ray.get(self.env_actor.close.remote())
                    logger.info("Called env_actor.close() to save recording")
                except Exception as e:
                    logger.warning(f"Failed to call env_actor.close(): {e}")
                
                if self.is_uuid_mode:
                    # UUID模式：归还环境到全局池中
                    from .global_pool import get_global_env_pool
                    env_pool = get_global_env_pool()
                    ray.get(env_pool.return_env.remote(self.client_uuid))
                    logger.info(f"Returned environment to pool: {self.client_uuid}")
                else:
                    # 传统模式：直接销毁环境
                    ray.kill(self.env_actor)
                    logger.info("Environment destroyed")

                self.env_actor = None
                self.created = False

                logger.info("Environment closed successfully")

        except Exception as e:
            logger.error(f"Close failed: {e}")
            raise

    def get_stats(self):
        """获取环境统计信息（仅用于测试和调试）

        Returns:
            统计信息字典
        """
        try:
            if not self.created or not self.env_actor:
                raise RuntimeError("Environment not initialized. Call reset() first.")

            stats = ray.get(self.env_actor.get_stats.remote())
            return stats

        except Exception as e:
            logger.error(f"Get stats failed: {e}")
            raise

    def _resolve_object(self, obj_or_ref):
        """解析对象引用

        如果对象是ObjectRef（大对象存储），则获取其值
        否则直接返回对象
        """
        if isinstance(obj_or_ref, ray.ObjectRef):
            return ray.get(obj_or_ref)
        return obj_or_ref

    def __enter__(self):
        """上下文管理器支持"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器自动清理"""
        self.close()