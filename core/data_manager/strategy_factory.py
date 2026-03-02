from typing import Dict, Type
from core.data_manager.strategy.base_strategy import StorageStrategy
from core.data_manager.strategy.sqlite_startegy_impl import SqliteStrategy
from core.data_manager.strategy.cloud_startegy_impl import CloudStrategy

class StorageFactory:
    # 注册:存储 "类型字符串" -> "策略类" 的映射
    _registry: Dict[str, Type[StorageStrategy]] = {}

    @classmethod
    def register(cls, name: str, strategy_cls: Type[StorageStrategy]):
        #注册新的存储策略
        cls._registry[name] = strategy_cls

    @classmethod
    def create(cls, job_id: str, storage_type: str, **kwargs) -> StorageStrategy:
        #根据类型创建实例，kwargs 是透传的配置参数
        if storage_type not in cls._registry:
            raise ValueError(f"Unknown storage type: {storage_type}. Available: {list(cls._registry.keys())}")
        
        strategy_cls = cls._registry[storage_type]
        # 实例化策略类，把所有参数传进去
        kwargs["job_id"]=job_id
        return strategy_cls(**kwargs)


StorageFactory.register("sqlite", SqliteStrategy)
StorageFactory.register("cloud", CloudStrategy)
# 未来想加 Kafka，只需要写好 KafkaStrategy 后在这里加一行：
# StorageFactory.register("kafka", KafkaStrategy)