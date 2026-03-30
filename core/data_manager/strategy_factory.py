from importlib import import_module
from typing import Dict, Type, Tuple, Union

from core.data_manager.strategy.base_strategy import StorageStrategy

StrategyEntry = Union[Type[StorageStrategy], Tuple[str, str]]

class StorageFactory:
    # 注册:存储 "类型字符串" -> "策略类" 或 （模块路径, 类名） 的映射
    _registry: Dict[str, StrategyEntry] = {}

    @classmethod
    def register(cls, name: str, strategy_cls: Type[StrategyEntry]):
        #注册新的存储策略
        cls._registry[name] = strategy_cls
        
    @classmethod
    def register_lazy(cls, name: str, module_path: str, class_name: str):
        # 注册新的存储策略（延迟导入）
        cls._registry[name] = (module_path, class_name)

    @classmethod
    def _resolve_strategy_cls(cls, storage_type: str) -> Type[StorageStrategy]:
        strategy_entry = cls._registry[storage_type]
        if isinstance(strategy_entry, tuple):
            module_path, class_name = strategy_entry
            module = import_module(module_path)
            strategy_cls = getattr(module, class_name)
            cls._registry[storage_type] = strategy_cls
            return strategy_cls
        return strategy_entry

    @classmethod
    def create(cls, job_id: str, storage_type: str, **kwargs) -> StorageStrategy:
        # 根据类型创建实例，kwargs 是透传的配置参数
        if storage_type not in cls._registry:
            raise ValueError(f"Unknown storage type: {storage_type}. Available: {list(cls._registry.keys())}")
        
        strategy_cls = cls._resolve_strategy_cls(storage_type)
        # 实例化策略类，把所有参数传进去
        kwargs["job_id"]=job_id
        return strategy_cls(**kwargs)


StorageFactory.register_lazy(
    "sqlite",
    "core.data_manager.strategy.sqlite_strategy_impl",
    "SqliteStrategy",
)
StorageFactory.register_lazy(
    "cloud",
    "core.data_manager.strategy.cloud_strategy_impl",
    "CloudStrategy",
)
# 未来想加 Kafka，只需要写好 KafkaStrategy 后在这里加一行：
# StorageFactory.register_lazy("kafka", "core.data_manager.strategy.kafka_strategy_impl", "KafkaStrategy")