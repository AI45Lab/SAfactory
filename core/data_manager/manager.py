from core.data_manager.strategy.sqlite_startegy_impl import SqliteStrategy
from core.data_manager.strategy_factory import StorageFactory

class DataManager:
    def __init__(
        self,
        storage_type: str = "sqlite", 
        **storage_config
    ):
        self.strategy = None
        
        try:
            print(f"Initializing DataManager with strategy: '{storage_type}'")
            self.strategy = StorageFactory.create(storage_type, **storage_config)
            print(f"DataManager initialized successfully using {self.strategy.__class__.__name__}")
        
        # 异常收口处理：告诉用户类型错误
        except ValueError as e:
            # Factory 抛出不支持该类型 异常
            error_msg = f"Unsupported storage type: '{storage_type}'. Please check registered types."
            print(f"{error_msg} Original Error: {e}")
            raise ValueError(error_msg) from e
        
        except TypeError as e:
            # Factory 找到了类型，但是透传的 **storage_config 参数不对
            error_msg = f"Invalid configuration for storage type '{storage_type}'."
            print(f"{error_msg} Missing or invalid arguments. Original Error: {e}")
            raise ValueError(error_msg) from e
        
        except Exception as e:
            # 具体的 Strategy 内部初始化炸了（比如数据库连不上、鉴权失败等）
            # 收口处理：包装成 RuntimeError
            error_msg = f"Failed to initialize storage strategy '{storage_type}' due to an internal error."
            print(f"{error_msg} Original Error: {e}")
            raise RuntimeError(error_msg) from e

    #所有方法直接委托给strategy
    async def init(self):
        await self.strategy.init()

    async def add_environment_config(self, env_name: str, **user_params):
        return await self.strategy.add_environment_config(env_name, **user_params)
    
    async def get_all_environments(self):
        return await self.strategy.get_all_environments()

    async def create_session(self, env_id, llm_model: str):
        return await self.strategy.create_session(env_id, llm_model)

    async def update_session(self, *args, **kwargs):
        return await self.strategy.update_session(*args, **kwargs)

    async def record_step(self, *args, **kwargs):
        return await self.strategy.record_step(*args, **kwargs)

    async def close(self):
        await self.strategy.close()
        
    def get_sync_connection(self):
        return self.strategy.get_sync_connection()
 
    @property
    def buffer_stats(self):
        # 只有 SqliteStrategy 才有 buffer
        if isinstance(self.strategy, SqliteStrategy):
            return self.strategy.buffer_stats
        return None

    async def fetch_done_steps_with_context(self, *args, **kwargs):
        # buffer相关代码
        if isinstance(self.strategy, SqliteStrategy):
            return await self.strategy.fetch_done_steps_with_context(*args, **kwargs)
        return None

    async def get_max_step_id(self):
        # buffer相关代码
        if isinstance(self.strategy, SqliteStrategy):
            return await self.strategy.get_max_step_id()
        return None
