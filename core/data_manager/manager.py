from core.data_manager.strategy.sqlite_startegy_impl import SqliteStrategy
from core.data_manager.strategy_factory import StorageFactory

class DataManager:
    def __init__(
        self,
        # 指定存储类型
        storage_type: str = "sqlite", 
        
        # 接收所有其他配置参数 (db_url, api_url, buffer_size等)
        **storage_config
    ):
        #无论未来加多少种策略，这里都不用改
        #storage_type: "sqlite" | "cloud" | "kafka" ...
        #**storage_config: 透传给具体 Strategy 的初始化参数
        self.strategy = StorageFactory.create(storage_type, **storage_config)

    #所有方法直接委托给strategy
    async def init(self):
        await self.strategy.init()

    async def add_environment_config(self, env_name: str, **user_params):
        return await self.strategy.add_environment_config(env_name, **user_params)
    
    async def get_all_environments(self):
        return await self.strategy.get_all_environments()

    async def create_session(self, env_config, llm_model: str):
        return await self.strategy.create_session(env_config, llm_model)

    async def update_session(self, *args, **kwargs):
        return await self.strategy.update_session(*args, **kwargs)

    async def record_step(self, *args, **kwargs):
        return await self.strategy.record_step(*args, **kwargs)

    async def close(self):
        await self.strategy.close()
 
    @property
    def buffer_stats(self):
        # 只有 SqliteStrategy 才有 buffer
        if isinstance(self.strategy, SqliteStrategy):
            return self.strategy.buffer_stats
        return None

    async def fetch_done_steps_with_context(self, *args, **kwargs):
        # buffer相关代码
        if isinstance(self.strategy, SqliteStrategy):
            return self.strategy.fetch_done_steps_with_context(*args, **kwargs)
        return None

    async def get_max_step_id(self):
        # buffer相关代码
        if isinstance(self.strategy, SqliteStrategy):
            return self.strategy.get_max_step_id()
        return None
