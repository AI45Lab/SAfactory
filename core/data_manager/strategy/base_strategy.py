from abc import ABC, abstractmethod
from typing import Any, Optional

class StorageStrategy(ABC):
    @abstractmethod
    async def init(self):      
        pass

    @abstractmethod
    async def add_environment_config(self, env_name: str, **user_params):
        pass
    
    @abstractmethod
    async def get_all_environments(self):
        pass

    @abstractmethod
    async def create_session(self, env_config, llm_model: str):   
        pass

    @abstractmethod
    async def update_session(self, session, trajectory, total_reward, is_completed):   
        pass

    @abstractmethod
    async def record_step(self, session, step_id, prompt, response, reward, env_state, done):
        pass

    @abstractmethod
    async def close(self):
        pass
    
    @abstractmethod
    def get_sync_connection(self):
        pass