import importlib
from typing import Any

import ray


@ray.remote(max_concurrency=1)
class GenericEnvActor:
    """
    A single actor hosts one env instance.
    The env class is loaded at runtime via 'module:Class' entrypoint.
    The env must implement: reset(), step(action), render(), close(), is_done(), health()
    """
    def __init__(self, envname: str, id_: int, entrypoint: str, create_kwargs: dict | None = None):
        self.envname = envname
        self.id = int(id_)
        module_name, cls_name = entrypoint.split(":")
        module = importlib.import_module(module_name)
        EnvCls = getattr(module, cls_name)
        self.env = EnvCls(id_=self.id, **(create_kwargs or {}))

    # delegate API
    def reset(self) -> dict:  return self.env.reset()
    def step(self, action: Any = None) -> dict:  return self.env.step(action)
    def render(self) -> dict: return self.env.render()
    def close(self) -> dict:  return self.env.close()
    def is_done(self) -> bool: return self.env.is_done()
    def health(self) -> bool:  return self.env.health()
    def describe(self) -> dict:
        return {"env": self.envname, "id": self.id, "class": self.env.__class__.__name__, "done": self.env.is_done()}


