import os
import sys
current_file_path = os.path.abspath(__file__)
examples_dir = os.path.dirname(current_file_path)
project_root = os.path.dirname(examples_dir)
sys.path.append(project_root)

import importlib
from typing import Any, Optional

import ray

from core.types.base import ResetOutput, StepOutput,RenderOutput, dumps_json_bytes


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
        self.env = EnvCls(**(create_kwargs or {}))

    # delegate API
    def reset(self, seed: Optional[int] = None) -> bytes:
        out: ResetOutput= self.env.reset(seed=seed)
        return dumps_json_bytes(out)
    def step(self, action: str) -> bytes:
        out: StepOutput=self.env.step(action)
        return dumps_json_bytes(out)
    def render(self) -> bytes:
        out: RenderOutput=self.env.render()
        return dumps_json_bytes(out)
    def close(self) -> dict:  return self.env.close()
    def is_done(self) -> bool: return self.env.is_done()
    def health(self) -> bool:  return self.env.health()
    def describe(self) -> dict:
        return {"env": self.envname, "id": self.id, "class": self.env.__class__.__name__, "done": self.env.is_done()}


