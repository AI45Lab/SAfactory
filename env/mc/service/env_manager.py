from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any, Dict, Optional

from core.types.base import RenderOutput, ResetOutput, StepOutput
from env.mc.mc_env import MCGym


class EnvManager:
    """Thread-safe singleton wrapper around MCGym."""

    def __init__(self):
        self._lock = threading.RLock()
        self._env: Optional[MCGym] = None
        self._active_config: Optional[str] = None
        self._project_root = Path(__file__).resolve().parents[3]

        self._default_config_input = os.getenv(
            "MC_ENV_CONFIG", "env/mc/config/kill/kill_zombie.yaml"
        )
        self._default_config_path = self._resolve_config_path(self._default_config_input)
        self._jar_path = self._normalize(os.getenv("MINECRAFT_JAR_PATH"))

    @staticmethod
    def _normalize(path: Optional[str]) -> Optional[str]:
        if not path:
            return None
        return os.path.abspath(os.path.expanduser(path))

    def _resolve_config_path(self, config_path: Optional[str]) -> Optional[str]:
        if not config_path:
            return None
        candidate = Path(config_path)
        if not candidate.is_absolute():
            candidate = (self._project_root / candidate).resolve()
        return str(candidate)

    def _ensure_env(self, env_config: Optional[str], force_recreate: bool) -> MCGym:
        with self._lock:
            resolved_config = self._resolve_config_path(env_config) or self._default_config_path
            if resolved_config and not os.path.exists(resolved_config):
                raise FileNotFoundError(
                    f"Env config not found: {resolved_config}. "
                    "Set MC_ENV_CONFIG or provide env_config in the request."
                )

            if force_recreate and self._env is not None:
                self._env.close()
                self._env = None
                self._active_config = None

            if self._env is None or self._active_config != resolved_config:
                kwargs: Dict[str, Any] = {}
                if resolved_config:
                    kwargs["env_config"] = resolved_config
                self._env = MCGym(**kwargs)
                self._active_config = resolved_config

            return self._env

    def reset(
        self,
        *,
        seed: Optional[int],
        env_config: Optional[str],
        force_recreate: bool,
    ) -> ResetOutput:
        env = self._ensure_env(env_config, force_recreate)
        return env.reset(seed=seed)

    def step(self, *, action: Any) -> StepOutput:
        env = self._ensure_env(env_config=None, force_recreate=False)
        return env.step(action)

    def render(self) -> RenderOutput:
        env = self._ensure_env(env_config=None, force_recreate=False)
        return env.render()

    def close(self) -> None:
        with self._lock:
            if self._env is not None:
                self._env.close()
                self._env = None
                self._active_config = None

    def health(self) -> Dict[str, Any]:
        jar_path = self._normalize(os.getenv("MINECRAFT_JAR_PATH")) or self._jar_path
        jar_exists = bool(jar_path and os.path.exists(jar_path))
        with self._lock:
            return {
                "env_ready": self._env is not None,
                "active_config": self._active_config,
                "default_config": self._default_config_path,
                "jar_path": jar_path,
                "jar_exists": jar_exists,
            }

