"""
Multiprocess environment pool.

This mirrors the public surface of ``myray.pool.GPUEnvPool`` but runs
each environment inside a dedicated ``multiprocessing.Process`` so
multiple step requests can execute truly in parallel.
"""

import logging
import multiprocessing as mp
import queue
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .resource_manager import GlobalResourceManager


logger = logging.getLogger("MPEnvPool")


def _ensure_sys_path():
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    minestudio_path = project_root / "MineStudio_gpu"
    if minestudio_path.exists() and str(minestudio_path) not in sys.path:
        sys.path.insert(0, str(minestudio_path))


def _env_worker(uuid: str, config: dict, command_q: mp.Queue, result_q: mp.Queue,
                reset_q: mp.Queue, resources: Dict[str, Any], mc_root: Optional[str]):
    """Worker loop that owns a single MCSimulator instance."""
    try:
        _ensure_sys_path()
        from mc_simulator import MCSimulator

        simulator = MCSimulator(
            config=config,
            display_port=resources.get("display_port"),
            working_dir=resources.get("working_dir"),
            mc_root=mc_root,
        )
    except Exception as e:
        logger.exception("[Env %s] init failed", uuid[:8])
        reset_q.put(("__error__", str(e)))
        return

    while True:
        msg = command_q.get()
        if msg is None:
            break

        cmd = msg.get("cmd")
        t0 = time.time()
        try:
            if cmd == "reset":
                obs, info = simulator.reset()
                reset_q.put((obs, info))
            elif cmd == "fast_reset":
                config = msg.get("config")
                obs, info = simulator.fast_reset(config)
                reset_q.put((obs, info))
            elif cmd == "step":
                action = msg.get("action")
                result_q.put(simulator.step(action))
            elif cmd == "close":
                break
            elif cmd == "fast_close":
                break
        except Exception as e:
            logger.exception("[Env %s] %s failed", uuid[:8], cmd)
            target_q = reset_q if cmd == "reset" else result_q
            target_q.put(("__error__", str(e)))
        finally:
            t1 = time.time()
            logger.info("[Env %s] cmd=%s dt=%.3fs", uuid[:8], cmd, t1 - t0)

    try:
        simulator.close()
    except Exception:
        pass


@dataclass
class EnvProcessContext:
    uuid: str
    process: mp.Process
    command_q: mp.Queue
    result_q: mp.Queue
    reset_q: mp.Queue
    status: str = "idle"  # idle / busy / failed
    ready: bool = False
    created_time: float = field(default_factory=time.time)
    resources: Dict[str, Any] = field(default_factory=dict)
    last_reset: Optional[Tuple[Any, Any]] = None


class MPEnvPool:
    """Manage multiple MCSimulator processes with a Ray-compatible API surface."""

    def __init__(self, resource_manager: Optional[GlobalResourceManager] = None):
        self.resource_manager = resource_manager or GlobalResourceManager(
            display_port_range=(1, 100),
            working_dir_base="/mnt/shared-storage-user/steai_share/luozhihao/mc_test/raycraft/record",
        )
        self.env_registry: Dict[str, EnvProcessContext] = {}
        self.step_res: Dict[str, Tuple] = {}

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def create_envs(self, env_list_id: int, uuids: List[str], configs: List[dict]) -> bool:
        ctx = mp.get_context("fork")
        for idx, (env_id, config) in enumerate(zip(uuids, configs)):
            resources = self.resource_manager.allocate_resources(env_id)

            command_q: mp.Queue = ctx.Queue()
            result_q: mp.Queue = ctx.Queue()
            reset_q: mp.Queue = ctx.Queue()

            # Keep mc_root per env to mirror the ray actor behaviour.
            mc_root = f"/mnt/shared-storage-user/steai_share/luozhihao/mc_test/env{env_list_id}/{idx}/.minecraft"

            proc = ctx.Process(
                target=_env_worker,
                args=(env_id, config, command_q, result_q, reset_q, resources, mc_root),
                daemon=True,
            )
            proc.start()

            self.env_registry[env_id] = EnvProcessContext(
                uuid=env_id,
                process=proc,
                command_q=command_q,
                result_q=result_q,
                reset_q=reset_q,
                status="idle",
                resources=resources,
            )

        return True

    def destroy_env(self, uuid: str) -> bool:
        ctx = self.env_registry.get(uuid)
        if not ctx:
            return False

        try:
            ctx.command_q.put({"cmd": "close"})
            ctx.command_q.put(None)
            ctx.process.join(timeout=10)
        except Exception as e:
            logger.warning("[Env %s] failed to close cleanly: %s", uuid[:8], e)

        self.resource_manager.release_resources(uuid)
        self.env_registry.pop(uuid, None)
        self.step_res.pop(uuid, None)
        return True

    def fast_destroy_env(self, uuid: str) -> bool:
        ctx = self.env_registry.get(uuid)
        if not ctx:
            return False

        try:
            ctx.command_q.put({"cmd": "fast_close"})
            ctx.command_q.put(None)
            ctx.process.join(timeout=10)
        except Exception as e:
            logger.warning("[Env %s] failed to fast close cleanly: %s", uuid[:8], e)

        return True

    def batch_destroy(self, uuids: List[str]) -> int:
        return sum(1 for env_id in uuids if self.destroy_env(env_id))

    def get_num_envs(self) -> int:
        return len(self.env_registry)

    # ------------------------------------------------------------------ #
    # State helpers
    # ------------------------------------------------------------------ #
    def _drain_reset_queue(self, ctx: EnvProcessContext):
        while True:
            try:
                res = ctx.reset_q.get_nowait()
            except queue.Empty:
                break

            if res and isinstance(res, tuple) and res[0] == "__error__":
                ctx.status = "failed"
                ctx.last_reset = None
                logger.error("[Env %s] reset error: %s", ctx.uuid[:8], res[1])
                continue

            ctx.last_reset = res
            ctx.ready = True
            ctx.status = "idle"

    def _drain_result_queue(self, ctx: EnvProcessContext):
        while True:
            try:
                res = ctx.result_q.get_nowait()
            except queue.Empty:
                break

            if res and isinstance(res, tuple) and res[0] == "__error__":
                ctx.status = "failed"
                logger.error("[Env %s] step error: %s", ctx.uuid[:8], res[1])
                continue

            self.step_res[ctx.uuid] = res
            ctx.status = "idle"

    def _drain_all(self):
        for ctx in self.env_registry.values():
            self._drain_reset_queue(ctx)
            self._drain_result_queue(ctx)

    # ------------------------------------------------------------------ #
    # Queries
    # ------------------------------------------------------------------ #
    def env_exists(self, uuid: str) -> bool:
        return uuid in self.env_registry and self.env_registry[uuid].status != "failed"

    def get_env(self, uuid: str) -> Optional[EnvProcessContext]:
        return self.env_registry.get(uuid)

    def get_env_by_uuid(self, uuid: str) -> Optional[EnvProcessContext]:
        return self.get_env(uuid)

    def get_env_status(self, uuid: str) -> Dict[str, Any]:
        ctx = self.env_registry.get(uuid)
        if not ctx:
            return {"status": "not_found", "exists": False}

        return {
            "status": ctx.status,
            "exists": True,
            "created_time": ctx.created_time,
        }

    def is_env_ready(self, uuid: str) -> bool:
        ctx = self.env_registry.get(uuid)
        if not ctx:
            return False

        self._drain_reset_queue(ctx)
        return ctx.ready

    def get_pool_stats(self) -> Dict:
        total_envs = len(self.env_registry)
        idle_envs = sum(1 for c in self.env_registry.values() if c.status == "idle")
        busy_envs = sum(1 for c in self.env_registry.values() if c.status == "busy")
        failed_envs = sum(1 for c in self.env_registry.values() if c.status == "failed")

        return {
            "total_environments": total_envs,
            "idle_environments": idle_envs,
            "busy_environments": busy_envs,
            "failed_environments": failed_envs,
            "environment_list": list(self.env_registry.keys()),
        }

    # ------------------------------------------------------------------ #
    # Reset / step
    # ------------------------------------------------------------------ #
    def trigger_reset(self, uuid: str):
        ctx = self.env_registry.get(uuid)
        if not ctx:
            return
        ctx.status = "busy"
        ctx.command_q.put({"cmd": "reset"})

    def reset_env(self, uuid: str, config, timeout: Optional[float] = None):
        ctx = self.env_registry.get(uuid)
        if not ctx:
            raise ValueError(f"Environment {uuid} not found")

        ctx.status = "busy"
        ctx.command_q.put({"cmd": "fast_reset", "config": config})
        # try:
        #     res = ctx.reset_q.get(timeout=timeout)
        # except queue.Empty:
        #     ctx.status = "failed"
        #     raise TimeoutError(f"Reset timeout for {uuid}")

        # if res and isinstance(res, tuple) and res[0] == "__error__":
        #     ctx.status = "failed"
        #     raise RuntimeError(res[1])

        # ctx.last_reset = res
        # ctx.ready = True
        # ctx.status = "idle"
        # return res

    def get_reset_result(self, uuid: str) -> Tuple[Any, Any]:
        ctx = self.env_registry.get(uuid)
        if not ctx:
            raise ValueError(f"Environment {uuid} not found")
        self._drain_reset_queue(ctx)
        return ctx.last_reset if ctx.last_reset else (None, None)

    def step(self, uuid: str, action: str):
        ctx = self.env_registry.get(uuid)
        if not ctx:
            raise ValueError(f"Environment {uuid} not found")
        ctx.status = "busy"
        ctx.command_q.put({"cmd": "step", "action": action})

    def step_sync(self, uuid: str, action: str, timeout: Optional[float] = None):
        ctx = self.env_registry.get(uuid)
        if not ctx:
            raise ValueError(f"Environment {uuid} not found")
        ctx.status = "busy"
        ctx.command_q.put({"cmd": "step", "action": action})
        try:
            res = ctx.result_q.get(timeout=timeout)
        except queue.Empty:
            ctx.status = "failed"
            raise TimeoutError(f"Step timeout for {uuid}")

        if res and isinstance(res, tuple) and res[0] == "__error__":
            ctx.status = "failed"
            raise RuntimeError(res[1])

        ctx.status = "idle"
        return res

    def get_step_res(self) -> Dict[str, Tuple]:
        self._drain_all()
        return dict(self.step_res)

    def rm_step_res(self, uuid: str):
        self.step_res.pop(uuid, None)

    def return_env(self, uuid: str) -> bool:
        ctx = self.env_registry.get(uuid)
        if not ctx:
            return False
        if ctx.status != "failed":
            ctx.status = "idle"
        return True

    def close_env(self, uuid: str) -> bool:
        return self.destroy_env(uuid)
    def fast_close_env(self, uuid: str) -> bool:
        return self.fast_destroy_env(uuid)

