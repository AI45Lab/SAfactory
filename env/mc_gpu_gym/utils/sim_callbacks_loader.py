import importlib
import inspect
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import yaml

logger = logging.getLogger(__name__)

# Type aliases
Context = Dict[str, Any]
Callback = Callable[[Context], None]


def _load_callable(dotted: str) -> Callable:
    """Dynamically import a callable from a dotted path 'pkg.mod:func' or 'pkg.mod.func'."""
    if ":" in dotted:
        module_path, func_name = dotted.split(":", 1)
    else:
        parts = dotted.split(".")
        module_path, func_name = ".".join(parts[:-1]), parts[-1]
    if not module_path or not func_name:
        raise ValueError(f"Invalid dotted path: {dotted}")
    module = importlib.import_module(module_path)
    fn = getattr(module, func_name)
    if not callable(fn):
        raise TypeError(f"Loaded object is not callable: {dotted}")
    return fn


def _wrap_with_kwargs(fn: Callable, kwargs: Optional[Dict[str, Any]]) -> Callback:
    """Return a callback ctx->None that calls fn(**kwargs, ctx=ctx) or fn(ctx, **kwargs) flexibly."""
    kwargs = kwargs or {}

    def _cb(ctx: Context) -> None:
        sig = inspect.signature(fn)
        try:
            if "ctx" in sig.parameters:
                fn(ctx=ctx, **kwargs)
            else:
                # Try passing ctx as first positional if accepted
                fn(ctx, **kwargs)
        except TypeError:
            # Fallback: no ctx in signature
            fn(**kwargs)
    return _cb


class CallbackManager:
    """Manage and execute simulator lifecycle callbacks loaded from YAML.

    Supported hooks:
      - on_reset: called after env reset
      - on_step: called after each step
      - on_episode_end: called when episode terminates
      - on_init: called once when simulator is created
    """

    def __init__(
        self,
        on_reset: Optional[Sequence[Callback]] = None,
        on_step: Optional[Sequence[Callback]] = None,
        on_episode_end: Optional[Sequence[Callback]] = None,
        on_init: Optional[Sequence[Callback]] = None,
        env_overrides: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._on_reset = list(on_reset or [])
        self._on_step = list(on_step or [])
        self._on_episode_end = list(on_episode_end or [])
        self._on_init = list(on_init or [])
        self.env_overrides = env_overrides or {}

    def run_on_init(self, ctx: Context) -> None:
        for cb in self._on_init:
            try:
                cb(ctx)
            except Exception as e:
                logger.exception(f"on_init callback failed: {e}")

    def run_on_reset(self, ctx: Context) -> None:
        for cb in self._on_reset:
            try:
                cb(ctx)
            except Exception as e:
                logger.exception(f"on_reset callback failed: {e}")

    def run_on_step(self, ctx: Context) -> None:
        for cb in self._on_step:
            try:
                cb(ctx)
            except Exception as e:
                logger.exception(f"on_step callback failed: {e}")

    def run_on_episode_end(self, ctx: Context) -> None:
        for cb in self._on_episode_end:
            try:
                cb(ctx)
            except Exception as e:
                logger.exception(f"on_episode_end callback failed: {e}")


def _normalize_callbacks(spec: Union[str, Dict[str, Any]]) -> Tuple[str, Dict[str, Any]]:
    """Accept 'module:func' or {'import': 'module:func', 'kwargs': {...}} and return (import_path, kwargs)."""
    if isinstance(spec, str):
        return spec, {}
    if isinstance(spec, dict):
        import_path = spec.get("import") or spec.get("callable") or spec.get("path")
        if not import_path:
            raise ValueError(f"Callback spec missing import path: {spec}")
        kwargs = spec.get("kwargs") or {}
        return import_path, kwargs
    raise TypeError(f"Unsupported callback spec: {spec}")


def _load_callbacks_list(items: Optional[Union[List[Any], Any]]) -> List[Callback]:
    if items is None:
        return []
    if not isinstance(items, list):
        items = [items]
    cbs: List[Callback] = []
    for it in items:
        import_path, kwargs = _normalize_callbacks(it)
        fn = _load_callable(import_path)
        cbs.append(_wrap_with_kwargs(fn, kwargs))
    return cbs


def load_callbacks_from_yaml(desc_path: Union[str, Path]) -> CallbackManager:
    """Load callbacks and environment overrides from a YAML description file.

    YAML schema example:

    env:
      hfov: 70
      vfov: 40
      seed: 123

    callbacks:
      on_init:
        - import: mypkg.sim.set_seed
          kwargs: {seed: 123}
      on_reset:
        - import: mypkg.sim.ensure_fov
          kwargs: {hfov: 70}
      on_step:
        - mypkg.sim.auto_attack_if_centered
      on_episode_end:
        - mypkg.sim.log_stats
    """
    desc_path = Path(desc_path)

    def _deep_merge(a: dict, b: dict) -> dict:
        out = dict(a)
        for k, v in (b or {}).items():
            if k in out and isinstance(out[k], dict) and isinstance(v, dict):
                out[k] = _deep_merge(out[k], v)
            elif k in ("sim_callbacks",):
                base_list = out.get(k) or []
                add_list = v or []
                out[k] = list(base_list) + list(add_list)
            else:
                out[k] = v
        return out

    def _load_yaml_with_defaults(path: Path, visited: set[str] | None = None) -> dict:
        visited = visited or set()
        key = str(path.resolve())
        if key in visited:
            raise RuntimeError(f"Cyclic defaults include detected at {path}")
        visited.add(key)
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        defaults = data.get("defaults") or []
        merged = {}
        if isinstance(defaults, list):
            for item in defaults:
                if isinstance(item, str):
                    if item == "_self_":
                        continue
                    base_path = path.parent / f"{item}.yaml"
                    if base_path.exists():
                        base_data = _load_yaml_with_defaults(base_path, visited)
                        merged = _deep_merge(merged, base_data)
                elif isinstance(item, dict):
                    # support {'base': 'xyz'} style if needed
                    for name, rel in item.items():
                        base_path = path.parent / f"{rel}.yaml"
                        if base_path.exists():
                            base_data = _load_yaml_with_defaults(base_path, visited)
                            merged = _deep_merge(merged, base_data)
        # finally overlay current
        merged = _deep_merge(merged, data)
        return merged

    data = _load_yaml_with_defaults(desc_path)

    env_overrides = data.get("env") or {}
    cb_cfg = data.get("callbacks") or {}

    mgr = CallbackManager(
        on_init=_load_callbacks_list(cb_cfg.get("on_init")),
        on_reset=_load_callbacks_list(cb_cfg.get("on_reset")),
        on_step=_load_callbacks_list(cb_cfg.get("on_step")),
        on_episode_end=_load_callbacks_list(cb_cfg.get("on_episode_end")),
        env_overrides=env_overrides,
    )
    logger.info(
        "Loaded callbacks from %s with env_overrides=%s (init=%d, reset=%d, step=%d, end=%d)",
        str(desc_path), env_overrides, len(mgr._on_init), len(mgr._on_reset), len(mgr._on_step), len(mgr._on_episode_end)
    )
    return mgr


def load_simulator_setup_from_yaml(desc_path: Union[str, Path]) -> tuple[list, dict]:
    """Build simulator (MinecraftSim) callbacks and env_overrides from YAML.

    YAML schema extension:
      sim_callbacks:
        - import: verl.workers.agent.envs.mc.MineStudio.minestudio.simulator.callbacks.SummonMobsCallback
          kwargs: {mobs: [{name: zombie, number: 3}]}
        - verl.workers.agent.envs.mc.MineStudio.minestudio.simulator.callbacks.MaskActionsCallback
    """

    desc_path = Path(desc_path)


    # Reuse defaults-aware loader from above
    def _deep_merge(a: dict, b: dict) -> dict:
        out = dict(a)
        for k, v in (b or {}).items():
            if k in out and isinstance(out[k], dict) and isinstance(v, dict):
                out[k] = _deep_merge(out[k], v)
            elif k in ("sim_callbacks",):
                base_list = out.get(k) or []
                add_list = v or []
                out[k] = list(base_list) + list(add_list)
            else:
                out[k] = v
        return out

    def _load_yaml_with_defaults(path: Path, visited: set[str] | None = None) -> dict:
        visited = visited or set()
        key = str(path.resolve())
        if key in visited:
            raise RuntimeError(f"Cyclic defaults include detected at {path}")
        visited.add(key)
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        defaults = data.get("defaults") or []
        merged = {}
        if isinstance(defaults, list):
            for item in defaults:
                if isinstance(item, str):
                    if item == "_self_":
                        continue
                    base_path = path.parent / f"{item}.yaml"
                    if base_path.exists():
                        base_data = _load_yaml_with_defaults(base_path, visited)
                        merged = _deep_merge(merged, base_data)
                elif isinstance(item, dict):
                    for name, rel in item.items():
                        base_path = path.parent / f"{rel}.yaml"
                        if base_path.exists():
                            base_data = _load_yaml_with_defaults(base_path, visited)
                            merged = _deep_merge(merged, base_data)
        merged = _deep_merge(merged, data)
        return merged

    data = _load_yaml_with_defaults(desc_path)

    env_overrides = data.get("env") or {}
    # Propagate top-level resolution/obs_size into env_overrides for convenience
    try:
        top_resolution = data.get("resolution")
        if "resolution" not in env_overrides and top_resolution is not None:
            env_overrides["resolution"] = top_resolution
        top_obs_size = data.get("obs_size")
        if top_obs_size and isinstance(top_obs_size, (list, tuple)) and len(top_obs_size) == 2:
            # map obs_size directly to resolution if not set
            env_overrides.setdefault("resolution", top_obs_size)
        # Some configs use nested 'sim' section
        sim_cfg = data.get("sim") or {}
        if sim_cfg:
            if "resolution" in sim_cfg and "resolution" not in env_overrides:
                env_overrides["resolution"] = sim_cfg["resolution"]
            if "obs_size" in sim_cfg and "resolution" not in env_overrides:
                osv = sim_cfg["obs_size"]
                if isinstance(osv, (list, tuple)) and len(osv) == 2:
                    env_overrides["resolution"] = osv
        # Propagate seed ONLY from env (ignore top-level or sim-level)
        if "seed" not in env_overrides:
            env_seed = (data.get("env") or {}).get("seed")
            if env_seed is not None:
                env_overrides["seed"] = env_seed
        # Fallback: propagate top-level seed if not provided under env
        if "seed" not in env_overrides:
            top_seed = data.get("seed")
            if top_seed is not None:
                env_overrides["seed"] = top_seed
        # Propagate teleport coordinates
        tp = data.get("teleport")
        if tp is None and sim_cfg:
            tp = sim_cfg.get("teleport")
        if tp is None:
            tp = (data.get("env") or {}).get("teleport")
        if isinstance(tp, dict) and all(k in tp for k in ("x","y","z")) and "teleport" not in env_overrides:
            env_overrides["teleport"] = {"x": tp["x"], "y": tp["y"], "z": tp["z"]}
    except Exception:
        pass

    sim_cb_specs = data.get("sim_callbacks") or []
    if not isinstance(sim_cb_specs, list):
        sim_cb_specs = [sim_cb_specs]

    cb_instances: list = []
    for spec in sim_cb_specs:
        import_path, kwargs = _normalize_callbacks(spec)
        cls_or_fn = _load_callable(import_path)
        try:
            inst = cls_or_fn(**(kwargs or {}))
        except TypeError:
            # no kwargs
            inst = cls_or_fn()
        cb_instances.append(inst)

    # Auto-inject SummonMobsCallback if 'mobs' section exists
    mobs_config = data.get("mobs")
    if mobs_config and isinstance(mobs_config, list) and len(mobs_config) > 0:
        try:
            summon_mobs_cls = _load_callable(
                "verl.workers.agent.envs.mc.MineStudio.minestudio.simulator.callbacks.SummonMobsCallback"
            )
            cb_instances.append(summon_mobs_cls(mobs=mobs_config))
            logger.info(f"Auto-injected SummonMobsCallback with {len(mobs_config)} mob type(s)")
        except Exception as e:
            logger.warning(f"Failed to inject SummonMobsCallback: {e}")

    # Auto-inject RewardsCallback if 'reward_conf' section exists
    reward_config = data.get("reward_conf")
    if reward_config and isinstance(reward_config, list) and len(reward_config) > 0:
        try:
            rewards_cls = _load_callable(
                "verl.workers.agent.envs.mc.MineStudio.minestudio.simulator.callbacks.RewardsCallback"
            )
            cb_instances.append(rewards_cls(reward_cfg=reward_config))
            logger.info(f"Auto-injected RewardsCallback with {len(reward_config)} reward(s)")
        except Exception as e:
            logger.warning(f"Failed to inject RewardsCallback: {e}")

    # Auto-inject CommandsCallback if 'command' section exists
    commands_config = data.get("command")
    if commands_config and isinstance(commands_config, list) and len(commands_config) > 0:
        try:
            commands_cls = _load_callable(
                "verl.workers.agent.envs.mc.MineStudio.minestudio.simulator.callbacks.CommandsCallback"
            )
            cb_instances.append(commands_cls(commands=commands_config))
            logger.info(f"Auto-injected CommandsCallback with {len(commands_config)} command(s)")
        except Exception as e:
            logger.warning(f"Failed to inject CommandsCallback: {e}")

    # Auto-inject InitInventoryCallback if 'init_inventory' section exists
    init_inventory_config = data.get("init_inventory")
    if init_inventory_config and isinstance(init_inventory_config, list) and len(init_inventory_config) > 0:
        try:
            init_inventory_cls = _load_callable(
                "verl.workers.agent.envs.mc.MineStudio.minestudio.simulator.callbacks.InitInventoryCallback"
            )
            cb_instances.append(init_inventory_cls(init_inventory=init_inventory_config))
            logger.info(f"Auto-injected InitInventoryCallback with {len(init_inventory_config)} item(s)")
        except Exception as e:
            logger.warning(f"Failed to inject InitInventoryCallback: {e}")

    # # Auto-inject teleport as CommandsCallback if 'teleport' section exists
    # tp = data.get("teleport")
    # if isinstance(tp, dict) and all(k in tp for k in ("x","y","z")):
    #     try:
    #         cmd = f"/tp @p {tp['x']} {tp['y']} {tp['z']}"
    #         commands_cb_cls = _load_callable(
    #             "verl.workers.agent.envs.mc.MineStudio.minestudio.simulator.callbacks.CommandsCallback"
    #         )
    #         cb_instances.append(commands_cb_cls(commands=[cmd]))
    #         logger.info(f"Auto-injected teleport CommandsCallback: {cmd}")
    #     except Exception as e:
    #         logger.warning(f"Failed to inject teleport CommandsCallback: {e}")

    return cb_instances, env_overrides


__all__ = [
    "CallbackManager",
    "load_callbacks_from_yaml",
    "load_simulator_setup_from_yaml",
] 