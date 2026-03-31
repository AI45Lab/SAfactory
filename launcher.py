import os
import sys
import argparse
import asyncio
import subprocess
import time
import uuid
import math

import requests
import yaml
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
from typing import Any, Dict, Optional, Set, Tuple, List

from utils import ActorPoolRuntimeState, discover_ready_actor_from_snapshot
from core.llm import StaticBaseURLProvider, SessionSuffixBaseURLProvider
from core.data_manager.manager import DataManager
from core.data_manager.yaml_aggregator import all_env_yaml_load, sync_configs_to_db, wait_for_pending_inserts

from interactor import Interactor, ActorHandle, ActorPool
from manager import EnvPoolManager


log = logging.getLogger("launcher")


# -------------------------
# Logging setup
# -------------------------
class ConsoleFilter(logging.Filter):
    """
    Gate on the console (stdout) handler.

    Rules (applied in order):
    1. INFO/DEBUG from core.llm.* are always suppressed.
    2. DEBUG records are suppressed unless the logger name is in *debug_loggers*.
    3. Everything else passes through.
    """
    def __init__(self, debug_loggers: Optional[List[str]] = None):
        super().__init__()
        self._debug_loggers: Set[str] = set(debug_loggers or [])

    def filter(self, record: logging.LogRecord) -> bool:
        if record.name.startswith("core.llm") and record.levelno < logging.WARNING:
            return False
        if record.levelno < logging.INFO and record.name not in self._debug_loggers:
            return False
        return True


def _setup_logging(
    *,
    log_dir: str,
    run_name: Optional[str],
    console_level: str = "INFO",
    file_level: str = "DEBUG",
    max_bytes: int = 50 * 1024 * 1024,
    backup_count: int = 5,
    debug_loggers: Optional[List[str]] = None,
) -> str:
    os.makedirs(log_dir, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = (run_name or "").strip()
    base = f"{run_name}-" if run_name else ""
    log_path = os.path.join(log_dir, f"{base}run-{stamp}.log")

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)  # let handlers filter

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler(stream=sys.stdout)
    # When debug_loggers are specified, lower the handler level so DEBUG records
    # reach the filter; ConsoleFilter then allowlists only the named loggers.
    ch_level = logging.DEBUG if debug_loggers else getattr(logging, console_level.upper(), logging.INFO)
    ch.setLevel(ch_level)
    ch.setFormatter(fmt)
    ch.addFilter(ConsoleFilter(debug_loggers))

    # File handler (keeps everything)
    fh = RotatingFileHandler(
        log_path,
        maxBytes=int(max_bytes),
        backupCount=int(backup_count),
        encoding="utf-8",
    )
    fh.setLevel(getattr(logging, file_level.upper(), logging.DEBUG))
    fh.setFormatter(fmt)

    # Avoid duplicated handlers
    root.handlers.clear()
    root.addHandler(ch)
    root.addHandler(fh)

    # Optional: reduce other noisy libs on console too
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("core.llm").setLevel(logging.WARNING)
    logging.getLogger("core.llm.base").setLevel(logging.WARNING)

    # Keep warnings in logs
    logging.captureWarnings(True)

    root.info(
        "logging initialized: console_level=%s file_level=%s log_path=%s",
        console_level.upper(), file_level.upper(), log_path
    )
    return log_path

# -------------------------
# Utils
# -------------------------
def _rebuild_sqlite_db(db_url: str) -> None:
    """Delete the SQLite file so Tortoise recreates all tables from scratch."""
    if not db_url.startswith("sqlite://"):
        return
    file_path = db_url[len("sqlite://"):].split("?")[0]
    if file_path and os.path.exists(file_path):
        os.remove(file_path)
        log.info("Removed existing SQLite DB for rebuild: %s", file_path)
    else:
        log.info("No existing SQLite DB found to remove (path=%s)", file_path)


def start_local_upstream_service(
    host: str,
    port: int,
    *,
    uvicorn_app: str,
    cwd: Optional[str] = None,
    app_dir: Optional[str] = None,          # ✅ add
    stdout_path: Optional[str] = None,
) -> subprocess.Popen:
    """
    Start local upstream env HTTP service (the one that exposes /envs + /reset/step/close).
    """
    cmd = [sys.executable, "-m", "uvicorn", uvicorn_app, "--host", host, "--port", str(port)]

    log.info("starting local upstream: %s", " ".join(cmd))

    env = os.environ.copy()

    if stdout_path:
        os.makedirs(os.path.dirname(stdout_path) or ".", exist_ok=True)
        f = open(stdout_path, "a", encoding="utf-8")
        proc = subprocess.Popen(cmd, cwd=cwd or os.getcwd(), env=env, stdout=f, stderr=f)
        log.info("local upstream pid=%s stdout/stderr -> %s", proc.pid, stdout_path)
        return proc

    proc = subprocess.Popen(cmd, cwd=cwd or os.getcwd(), env=env)
    log.info("local upstream pid=%s", proc.pid)
    return proc



async def wait_http_ready(base_url: str, timeout_s: float = 30.0) -> None:
    url = base_url.rstrip("/") + "/envs"
    deadline = time.time() + timeout_s
    attempt = 0

    while time.time() < deadline:
        attempt += 1
        try:
            def _probe():
                r = requests.get(url, timeout=2.0)
                return r.status_code == 200, r.status_code

            ok, code = await asyncio.to_thread(_probe)
            if ok:
                log.info("HTTP upstream ready: %s (status=%s)", url, code)
                return
            log.debug("HTTP upstream not ready: %s (status=%s) attempt=%d", url, code, attempt)
        except Exception as e:
            log.debug("HTTP upstream probe error: %s attempt=%d err=%s", url, attempt, e)

        await asyncio.sleep(0.3)

    raise TimeoutError(f"HTTP service not ready: {url}")


async def stop_process(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    log.info("stopping process pid=%s ...", proc.pid)
    proc.terminate()
    try:
        await asyncio.to_thread(proc.wait, timeout=10)
    except Exception:
        log.warning("terminate timeout; killing pid=%s", proc.pid)
        proc.kill()


def _set_nested(cfg: Dict[str, Any], path: List[str], value: Any) -> None:
    cur: Dict[str, Any] = cfg
    for k in path[:-1]:
        nxt = cur.get(k)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[k] = nxt
        cur = nxt
    cur[path[-1]] = value


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid yaml root (expected dict): {path}")
    return cfg


# -------------------------
# Adapter Pool: EnvPoolManager -> ActorPool
# -------------------------
class ManagerActorPool(ActorPool):
    """
    Wrap EnvPoolManager using only PUBLIC APIs:
      - start()
      - list_pool_actors()
      - get_actor_route()
      - close_and_refill()
      - close_all()
    """

    def __init__(self, mgr: EnvPoolManager, *, pool_size: Optional[int] = None):
        self.mgr = mgr

        ps = pool_size
        if ps is None:
            for attr in ("pool_size", "_pool_size"):
                v = getattr(mgr, attr, None)
                if v is not None:
                    try:
                        ps = int(v)
                        break
                    except Exception:
                        pass
        self.pool_size = max(1, int(ps or 1))

        self._runtime = ActorPoolRuntimeState[ActorHandle]()
        self._bg_refill_tasks: Set[asyncio.Task] = set()
        self._lifecycle_lock = asyncio.Lock()
        self._closing = False
        self._closed = False

    @staticmethod
    def _route_to_base_url(route: Any) -> Optional[str]:
        if route is None:
            return None
        if isinstance(route, (tuple, list)) and len(route) == 2:
            host, port = route
            return f"http://{str(host)}:{int(port)}"
        host = getattr(route, "head_ip", None) or getattr(route, "host", None)
        port = getattr(route, "port", None)
        if host is None or port is None:
            return None
        return f"http://{str(host)}:{int(port)}"

    def _actor_base_url(self, env: str, env_id: str) -> Optional[str]:
        route = self.mgr.get_actor_route(env, env_id)
        return self._route_to_base_url(route)

    async def start(self) -> None:
        log.info("ManagerActorPool.start(): starting EnvPoolManager ...")
        await self.mgr.start()

        actors = await self.mgr.list_pool_actors()
        log.info("ManagerActorPool.start(): manager reports %d warmed actors", len(actors))

        for item in actors:
            env = str(item["env_name"])
            env_id = str(item["env_id"])
            group_id = str(item.get("group_id") or "")
            base_url = self._actor_base_url(env, env_id)
            if base_url:
                added = await self._runtime.add_ready_actor(
                    (env, env_id),
                    ActorHandle(base_url=base_url, env_name=env, env_id=env_id, group_id=group_id),
                )
                if added:
                    log.debug("enqueued actor: %s/%s -> %s", env, env_id, base_url)
            else:
                log.warning("actor route missing for %s/%s (skipped)", env, env_id)

        await self._runtime.mark_initial_load_done()

    async def acquire(self) -> Optional[ActorHandle]:
        actor = await self._runtime.acquire()
        if actor is None:
            log.debug("acquire(): exhausted")
        else:
            log.debug("acquire(): got %s/%s", actor.env_name, actor.env_id)
        return actor

    async def done(self, actor: ActorHandle) -> None:
        old_key = (str(actor.env_name), str(actor.env_id))
        async with self._lifecycle_lock:
            if self._closing or self._closed:
                log.info(
                    "done(): pool is closing; skip refill for env=%s id=%s",
                    actor.env_name,
                    actor.env_id,
                )
                await self._finish_without_replacement(old_key)
                return

            log.info(
                "done(): scheduling background close_and_refill for env=%s id=%s",
                actor.env_name,
                actor.env_id,
            )
            await self._runtime.begin_refill(old_key)
            try:
                task = asyncio.create_task(
                    self._bg_done(actor, old_key),
                    name=f"manager-close-and-refill:{actor.env_name}:{actor.env_id}",
                )
            except Exception:
                await self._runtime.fail_refill(old_key)
                raise
            self._register_bg_task(task)

    async def _finish_without_replacement(self, old_key: Tuple[str, str]) -> None:
        await self._runtime.begin_refill(old_key)
        await self._runtime.fail_refill(old_key)

    def _register_bg_task(self, task: asyncio.Task) -> None:
        self._bg_refill_tasks.add(task)
        task.add_done_callback(self._on_bg_task_done)

    def _on_bg_task_done(self, task: asyncio.Task) -> None:
        self._bg_refill_tasks.discard(task)
        try:
            task.exception()
        except asyncio.CancelledError:
            log.debug("background refill task cancelled: %s", task.get_name())
        except Exception:
            log.exception("background refill task failed unexpectedly: %s", task.get_name())

    async def _bg_done(self, actor: ActorHandle, old_key: Tuple[str, str]) -> None:
        try:
            await self.mgr.close_and_refill(actor.env_name, actor.env_id)
        except asyncio.CancelledError:
            log.info(
                "_bg_done cancelled during shutdown for %s/%s",
                actor.env_name,
                actor.env_id,
            )
            await self._runtime.fail_refill(old_key)
            raise
        except Exception:
            log.exception("close_and_refill failed for %s/%s", actor.env_name, actor.env_id)
            await self._runtime.fail_refill(old_key)
            return

        actors = await self.mgr.list_pool_actors()
        known_keys = await self._runtime.known_keys_snapshot()
        new_key, new_actor = discover_ready_actor_from_snapshot(
            actors,
            known_keys=known_keys,
            base_url_resolver=self._actor_base_url,
            actor_builder=lambda base_url, env, env_id, group_id: ActorHandle(
                base_url=base_url,
                env_name=env,
                env_id=env_id,
                group_id=group_id,
            ),
        )
        added = await self._runtime.finish_refill(old_key, new_key, new_actor)
        if added and new_key is not None and new_actor is not None:
            log.debug("discovered new actor: %s/%s -> %s", new_key[0], new_key[1], new_actor.base_url)

    async def _wait_for_bg_refill_tasks(self) -> None:
        while True:
            async with self._lifecycle_lock:
                tasks = list(self._bg_refill_tasks)

            if not tasks:
                return

            log.info(
                "ManagerActorPool.aclose(): waiting for %d background refill task(s)",
                len(tasks),
            )
            try:
                await asyncio.gather(*tasks, return_exceptions=True)
            except asyncio.CancelledError:
                log.warning(
                    "ManagerActorPool.aclose(): cancelled while waiting; cancelling %d refill task(s)",
                    len(tasks),
                )
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                raise

    async def aclose(self) -> None:
        async with self._lifecycle_lock:
            if self._closed:
                log.info("ManagerActorPool.aclose(): already closed")
                return
            self._closing = True

        log.info("ManagerActorPool.aclose(): waiting for background refill tasks ...")
        await self._wait_for_bg_refill_tasks()

        async with self._lifecycle_lock:
            if self._closed:
                log.info("ManagerActorPool.aclose(): already closed")
                return
            log.info("ManagerActorPool.aclose(): closing EnvPoolManager ...")
            await self.mgr.close_all()
            self._closed = True


# -------------------------
# CLI
# -------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="RL env launcher",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    #Task identifier
    p.add_argument("--job-id", type=str, default="", help="job id is used to identify each task and record in the environmentconfig table as session")
    # YAML
    p.add_argument("--manager-config", type=str, default="./manager/config.yaml", help="Path to unified YAML config")
    p.add_argument("--exp-config", type=str, default="./core/exp/config.yaml", help="Path to experience injection YAML config")
    p.add_argument("--mode", choices=["local", "remote"], default="local")

    # DB / YAML-aggregator
    p.add_argument("--env-config", type=str, default=None, help="env config which used for specify the input env configs, and it's incompatible with  env-root")
    p.add_argument("--env-root", type=str, default="env", help="only works when env-config is not specified")
    p.add_argument("--storage-type", type=str, default="sqlite", choices=["sqlite", "cloud"], help="Storage backend for environment configs and results (affects DataManager implementation)")
    p.add_argument("--warmup-count", type=int, default=100, help="The number of environment configs to pre-store in the manager.")
    p.add_argument("--save-batch-size", type=int, default=100, help="Size of environment configs to store in the manager.")
    p.add_argument("--enable-buffer", type=bool, default=False, help="Enable buffer record storage")
    p.add_argument("--buffer-size", type=int, default=100, help="Size of the buffer for storing records")
    p.add_argument("--flush-interval", type=float, default=5.0, help="Interval (in seconds) for flushing buffered records")
    p.add_argument("--db-path", type=str, default="sqlite://android_envs.db")
    p.add_argument("--rebuild-table", action=argparse.BooleanOptionalAction, default=False,
                   help="Delete and recreate the SQLite DB file before loading configs (SQLite only)")

    # Pool overrides
    p.add_argument("--pool-size", type=int, default=0, help="Override pool_size in YAML (0 = keep YAML)")
    p.add_argument("--multiplier", type=float, default=1.2, help="Buffer multiplier for pool size. Total pre-warmed actors = [multiplier * pool_size]")

    # Local upstream service (optional)
    p.add_argument("--start-local-upstream", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--local-upstream-app", type=str, default="env.app:app")
    p.add_argument("--local-upstream-host", type=str, default="0.0.0.0")
    p.add_argument("--local-upstream-port", type=int, default=36663)
    p.add_argument("--local-upstream-url", type=str, default="http://127.0.0.1:36663")
    p.add_argument("--wait-timeout", type=float, default=60.0)

    # Interactor
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--message-cut", type=int, default=-1, help="Number of recent messages to keep in the conversation history (<= 0 keep all)")
    p.add_argument("--env-http-timeout-s", type=float, default=300.0)
    p.add_argument("--http-retries", type=int, default=2)

    # LLM
    p.add_argument("--llm-base-url", type=str, default="http://100.99.119.227:30000/v1")
    p.add_argument("--llm-api-key", type=str, default="EMPTY")
    p.add_argument("--llm-model", type=str, default="Qwen2.5-VL-72B-Instruct")
    p.add_argument("--llm-temperature", type=float, default=0.3)
    p.add_argument("--rl-use-session-suffix-url", action="store_true", default=False,
                   help="Use SessionSuffixBaseURLProvider (for LLM Proxy with session routing)")
    p.add_argument("--rl-group-size", type=int, default=0,
                   help="Override env_num for all environments (0 = keep YAML config, used for num_repeat_per_sample)")
    p.add_argument("--rl-epoch", type=int, default=1,
                   help="Number of epochs: duplicates all env configs N times with distinct group_ids")

    # Logging
    p.add_argument("--log-dir", type=str, default="logs", help="Directory to store log files")
    p.add_argument("--run-name", type=str, default="", help="Optional run name prefix for log file")
    p.add_argument("--console-log-level", type=str, default="INFO", help="Console log level")
    p.add_argument("--file-log-level", type=str, default="DEBUG", help="File log level")
    p.add_argument("--log-max-bytes", type=int, default=50 * 1024 * 1024, help="Rotate log after this size")
    p.add_argument("--log-backup-count", type=int, default=5, help="How many rotated logs to keep")
    p.add_argument("--debug-log", action="store_true", default=False,
                   help="Enable DEBUG-level console logging for the storage layer")

    return p.parse_args()


async def main():
    args = parse_args()

    # Setup logs FIRST (so everything goes to file)
    main_log_path = _setup_logging(
        log_dir=args.log_dir,
        run_name=args.run_name,
        console_level=args.console_log_level,
        file_level=args.file_log_level,
        max_bytes=args.log_max_bytes,
        backup_count=args.log_backup_count,
        debug_loggers=["sqlite_strategy", "yaml_aggregator"] if args.debug_log else None,
    )

    # Optional: upstream service log file
    upstream_log_path = os.path.join(args.log_dir, "upstream.log")
    log.info("main log file: %s", main_log_path)

    job_id=args.job_id
    if job_id=="":
       job_id =  uuid.uuid4().hex

    # Rebuild the DB file if requested (SQLite only)
    if args.rebuild_table and args.storage_type == "sqlite":
        _rebuild_sqlite_db(args.db_path)

    data_manager = DataManager(job_id=job_id, storage_type=args.storage_type, db_url=args.db_path, enable_buffer=args.enable_buffer, buffer_size=args.buffer_size, flush_interval=args.flush_interval)
    yaml_config_list = all_env_yaml_load(env_root=args.env_root, env_config=args.env_config)

    # 如果指定了 --rl-group-size，覆盖所有环境的 env_num
    if args.rl_group_size > 0:
        for cfg in yaml_config_list:
            cfg["env_num"] = args.rl_group_size
        log.info("Override env_num=%d for all %d environments", args.rl_group_size, len(yaml_config_list))

    # 如果 --rl-epoch > 1，复制配置 N 次，每次用不同 task_idx 产生不同 group_id
    if args.rl_epoch > 1:
        base_configs = list(yaml_config_list)
        num_tasks = len(base_configs)
        for epoch_idx in range(1, args.rl_epoch):
            for cfg in base_configs:
                epoch_cfg = dict(cfg)
                epoch_cfg["task_idx"] = cfg.get("task_idx", 1) + epoch_idx * num_tasks
                yaml_config_list.append(epoch_cfg)
        log.info("rl_epoch=%d: expanded %d configs to %d configs", args.rl_epoch, num_tasks, len(yaml_config_list))

    conn = await sync_configs_to_db(data_manager, yaml_config_list, args.storage_type, args.warmup_count, args.save_batch_size)

    local_proc: Optional[subprocess.Popen] = None
    pool: Optional[ActorPool] = None

    try:
        # 3) load YAML config
        cfg = _load_yaml(args.manager_config)
        log.info("loaded config: %s", args.manager_config)

        # Ensure config points to our DB path
        _set_nested(cfg, ["database", "driver"], cfg.get("database", {}).get("driver", "sqlite"))
        _set_nested(cfg, ["database", "sqlite_path"], args.db_path)

        # Override pool size if requested
        original_pool_size = int(cfg.get("pool_size", 1) or 1)
        if int(args.pool_size) > 0:
            original_pool_size = int(args.pool_size)
        buffer_multiplier = float(args.multiplier) if args.multiplier >0.0 else 1.2
        total_pool_size = math.ceil(original_pool_size * buffer_multiplier)
        cfg["pool_size"] = total_pool_size
        log.info("override pool_size=%d (original=%d, multiplier=%.2f)", total_pool_size, original_pool_size, buffer_multiplier)
        # Mode override
        if args.mode in ("local", "remote"):
            cfg["mode"] = args.mode
            log.info("override mode=%s", args.mode)

        # Local mode upstream service handling
        mode = str(cfg.get("mode") or (cfg.get("cluster") or {}).get("mode", "")).lower()
        if mode == "local" or (args.mode == "local"):
            cfg["mode"] = "local"
            _set_nested(cfg, ["cluster", "local", "host"], "127.0.0.1")
            _set_nested(cfg, ["cluster", "http", "port"], int(args.local_upstream_port))

            start_local = args.start_local_upstream
            if start_local is None:
                start_local = True

            if start_local:
                local_proc = start_local_upstream_service(
                    host=str(args.local_upstream_host),
                    port=int(args.local_upstream_port),
                    uvicorn_app=str(args.local_upstream_app),
                    stdout_path=upstream_log_path,
                )
                await wait_http_ready(str(args.local_upstream_url), float(args.wait_timeout))
            else:
                log.info("local upstream auto-start disabled; assuming already running at %s", args.local_upstream_url)

        # 4) start manager + adapter pool
        mgr = EnvPoolManager(cfg, conn, job_id=job_id)
        pool = ManagerActorPool(mgr, pool_size=total_pool_size)

        # 5) build EpisodeHandler from config
        _exp_cfg = _load_yaml(args.exp_config)
        from core.exp.handler import build_episode_handler
        episode_handler = build_episode_handler(
            exp_dir=str(_exp_cfg.get("dir", "./experiences")),
            enabled=bool(_exp_cfg.get("enabled", False)),
            top_k=int(_exp_cfg.get("top_k", 3)),
            mode=str(_exp_cfg.get("mode", "template")),
            embedding_model=_exp_cfg.get("embedding_model"),
        )
        log.info(
            "experience injection: config=%s enabled=%s dir=%s top_k=%d mode=%s",
            args.exp_config,
            bool(_exp_cfg.get("enabled", False)),
            _exp_cfg.get("dir", "./experiences"),
            int(_exp_cfg.get("top_k", 3)),
            str(_exp_cfg.get("mode", "template")),
        )

        # 根据参数选择 BaseURLProvider
        if args.rl_use_session_suffix_url:
            base_url_provider = SessionSuffixBaseURLProvider(base_url_root=args.llm_base_url)
            log.info("Using SessionSuffixBaseURLProvider with root: %s", args.llm_base_url)
        else:
            base_url_provider = StaticBaseURLProvider(base_url=args.llm_base_url)
            log.info("Using StaticBaseURLProvider: %s", args.llm_base_url)

        interactor = Interactor(
            pool=pool,
            base_url_provider=base_url_provider,
            api_key=args.llm_api_key,
            model=args.llm_model,
            data_manager=data_manager,
            temperature=args.llm_temperature,
            max_steps=args.max_steps,
            message_cut=args.message_cut,
            env_http_timeout_s=args.env_http_timeout_s,
            max_workers=original_pool_size,
            http_retries=int(args.http_retries),
            verbose=True,
            episode_handler=episode_handler,
        )

        log.info("starting interactor.run() ...")
        results = await interactor.run_all_environments()
        await interactor.aclose()

        # Summary (also printed to console via INFO)
        log.info("RUN SUMMARY: episodes=%d", len(results))
        for k, v in results.items():
            log.info("  %s reward=%.6f", k, v)

        log.info("done. main_log=%s upstream_log=%s", main_log_path, upstream_log_path)

    finally:
        if pool is not None:
            try:
                log.info("Closing actor pool...")
                await pool.aclose()
            except Exception:
                log.exception("pool.aclose failed (ignored)")

        if local_proc is not None:
            await stop_process(local_proc)
            
        await asyncio.sleep(0.5)

        try:
            await wait_for_pending_inserts()
        except Exception:
            log.exception("wait_for_pending_inserts failed (ignored)")

        try:
            log.info("Closing data manager...")
            await data_manager.close()
        except Exception:
            log.exception("db manager failed close")
        
        try:
            conn.close()
        except Exception:
            log.exception("conn close failed (ignored)")

        if local_proc is not None:
            await stop_process(local_proc)


if __name__ == "__main__":
    asyncio.run(main())
