import asyncio
import sqlite3
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Any, List, Set

import ray
from ray_base import GenericEnvActor
from db_loader import get_active_data, compose_create_kwargs


@dataclass(slots=True)
class ActorRec:
    env: str
    id: int
    handle: ray.actor.ActorHandle
    state: str  # "READY" | "CLOSING"


class EnvPoolManager:
    """
    Scalable pool manager for thousands of Ray actors.

    Key properties for scale:
    - Thread-/task-safe DB cursor: self._reserve_rows() holds a lock around (offset read -> query -> offset advance).
    - Bounded spawn concurrency (self._spawn_sem) to avoid startup thundering herd.
    - De-dupe against both the active registry and an 'in-flight' set so concurrent refills can't double-spawn.
    - One-at-a-time refill loop (self._refill_lock) so multiple callers don't race. Uses 'in-flight' to size accurately.
    - Concurrency-limited 'describe' and 'close' for large N.
    """

    def __init__(self, cfg: dict, conn: sqlite3.Connection):
        self.cfg = cfg
        self.conn = conn

        # ---- pool sizing ----
        # default pool size 5
        self.pool_size: int = int(self.cfg.get("pool_size", 0) or 5)
        # Batch size and max inflight spawns - tune for your cluster size:
        spawn_cfg = dict(self.cfg.get("spawn", {}) or {})
        self._batch_size: int = int(spawn_cfg.get("batch_size", 64))       # spawn this many at a time
        self._max_inflight: int = int(spawn_cfg.get("max_inflight", 64))   # cap concurrent spawn/reset

        # ---- state ----
        self.registry: Dict[Tuple[str, int], ActorRec] = {}
        self._inflight: Set[Tuple[str, int]] = set()  # keys being spawned/reset but not READY yet
        self._closed: bool = False

        # ---- concurrency primitives ----
        self._spawn_sem = asyncio.Semaphore(self._max_inflight)
        self._state_lock = asyncio.Lock()   # protects registry + inflight + closed
        self._cursor_lock = asyncio.Lock()  # protects DB cursor (offset) reservation
        self._refill_lock = asyncio.Lock()  # ensures only one refill loop at a time

        # ---- DB cursor (OFFSET-based) ----
        # We keep OFFSET for compatibility and guard it with _cursor_lock so it is task-safe.
        # For very large tables, consider switching to an ID-based cursor (see note below).
        self._offset: int = 0

    # -------------------- lifecycle --------------------

    async def start(self) -> None:
        """Initialize Ray (if needed) and fill pool to target size."""
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True, include_dashboard=False, logging_level="ERROR")
        await self._ensure_capacity()

    async def close_all(self, timeout_s: float = 15.0, parallelism: int = 128) -> None:
        """Gracefully close all actors with bounded parallelism."""
        async with self._state_lock:
            self._closed = True
            recs = list(self.registry.values())
            self.registry.clear()
            self._inflight.clear()

        if not recs:
            return

        sem = asyncio.Semaphore(parallelism)

        async def _close(rec: ActorRec) -> None:
            async with sem:
                try:
                    await asyncio.to_thread(ray.get, rec.handle.close.remote())
                except Exception:
                    pass

        try:
            await asyncio.wait_for(asyncio.gather(*(_close(r) for r in recs), return_exceptions=True), timeout_s)
        except asyncio.TimeoutError:
            # best-effort close
            pass

    # -------------------- public API (used by app.py) --------------------

    def get(self, env: str, id_: int) -> Optional[ActorRec]:
        return self.registry.get((env, int(id_)))

    async def list_status(self, parallelism: int = 128) -> List[dict]:
        """
        Describe all actors with bounded parallelism to avoid 1000 concurrent ray.get calls.
        """
        async with self._state_lock:
            items = list(self.registry.items())

        sem = asyncio.Semaphore(parallelism)

        async def _describe(key: Tuple[str, int], rec: ActorRec) -> dict:
            async with sem:
                try:
                    desc = await asyncio.to_thread(ray.get, rec.handle.describe.remote())
                except Exception as e:
                    desc = {"env": key[0], "id": key[1], "error": str(e)}
                desc["state"] = rec.state
                return desc

        return await asyncio.gather(*(_describe(k, r) for k, r in items))

    async def close_and_remove(self, env: str, id_: int) -> None:
        """
        Close one actor and refill the pool (idempotent under concurrency).
        """
        key = (env, int(id_))
        rec: Optional[ActorRec] = None

        async with self._state_lock:
            rec = self.registry.pop(key, None)

        if rec:
            rec.state = "CLOSING"
            try:
                await asyncio.to_thread(ray.get, rec.handle.close.remote())
            except Exception:
                pass

        # Top up the pool safely under concurrency
        await self._ensure_capacity()

    # -------------------- internals --------------------

    async def _ensure_capacity(self) -> None:
        """
        Bring (READY + inflight) up to pool_size, batching and bounding spawn concurrency.
        Only one caller runs this loop at a time; others will return quickly.
        """
        # Ensure only one refill loop runs; others will return without work.
        if not self.pool_size or self.pool_size <= 0:
            return

        async with self._refill_lock:
            while True:
                async with self._state_lock:
                    if self._closed:
                        return
                    ready = len(self.registry)
                    inflight = len(self._inflight)
                    deficit = self.pool_size - (ready + inflight)

                if deficit <= 0:
                    return

                # Reserve a batch of rows from DB (task-safe)
                batch = min(deficit, self._batch_size)
                rows = await self._reserve_rows(batch)
                if not rows:
                    # No more rows to schedule right now
                    return

                # Spawn this batch; bounded by _spawn_sem inside _spawn_row()
                await asyncio.gather(*(self._spawn_row(row) for row in rows))

    async def _reserve_rows(self, n: int) -> List[dict]:
        """
        Atomically reserve 'n' rows by advancing the OFFSET under lock.
        This prevents double-reservation when multiple requests call refill concurrently.
        """
        async with self._cursor_lock:
            rows = get_active_data(self.conn, n, self._offset)
            self._offset += len(rows)
            return rows

    async def _spawn_row(self, row: dict) -> None:
        """
        Schedule spawn for one DB row, with de-dupe against registry and inflight.
        """
        env_name = str(row.get("env_name"))
        env_spec = self._get_env_spec(env_name)
        if env_spec is None:
            print(f"[pool] skip row id={row.get('id')}: env '{env_name}' not declared in config")
            return

        entry = env_spec["entrypoint"]
        actor_id = int(row.get("id") or row.get("env_id") or 0)
        if actor_id == 0:
            print(f"[pool] skip row with missing id/env_id for env '{env_name}'")
            return

        key = (env_name, actor_id)

        # De-dupe against registry and inflight
        async with self._state_lock:
            if self._closed:
                return
            if key in self.registry or key in self._inflight:
                return
            self._inflight.add(key)

        # Merge create kwargs (config defaults overridden by DB)
        cfg_defaults = dict((env_spec.get("default_create_kwargs") or {}))
        db_kwargs = compose_create_kwargs(row)
        kwargs = {**cfg_defaults, **db_kwargs}
        if "env_id" not in kwargs and row.get("env_id"):
            kwargs["env_id"] = row["env_id"]

        ok, actor = await self._spawn_one(env_name, actor_id, entry, kwargs)

        async with self._state_lock:
            self._inflight.discard(key)
            if ok and not self._closed:
                self.registry[key] = ActorRec(env_name, actor_id, actor, "READY")
                print(f"[pool] READY -> ({env_name}, {actor_id}) [{entry} {kwargs}]")

    async def _spawn_one(self, env: str, id_: int, entry: str, kwargs: dict) -> Tuple[bool, Optional[ray.actor.ActorHandle]]:
        """
        Create and reset one actor with bounded concurrency.
        """
        opts: Dict[str, Any] = {"num_cpus": float(self.cfg["ray"].get("actor_num_cpus", 0.2))}
        max_conc = self.cfg["ray"].get("max_concurrency", None)
        if max_conc:
            opts["max_concurrency"] = int(max_conc)

        async with self._spawn_sem:
            actor = None
            try:
                actor = GenericEnvActor.options(**opts).remote(env, id_, entry, kwargs)
                # Keep this blocking call off the event loop thread
                await asyncio.to_thread(ray.get, actor.reset.remote())
                return True, actor
            except Exception as e:
                print(f"[pool] FAILED spawn/reset -> ({env}, {id_}) [{entry} {kwargs}] :: {e}")
                try:
                    if actor is not None:
                        ray.kill(actor, no_restart=True)
                except Exception:
                    pass
                return False, None

    def _get_env_spec(self, env_name: str) -> Optional[dict]:
        envs = self.cfg.get("environments", {})
        return envs.get(env_name)
