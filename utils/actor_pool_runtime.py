from __future__ import annotations

import asyncio
from collections import deque
from typing import Any, Callable, Deque, Dict, Generic, List, Optional, Set, Tuple, TypeVar

ActorT = TypeVar("ActorT")
ActorKey = Tuple[str, str]
ActorSnapshot = Dict[str, Any]
BaseURLResolver = Callable[[str, str], Optional[str]]
ActorBuilder = Callable[[str, str, str, str], ActorT]


class ActorPoolRuntimeState(Generic[ActorT]):
    """
    Runtime state for an actor pool adapter.

    Responsibilities:
      - Hold ready actors in memory
      - Track actors currently leased to workers
      - Track refills in progress
      - Decide when the pool is truly exhausted

    Important:
      - The ready queue no longer uses `None` as a runtime sentinel.
      - `acquire()` returns `None` only when the pool is truly exhausted.
    """

    def __init__(self) -> None:
        self._ready: Deque[ActorT] = deque()
        self._known: Set[ActorKey] = set()
        self._leased: int = 0
        self._refills_in_flight: int = 0
        self._initial_load_done: bool = False
        self._cond = asyncio.Condition()

    async def add_ready_actor(self, key: ActorKey, actor: ActorT) -> bool:
        async with self._cond:
            if key in self._known:
                return False
            self._known.add(key)
            self._ready.append(actor)
            self._cond.notify(1)
            return True

    async def mark_initial_load_done(self) -> None:
        async with self._cond:
            self._initial_load_done = True
            self._cond.notify_all()

    async def acquire(self) -> Optional[ActorT]:
        async with self._cond:
            while True:
                if self._ready:
                    actor = self._ready.popleft()
                    self._leased += 1
                    return actor
                if self._is_truly_exhausted_locked():
                    return None
                await self._cond.wait()

    async def begin_refill(self, _old_key: ActorKey) -> None:
        async with self._cond:
            if self._leased > 0:
                self._leased -= 1
            self._refills_in_flight += 1
            self._cond.notify_all()

    async def finish_refill(
        self,
        old_key: ActorKey,
        new_key: Optional[ActorKey] = None,
        actor: Optional[ActorT] = None,
    ) -> bool:
        async with self._cond:
            self._known.discard(old_key)
            if self._refills_in_flight > 0:
                self._refills_in_flight -= 1

            added = False
            if new_key is not None and actor is not None and new_key not in self._known:
                self._known.add(new_key)
                self._ready.append(actor)
                added = True

            self._cond.notify_all()
            return added

    async def fail_refill(self, old_key: ActorKey) -> None:
        async with self._cond:
            self._known.discard(old_key)
            if self._refills_in_flight > 0:
                self._refills_in_flight -= 1
            self._cond.notify_all()

    async def known_keys_snapshot(self) -> Set[ActorKey]:
        async with self._cond:
            return set(self._known)

    def _is_truly_exhausted_locked(self) -> bool:
        return (
            self._initial_load_done
            and not self._ready
            and self._leased == 0
            and self._refills_in_flight == 0
        )


def discover_ready_actor_from_snapshot(
    actors: List[ActorSnapshot],
    *,
    known_keys: Set[ActorKey],
    base_url_resolver: BaseURLResolver,
    actor_builder: ActorBuilder[ActorT],
) -> Tuple[Optional[ActorKey], Optional[ActorT]]:
    """
    Find the first discoverable actor from a manager snapshot.

    An actor is discoverable when:
      - it is not already known
      - its route can be resolved to a usable base URL
    """
    for item in actors:
        env = str(item.get("env_name") or "")
        env_id = str(item.get("env_id") or "")
        if not env or not env_id:
            continue

        key = (env, env_id)
        if key in known_keys:
            continue

        base_url = base_url_resolver(env, env_id)
        if not base_url:
            continue

        group_id = str(item.get("group_id") or "")
        actor = actor_builder(base_url, env, env_id, group_id)
        return key, actor

    return None, None
