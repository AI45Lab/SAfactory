from __future__ import annotations

import asyncio
import os
import shutil
import tempfile
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol

from evaluator.eval_types import (
    EvaluatorAgentLease,
    EvaluatorAgentPoolSpec,
    EvaluatorAgentSpec,
    normalize_evaluator_agent_type,
)


class EvaluatorLeaseManager(Protocol):
    async def start_lease(self, spec: EvaluatorAgentSpec, index: int) -> EvaluatorAgentLease:
        ...

    async def stop_lease(self, lease: EvaluatorAgentLease) -> None:
        ...


class SyntheticEvaluatorLeaseManager:
    """Test-friendly manager that creates leases for already-managed runtimes."""

    async def start_lease(self, spec: EvaluatorAgentSpec, index: int) -> EvaluatorAgentLease:
        name = f"eval-{spec.evaluator_agent_id}-{index}"
        return EvaluatorAgentLease(
            lease_id=str(uuid.uuid4()),
            pool_id=spec.pool_id,
            evaluator_agent_id=spec.evaluator_agent_id,
            base_agent=spec.base_agent,
            container_id=name if spec.agent_type == "docker_container" else "",
            container_name=name,
            spec=spec,
        )

    async def stop_lease(self, lease: EvaluatorAgentLease) -> None:
        return None

    async def start_container(self, spec: EvaluatorAgentSpec, index: int) -> EvaluatorAgentLease:
        return await self.start_lease(spec, index)

    async def stop_container(self, lease: EvaluatorAgentLease) -> None:
        await self.stop_lease(lease)


class DockerEvaluatorLeaseManager:
    def __init__(
        self,
        *,
        docker_bin: str = "docker",
        idle_command: str = "tail -f /dev/null",
        name_prefix: str = "sf-evaluator",
    ) -> None:
        self.docker_bin = docker_bin
        self.idle_command = idle_command
        self.name_prefix = name_prefix

    async def start_lease(self, spec: EvaluatorAgentSpec, index: int) -> EvaluatorAgentLease:
        if not spec.image:
            raise ValueError(f"docker evaluator agent {spec.evaluator_agent_id!r} requires image")
        name = f"{self.name_prefix}-{spec.evaluator_agent_id}-{index}-{uuid.uuid4().hex[:8]}"
        args = [self.docker_bin, "run", "-d", "--name", name]
        for key, value in spec.env.items():
            args.extend(["-e", f"{key}={value}"])
        for mount in spec.mounts:
            source = mount.get("source")
            target = mount.get("target")
            if not source or not target:
                continue
            mode = mount.get("mode", "ro")
            args.extend(["-v", f"{source}:{target}:{mode}"])
        args.extend([spec.image, "sh", "-lc", self.idle_command])
        container_id = (await _run_text(args, timeout_s=120.0)).strip()
        return EvaluatorAgentLease(
            lease_id=str(uuid.uuid4()),
            pool_id=spec.pool_id,
            evaluator_agent_id=spec.evaluator_agent_id,
            base_agent=spec.base_agent,
            container_id=container_id,
            container_name=name,
            spec=spec,
        )

    async def stop_lease(self, lease: EvaluatorAgentLease) -> None:
        await _run_text(
            [self.docker_bin, "rm", "-f", lease.container_id or lease.container_name],
            timeout_s=60.0,
            check=False,
        )

    async def start_container(self, spec: EvaluatorAgentSpec, index: int) -> EvaluatorAgentLease:
        return await self.start_lease(spec, index)

    async def stop_container(self, lease: EvaluatorAgentLease) -> None:
        await self.stop_lease(lease)


class CodexCliEvaluatorLeaseManager:
    """Creates local process slots for Codex CLI evaluator runs."""

    def __init__(
        self,
        *,
        runtime_root: str | None = None,
        name_prefix: str = "sf-evaluator-cli",
    ) -> None:
        self.runtime_root = runtime_root
        self.name_prefix = name_prefix

    async def start_lease(self, spec: EvaluatorAgentSpec, index: int) -> EvaluatorAgentLease:
        root = Path(spec.runtime_dir or self.runtime_root or _default_cli_runtime_root()).expanduser()
        slot_name = f"{self.name_prefix}-{_safe_name(spec.evaluator_agent_id)}-{index}-{uuid.uuid4().hex[:8]}"
        runtime_dir = root / slot_name
        await asyncio.to_thread(runtime_dir.mkdir, parents=True, exist_ok=True)
        lease_spec = replace(
            spec,
            runtime_dir=str(runtime_dir),
            task_input_path=str(runtime_dir / _path_name(spec.task_input_path, "agent_eval_task.json")),
            prompt_path=str(runtime_dir / _path_name(spec.prompt_path, "agent_eval_prompt.md")),
            result_path=str(runtime_dir / _path_name(spec.result_path, "eval_result.json")),
        )
        return EvaluatorAgentLease(
            lease_id=str(uuid.uuid4()),
            pool_id=lease_spec.pool_id,
            evaluator_agent_id=lease_spec.evaluator_agent_id,
            base_agent=lease_spec.base_agent,
            container_id="",
            container_name=slot_name,
            spec=lease_spec,
        )

    async def stop_lease(self, lease: EvaluatorAgentLease) -> None:
        runtime_dir = lease.spec.runtime_dir
        if not runtime_dir or not lease.spec.cleanup_runtime_dir:
            return
        await asyncio.to_thread(shutil.rmtree, runtime_dir, ignore_errors=True)


class CompositeEvaluatorLeaseManager:
    def __init__(self, managers: dict[str, EvaluatorLeaseManager]) -> None:
        self.managers = {
            normalize_evaluator_agent_type(agent_type): manager
            for agent_type, manager in managers.items()
        }

    async def start_lease(self, spec: EvaluatorAgentSpec, index: int) -> EvaluatorAgentLease:
        return await self._manager_for(spec.agent_type).start_lease(spec, index)

    async def stop_lease(self, lease: EvaluatorAgentLease) -> None:
        await self._manager_for(lease.spec.agent_type).stop_lease(lease)

    def _manager_for(self, agent_type: str) -> EvaluatorLeaseManager:
        normalized = normalize_evaluator_agent_type(agent_type)
        try:
            return self.managers[normalized]
        except KeyError as exc:
            raise ValueError(f"no evaluator lease manager registered for agent_type={normalized!r}") from exc


# Backward-compatible names used by existing imports.
EvaluatorContainerManager = EvaluatorLeaseManager
SyntheticEvaluatorContainerManager = SyntheticEvaluatorLeaseManager
DockerEvaluatorContainerManager = DockerEvaluatorLeaseManager


@dataclass
class _LeaseState:
    lease: EvaluatorAgentLease
    active_slots: int = 0
    reusable: bool = True


class EvaluatorAgentPool:
    def __init__(
        self,
        *,
        pool_specs: list[EvaluatorAgentPoolSpec],
        container_manager: EvaluatorLeaseManager | None = None,
        lease_manager: EvaluatorLeaseManager | None = None,
    ) -> None:
        self.pool_specs = {pool.pool_id: pool for pool in pool_specs}
        self.lease_manager = lease_manager or container_manager or SyntheticEvaluatorLeaseManager()
        self.container_manager = self.lease_manager
        self._states: list[_LeaseState] = []
        self._cond = asyncio.Condition()
        self._started = False
        self._rr_index = 0

    async def start(self) -> None:
        async with self._cond:
            if self._started:
                return
            leases: list[_LeaseState] = []
            for pool in self.pool_specs.values():
                for spec in pool.members:
                    spec.pool_id = pool.pool_id
                    for index in range(spec.pool_size):
                        lease = await self._start_lease(spec, index)
                        leases.append(_LeaseState(lease=lease))
            self._states = leases
            self._started = True
            self._cond.notify_all()

    async def acquire(
        self,
        *,
        evaluator_pool_id: str | None = None,
        evaluator_agent_id: str | None = None,
        evaluator_agent_type: str | None = None,
        required_capabilities: set[str] | None = None,
        allowed_base_agents: list[str] | None = None,
        target_access_mode: str = "snapshot",
        timeout_s: float = 60.0,
    ) -> EvaluatorAgentLease:
        if not self._started:
            await self.start()

        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_s
        async with self._cond:
            while True:
                candidates = self._eligible_states(
                    evaluator_pool_id=evaluator_pool_id,
                    evaluator_agent_id=evaluator_agent_id,
                    evaluator_agent_type=evaluator_agent_type,
                    required_capabilities=required_capabilities or set(),
                    allowed_base_agents=allowed_base_agents or [],
                    target_access_mode=target_access_mode,
                )
                if candidates:
                    state = self._choose(candidates, evaluator_pool_id)
                    state.active_slots += 1
                    state.lease.active_slots = state.active_slots
                    return state.lease

                remaining = deadline - loop.time()
                if remaining <= 0:
                    raise TimeoutError("timed out waiting for evaluator agent lease")
                await asyncio.wait_for(self._cond.wait(), timeout=remaining)

    async def release(
        self,
        lease: EvaluatorAgentLease,
        *,
        reusable: bool = True,
    ) -> None:
        async with self._cond:
            state = self._find_state(lease.lease_id)
            if state is None:
                return
            state.active_slots = max(0, state.active_slots - 1)
            state.lease.active_slots = state.active_slots
            if not reusable:
                state.reusable = False
                if state.active_slots == 0:
                    await self._stop_lease(state.lease)
            self._cond.notify_all()

    async def stop(self) -> None:
        async with self._cond:
            states = list(self._states)
            self._states.clear()
            self._started = False
        for state in states:
            await self._stop_lease(state.lease)

    async def snapshot(self) -> dict[str, object]:
        async with self._cond:
            return {
                "started": self._started,
                "leases": [
                    {
                        "lease_id": state.lease.lease_id,
                        "pool_id": state.lease.pool_id,
                        "evaluator_agent_id": state.lease.evaluator_agent_id,
                        "agent_type": state.lease.spec.agent_type,
                        "base_agent": state.lease.base_agent,
                        "active_slots": state.active_slots,
                        "capacity": state.lease.spec.max_concurrency_per_container,
                        "runtime_dir": state.lease.spec.runtime_dir,
                        "reusable": state.reusable,
                    }
                    for state in self._states
                ],
            }

    def _eligible_states(
        self,
        *,
        evaluator_pool_id: str | None,
        evaluator_agent_id: str | None,
        evaluator_agent_type: str | None,
        required_capabilities: set[str],
        allowed_base_agents: list[str],
        target_access_mode: str,
    ) -> list[_LeaseState]:
        allowed_base_set = set(allowed_base_agents or [])
        normalized_agent_type = normalize_evaluator_agent_type(evaluator_agent_type) if evaluator_agent_type else None
        result: list[_LeaseState] = []
        for state in self._states:
            spec = state.lease.spec
            if not state.reusable:
                continue
            if evaluator_pool_id and state.lease.pool_id != evaluator_pool_id:
                continue
            if evaluator_agent_id and state.lease.evaluator_agent_id != evaluator_agent_id:
                continue
            if normalized_agent_type and spec.agent_type != normalized_agent_type:
                continue
            if allowed_base_set and state.lease.base_agent not in allowed_base_set:
                continue
            if required_capabilities and not required_capabilities.issubset(spec.capabilities):
                continue
            if target_access_mode not in spec.allowed_target_access_modes:
                continue
            if target_access_mode == "direct_docker" and not spec.allow_direct_docker:
                continue
            if state.active_slots >= spec.max_concurrency_per_container:
                continue
            result.append(state)
        return result

    def _choose(self, candidates: list[_LeaseState], evaluator_pool_id: str | None) -> _LeaseState:
        pool_id = evaluator_pool_id or candidates[0].lease.pool_id
        policy = self.pool_specs.get(pool_id, EvaluatorAgentPoolSpec(pool_id, [])).selection_policy
        if policy == "weighted_round_robin":
            expanded: list[_LeaseState] = []
            for state in candidates:
                expanded.extend([state] * max(1, int(state.lease.spec.weight)))
            chosen = expanded[self._rr_index % len(expanded)]
            self._rr_index += 1
            return chosen
        return min(
            candidates,
            key=lambda state: (
                state.active_slots / state.lease.spec.max_concurrency_per_container,
                -state.lease.spec.weight,
                state.lease.evaluator_agent_id,
            ),
        )

    def _find_state(self, lease_id: str) -> _LeaseState | None:
        for state in self._states:
            if state.lease.lease_id == lease_id:
                return state
        return None

    async def _start_lease(self, spec: EvaluatorAgentSpec, index: int) -> EvaluatorAgentLease:
        manager = self.lease_manager
        if hasattr(manager, "start_lease"):
            return await manager.start_lease(spec, index)  # type: ignore[attr-defined]
        return await manager.start_container(spec, index)  # type: ignore[attr-defined]

    async def _stop_lease(self, lease: EvaluatorAgentLease) -> None:
        manager = self.lease_manager
        if hasattr(manager, "stop_lease"):
            await manager.stop_lease(lease)  # type: ignore[attr-defined]
            return
        await manager.stop_container(lease)  # type: ignore[attr-defined]


async def _run_text(args: list[str], *, timeout_s: float, check: bool = True) -> str:
    proc = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.communicate()
        raise TimeoutError(f"command timed out after {timeout_s}s: {args!r}")
    stdout = stdout_b.decode("utf-8", errors="replace")
    stderr = stderr_b.decode("utf-8", errors="replace")
    if check and proc.returncode != 0:
        raise RuntimeError(f"command failed with exit code {proc.returncode}: {stderr}")
    return stdout


def _default_cli_runtime_root() -> str:
    return os.path.join(tempfile.gettempdir(), "safactory-codex-evaluator")


def _path_name(path: str, fallback: str) -> str:
    name = Path(str(path or "")).name
    return name or fallback


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in str(value))[:96] or "agent"
