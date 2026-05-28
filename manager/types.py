from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

AgentKey = Tuple[str, str]     # (agent_name, agent_id)


@dataclass(slots=True)
class PoolEntry:
    """
    Local record of one DB row assigned to a Docker container.
    """
    env_name: str
    env_id: str
    row_id: Optional[int]
    image: str
    job_name: str
    env_params: Dict[str, Any] = field(default_factory=dict)
    group_id: str = ""
    status: str = "ready"
    container_id: str = ""
    container_name: str = ""
    docker_bin: str = "docker"
    workdir: str = ""
    run_command: str = "node /tmp/safactory-openclaw-runner.mjs"
    cleanup_command: str = ""
    healthcheck_command: str = ""
    reuse_container: bool = False


@dataclass(frozen=True, slots=True)
class SimulationRunConfig:
    job_id: str
    exp_config_path: str
    agent_root: str
    agent_config: Optional[str]
    agent_start_config: Optional[str]
    storage_type: str
    db_url: str
    pool_size: int
    warm_pool_size: int
    startup_submit_count: int
    followup_submit_batch: int
    mode: str
    gateway_base_url: str
    llm_model: str
    llm_temperature: float
    max_steps: int
    agent_start_timeout_s: float
    docker_bin: str
    docker_pull_policy: str
    docker_startup_concurrency: int
    max_workers: Optional[int] = None
    agent_runtime: str = "agent_start"
    rebuild_table: bool = False
    enable_buffer: bool = True
    buffer_size: int = 100
    flush_interval: float = 5.0
    rl_group_size: int = 0
    rl_epoch: int = 1
    evaluation_enabled: bool = False
    evaluation_config: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SimulationAgentLease:
    agent_name: str
    agent_id: str
    group_id: str
    image: str
    row_id: Optional[int]
    env_params: Dict[str, Any] = field(default_factory=dict)
    container_id: str = ""
    container_name: str = ""
    docker_bin: str = "docker"
    workdir: str = ""
    run_command: str = "node /tmp/safactory-openclaw-runner.mjs"
    cleanup_command: str = ""
    healthcheck_command: str = ""
    reuse_container: bool = False


@dataclass(frozen=True, slots=True)
class SimulationStartRequest:
    job_id: str
    session_id: str
    agent_name: str
    agent_id: str
    group_id: str
    gateway_base_url: str
    model: str
    temperature: float
    max_steps: int
    storage_type: str
    env_params: Dict[str, Any] = field(default_factory=dict)
    storage_config: Dict[str, Any] = field(default_factory=dict)
    agent_start_timeout_s: float = 600.0
    record_mode: str = "agent_runtime"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.agent_name:
            raise ValueError("SimulationStartRequest requires agent_name")
        if not self.agent_id:
            raise ValueError("SimulationStartRequest requires agent_id")


@dataclass(slots=True)
class SimulationStartResult:
    session_id: str
    status: str
    total_reward: float
    step_count: int
    terminated: bool
    truncated: bool
    error_text: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SimulationRunSummary:
    job_id: str
    status: str
    total_episodes: int
    succeeded_episodes: int
    failed_episodes: int
    cancelled: bool
    results: Dict[str, float] = field(default_factory=dict)
