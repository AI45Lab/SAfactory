from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

AgentKey = Tuple[str, str]     # (agent_name, agent_id)


@dataclass(slots=True)
class PoolEntry:
    """
    Local record of one DB row assigned to a runtime resource.
    """
    env_name: str
    env_id: str
    row_id: Optional[int]
    image: str
    job_name: str
    env_params: Dict[str, Any] = field(default_factory=dict)
    group_id: str = ""
    status: str = "ready"
    runtime: str = "docker"
    runtime_config: Dict[str, Any] = field(default_factory=dict)
    runtime_handle: Any = field(default=None, repr=False)
    resource_id: str = ""
    resource_name: str = ""
    container_id: str = ""
    container_name: str = ""
    docker_bin: str = "docker"
    workdir: str = ""
    run_command: str = "node /tmp/safactory-openclaw-runner.mjs"
    result_mode: str = "json"
    cleanup_command: str = ""
    healthcheck_command: str = ""
    reuse_container: bool = False


@dataclass(frozen=True, slots=True)
class SimulationRunConfig:
    job_id: str
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
    docker_image_archive_dir: str
    cleanup_docker_image: bool
    docker_startup_concurrency: int
    agent_start_timeout_grace_s: float = 120.0
    container_refill_timeout_s: float = 300.0
    row_wait_timeout_s: float = 60.0
    row_fetch_timeout_s: float = 30.0
    gateway_close_timeout_s: float = 15.0
    gateway_close_retries: int = 1
    gateway_close_retry_backoff_s: float = 1.0
    shutdown_timeout_s: float = 120.0
    docker_command_timeout_s: float = 300.0
    docker_start_timeout_s: float = 300.0
    docker_remove_timeout_s: float = 120.0
    docker_stop_timeout_s: float = 10.0
    docker_inspect_timeout_s: float = 10.0
    docker_remove_retries: int = 3
    docker_remove_retry_delay_s: float = 2.0
    docker_lifecycle_timeout_s: float = 60.0
    rjob_config: Dict[str, Any] = field(default_factory=dict)
    sandbox_config: Dict[str, Any] = field(default_factory=dict)
    cleanup_docker_container: bool = True
    cleanup_stale_docker_containers: bool = True
    max_workers: Optional[int] = None
    rebuild_table: bool = False
    resume: bool = False
    confirm_cloud_delete_job_id: str = ""
    confirm_production: bool = False
    cloud_delete_archive_dir: str = ""
    cloud_job_claim_dir: str = ""
    enable_buffer: bool = True
    buffer_size: int = 100
    flush_interval: float = 5.0
    rl_group_size: int = 0
    rl_epoch: int = 1
    evaluation_enabled: bool = False
    circuit_breaker_enabled: bool = True
    circuit_breaker_window: int = 50
    circuit_breaker_min_samples: int = 20
    circuit_breaker_failure_rate: float = 0.8
    circuit_breaker_timeout_rate: float = 0.5
    circuit_breaker_consecutive_timeouts: int = 5


@dataclass(frozen=True, slots=True)
class SimulationAgentLease:
    agent_name: str
    agent_id: str
    group_id: str
    image: str
    row_id: Optional[int]
    env_params: Dict[str, Any] = field(default_factory=dict)
    runtime: str = "docker"
    runtime_config: Dict[str, Any] = field(default_factory=dict)
    runtime_handle: Any = field(default=None, repr=False)
    resource_id: str = ""
    resource_name: str = ""
    container_id: str = ""
    container_name: str = ""
    docker_bin: str = "docker"
    workdir: str = ""
    run_command: str = "node /tmp/safactory-openclaw-runner.mjs"
    result_mode: str = "json"
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
    total_reward: Optional[float]
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
    truncated_episodes: int
    failed_episodes: int
    cancelled: bool
    results: Dict[str, Optional[float]] = field(default_factory=dict)
