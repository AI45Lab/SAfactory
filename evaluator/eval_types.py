from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
import shlex
from typing import Any


class EvalMethod(str, Enum):
    LLM_JUDGE = "llm_judge"
    AGENT_EVAL = "agent_eval"
    RULE_EVALUATOR = "rule_evaluator"


class EvalStatus(str, Enum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"
    TIMEOUT = "timeout"


SUPPORTED_EVALUATOR_AGENT_TYPES = {"docker_container", "codex_cli"}


@dataclass
class EvalSpec:
    eval_id: str
    method: EvalMethod | str
    requires_container: bool = False
    timeout_s: float = 600.0

    judge_id: str | None = None
    judge_version: str | None = None
    judge_model: str | None = None
    prompt_template_id: str | None = None
    judge_prompt_template: str | None = None
    rubric: dict[str, Any] | None = None
    variables: dict[str, Any] = field(default_factory=dict)
    input_builder: str = "trajectory_final_answer"
    output_parser: str = "json_score_reason"

    evaluator_agent_id: str | None = None
    evaluator_pool_id: str | None = None
    evaluator_agent_type: str | None = None
    evaluator_agent_image: str | None = None
    evaluator_base_agents: list[str] = field(default_factory=list)
    evaluator_required_capabilities: set[str] = field(default_factory=set)
    evaluator_task_input: dict[str, Any] = field(default_factory=dict)
    evaluator_task_template_id: str | None = None
    evaluator_task_template: str | None = None
    evaluator_output_path: str | None = None
    target_access_mode: str = "snapshot"
    target_container_alias: str | None = None

    rule_evaluator: str | None = None

    score_min: float = 0.0
    score_max: float = 10.0
    weight: float = 1.0

    def __post_init__(self) -> None:
        self.method = coerce_eval_method(self.method)
        if self.evaluator_agent_type:
            self.evaluator_agent_type = normalize_evaluator_agent_type(self.evaluator_agent_type)
        self.evaluator_required_capabilities = set(self.evaluator_required_capabilities or set())
        self.evaluator_base_agents = list(self.evaluator_base_agents or [])
        self.variables = dict(self.variables or {})
        self.evaluator_task_input = dict(self.evaluator_task_input or {})


@dataclass
class EvalRequest:
    job_id: str
    session_id: str
    lease: Any | None = None
    start_result: Any | None = None
    env_params: dict[str, Any] = field(default_factory=dict)
    eval_specs: list[EvalSpec] = field(default_factory=list)


@dataclass
class EvalResult:
    session_id: str
    status: str
    normalized_score_10: float
    raw_score: float | None = None
    reason: str = ""
    method_results: list[dict[str, Any]] = field(default_factory=list)
    artifacts: dict[str, Any] = field(default_factory=dict)
    error_text: str | None = None
    eval_id: str | None = None
    method: str | None = None

    @classmethod
    def failed(
        cls,
        *,
        session_id: str,
        reason: str,
        eval_id: str | None = None,
        method: str | None = None,
        status: str = EvalStatus.FAILED.value,
        error_text: str | None = None,
        artifacts: dict[str, Any] | None = None,
    ) -> "EvalResult":
        return cls(
            session_id=session_id,
            status=status,
            normalized_score_10=0.0,
            reason=reason,
            error_text=error_text or reason,
            artifacts=artifacts or {},
            eval_id=eval_id,
            method=method,
        )

    def to_method_result(self) -> dict[str, Any]:
        data = to_jsonable(self)
        data.pop("method_results", None)
        return data


@dataclass
class Trajectory:
    session_id: str
    steps: list[dict[str, Any]] = field(default_factory=list)
    final_response: str | None = None
    token_usage: dict[str, int] = field(default_factory=dict)
    raw_rows: list[dict[str, Any]] = field(default_factory=list)
    sealed: bool = False
    warnings: list[str] = field(default_factory=list)

    def compact(self, max_chars: int = 60000) -> str:
        chunks: list[str] = []
        for step in self.steps:
            step_id = step.get("step_id", step.get("id", "?"))
            messages = step.get("messages")
            response = step.get("response")
            if messages:
                chunks.append(f"[step {step_id} messages]\n{safe_dumps(messages)}")
            if response:
                chunks.append(f"[step {step_id} response]\n{response}")
        text = "\n\n".join(chunks)
        if len(text) <= max_chars:
            return text
        head = max_chars // 2
        tail = max_chars - head
        return text[:head] + "\n\n[trajectory truncated]\n\n" + text[-tail:]


@dataclass
class TargetAgentRef:
    session_id: str

    container_id: str
    container_name: str
    container_alias: str
    image: str

    workspace_path: str | None = None
    artifact_paths: dict[str, str] = field(default_factory=dict)

    access_mode: str = "snapshot"
    broker_base_url: str | None = None
    broker_token: str | None = None
    docker_host: str | None = None
    docker_socket_path: str | None = None
    alias_map: dict[str, str] = field(default_factory=dict)
    exec_hint: str | None = None


@dataclass
class AgentEvalTask:
    eval_task_id: str
    session_id: str

    instruction: str
    task_input: dict[str, Any]
    rubric: dict[str, Any]
    variables: dict[str, Any]

    target: TargetAgentRef
    trajectory: Trajectory | None
    artifacts: dict[str, Any]

    output_contract: dict[str, Any]
    timeout_s: float
    evaluation_model: str | None = None
    gateway_base_url: str | None = None


@dataclass
class EvaluatorAgentSpec:
    evaluator_agent_id: str
    image: str = ""
    command_template: str = ""
    pool_id: str = "default"
    base_agent: str = "codex"
    agent_type: str = "docker_container"
    pool_size: int = 1
    max_concurrency_per_container: int = 1
    weight: float = 1.0
    capabilities: set[str] = field(default_factory=set)

    workdir: str = "/workspace"
    input_mode: str = "prompt_file"
    task_input_path: str = "/tmp/agent_eval_task.json"
    prompt_path: str = "/tmp/agent_eval_prompt.md"
    gateway_base_url: str | None = None
    model: str | None = None
    max_eval_llm_calls: int = -1
    result_path: str = "/tmp/eval_result.json"
    env: dict[str, str] = field(default_factory=dict)
    mounts: list[dict[str, str]] = field(default_factory=list)

    cli_bin: str = "codex"
    cli_args: list[str] = field(default_factory=list)
    cli_profile: str | None = None
    sandbox: str = "read-only"
    approval_policy: str = "never"
    ephemeral: bool = True
    runtime_dir: str | None = None
    cleanup_runtime_dir: bool = True

    requires_docker_socket: bool = False
    allow_direct_docker: bool = False
    allowed_target_access_modes: set[str] = field(default_factory=lambda: {"snapshot"})

    def __post_init__(self) -> None:
        self.agent_type = normalize_evaluator_agent_type(self.agent_type)
        self.evaluator_agent_id = str(self.evaluator_agent_id)
        self.image = str(self.image or "")
        self.command_template = str(self.command_template or "")
        self.base_agent = str(self.base_agent or "codex").strip() or "codex"
        self.workdir = str(self.workdir or "").strip()
        self.capabilities = set(self.capabilities or set())
        self.allowed_target_access_modes = set(self.allowed_target_access_modes or {"snapshot"})
        self.pool_size = max(1, int(self.pool_size or 1))
        self.max_concurrency_per_container = max(1, int(self.max_concurrency_per_container or 1))
        if self.agent_type == "codex_cli":
            # A CLI lease represents one local process slot; use pool_size for
            # parallelism so prompt/result files never collide inside a slot.
            self.max_concurrency_per_container = 1
            if self.workdir == "/workspace":
                self.workdir = ""
        self.env = {str(key): str(value) for key, value in dict(self.env or {}).items()}
        self.mounts = [dict(mount) for mount in (self.mounts or [])]
        if isinstance(self.cli_args, str):
            self.cli_args = shlex.split(self.cli_args)
        else:
            self.cli_args = [str(arg) for arg in (self.cli_args or [])]
        self.cli_bin = str(self.cli_bin or "codex")
        self.sandbox = str(self.sandbox or "").strip()
        self.approval_policy = str(self.approval_policy or "").strip()
        if self.cli_profile is not None:
            self.cli_profile = str(self.cli_profile).strip() or None
        if self.runtime_dir is not None:
            self.runtime_dir = str(self.runtime_dir).strip() or None


@dataclass
class EvaluatorAgentPoolSpec:
    pool_id: str
    members: list[EvaluatorAgentSpec]
    selection_policy: str = "least_busy"
    acquire_timeout_s: float = 60.0
    max_queue_size: int = 1024


@dataclass
class EvaluatorAgentLease:
    lease_id: str
    pool_id: str
    evaluator_agent_id: str
    base_agent: str
    container_id: str
    container_name: str
    spec: EvaluatorAgentSpec
    active_slots: int = 0

    @property
    def agent_type(self) -> str:
        return self.spec.agent_type


@dataclass
class EvaluatorRunResult:
    stdout: str = ""
    stderr: str = ""
    result_text: str | None = None
    result_path: str | None = None
    exit_code: int | None = None
    timed_out: bool = False
    artifacts: dict[str, Any] = field(default_factory=dict)


def normalize_score(raw_score: float, min_score: float, max_score: float) -> float:
    if max_score <= min_score:
        raise ValueError("score_max must be greater than score_min")
    normalized = (float(raw_score) - float(min_score)) / (float(max_score) - float(min_score)) * 10.0
    return max(0.0, min(10.0, normalized))


def parse_eval_specs(
    env_params: dict[str, Any] | None,
    *,
    default_specs: list[EvalSpec | dict[str, Any]] | None = None,
) -> list[EvalSpec]:
    env_params = env_params or {}
    evaluation = env_params.get("evaluation")
    evaluation = evaluation if isinstance(evaluation, dict) else {}
    raw_specs = (
        env_params.get("eval")
        or evaluation.get("eval")
        or evaluation.get("specs")
        or default_specs
        or []
    )
    if isinstance(raw_specs, dict):
        raw_specs = [raw_specs]
    return [coerce_eval_spec(item) for item in raw_specs]


def coerce_eval_spec(value: EvalSpec | dict[str, Any]) -> EvalSpec:
    if isinstance(value, EvalSpec):
        return value
    data = dict(value)
    if "method" not in data:
        raise ValueError("EvalSpec requires method")
    if "eval_id" not in data:
        data["eval_id"] = str(data["method"])
    return EvalSpec(**data)


def coerce_eval_method(value: EvalMethod | str) -> EvalMethod:
    if isinstance(value, EvalMethod):
        return value
    text = str(value).strip().lower()
    aliases = {
        "llm": EvalMethod.LLM_JUDGE,
        "judge": EvalMethod.LLM_JUDGE,
        "llm_judge": EvalMethod.LLM_JUDGE,
        "agent": EvalMethod.AGENT_EVAL,
        "agent_judge": EvalMethod.AGENT_EVAL,
        "agent_eval": EvalMethod.AGENT_EVAL,
        "rule": EvalMethod.RULE_EVALUATOR,
        "rule_eval": EvalMethod.RULE_EVALUATOR,
        "rule_judge": EvalMethod.RULE_EVALUATOR,
        "rule_evaluator": EvalMethod.RULE_EVALUATOR,
    }
    try:
        return aliases[text]
    except KeyError as exc:
        supported = ", ".join(sorted(aliases))
        raise ValueError(f"unsupported eval method {value!r}; supported methods: {supported}") from exc


def normalize_evaluator_agent_type(value: str | None) -> str:
    text = str(value or "docker_container").strip().lower()
    aliases = {
        "container": "docker_container",
        "docker": "docker_container",
        "docker_container": "docker_container",
        "docker-agent": "docker_container",
        "docker_agent": "docker_container",
        "cli": "codex_cli",
        "codex": "codex_cli",
        "codex_cli": "codex_cli",
        "codex-cli": "codex_cli",
    }
    try:
        return aliases[text]
    except KeyError as exc:
        supported = ", ".join(sorted(SUPPORTED_EVALUATOR_AGENT_TYPES))
        raise ValueError(f"unsupported evaluator agent_type {value!r}; supported types: {supported}") from exc


def validate_eval_specs(specs: list[EvalSpec]) -> None:
    seen: set[str] = set()
    for spec in specs:
        if not spec.eval_id:
            raise ValueError("eval_id must not be empty")
        if spec.eval_id in seen:
            raise ValueError(f"duplicate eval_id: {spec.eval_id}")
        seen.add(spec.eval_id)
        if spec.score_max <= spec.score_min:
            raise ValueError(f"{spec.eval_id}: score_max must be greater than score_min")
        if spec.weight < 0:
            raise ValueError(f"{spec.eval_id}: weight must be >= 0")
        if spec.method == EvalMethod.AGENT_EVAL:
            if spec.evaluator_agent_type:
                normalize_evaluator_agent_type(spec.evaluator_agent_type)
            invalid = set(spec.evaluator_base_agents) - {"codex", "claude_code"}
            if invalid:
                raise ValueError(f"{spec.eval_id}: unsupported evaluator_base_agents: {sorted(invalid)}")
            if spec.target_access_mode == "direct_docker" and not spec.requires_container:
                raise ValueError(f"{spec.eval_id}: direct_docker requires requires_container=true")


def merge_eval_results(
    results: list[EvalResult],
    specs: list[EvalSpec],
    *,
    fail_policy: str = "zero_reward",
) -> EvalResult:
    if not results:
        return EvalResult.failed(session_id="", reason="no eval results")

    session_id = results[0].session_id
    weight_by_id = {spec.eval_id: spec.weight for spec in specs}
    succeeded: list[tuple[EvalResult, float]] = []
    for index, result in enumerate(results):
        eval_id = result.eval_id or (specs[index].eval_id if index < len(specs) else None)
        weight = weight_by_id.get(eval_id or "", specs[index].weight if index < len(specs) else 1.0)
        if result.status == EvalStatus.SUCCEEDED.value and weight > 0:
            succeeded.append((result, weight))

    method_results = [result.to_method_result() for result in results]
    artifacts = {"method_count": len(results)}
    if not succeeded:
        reason = "all eval specs failed"
        return EvalResult(
            session_id=session_id,
            status=EvalStatus.FAILED.value,
            normalized_score_10=0.0,
            reason=reason,
            method_results=method_results,
            artifacts=artifacts,
            error_text=reason,
        )

    total_weight = sum(weight for _, weight in succeeded)
    score = sum(result.normalized_score_10 * weight for result, weight in succeeded) / total_weight
    failed_count = len(results) - len(succeeded)
    return EvalResult(
        session_id=session_id,
        status=EvalStatus.SUCCEEDED.value,
        normalized_score_10=normalize_score(score, 0, 10),
        raw_score=score,
        reason=f"merged {len(succeeded)} succeeded eval result(s), {failed_count} failed/skipped",
        method_results=method_results,
        artifacts=artifacts,
    )


def safe_dumps(value: Any) -> str:
    import json

    return json.dumps(to_jsonable(value), ensure_ascii=False, sort_keys=True)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return to_jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(item) for item in value]
    return value
