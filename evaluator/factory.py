from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any

from evaluator.backends import (
    AgentEvalBackend,
    LLMJudgeBackend,
    OpenAICompatibleJudgeClient,
    RuleEvaluatorBackend,
)
from evaluator.config_loader import (
    load_evaluation_config,
    parse_default_specs,
    parse_evaluator_pool_specs,
    parse_judge_definition_dirs,
)
from evaluator.eval_types import EvalMethod
from evaluator.evaluator_pool import (
    CodexCliEvaluatorLeaseManager,
    CompositeEvaluatorLeaseManager,
    DockerEvaluatorContainerManager,
    EvaluatorAgentPool,
    EvaluatorContainerManager,
    EvaluatorLeaseManager,
    SyntheticEvaluatorContainerManager,
)
from evaluator.judges import JudgeRegistry
from evaluator.service import EvaluationService
from evaluator.trajectory_reader import TrajectoryReader

log = logging.getLogger("evaluator.factory")


@dataclass
class EvaluationRuntime:
    service: EvaluationService
    evaluator_pool: EvaluatorAgentPool
    judge_registry: JudgeRegistry

    async def start(self) -> None:
        await self.evaluator_pool.start()

    async def stop(self) -> None:
        await self.evaluator_pool.stop()


def build_evaluation_runtime(
    *,
    config: dict[str, Any] | None = None,
    config_path: str | Path | None = None,
    gateway_base_url: str | None = None,
    evaluation_model: str | None = None,
    trajectory_reader: TrajectoryReader | None = None,
    judge_client: Any | None = None,
    evaluator_container_manager: EvaluatorContainerManager | None = None,
    evaluator_lease_manager: EvaluatorLeaseManager | None = None,
    registry: Any | None = None,
) -> EvaluationRuntime:
    """Build evaluator runtime objects from evaluator-only config.

    The function does not touch manager/gateway internals. Runtime integration
    can inject a TrajectoryReader, judge_client, and container manager.
    """

    cfg = dict(config or {})
    if config_path is not None:
        cfg.update(load_evaluation_config(config_path))
    evaluation_model = str(evaluation_model or cfg.get("evaluation_model") or "").strip()
    gateway_base_url = str(gateway_base_url or cfg.get("gateway_base_url") or "").strip()
    log.info(
        "EVAL RUNTIME build: config_path=%s max_concurrency=%s fail_policy=%s start_real_evaluator_containers=%s evaluation_model=%s",
        config_path,
        cfg.get("max_concurrency") or 64,
        cfg.get("fail_policy") or "zero_reward",
        bool(cfg.get("start_real_evaluator_containers")),
        evaluation_model or "",
    )

    judge_registry = JudgeRegistry(definition_dirs=parse_judge_definition_dirs(cfg))
    judge_base_url = str(
        cfg.get("judge_base_url") or _standard_openai_base_url(gateway_base_url) or ""
    ).strip()
    judge_client = judge_client or OpenAICompatibleJudgeClient(
        base_url=judge_base_url or None,
        api_key=cfg.get("judge_api_key"),
    )
    pool_specs = parse_evaluator_pool_specs(cfg)
    _apply_evaluation_defaults(
        pool_specs,
        gateway_base_url=gateway_base_url,
        evaluation_model=evaluation_model,
    )
    log.info(
        "EVAL RUNTIME pools: count=%d pools=%s judge_definition_dirs=%s",
        len(pool_specs),
        [
            {
                "pool_id": pool.pool_id,
                "members": [
                    {
                        "evaluator_agent_id": member.evaluator_agent_id,
                        "agent_type": member.agent_type,
                    }
                    for member in pool.members
                ],
            }
            for pool in pool_specs
        ],
        parse_judge_definition_dirs(cfg),
    )
    container_manager = evaluator_lease_manager or evaluator_container_manager
    if container_manager is None:
        docker_manager = (
            DockerEvaluatorContainerManager()
            if cfg.get("start_real_evaluator_containers")
            else SyntheticEvaluatorContainerManager()
        )
        container_manager = CompositeEvaluatorLeaseManager(
            {
                "docker_container": docker_manager,
                "codex_cli": CodexCliEvaluatorLeaseManager(
                    runtime_root=cfg.get("codex_cli_runtime_dir") or cfg.get("cli_runtime_dir")
                ),
            }
        )
    evaluator_pool = EvaluatorAgentPool(
        pool_specs=pool_specs,
        container_manager=container_manager,
    )

    backends = {
        EvalMethod.LLM_JUDGE: LLMJudgeBackend(
            judge_registry=judge_registry,
            judge_client=judge_client,
            default_judge_model=str(evaluation_model or cfg.get("default_judge_model") or "gpt-4o-mini"),
            judge_model_override=evaluation_model or None,
        ),
        EvalMethod.AGENT_EVAL: AgentEvalBackend(
            evaluator_pool=evaluator_pool,
            gateway_base_url=gateway_base_url or None,
            evaluation_model=evaluation_model or None,
        ),
        EvalMethod.RULE_EVALUATOR: RuleEvaluatorBackend(
            env_root=cfg.get("rule_evaluator_env_root") or "env",
        ),
    }

    service = EvaluationService(
        trajectory_reader=trajectory_reader,
        backends=backends,
        registry=registry,
        max_concurrency=int(cfg.get("max_concurrency") or 64),
        fail_policy=str(cfg.get("fail_policy") or "zero_reward"),
        default_specs=parse_default_specs(cfg),
    )
    return EvaluationRuntime(
        service=service,
        evaluator_pool=evaluator_pool,
        judge_registry=judge_registry,
    )


def build_evaluation_service(**kwargs: Any) -> EvaluationService:
    return build_evaluation_runtime(**kwargs).service


def _apply_evaluation_defaults(
    pool_specs: list[Any],
    *,
    gateway_base_url: str,
    evaluation_model: str,
) -> None:
    for pool in pool_specs:
        for member in pool.members:
            if gateway_base_url:
                member.gateway_base_url = gateway_base_url
                member.env.setdefault("SAFACTORY_GATEWAY_BASE_URL", gateway_base_url)
            if evaluation_model:
                if member.agent_type == "docker_container":
                    member.model = evaluation_model
                member.env.setdefault("SAFACTORY_EVALUATION_MODEL", evaluation_model)


def _standard_openai_base_url(gateway_base_url: str | None) -> str | None:
    if not gateway_base_url:
        return None
    normalized = str(gateway_base_url).rstrip("/")
    if normalized.endswith("/sessions"):
        return normalized[: -len("/sessions")]
    marker = "/sessions/"
    if marker in normalized:
        return normalized.split(marker, 1)[0]
    return normalized
