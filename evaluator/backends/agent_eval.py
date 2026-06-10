from __future__ import annotations

import logging
import time

from evaluator.agent_adapters import EvaluatorAgentAdapterRegistry
from evaluator.container_runner import DockerContainerRunner
from evaluator.eval_types import (
    AgentEvalTask,
    EvalRequest,
    EvalResult,
    EvalSpec,
    EvalStatus,
    EvaluatorAgentLease,
    EvaluatorRunResult,
    TargetAgentRef,
    Trajectory,
    normalize_evaluator_agent_type,
)
from evaluator.evaluator_pool import EvaluatorAgentPool
from evaluator.local_process_runner import LocalProcessRunner
from evaluator.output_parsers import JsonScoreReasonParser
from evaluator.target_access import TargetContainerAccessService
from evaluator.task_renderer import EvaluatorTaskRenderer

log = logging.getLogger("evaluator.backends.agent_eval")


class EvaluatorRunnerRegistry:
    def __init__(
        self,
        *,
        docker_runner: DockerContainerRunner | None = None,
        cli_runner: LocalProcessRunner | None = None,
    ) -> None:
        self._runners = {
            "docker_container": docker_runner or DockerContainerRunner(),
            "codex_cli": cli_runner or LocalProcessRunner(),
        }

    def get(self, agent_type: str):
        agent_type = normalize_evaluator_agent_type(agent_type)
        try:
            return self._runners[agent_type]
        except KeyError as exc:
            raise KeyError(f"no evaluator runner registered for agent_type={agent_type!r}") from exc


class AgentEvalBackend:
    def __init__(
        self,
        *,
        evaluator_pool: EvaluatorAgentPool,
        target_access_service: TargetContainerAccessService | None = None,
        task_renderer: EvaluatorTaskRenderer | None = None,
        adapter_registry: EvaluatorAgentAdapterRegistry | None = None,
        runner: DockerContainerRunner | None = None,
        runner_registry: EvaluatorRunnerRegistry | None = None,
        gateway_base_url: str | None = None,
        evaluation_model: str | None = None,
        timeout_s: float = 900.0,
    ) -> None:
        self.evaluator_pool = evaluator_pool
        self.target_access_service = target_access_service or TargetContainerAccessService()
        self.task_renderer = task_renderer or EvaluatorTaskRenderer()
        self.adapter_registry = adapter_registry or EvaluatorAgentAdapterRegistry()
        self.runner_registry = runner_registry or EvaluatorRunnerRegistry(docker_runner=runner)
        self.runner = runner or self.runner_registry.get("docker_container")
        self.gateway_base_url = str(gateway_base_url or "").rstrip("/") or None
        self.evaluation_model = str(evaluation_model or "").strip() or None
        self.timeout_s = timeout_s
        self.parser = JsonScoreReasonParser()

    async def evaluate(
        self,
        *,
        request: EvalRequest,
        spec: EvalSpec,
        trajectory: Trajectory,
    ) -> EvalResult:
        target: TargetAgentRef | None = None
        lease: EvaluatorAgentLease | None = None
        reusable = True
        started_at = time.perf_counter()
        try:
            log.info(
                "EVAL BACKEND agent_eval start: session=%s eval_id=%s pool=%s evaluator_agent=%s target_access=%s",
                request.session_id,
                spec.eval_id,
                spec.evaluator_pool_id,
                spec.evaluator_agent_id,
                spec.target_access_mode,
            )
            target = await self.target_access_service.prepare_target_access(request=request, spec=spec)
            log.info(
                "EVAL BACKEND agent_eval target ready: session=%s eval_id=%s container=%s access=%s",
                request.session_id,
                spec.eval_id,
                target.container_name,
                target.access_mode,
            )
            lease = await self.evaluator_pool.acquire(
                evaluator_pool_id=spec.evaluator_pool_id,
                evaluator_agent_id=spec.evaluator_agent_id,
                evaluator_agent_type=spec.evaluator_agent_type,
                required_capabilities=spec.evaluator_required_capabilities,
                allowed_base_agents=spec.evaluator_base_agents,
                target_access_mode=spec.target_access_mode,
                timeout_s=min(spec.timeout_s, self.timeout_s),
            )
            log.info(
                "EVAL BACKEND agent_eval lease acquired: session=%s eval_id=%s evaluator_agent=%s agent_type=%s base_agent=%s runtime=%s",
                request.session_id,
                spec.eval_id,
                lease.evaluator_agent_id,
                lease.spec.agent_type,
                lease.base_agent,
                lease.container_name,
            )
            task = self.build_evaluator_task(
                request=request,
                spec=spec,
                trajectory=trajectory,
                target=target,
            )
            run_result = await self.run_evaluator_agent(
                evaluator_lease=lease,
                task=task,
                spec=spec,
                timeout_s=min(spec.timeout_s, self.timeout_s),
            )
            result = self.parse_evaluator_output(run_result, request=request, spec=spec)
            result.artifacts.update(
                {
                    "evaluator_agent_id": lease.evaluator_agent_id,
                    "evaluator_agent_type": lease.spec.agent_type,
                    "evaluator_base_agent": lease.base_agent,
                    "evaluator_container": lease.container_name,
                    "evaluator_runtime_dir": lease.spec.runtime_dir,
                    "target": target,
                }
            )
            reusable = result.status == EvalStatus.SUCCEEDED.value
            log.info(
                "EVAL BACKEND agent_eval complete: session=%s eval_id=%s status=%s score=%.4f elapsed=%.2fs",
                request.session_id,
                spec.eval_id,
                result.status,
                result.normalized_score_10,
                time.perf_counter() - started_at,
            )
            return result
        except TimeoutError as exc:
            reusable = False
            log.warning(
                "EVAL BACKEND agent_eval timeout: session=%s eval_id=%s elapsed=%.2fs",
                request.session_id,
                spec.eval_id,
                time.perf_counter() - started_at,
            )
            return EvalResult.failed(
                session_id=request.session_id,
                eval_id=spec.eval_id,
                method=spec.method.value,
                status=EvalStatus.TIMEOUT.value,
                reason="agent_eval timed out",
                error_text=str(exc),
            )
        except Exception as exc:
            reusable = False
            log.exception(
                "EVAL BACKEND agent_eval failed: session=%s eval_id=%s elapsed=%.2fs",
                request.session_id,
                spec.eval_id,
                time.perf_counter() - started_at,
            )
            return EvalResult.failed(
                session_id=request.session_id,
                eval_id=spec.eval_id,
                method=spec.method.value,
                reason="agent_eval failed",
                error_text=str(exc),
            )
        finally:
            if lease is not None:
                await self.evaluator_pool.release(lease, reusable=reusable)
            if target is not None:
                await self.target_access_service.cleanup_target_access(target)

    def build_evaluator_task(
        self,
        *,
        request: EvalRequest,
        spec: EvalSpec,
        trajectory: Trajectory,
        target: TargetAgentRef,
    ) -> AgentEvalTask:
        instruction = (
            spec.evaluator_task_template
            or request.env_params.get("task")
            or request.env_params.get("instruction")
            or "Evaluate the target agent and return the required JSON result."
        )
        variables = dict(spec.variables or {})
        if self.evaluation_model:
            variables["evaluation_model"] = self.evaluation_model
        if self.gateway_base_url:
            variables["gateway_base_url"] = self.gateway_base_url
        return AgentEvalTask(
            eval_task_id=f"{request.session_id}:{spec.eval_id}",
            session_id=request.session_id,
            instruction=str(instruction),
            task_input=dict(spec.evaluator_task_input or {}),
            rubric=dict(spec.rubric or request.env_params.get("rubric") or {}),
            variables=variables,
            target=target,
            trajectory=trajectory,
            artifacts={
                "start_result": request.start_result,
                "env_artifacts": request.env_params.get("artifact_paths") or {},
            },
            output_contract={
                "score": "number",
                "passed": "boolean",
                "reason": "string",
                "findings": "array",
                "artifacts": "object",
            },
            timeout_s=spec.timeout_s,
            evaluation_model=self.evaluation_model,
            gateway_base_url=self.gateway_base_url,
        )

    async def run_evaluator_agent(
        self,
        *,
        evaluator_lease: EvaluatorAgentLease,
        task: AgentEvalTask,
        spec: EvalSpec,
        timeout_s: float,
    ) -> EvaluatorRunResult:
        adapter = self.adapter_registry.get(evaluator_lease.base_agent)
        runner = self.runner_registry.get(evaluator_lease.spec.agent_type)
        rendered_prompt = self.task_renderer.render(task=task, spec=spec)
        log.info(
            "EVAL BACKEND agent_eval preparing inputs: session=%s eval_id=%s prompt_chars=%d",
            task.session_id,
            spec.eval_id,
            len(rendered_prompt),
        )
        await adapter.prepare_inputs(
            lease=evaluator_lease,
            task=task,
            rendered_prompt=rendered_prompt,
            runner=runner,
        )
        command = adapter.build_command(spec=evaluator_lease.spec, task=task)
        log.info(
            "EVAL BACKEND agent_eval running evaluator: session=%s eval_id=%s agent_type=%s command_len=%d timeout_s=%.2f",
            task.session_id,
            spec.eval_id,
            evaluator_lease.spec.agent_type,
            len(command),
            timeout_s,
        )
        stdout = await runner.exec(lease=evaluator_lease, command=command, timeout_s=timeout_s)
        last_result = runner.get_last_result(evaluator_lease) if hasattr(runner, "get_last_result") else None
        stderr = getattr(last_result, "stderr", "") if last_result is not None else ""
        exit_code = getattr(last_result, "exit_code", 0) if last_result is not None else 0
        result_text = None
        try:
            result_text = await runner.read_file(
                lease=evaluator_lease,
                path=spec.evaluator_output_path or evaluator_lease.spec.result_path,
                timeout_s=30.0,
            )
        except Exception:
            log.info(
                "EVAL BACKEND agent_eval result file unavailable, using stdout: session=%s eval_id=%s path=%s",
                task.session_id,
                spec.eval_id,
                spec.evaluator_output_path or evaluator_lease.spec.result_path,
            )
            result_text = None
        log.info(
            "EVAL BACKEND agent_eval raw output: session=%s eval_id=%s stdout_chars=%d result_chars=%d",
            task.session_id,
            spec.eval_id,
            len(stdout or ""),
            len(result_text or ""),
        )
        return adapter.normalize_output(
            run_result=EvaluatorRunResult(
                stdout=stdout,
                stderr=stderr,
                result_text=result_text,
                result_path=spec.evaluator_output_path or evaluator_lease.spec.result_path,
                exit_code=exit_code,
            ),
            spec=evaluator_lease.spec,
        )

    def parse_evaluator_output(
        self,
        output: EvaluatorRunResult,
        *,
        request: EvalRequest,
        spec: EvalSpec,
    ) -> EvalResult:
        text = output.result_text or output.stdout
        result = self.parser.parse(text=text, request=request, spec=spec, definition=None)
        result.artifacts.update(
            {
                "evaluator_stdout": output.stdout[:8000],
                "evaluator_stderr": output.stderr[:8000],
                "evaluator_result_path": output.result_path,
                "evaluator_artifacts": output.artifacts,
            }
        )
        return result
