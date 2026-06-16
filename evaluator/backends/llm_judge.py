from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any

from evaluator.backends.base import LLMJudgeClient
from evaluator.eval_types import EvalRequest, EvalResult, EvalSpec, EvalStatus, Trajectory
from evaluator.input_builders import JudgeInputBuilder, default_input_builders
from evaluator.judges import JudgeDefinition, JudgeRegistry
from evaluator.output_parsers import JudgeOutputParser, default_output_parsers
from evaluator.templating import render_template

log = logging.getLogger("evaluator.backends.llm_judge")


class OpenAICompatibleJudgeClient:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        gateway_session_base_url: str | None = None,
        temperature: float = 0.0,
        close_gateway_session: bool = True,
        trust_env: bool = False,
    ) -> None:
        self.api_key = api_key
        normalized_base_url = _without_session_suffix(base_url) or _without_session_suffix(
            gateway_session_base_url
        )
        self.base_url = normalized_base_url.rstrip("/") if normalized_base_url else None
        self.temperature = temperature
        self.close_gateway_session = close_gateway_session
        self.trust_env = trust_env
        self._clients: dict[str, Any] = {}

    def _get_client(self, base_url: str | None) -> Any:
        key = base_url or ""
        if key not in self._clients:
            import openai
            import httpx

            api_key = self.api_key or os.environ.get("OPENAI_API_KEY") or "safactory"
            kwargs: dict[str, Any] = {
                "api_key": api_key,
                "http_client": httpx.Client(trust_env=self.trust_env),
            }
            if base_url:
                kwargs["base_url"] = base_url
            self._clients[key] = openai.OpenAI(**kwargs)
        return self._clients[key]

    async def complete(
        self,
        *,
        messages: list[dict[str, Any]],
        model: str,
        timeout_s: float,
        request: EvalRequest | None = None,
        eval_id: str | None = None,
    ) -> str:
        def _call() -> str:
            response = self._get_client(self.base_url).chat.completions.create(
                model=model,
                messages=messages,
                temperature=self.temperature,
                timeout=timeout_s,
            )
            return response.choices[0].message.content or ""

        return await asyncio.wait_for(asyncio.to_thread(_call), timeout=timeout_s + 5.0)


def _without_session_suffix(base_url: str | None) -> str | None:
    if not base_url:
        return None
    normalized = str(base_url).rstrip("/")
    if normalized.endswith("/sessions"):
        return normalized[: -len("/sessions")]
    marker = "/sessions/"
    if marker in normalized:
        return normalized.split(marker, 1)[0]
    return normalized


class LLMJudgeBackend:
    def __init__(
        self,
        *,
        judge_registry: JudgeRegistry,
        judge_client: LLMJudgeClient,
        input_builders: dict[str, JudgeInputBuilder] | None = None,
        output_parsers: dict[str, JudgeOutputParser] | None = None,
        default_judge_model: str = "gpt-4o-mini",
        judge_model_override: str | None = None,
        timeout_s: float = 120.0,
    ) -> None:
        self.judge_registry = judge_registry
        self.judge_client = judge_client
        self.input_builders = input_builders or default_input_builders()
        self.output_parsers = output_parsers or default_output_parsers()
        self.default_judge_model = default_judge_model
        self.judge_model_override = str(judge_model_override or "").strip() or None
        self.timeout_s = timeout_s

    async def evaluate(
        self,
        *,
        request: EvalRequest,
        spec: EvalSpec,
        trajectory: Trajectory,
    ) -> EvalResult:
        started_at = time.perf_counter()
        try:
            definition = self.resolve_judge_definition(spec=spec, request=request)
            messages = self.build_judge_messages(
                request=request,
                spec=spec,
                definition=definition,
                trajectory=trajectory,
            )
            model = self.judge_model_override or spec.judge_model or definition.model or self.default_judge_model
            prompt_chars = sum(len(str(message.get("content") or "")) for message in messages)
            log.info(
                "EVAL BACKEND llm_judge start: session=%s eval_id=%s judge_id=%s version=%s model=%s prompt_chars=%d timeout_s=%.2f",
                request.session_id,
                spec.eval_id,
                definition.judge_id,
                definition.version,
                model,
                prompt_chars,
                min(spec.timeout_s, self.timeout_s),
            )
            text = await self._complete_with_context(
                messages=messages,
                model=model,
                timeout_s=min(spec.timeout_s, self.timeout_s),
                request=request,
                eval_id=spec.eval_id,
            )
            log.info(
                "EVAL BACKEND llm_judge response: session=%s eval_id=%s response_chars=%d",
                request.session_id,
                spec.eval_id,
                len(text or ""),
            )
            result = self.parse_judge_response(text, request=request, spec=spec, definition=definition)
            result.artifacts.update(
                {
                    "judge_id": definition.judge_id,
                    "judge_version": definition.version,
                    "judge_model": model,
                    "input_builder": definition.input_builder,
                    "output_parser": definition.output_parser,
                    "raw_judge_response": text,
                }
            )
            log.info(
                "EVAL BACKEND llm_judge complete: session=%s eval_id=%s status=%s score=%.4f elapsed=%.2fs",
                request.session_id,
                spec.eval_id,
                result.status,
                result.normalized_score_10,
                time.perf_counter() - started_at,
            )
            return result
        except Exception as exc:
            log.exception(
                "EVAL BACKEND llm_judge failed: session=%s eval_id=%s elapsed=%.2fs",
                request.session_id,
                spec.eval_id,
                time.perf_counter() - started_at,
            )
            return EvalResult.failed(
                session_id=request.session_id,
                eval_id=spec.eval_id,
                method=spec.method.value,
                reason="llm_judge failed",
                error_text=str(exc),
            )

    def resolve_judge_definition(
        self,
        *,
        spec: EvalSpec,
        request: EvalRequest,
    ) -> JudgeDefinition:
        if spec.judge_id:
            return self.judge_registry.get(spec.judge_id, spec.judge_version)
        if spec.judge_prompt_template:
            return JudgeDefinition(
                judge_id=spec.prompt_template_id or spec.eval_id,
                version=spec.judge_version or "inline",
                model=spec.judge_model or self.default_judge_model,
                prompt_template=spec.judge_prompt_template,
                input_builder=spec.input_builder,
                output_parser=spec.output_parser,
                score_min=spec.score_min,
                score_max=spec.score_max,
            )
        default_judge = request.env_params.get("default_judge_id") or "general_task_judge"
        return self.judge_registry.get(str(default_judge), None)

    def build_judge_messages(
        self,
        *,
        request: EvalRequest,
        spec: EvalSpec,
        definition: JudgeDefinition,
        trajectory: Trajectory,
    ) -> list[dict[str, Any]]:
        builder_name = spec.input_builder or definition.input_builder
        builder = self.input_builders[builder_name]
        variables = builder.build(request=request, spec=spec, trajectory=trajectory)
        prompt = render_template(definition.prompt_template, variables)
        return [
            {"role": "system", "content": "You are a strict evaluator. Output only the requested JSON."},
            {"role": "user", "content": prompt},
        ]

    async def _complete_with_context(
        self,
        *,
        messages: list[dict[str, Any]],
        model: str,
        timeout_s: float,
        request: EvalRequest,
        eval_id: str,
    ) -> str:
        try:
            return await self.judge_client.complete(
                messages=messages,
                model=model,
                timeout_s=timeout_s,
                request=request,
                eval_id=eval_id,
            )
        except TypeError as exc:
            if "unexpected keyword" not in str(exc):
                raise
            return await self.judge_client.complete(
                messages=messages,
                model=model,
                timeout_s=timeout_s,
            )

    def parse_judge_response(
        self,
        text: str,
        *,
        request: EvalRequest,
        spec: EvalSpec,
        definition: JudgeDefinition,
    ) -> EvalResult:
        parser_name = spec.output_parser or definition.output_parser
        parser = self.output_parsers[parser_name]
        result = parser.parse(text=text, request=request, spec=spec, definition=definition)
        result.status = EvalStatus.SUCCEEDED.value
        return result
