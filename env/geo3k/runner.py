#!/usr/bin/env python3
"""Safactory v2 runtime for the Geo3K-VL geometry environment.

This is a port of the v1 ``Geo3kVLTestEnv`` (a ``core.env.BaseEnv`` multi-turn
VL env) onto the v2 external-runtime contract. In v1 the rollout loop lived
outside the env and drove ``reset``/``step``; here the runner owns the whole
multi-turn loop: it calls the gateway session repeatedly, tracks messages,
handles the ``calc_score`` self-check tool, scores with the sympy grader, and
prints one result JSON.

Grading logic (``math_utils.grade_answer_verl``), tool semantics, ``<think>``
stripping, boxed extraction and the turn-cap fallback are reproduced verbatim
from the v1 environment.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import requests

# math_utils.py is embedded next to this file in RJob mode.
sys.path.insert(0, str(Path(__file__).resolve().parent))
extract_boxed_answer: Any = None
grade_answer_verl: Any = None


TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
SUPPORTED_TOOL_NAMES = {"calc_score", "calc_geo3k_reward"}

DEFAULT_MAX_TURNS = 3
DEFAULT_MAX_IMAGES = 1
DEFAULT_TEMPERATURE = 0.3
RESULT_JSON_PREFIX = "SAFACTORY_RESULT_JSON "
RESULT_PATH_ENV = "SAFACTORY_RESULT_PATH"


def main() -> int:
    started_at = time.perf_counter()
    session_id = os.environ.get("SAFACTORY_SESSION_ID", "")

    try:
        request = _read_request()
        session_id = _required_text(request.get("session_id"), "session_id")
        env_params = request.get("env_params") if isinstance(request.get("env_params"), dict) else {}
        dataset = env_params.get("dataset") if isinstance(env_params.get("dataset"), dict) else {}

        max_turns = _int(env_params.get("max_turns"), DEFAULT_MAX_TURNS)
        max_images = _int(env_params.get("max_images"), DEFAULT_MAX_IMAGES)
        echo_images_on_feedback = bool(env_params.get("echo_images_on_feedback", False))

        question = dataset.get("problem", dataset.get("question"))
        if question is None:
            raise RuntimeError("geo3k dataset row requires `problem` (or `question`)")
        question = str(question)

        ground_truth = _normalize_ground_truth(dataset.get("answer", dataset.get("golden_answers")))
        image_urls = _normalize_image_urls(dataset.get("images"))

        base_url = _resolve_base_url(request, session_id)
        model = _first_text(request.get("model"), env_params.get("route_model"), os.environ.get("SAFACTORY_ROUTE_MODEL"))
        if not model:
            raise RuntimeError("geo3k runner could not resolve a target model")
        temperature = _float(request.get("temperature"), DEFAULT_TEMPERATURE)
        timeout_s = _float(request.get("agent_start_timeout_s"), 300.0)

        _load_math_utils()
        state = _RunState(
            base_url=base_url,
            model=model,
            temperature=temperature,
            timeout_s=timeout_s,
            question=question,
            ground_truth=ground_truth,
            image_urls=image_urls,
            max_turns=max_turns,
            max_images=max_images,
            echo_images_on_feedback=echo_images_on_feedback,
        )
        result = state.run()

        _write_result(
            {
                "session_id": session_id,
                "status": "succeeded",
                "total_reward": float(result["score"]),
                "step_count": int(result["turns"]),
                "terminated": bool(result["terminated"]),
                "truncated": bool(result["truncated"]),
                "error_text": None,
                "metrics": {
                    "bench": "geo3k",
                    "task_id": dataset.get("task_id", dataset.get("id")),
                    "score": float(result["score"]),
                    "passed": bool(result["score"] >= 1.0),
                    "ground_truth": ground_truth,
                    "final_answer": result["final_answer"],
                    "latest_boxed_answer": result["latest_boxed_answer"],
                    "reward_source": result["reward_source"],
                    "turns": int(result["turns"]),
                    "max_turns": max_turns,
                    "total_tool_calls": int(result["total_tool_calls"]),
                    "tool_calls": result["tool_calls"],
                    "image_count": len(image_urls),
                    "duration_ms": round((time.perf_counter() - started_at) * 1000, 3),
                },
            }
        )
        return 0
    except Exception as exc:  # noqa: BLE001 - runtime must always emit a result
        _write_result(_failure_result(session_id, str(exc), started_at))
        return 0


class _RunState:
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        temperature: float,
        timeout_s: float,
        question: str,
        ground_truth: str,
        image_urls: list[str],
        max_turns: int,
        max_images: int,
        echo_images_on_feedback: bool,
    ) -> None:
        self.base_url = base_url
        self.model = model
        self.temperature = temperature
        self.timeout_s = timeout_s
        self.question = question
        self.ground_truth = ground_truth
        self.image_urls = image_urls
        self.max_turns = max_turns
        self.max_images = max_images
        self.echo_images_on_feedback = echo_images_on_feedback

        self.messages: list[dict[str, Any]] = []
        self.latest_boxed_answer: str | None = None
        self.last_tool_score: float | None = None
        self.total_tool_calls = 0
        self.tool_calls: list[dict[str, Any]] = []
        self.final_answer: str | None = None
        self.step_count = 0

    def run(self) -> dict[str, Any]:
        user_content: list[dict[str, Any]] = [{"type": "text", "text": self.question}]
        if self.max_images > 0:
            for image_url in self.image_urls[: self.max_images]:
                user_content.append({"type": "image_url", "image_url": {"url": image_url}})
        self.messages = [{"role": "user", "content": user_content}]

        score = 0.0
        reward_source = "missing_boxed_answer"
        terminated = False
        truncated = False

        while True:
            self.step_count += 1
            assistant_msg = self._call_gateway()
            done, score, reward_source, tool_executed = self._process_action(assistant_msg)
            if done:
                terminated = True
                break
            # A tool call was handled; feedback is already appended. Enforce the
            # hard turn cap exactly like the v1 env-level guard.
            if self.step_count >= self.max_turns:
                if self.latest_boxed_answer:
                    score = self._score_answer(self.latest_boxed_answer)
                    reward_source = "latest_boxed_answer"
                else:
                    score = 0.0
                    reward_source = "missing_boxed_answer"
                self.final_answer = self.latest_boxed_answer
                terminated = True
                break

        return {
            "score": score,
            "reward_source": reward_source,
            "terminated": terminated,
            "truncated": truncated,
            "final_answer": self.final_answer,
            "latest_boxed_answer": self.latest_boxed_answer,
            "total_tool_calls": self.total_tool_calls,
            "tool_calls": list(self.tool_calls),
            "turns": self.step_count,
        }

    def _call_gateway(self) -> str:
        response = requests.post(
            f"{self.base_url}/chat/completions",
            json={
                "model": self.model,
                "messages": self.messages,
                "temperature": self.temperature,
            },
            timeout=self.timeout_s,
        )
        response.raise_for_status()
        body = response.json()
        return str(body["choices"][0]["message"].get("content", "") or "")

    def _process_action(self, assistant_msg: str) -> tuple[bool, float, str, bool]:
        """Mirror of v1 ``_process_action``. Returns (done, score, reward_source, tool_executed)."""
        msg = (assistant_msg or "").strip()
        self.messages.append({"role": "assistant", "content": msg})
        msg_wo_think = re.sub(r"<think>.*?</think>", "", msg, flags=re.DOTALL).strip()

        boxed_answer = extract_boxed_answer(msg_wo_think)
        if boxed_answer is not None:
            boxed_answer = boxed_answer.strip()
            if boxed_answer:
                self.latest_boxed_answer = boxed_answer

        tool_call = self._extract_tool_call(msg_wo_think)
        if tool_call is not None:
            self._handle_tool_call(tool_call)
            return False, 0.0, "", True

        self.final_answer = self.latest_boxed_answer or msg_wo_think
        if self.latest_boxed_answer:
            score = self._score_answer(self.latest_boxed_answer)
            reward_source = "latest_boxed_answer"
        else:
            score = 0.0
            reward_source = "missing_boxed_answer"
        return True, score, reward_source, False

    def _handle_tool_call(self, tool_call: dict[str, Any]) -> None:
        name = str(tool_call.get("name", "")).strip()
        arguments = tool_call.get("arguments", {})

        if name not in SUPPORTED_TOOL_NAMES:
            self._append_tool_feedback(
                f"Tool `{name}` is not supported. "
                'Use `<tool_call>{"name":"calc_score","arguments":{"answer":"..."}}</tool_call>`.'
            )
            return

        if not isinstance(arguments, dict):
            self._append_tool_feedback("Tool arguments must be a JSON object.")
            return

        raw_answer = arguments.get("answer")
        parsed_answer = "" if raw_answer is None else str(raw_answer).strip()
        if not parsed_answer:
            self._append_tool_feedback("Tool call detected but no `answer` was provided.")
            return

        score = self._score_answer(parsed_answer)
        self.last_tool_score = score
        self.total_tool_calls += 1
        self.tool_calls.append({"name": name, "answer": parsed_answer, "score": score})
        self._append_tool_feedback(self._build_tool_feedback(score, parsed_answer))

    def _append_tool_feedback(self, text: str) -> None:
        if self.echo_images_on_feedback and self.max_images > 0 and self.image_urls:
            content: list[dict[str, Any]] = [{"type": "text", "text": text}]
            for image_url in self.image_urls[: self.max_images]:
                content.append({"type": "image_url", "image_url": {"url": image_url}})
            self.messages.append({"role": "user", "content": content})
        else:
            self.messages.append({"role": "user", "content": text})

    def _extract_tool_call(self, text: str) -> dict[str, Any] | None:
        matches = list(TOOL_CALL_RE.finditer(text))
        if not matches:
            return None

        raw_json = matches[-1].group(1).strip()
        try:
            payload = json.loads(raw_json)
        except json.JSONDecodeError:
            return None

        name = payload.get("name") or payload.get("function", {}).get("name")
        arguments = payload.get("arguments") or payload.get("function", {}).get("arguments") or {}
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                return None

        if not name:
            return None
        return {"name": name, "arguments": arguments}

    def _score_answer(self, answer: str) -> float:
        if not self.ground_truth:
            return 0.0
        answer = answer.strip()
        candidates = [answer]
        if "\\boxed" not in answer:
            candidates.append(f"\\boxed{{{answer}}}")
        for candidate in candidates:
            if grade_answer_verl(candidate, self.ground_truth):
                return 1.0
        return 0.0

    def _build_tool_feedback(self, score: float, parsed_answer: str) -> str:
        turn_idx = self.step_count - 1  # zero-based
        last_warning_turn = None
        if self.max_turns is not None:
            if self.max_turns >= 2:
                last_warning_turn = self.max_turns - 2
            else:
                last_warning_turn = self.max_turns - 1
        is_final_turn = last_warning_turn is not None and turn_idx >= last_warning_turn

        if score == 1.0:
            return (
                f"calc_score result: {score}. Parsed answer '{parsed_answer}' matches the reference. "
                "You can now stop reasoning and provide the final solution in \\boxed{}."
            )
        if is_final_turn:
            return (
                f"calc_score result: {score}. Parsed answer '{parsed_answer}' does not match the reference. "
                "Your answer is wrong. You may need to reason in a different way. Don't repeat your answer unless necessary. "
                "Since you only have one chance to answer, don't call tool again. "
                "You should provide your final answer in the form Answer: \\boxed{$Answer} where $Answer is your final answer to this problem."
            )
        return (
            f"calc_score result: {score}. Parsed answer '{parsed_answer}' does not match the reference. "
            "Your answer is wrong. You may need to reason in a different way. Don't repeat your answer unless necessary."
        )


def _resolve_base_url(request: dict[str, Any], session_id: str) -> str:
    base_url = _first_text(
        os.environ.get("SAFACTORY_GATEWAY_SESSION_URL_CONTAINER"),
        _gateway_session_url(request, session_id),
        os.environ.get("OPENROUTER_BASE_URL"),
        os.environ.get("OPENAI_BASE_URL"),
    )
    if not base_url:
        raise RuntimeError("geo3k runner could not resolve an OpenAI-compatible base URL")
    return base_url


def _gateway_session_url(request: dict[str, Any], session_id: str) -> str:
    base = str(request.get("gateway_base_url") or "").rstrip("/")
    if not base:
        return ""
    return _containerize_local_gateway_url(f"{base}/{session_id}")


def _containerize_local_gateway_url(url: str) -> str:
    try:
        parts = urlsplit(str(url))
    except Exception:
        return str(url)
    if parts.hostname not in {"127.0.0.1", "localhost", "::1"}:
        return str(url)
    netloc = "host.docker.internal"
    if parts.port is not None:
        netloc = f"{netloc}:{parts.port}"
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))


def _normalize_ground_truth(raw_answer: Any) -> str:
    if raw_answer is None:
        return ""
    value = raw_answer
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        for item in value:
            s = str(item).strip()
            if s:
                return s
        return ""
    s = str(value).strip()
    if not s:
        return ""
    if s.startswith("[") and s.endswith("]"):
        try:
            obj = json.loads(s)
            if isinstance(obj, list):
                for item in obj:
                    t = str(item).strip()
                    if t:
                        return t
                return ""
        except json.JSONDecodeError:
            return s
    return s


def _normalize_image_urls(raw_images: Any) -> list[str]:
    if raw_images is None:
        return []
    value = raw_images
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return []
        if s.startswith("[") and s.endswith("]"):
            value = json.loads(s)
        else:
            value = [s]
    if not isinstance(value, (list, tuple)):
        return []
    urls: list[str] = []
    for item in value:
        if isinstance(item, str):
            s = item.strip()
            if s:
                urls.append(s)
            continue
        if isinstance(item, dict):
            src = item.get("src") or item.get("url")
            if isinstance(src, str) and src.strip():
                urls.append(src.strip())
            continue
        s = str(item).strip()
        if s:
            urls.append(s)
    return urls


def _read_request() -> dict[str, Any]:
    raw = sys.stdin.read().strip() or os.environ.get("SAFACTORY_START_REQUEST_JSON", "").strip()
    if not raw:
        raise RuntimeError("SimulationStartRequest JSON was not provided on stdin")
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise RuntimeError("SimulationStartRequest must be a JSON object")
    return data


def _load_math_utils() -> None:
    global extract_boxed_answer, grade_answer_verl
    if extract_boxed_answer is not None and grade_answer_verl is not None:
        return
    from math_utils import extract_answer as _extract_answer  # noqa: PLC0415
    from math_utils import grade_answer_verl as _grade_answer_verl  # noqa: PLC0415

    extract_boxed_answer = _extract_answer
    grade_answer_verl = _grade_answer_verl


def _required_text(value: Any, name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise RuntimeError(f"SimulationStartRequest missing {name}")
    return text


def _first_text(*values: Any) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _failure_result(session_id: str, error_text: str, started_at: float) -> dict[str, Any]:
    return {
        "session_id": session_id,
        "status": "failed",
        "total_reward": 0.0,
        "step_count": 0,
        "terminated": True,
        "truncated": False,
        "error_text": error_text,
        "metrics": {
            "bench": "geo3k",
            "duration_ms": round((time.perf_counter() - started_at) * 1000, 3),
        },
    }


def _write_result(result: dict[str, Any]) -> None:
    _persist_result_artifact(result)
    print(RESULT_JSON_PREFIX + json.dumps(result, ensure_ascii=False), flush=True)


def _persist_result_artifact(result: dict[str, Any]) -> None:
    raw_path = str(os.environ.get(RESULT_PATH_ENV) or "").strip()
    if not raw_path:
        return
    try:
        path = Path(raw_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(path.name + ".tmp")
        tmp_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        tmp_path.replace(path)
    except Exception as exc:
        print(f"SAFACTORY_RUNNER_DIAGNOSTIC result_artifact_write_failed: {exc}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
