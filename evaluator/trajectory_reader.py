from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from core.data_manager.manager import DataManager
from evaluator.eval_types import Trajectory


class TrajectoryReader:
    def __init__(
        self,
        *,
        db_url: str,
        storage_type: str = "sqlite",
        data_manager: Any | None = None,
    ) -> None:
        self.storage_type = str(storage_type or "sqlite").strip().lower()
        if self.storage_type not in {"sqlite", "cloud"}:
            raise ValueError(f"TrajectoryReader does not support storage type {storage_type!r}")
        if data_manager is None:
            if self.storage_type == "cloud":
                raise ValueError("TrajectoryReader cloud mode requires a data manager")
            data_manager = DataManager(job_id="", storage_type="sqlite", db_url=db_url)
        self.data_manager = data_manager

    async def read_by_session(self, session_id: str) -> Trajectory:
        init = getattr(self.data_manager, "init", None)
        if callable(init):
            await init()
        rows = await self.data_manager.list_session_steps(
            session_id,
            checkout_latest=True,
        )
        return self._trajectory_from_rows(session_id, rows)

    async def wait_until_sealed(
        self,
        session_id: str,
        *,
        timeout_s: float = 30.0,
        poll_interval_s: float = 0.2,
    ) -> Trajectory:
        deadline = time.monotonic() + timeout_s
        last = Trajectory(session_id=session_id, warnings=["trajectory not read yet"])
        while True:
            last = await self.read_by_session(session_id)
            if last.sealed or time.monotonic() >= deadline:
                if not last.sealed:
                    last.warnings.append("trajectory may be incomplete: session was not sealed before timeout")
                return last
            await asyncio.sleep(poll_interval_s)

    def _trajectory_from_rows(
        self,
        session_id: str,
        rows: list[dict[str, Any]],
    ) -> Trajectory:
        all_steps = [self.parse_gateway_row(row) for row in rows]
        token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        sealed = False
        final_response = None
        steps: list[dict[str, Any]] = []
        for step in all_steps:
            env_state = step.get("env_state") or {}
            if _is_session_sealing_event(step):
                sealed = True
            if not _is_trajectory_step(step):
                continue
            steps.append(step)
            response = step.get("response") or _last_assistant_content(step.get("messages"))
            if response:
                final_response = response
            for key in token_usage:
                token_usage[key] += int(env_state.get(key) or 0)
        return Trajectory(
            session_id=session_id,
            steps=steps,
            final_response=final_response,
            token_usage=token_usage,
            raw_rows=[dict(row) for row in rows],
            sealed=sealed,
        )

    def parse_gateway_row(self, row: dict[str, Any]) -> dict[str, Any]:
        parsed = dict(row)
        parsed["messages"] = _json_loads(row.get("messages"), default=[])
        parsed["env_state"] = _json_loads(row.get("env_state"), default={})
        parsed["response"] = _extract_response_text(row.get("response"))
        return parsed


def _sqlite_path(db_url: str) -> str:
    if db_url.startswith("sqlite://"):
        return db_url[len("sqlite://") :]
    return db_url


def _json_loads(value: Any, *, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except Exception:
        return default


def _extract_response_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        parsed = value
    elif not isinstance(value, str):
        return str(value)
    else:
        text = value.strip()
        parsed = _json_loads(text, default=None)
    if isinstance(parsed, list):
        text_chunks: list[str] = []
        for item in parsed:
            if isinstance(item, str):
                text_chunks.append(item)
                continue
            if not isinstance(item, dict):
                continue
            for key in ("output_text", "text", "content"):
                content = item.get(key)
                if isinstance(content, str):
                    text_chunks.append(content)
                    break
                if isinstance(content, list):
                    chunks = [
                        str(part.get("text") or part.get("content") or "")
                        for part in content
                        if isinstance(part, dict)
                    ]
                    combined = "".join(chunks)
                    if combined:
                        text_chunks.append(combined)
                        break
        if text_chunks:
            return "".join(text_chunks)
        return json.dumps(parsed, ensure_ascii=False, default=str)
    if not isinstance(parsed, dict):
        return value

    choice_text = _extract_choice_text(parsed)
    if choice_text:
        return choice_text

    for key in ("output_text", "text", "content"):
        item = parsed.get(key)
        if isinstance(item, str) and item.strip():
            return item
        if isinstance(item, list):
            chunks = [
                str(part.get("text") or part.get("content") or "")
                for part in item
                if isinstance(part, dict)
            ]
            combined = "".join(chunks)
            if combined:
                return combined

    stream_text = parsed.get("stream_text")
    if isinstance(stream_text, str):
        extracted = _extract_stream_text(stream_text)
        if extracted:
            return extracted

    if parsed.get("content") == "":
        has_non_text_output = any(
            item not in (None, "", [], {})
            for key, item in parsed.items()
            if key not in {"role", "content", "name"}
        )
        if not has_non_text_output:
            return ""
    return json.dumps(parsed, ensure_ascii=False, default=str)


def _extract_choice_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    chunks: list[str] = []
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        message = choice.get("message")
        if isinstance(message, dict) and isinstance(message.get("content"), str):
            chunks.append(message["content"])
            continue
        delta = choice.get("delta")
        if isinstance(delta, dict) and isinstance(delta.get("content"), str):
            chunks.append(delta["content"])
    return "".join(chunks)


def _extract_stream_text(stream_text: str) -> str:
    chunks: list[str] = []
    for raw_line in stream_text.splitlines():
        line = raw_line.strip()
        if not line.startswith("data:"):
            continue
        data = line[len("data:") :].strip()
        if not data or data == "[DONE]":
            continue
        parsed = _json_loads(data, default=None)
        if isinstance(parsed, dict):
            chunks.append(_extract_choice_text(parsed))
    return "".join(chunks)


def _last_assistant_content(messages: Any) -> str:
    if not isinstance(messages, list):
        return ""
    for message in reversed(messages):
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text") or item.get("content")
                    if isinstance(text, str):
                        parts.append(text)
                elif isinstance(item, str):
                    parts.append(item)
            return "".join(parts)
    return ""


_NON_TRAJECTORY_EVENT_TYPES = {
    "gateway_session_close",
    "episode_summary",
    "evaluation_summary",
}


def _is_session_sealing_event(step: dict[str, Any]) -> bool:
    env_state = step.get("env_state") or {}
    event_type = env_state.get("event_type")
    return bool(
        step.get("is_session_completed")
        or step.get("is_terminal")
        or env_state.get("is_session_completed")
        or event_type in _NON_TRAJECTORY_EVENT_TYPES
    )


def _is_trajectory_step(step: dict[str, Any]) -> bool:
    env_state = step.get("env_state") or {}
    event_type = env_state.get("event_type")
    if event_type in _NON_TRAJECTORY_EVENT_TYPES:
        return False
    if env_state.get("synthetic_stop"):
        return False
    if event_type == "gateway_inference":
        try:
            return int(env_state.get("status_code") or 200) < 400
        except (TypeError, ValueError):
            return True
    return bool(step.get("messages") or step.get("response"))
