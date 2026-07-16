from __future__ import annotations

import asyncio
import json
import sqlite3
import time
from pathlib import Path
from typing import Any

from evaluator.eval_types import Trajectory


class TrajectoryReader:
    def __init__(self, *, db_url: str, storage_type: str = "sqlite") -> None:
        if storage_type != "sqlite":
            raise ValueError("TrajectoryReader MVP only supports sqlite")
        self.db_path = _sqlite_path(db_url)

    async def read_by_session(self, session_id: str) -> Trajectory:
        return await asyncio.to_thread(self._read_by_session_sync, session_id)

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

    def _read_by_session_sync(self, session_id: str) -> Trajectory:
        if not Path(self.db_path).exists():
            return Trajectory(session_id=session_id, warnings=[f"db not found: {self.db_path}"])
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT *
                FROM session_steps
                WHERE session_id = ?
                ORDER BY step_id ASC, id ASC
                """,
                (session_id,),
            ).fetchall()

        all_steps = [self.parse_gateway_row(dict(row)) for row in rows]
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
    if not isinstance(value, str):
        return str(value)

    text = value.strip()
    parsed = _json_loads(text, default=None)
    if not isinstance(parsed, dict):
        return value

    choice_text = _extract_choice_text(parsed)
    if choice_text:
        return choice_text

    for key in ("output_text", "text", "content"):
        item = parsed.get(key)
        if isinstance(item, str) and item.strip():
            return item

    stream_text = parsed.get("stream_text")
    if isinstance(stream_text, str):
        extracted = _extract_stream_text(stream_text)
        if extracted:
            return extracted

    return value


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
