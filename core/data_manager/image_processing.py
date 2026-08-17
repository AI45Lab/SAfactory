"""Message image persistence kept at the data-manager boundary.

Storage DAOs receive message payloads whose binary images have already been
externalized. SQLite bypasses this component and remains self-contained.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import re
from typing import Any, Dict, List


log = logging.getLogger("core.data_manager.image_processing")


class MessageImageProcessor:
    def __init__(
        self,
        *,
        job_id: str,
        uploader: Any = None,
        fallback_dir: str = "saved_images",
        max_retries: int = 3,
    ) -> None:
        self.job_id = job_id
        self.uploader = uploader
        self.fallback_dir = fallback_dir
        self.max_retries = max(1, int(max_retries))
        self._history: Dict[tuple[str, str], List[Dict[str, Any]]] = {}

    async def process_rows(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        processed: List[Dict[str, Any]] = []
        for source in rows:
            row = dict(source)
            messages = row.get("messages")
            if isinstance(messages, str):
                try:
                    messages = json.loads(messages)
                except Exception:
                    messages = None
            if not isinstance(messages, list):
                processed.append(row)
                continue

            cache_key = (
                str(row.get("session_id") or ""),
                str(row.get("llm_model") or ""),
            )
            previous = self._history.get(cache_key, [])
            if previous and len(messages) >= len(previous):
                prefix = list(previous)
                pending = messages[len(previous):]
            else:
                prefix = []
                pending = messages
            env_key = f"{row.get('env_name') or 'gateway'}_{row.get('session_id') or ''}"
            suffix = await self._process_messages(
                pending,
                env_key=env_key,
                step_id=int(row.get("step_id") or 0),
            )
            row["messages"] = prefix + suffix
            self._history[cache_key] = row["messages"]
            processed.append(row)
        return processed

    async def _process_messages(
        self,
        messages: List[Dict[str, Any]],
        *,
        env_key: str,
        step_id: int,
    ) -> List[Dict[str, Any]]:
        processed: List[Dict[str, Any]] = []
        image_count = 0
        for message_index, message in enumerate(messages):
            if not isinstance(message, dict) or not isinstance(message.get("content"), list):
                processed.append(message)
                continue
            new_message = dict(message)
            new_content: List[Any] = []
            for item in message["content"]:
                if not isinstance(item, dict) or item.get("type") != "image_url":
                    new_content.append(item)
                    continue
                image_url = item.get("image_url")
                url = image_url.get("url", "") if isinstance(image_url, dict) else ""
                match = re.fullmatch(r"data:image/([\w.+-]+);base64,(.+)", url, re.DOTALL)
                if not match:
                    new_content.append(item)
                    continue
                extension, payload = match.groups()
                file_name = f"step_{step_id}_m{message_index}_i{image_count}.{extension}"
                final_url = await self._store_image(
                    payload,
                    env_key=env_key,
                    file_name=file_name,
                )
                new_item = dict(item)
                new_item["image_url"] = dict(image_url)
                new_item["image_url"]["url"] = final_url
                new_content.append(new_item)
                image_count += 1
            new_message["content"] = new_content
            processed.append(new_message)
        return processed

    async def _store_image(self, payload: str, *, env_key: str, file_name: str) -> str:
        try:
            image = base64.b64decode(payload)
        except Exception as exc:
            log.warning("Cannot decode message image %s: %s", file_name, exc)
            return f"[IMAGE_DECODE_FAILED:{file_name}]"

        local_dir = os.path.join(self.fallback_dir, env_key)
        local_path = os.path.join(local_dir, file_name)
        os.makedirs(local_dir, exist_ok=True)
        with open(local_path, "wb") as file:
            file.write(image)

        if self.uploader is None:
            return local_path
        key = f"aievobox/{self.job_id}/{env_key}/{file_name}"
        for attempt in range(self.max_retries):
            try:
                uploaded = await asyncio.to_thread(
                    self.uploader.upload_file,
                    file_path=local_path,
                    key=key,
                )
                if uploaded:
                    return str(uploaded)
            except Exception as exc:
                if attempt + 1 >= self.max_retries:
                    log.warning("Image upload failed; using local fallback %s: %s", local_path, exc)
                    break
                await asyncio.sleep(2 ** attempt)
        return local_path
