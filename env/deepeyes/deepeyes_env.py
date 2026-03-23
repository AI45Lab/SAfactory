import base64
import copy
import io
import json
import logging
import os
import re
import shutil
import time
import uuid
from math import ceil, floor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import matplotlib.pyplot as plt
import requests
import yaml
from openai.types.chat import ChatCompletionMessageParam
from PIL import Image

from core.env.base_env import BaseEnv
from core.env.env_register import register_env
from core.types.base import RenderOutput, ResetOutput, StepOutput
from env.deepeyes.prompt import (
    SYSTEM_PROMPT_V2,
    TOOL_RESPONSE_PREFIX,
    TOOL_RESPONSE_SUFFIX,
    USER_PROMPT_V2,
)
from env.deepeyes.reward import JudgeClient, RewardResult, compute_reward

logger = logging.getLogger(__name__)

TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
SLOW_PARQUET_READ_MS = 3000
SLOW_IMAGE_LOAD_MS = 3000


def _elapsed_ms(start_time: float) -> int:
    return int((time.perf_counter() - start_time) * 1000)


def _short_source_repr(source: Any, *, max_len: int = 160) -> str:
    text = str(source)
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


@register_env("deepeyes_env")
class DeepEyesEnv(BaseEnv):
    """
    A DeepEyes-style multi-turn visual tool environment for AIEvoBox.

    Expected dataset columns:
    - prompt: DeepEyes/VeRL-style chat messages
    - images: list[str] image paths/urls/data-urls
    - data_source: reward branch selector
    - reward_model.ground_truth: reference answer
    - extra_info.question: original question text
    """

    def __init__(
        self,
        dataset: Dict[str, Any],
        config_path: Optional[str] = None,
        env_id: str = "",
        env_name: str = "",
    ) -> None:
        init_start = time.perf_counter()
        super().__init__(env_id=env_id, env_name=env_name, dataset=dataset)
        dataset_ref = dataset.get("__dataset_ref__") if isinstance(dataset, dict) else None
        logger.info(
            "DeepEyes Init Start: env=%s/%s config_path=%s dataset_ref_kind=%s",
            env_name,
            env_id,
            config_path,
            dataset_ref.get("kind") if isinstance(dataset_ref, dict) else None,
        )

        if config_path is None:
            config_path = str(Path(__file__).with_name("deepeyes_env_runtime.yaml"))
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                runtime_cfg = yaml.safe_load(f) or {}
        except FileNotFoundError:
            logger.warning("Runtime config not found: %s, using defaults", config_path)
            runtime_cfg = {}

        self.max_turns: int = int(runtime_cfg.get("max_turns", 5))
        self.allow_rotate_tool: bool = bool(runtime_cfg.get("allow_rotate_tool", True))
        self.http_timeout_s: float = float(runtime_cfg.get("http_timeout_s", 30.0))
        self.cleanup_temp_dir_on_close: bool = bool(runtime_cfg.get("cleanup_temp_dir_on_close", False))
        config_dir = Path(config_path).expanduser().resolve().parent
        default_crop_root = (
            Path(os.environ.get("AIEVOBOX_ROOT", str(Path(__file__).resolve().parents[2]))).expanduser()
            / "runtime"
            / "deepeyes_crops"
        )
        crop_root = str(runtime_cfg.get("crop_root", "")).strip()
        if crop_root:
            configured_crop_root = Path(crop_root).expanduser()
            self.crop_root = (
                configured_crop_root
                if configured_crop_root.is_absolute()
                else (config_dir / configured_crop_root)
            )
        else:
            self.crop_root = default_crop_root
        self.crop_root.mkdir(parents=True, exist_ok=True)

        judge_base_url = str(runtime_cfg.get("judge_base_url", "")).strip() or None
        judge_model = str(runtime_cfg.get("judge_model", "")).strip() or None
        judge_timeout_s = float(runtime_cfg.get("judge_timeout_s", 60.0))
        judge_api_key = str(runtime_cfg.get("judge_api_key", "EMPTY"))
        self.judge_client = JudgeClient(
            base_url=judge_base_url,
            model_name=judge_model,
            timeout=judge_timeout_s,
            api_key=judge_api_key,
        )

        dataset_row_start = time.perf_counter()
        self.dataset_row = self._resolve_dataset_row(dataset or {})
        logger.info(
            "DeepEyes Dataset Row Done: env=%s/%s dataset_path=%s row_group=%s row_in_group=%s elapsed_ms=%s",
            env_name,
            env_id,
            dataset_ref.get("path") if isinstance(dataset_ref, dict) else None,
            dataset_ref.get("row_group") if isinstance(dataset_ref, dict) else None,
            dataset_ref.get("row_in_group") if isinstance(dataset_ref, dict) else None,
            _elapsed_ms(dataset_row_start),
        )
        self.data_source: str = str(self.dataset_row.get("data_source", "vl_agent")).strip() or "vl_agent"
        self.extra_info: Dict[str, Any] = self._normalize_mapping(self.dataset_row.get("extra_info"))
        self.question: str = self._resolve_question(self.dataset_row)
        self.ground_truth: str = self._normalize_ground_truth(
            self._resolve_ground_truth(self.dataset_row)
        )
        self.image_sources: List[Any] = self._normalize_image_sources(self.dataset_row)
        self.message_image_sources: List[str] = []
        self.raw_prompt_messages = copy.deepcopy(self._normalize_prompt_messages(self.dataset_row.get("prompt")))

        self.messages: List[ChatCompletionMessageParam] = []
        self.original_images: List[Image.Image] = []
        self.step_count: int = 0
        self.total_tool_calls: int = 0
        self.final_answer: Optional[str] = None
        self.last_reward_result: Optional[RewardResult] = None
        self.tool_calls: List[Dict[str, Any]] = []
        self._crop_session_dir: Optional[Path] = None

        self.action_space = gym.spaces.Text(max_length=12000)
        self.observation_space = gym.spaces.Dict({})
        logger.info(
            "DeepEyes Init Done: env=%s/%s data_source=%s image_count=%s total_ms=%s",
            env_name,
            env_id,
            self.data_source,
            len(self.image_sources),
            _elapsed_ms(init_start),
        )

    def reset(self, seed: Optional[int] = None) -> ResetOutput:
        reset_start = time.perf_counter()
        logger.info(
            "DeepEyes Reset Start: env=%s/%s seed=%s image_count=%s",
            self.env_name,
            self.env_id,
            seed,
            len(self.image_sources),
        )
        try:
            del seed
            self.step_count = 0
            self.total_tool_calls = 0
            self.final_answer = None
            self.last_reward_result = None
            self.tool_calls = []
            self.done = False

            self._cleanup_temp_dir()
            self._crop_session_dir = self.crop_root / f"{self.env_id or 'deepeyes'}-{uuid.uuid4().hex}"
            self._crop_session_dir.mkdir(parents=True, exist_ok=True)

            materialize_start = time.perf_counter()
            self.message_image_sources = self._materialize_message_image_sources(self.image_sources)
            logger.info(
                "DeepEyes Reset MaterializeImages Done: env=%s/%s message_image_count=%s elapsed_ms=%s",
                self.env_name,
                self.env_id,
                len(self.message_image_sources),
                _elapsed_ms(materialize_start),
            )

            load_images_start = time.perf_counter()
            self.original_images = self._load_images(self.image_sources)
            logger.info(
                "DeepEyes Reset LoadImages Done: env=%s/%s loaded_image_count=%s elapsed_ms=%s",
                self.env_name,
                self.env_id,
                len(self.original_images),
                _elapsed_ms(load_images_start),
            )

            build_messages_start = time.perf_counter()
            self.messages = self._build_initial_messages()
            logger.info(
                "DeepEyes Reset BuildMessages Done: env=%s/%s message_count=%s elapsed_ms=%s",
                self.env_name,
                self.env_id,
                len(self.messages),
                _elapsed_ms(build_messages_start),
            )

            info = {
                "question": self.question,
                "ground_truth": self.ground_truth,
                "data_source": self.data_source,
                "image_count": len(self.image_sources),
                "loaded_image_count": len(self.original_images),
                "step": self.step_count,
                "max_turns": self.max_turns,
                "total_tool_calls": self.total_tool_calls,
            }
            logger.info(
                "DeepEyes Reset Done: env=%s/%s loaded_image_count=%s total_ms=%s",
                self.env_name,
                self.env_id,
                len(self.original_images),
                _elapsed_ms(reset_start),
            )
            return ResetOutput(observation={}, info=info)
        except Exception:
            logger.exception(
                "DeepEyes Reset Failed: env=%s/%s total_ms=%s",
                self.env_name,
                self.env_id,
                _elapsed_ms(reset_start),
            )
            raise

    def step(self, action: str) -> StepOutput:
        self.step_count += 1
        reward, terminated, truncated, extra_info = self._process_action(action or "")

        if not (terminated or truncated) and self.step_count >= self.max_turns:
            reward = 0.0
            terminated = True
            self.done = True
            extra_info = dict(extra_info)
            extra_info["reward_result"] = None
            extra_info["termination_reason"] = "max_turns_reached"

        info = {
            "question": self.question,
            "ground_truth": self.ground_truth,
            "data_source": self.data_source,
            "step": self.step_count,
            "max_turns": self.max_turns,
            "total_tool_calls": self.total_tool_calls,
            "final_answer": self.final_answer,
            "tool_calls": list(self.tool_calls),
            "reward_result": self._reward_result_to_dict(self.last_reward_result),
            "extra": extra_info,
        }
        return StepOutput(
            observation={},
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )

    def get_task_prompt(self) -> List[ChatCompletionMessageParam]:
        return copy.deepcopy(self.messages)

    def render(self) -> RenderOutput:
        recent = self.messages[-8:] if self.messages else []
        lines: List[str] = []
        for idx, msg in enumerate(recent, 1):
            role = str(msg.get("role", "?"))
            content = self._message_content_to_text(msg.get("content"))
            snippet = content[:180] + ("..." if len(content) > 180 else "")
            lines.append(f"[{idx}] {role}: {snippet}")

        if not lines:
            lines = ["(no messages yet)"]

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.axis("off")
        ax.text(0.01, 0.99, "\n\n".join(lines), va="top", ha="left", wrap=True, fontsize=10)
        ax.set_title(f"DeepEyesEnv | step={self.step_count}", fontsize=12)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
        buf.seek(0)
        image_data = buf.read()
        buf.close()
        plt.close(fig)
        return RenderOutput(step=self.step_count, image_data=image_data)

    def close(self) -> Dict[str, Any]:
        self._cleanup_temp_dir()
        return {"status": "closed"}

    def is_done(self) -> bool:
        return self.done

    def health(self) -> bool:
        return True

    def _process_action(self, assistant_msg: str) -> Tuple[float, bool, bool, Dict[str, Any]]:
        msg = (assistant_msg or "").strip()
        self.messages.append({"role": "assistant", "content": msg})  # type: ignore[typeddict-item]

        final_answer = self._extract_answer(msg)
        if final_answer:
            self.final_answer = final_answer
            reward_result = self._compute_terminal_reward(msg)
            self.done = True
            return reward_result.score, True, False, {
                "termination_reason": "final_answer",
                "reward_result": self._reward_result_to_dict(reward_result),
            }

        tool_call, tool_call_error = self._extract_tool_call(msg)
        if tool_call_error is not None:
            self._append_error_feedback(tool_call_error)
            return 0.0, False, False, {
                "tool_executed": False,
                "malformed_tool_call": True,
                "error": tool_call_error,
            }

        if tool_call is not None:
            return self._handle_tool_call(tool_call)

        self.done = True
        self.final_answer = None
        self.last_reward_result = None
        return 0.0, True, False, {
            "termination_reason": "missing_tool_call_or_answer",
            "reward_result": None,
        }

    def _handle_tool_call(self, tool_call: Dict[str, Any]) -> Tuple[float, bool, bool, Dict[str, Any]]:
        name = str(tool_call.get("name", "")).strip()
        arguments = tool_call.get("arguments", {})

        if not isinstance(arguments, dict):
            self._append_error_feedback("Tool arguments must be a JSON object.")
            return 0.0, False, False, {"tool_executed": False, "bad_arguments": True}

        if name == "image_zoom_in_tool":
            return self._handle_zoom_in(arguments)

        if name == "image_rotate_tool" and self.allow_rotate_tool:
            return self._handle_rotate(arguments)

        self._append_error_feedback(f"Unknown tool name: {name}")
        return 0.0, False, False, {"tool_executed": False, "unsupported_tool": name}

    def _handle_zoom_in(self, arguments: Dict[str, Any]) -> Tuple[float, bool, bool, Dict[str, Any]]:
        bbox = arguments.get("bbox_2d", arguments.get("bbox"))
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            self._append_error_feedback("ZOOM IN ARGUMENTS ARE INVALID")
            return 0.0, False, False, {"tool_executed": False, "invalid_bbox": True}

        resized_bbox = self._maybe_resize_bbox(*bbox)
        if not resized_bbox:
            self._append_error_feedback("ZOOM IN ARGUMENTS ARE INVALID")
            return 0.0, False, False, {"tool_executed": False, "invalid_bbox": True}

        if not self.original_images:
            self._append_error_feedback("No source image is available for zoom-in.")
            return 0.0, False, False, {"tool_executed": False, "missing_image": True}

        crop = self.original_images[0].crop(tuple(resized_bbox))
        crop_path = self._save_image(crop, stem=f"turn{self.step_count:03d}_zoom")

        self.total_tool_calls += 1
        self.tool_calls.append(
            {
                "name": "image_zoom_in_tool",
                "bbox_2d": list(resized_bbox),
                "label": str(arguments.get("label", "")).strip(),
                "crop_path": crop_path,
            }
        )
        self._append_tool_feedback(crop_path)
        return 0.0, False, False, {
            "tool_executed": True,
            "tool_name": "image_zoom_in_tool",
            "bbox_2d": list(resized_bbox),
        }

    def _handle_rotate(self, arguments: Dict[str, Any]) -> Tuple[float, bool, bool, Dict[str, Any]]:
        angle = arguments.get("angle")
        try:
            angle_value = int(angle)
        except Exception:
            self._append_error_feedback("Rotation angle must be an integer.")
            return 0.0, False, False, {"tool_executed": False, "invalid_angle": True}

        if not self.original_images:
            self._append_error_feedback("No source image is available for rotation.")
            return 0.0, False, False, {"tool_executed": False, "missing_image": True}

        rotated = self.original_images[0].rotate(angle_value)
        rotated_path = self._save_image(rotated, stem=f"turn{self.step_count:03d}_rotate")

        self.total_tool_calls += 1
        self.tool_calls.append(
            {
                "name": "image_rotate_tool",
                "angle": angle_value,
                "crop_path": rotated_path,
            }
        )
        self._append_tool_feedback(rotated_path)
        return 0.0, False, False, {
            "tool_executed": True,
            "tool_name": "image_rotate_tool",
            "angle": angle_value,
        }

    def _append_tool_feedback(self, image_path: str) -> None:
        content = [
            {"type": "text", "text": TOOL_RESPONSE_PREFIX},
            {"type": "image_url", "image_url": {"url": image_path}},
            {"type": "text", "text": f"{USER_PROMPT_V2}{TOOL_RESPONSE_SUFFIX}"},
        ]
        self.messages.append({"role": "user", "content": content})  # type: ignore[typeddict-item]

    def _append_error_feedback(self, error_text: str) -> None:
        self.messages.append({"role": "user", "content": f"Error: {error_text}"})  # type: ignore[typeddict-item]

    def _build_initial_messages(self) -> List[ChatCompletionMessageParam]:
        if self.raw_prompt_messages:
            messages = self._expand_prompt_messages(self.raw_prompt_messages, self.message_image_sources)
            if messages:
                return messages

        image_markers = "".join("<image>" for _ in self.message_image_sources) or "<image>"
        user_text = f"{image_markers}\n{self.question}{USER_PROMPT_V2}"
        return self._expand_prompt_messages(
            [
                {"role": "system", "content": SYSTEM_PROMPT_V2},
                {"role": "user", "content": user_text},
            ],
            self.message_image_sources,
        )

    def _compute_terminal_reward(self, final_response: str) -> RewardResult:
        reward_result = compute_reward(
            self.data_source,
            final_response,
            self.ground_truth,
            extra_info={"question": self.question, **self.extra_info},
            tool_used=self.total_tool_calls > 0,
            judge_client=self.judge_client,
        )
        self.last_reward_result = reward_result
        return reward_result

    def _expand_prompt_messages(
        self,
        raw_messages: List[Dict[str, Any]],
        image_sources: List[str],
    ) -> List[ChatCompletionMessageParam]:
        messages: List[ChatCompletionMessageParam] = []
        image_idx = 0

        for raw_msg in raw_messages:
            if not isinstance(raw_msg, dict):
                continue
            role = str(raw_msg.get("role", "user")).strip() or "user"
            content, image_idx = self._expand_message_content(raw_msg.get("content"), image_sources, image_idx)
            messages.append({"role": role, "content": content})  # type: ignore[typeddict-item]

        if image_sources and image_idx == 0:
            fallback_content = [{"type": "image_url", "image_url": {"url": src}} for src in image_sources]
            if messages:
                for idx in range(len(messages) - 1, -1, -1):
                    if messages[idx].get("role") == "user":
                        existing = self._ensure_content_list(messages[idx].get("content"))
                        messages[idx]["content"] = fallback_content + existing
                        break
            else:
                messages.append({"role": "user", "content": fallback_content})  # type: ignore[typeddict-item]

        return messages

    def _expand_message_content(
        self,
        content: Any,
        image_sources: List[str],
        image_idx: int,
    ) -> Tuple[Any, int]:
        if isinstance(content, str):
            items: List[Dict[str, Any]] = []
            for segment in re.split(r"(<image>|<video>)", content):
                if segment == "<image>":
                    if image_idx < len(image_sources):
                        items.append({"type": "image_url", "image_url": {"url": image_sources[image_idx]}})
                        image_idx += 1
                    else:
                        items.append({"type": "text", "text": "<image>"})
                elif segment == "<video>":
                    items.append({"type": "text", "text": "<video>"})
                elif segment:
                    items.append({"type": "text", "text": segment})
            return items or content, image_idx

        if isinstance(content, list):
            items: List[Dict[str, Any]] = []
            for item in content:
                if isinstance(item, str):
                    items.append({"type": "text", "text": item})
                    continue
                if not isinstance(item, dict):
                    items.append({"type": "text", "text": str(item)})
                    continue

                item_type = item.get("type")
                if item_type == "text":
                    items.append({"type": "text", "text": str(item.get("text", ""))})
                elif item_type in {"image_url", "image"}:
                    url_value = None
                    image_url = item.get("image_url")
                    if isinstance(image_url, dict):
                        url_value = image_url.get("url")
                    elif image_url is not None:
                        url_value = image_url
                    elif "image" in item:
                        url_value = item.get("image")
                    if isinstance(url_value, str) and url_value.strip():
                        items.append({"type": "image_url", "image_url": {"url": url_value.strip()}})
                    elif image_idx < len(image_sources):
                        items.append({"type": "image_url", "image_url": {"url": image_sources[image_idx]}})
                        image_idx += 1
                else:
                    items.append({"type": "text", "text": json.dumps(item, ensure_ascii=False)})
            return items, image_idx

        if isinstance(content, dict):
            return self._expand_message_content([content], image_sources, image_idx)

        return str(content), image_idx

    def _load_images(self, image_sources: List[Any]) -> List[Image.Image]:
        images: List[Image.Image] = []
        for idx, source in enumerate(image_sources):
            image_start = time.perf_counter()
            try:
                image = self._load_single_image(source)
            except Exception as exc:
                logger.warning(
                    "Failed to load DeepEyes image env=%s/%s index=%s source='%s' elapsed_ms=%s: %s",
                    self.env_name,
                    self.env_id,
                    idx,
                    _short_source_repr(source),
                    _elapsed_ms(image_start),
                    exc,
                )
                continue
            image_elapsed_ms = _elapsed_ms(image_start)
            if image_elapsed_ms >= SLOW_IMAGE_LOAD_MS:
                logger.warning(
                    "Slow DeepEyes image load env=%s/%s index=%s source='%s' elapsed_ms=%s",
                    self.env_name,
                    self.env_id,
                    idx,
                    _short_source_repr(source),
                    image_elapsed_ms,
                )
            images.append(image)
        return images

    def _load_single_image(self, source: Any) -> Image.Image:
        if isinstance(source, dict):
            image_bytes = source.get("bytes")
            if isinstance(image_bytes, (bytes, bytearray)):
                image = Image.open(io.BytesIO(bytes(image_bytes)))
                return image.convert("RGB")

            path_value = source.get("path")
            if isinstance(path_value, str) and path_value.strip():
                path = Path(path_value).expanduser()
                if path.exists():
                    image = Image.open(path)
                    return image.convert("RGB")

            for key in ("src", "url", "image"):
                value = source.get(key)
                if isinstance(value, str) and value.strip():
                    return self._load_single_image(value.strip())

            raise ValueError("Unsupported image source mapping.")

        source = str(source).strip()
        if source.startswith("data:image"):
            header, encoded = source.split(",", 1)
            del header
            image_bytes = base64.b64decode(encoded)
            image = Image.open(io.BytesIO(image_bytes))
            return image.convert("RGB")

        if source.startswith("http://") or source.startswith("https://"):
            response = requests.get(source, timeout=self.http_timeout_s)
            response.raise_for_status()
            image = Image.open(io.BytesIO(response.content))
            return image.convert("RGB")

        image = Image.open(Path(source).expanduser())
        return image.convert("RGB")

    def _materialize_message_image_sources(self, image_sources: List[Any]) -> List[str]:
        materialized: List[str] = []
        for index, source in enumerate(image_sources):
            url = self._source_to_message_url(source, index=index)
            if url:
                materialized.append(url)
        return materialized

    def _source_to_message_url(self, source: Any, *, index: int) -> Optional[str]:
        if isinstance(source, str):
            source = source.strip()
            return source or None

        if isinstance(source, dict):
            path_value = source.get("path")
            if isinstance(path_value, str) and path_value.strip():
                path = Path(path_value).expanduser()
                if path.exists():
                    return str(path)

            for key in ("src", "url", "image"):
                value = source.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()

            image_bytes = source.get("bytes")
            if isinstance(image_bytes, (bytes, bytearray)):
                if self._crop_session_dir is None:
                    raise RuntimeError("Crop session directory is not initialized.")
                suffix = self._guess_image_suffix(bytes(image_bytes), path_hint=path_value)
                path = self._crop_session_dir / f"source_{index:02d}{suffix}"
                if not path.exists():
                    path.write_bytes(bytes(image_bytes))
                return str(path)

        return None

    @staticmethod
    def _guess_image_suffix(image_bytes: bytes, *, path_hint: Any = None) -> str:
        if isinstance(path_hint, str):
            suffix = Path(path_hint).suffix
            if suffix:
                return suffix
        if image_bytes.startswith(b"\xff\xd8\xff"):
            return ".jpg"
        if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
            return ".png"
        if image_bytes.startswith((b"GIF87a", b"GIF89a")):
            return ".gif"
        if image_bytes.startswith(b"RIFF") and image_bytes[8:12] == b"WEBP":
            return ".webp"
        return ".img"

    def _save_image(self, image: Image.Image, *, stem: str) -> str:
        if self._crop_session_dir is None:
            raise RuntimeError("Crop session directory is not initialized.")
        filename = f"{stem}.png"
        path = self._crop_session_dir / filename
        image.save(path, format="PNG")
        return str(path)

    def _cleanup_temp_dir(self) -> None:
        if not self._crop_session_dir:
            return
        if self.cleanup_temp_dir_on_close and self._crop_session_dir.exists():
            shutil.rmtree(self._crop_session_dir, ignore_errors=True)
        self._crop_session_dir = None

    def _validate_bbox(self, left: int, top: int, right: int, bottom: int) -> bool:
        try:
            assert left < right and bottom > top
            height = bottom - top
            width = right - left
            assert max(height, width) / min(height, width) <= 100
            assert min(height, width) > 30
            return True
        except Exception:
            return False

    def _maybe_resize_bbox(self, left: Any, top: Any, right: Any, bottom: Any) -> Optional[List[int]]:
        if not self.original_images:
            return None
        try:
            left = int(left)
            top = int(top)
            right = int(right)
            bottom = int(bottom)
        except Exception:
            return None

        width = self.original_images[0].width
        height = self.original_images[0].height
        left = max(0, left)
        top = max(0, top)
        right = min(width, right)
        bottom = min(height, bottom)
        if not self._validate_bbox(left, top, right, bottom):
            return None

        box_height = bottom - top
        box_width = right - left
        if box_height < 28 or box_width < 28:
            center_x = (left + right) / 2.0
            center_y = (top + bottom) / 2.0
            ratio = 28 / min(box_height, box_width)
            new_half_height = ceil(box_height * ratio * 0.5)
            new_half_width = ceil(box_width * ratio * 0.5)
            new_left = floor(center_x - new_half_width)
            new_right = ceil(center_x + new_half_width)
            new_top = floor(center_y - new_half_height)
            new_bottom = ceil(center_y + new_half_height)
            if not self._validate_bbox(new_left, new_top, new_right, new_bottom):
                return None
            return [new_left, new_top, new_right, new_bottom]
        return [left, top, right, bottom]

    @staticmethod
    def _extract_tool_call(text: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        matches = list(TOOL_CALL_RE.finditer(text))
        if not matches:
            return None, None

        raw_json = matches[-1].group(1).strip()
        try:
            payload = json.loads(raw_json)
        except json.JSONDecodeError as exc:
            return None, f"Invalid tool call format: {raw_json}. Error: {exc}"

        name = payload.get("name") or payload.get("function", {}).get("name")
        arguments = payload.get("arguments") or payload.get("function", {}).get("arguments") or {}
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError as exc:
                return None, f"Invalid tool call arguments: {arguments}. Error: {exc}"

        if not name:
            return None, "Invalid tool call format: missing tool name."
        return {"name": name, "arguments": arguments}, None

    @staticmethod
    def _extract_answer(text: str) -> str:
        matches = re.findall(r"<answer>(.*?)</answer>", text, flags=re.DOTALL)
        return matches[-1].strip() if matches else ""

    @staticmethod
    def _ensure_content_list(content: Any) -> List[Dict[str, Any]]:
        if isinstance(content, list):
            return content
        if isinstance(content, str):
            return [{"type": "text", "text": content}]
        if isinstance(content, dict):
            return [content]
        return [{"type": "text", "text": str(content)}]

    @staticmethod
    def _normalize_mapping(value: Any) -> Dict[str, Any]:
        if value is None:
            return {}
        if hasattr(value, "tolist"):
            value = value.tolist()
        return dict(value) if isinstance(value, dict) else {}

    @staticmethod
    def _resolve_dataset_row(dataset: Any) -> Dict[str, Any]:
        if not isinstance(dataset, dict):
            return {}

        dataset_ref = dataset.get("__dataset_ref__")
        if isinstance(dataset_ref, dict) and dataset_ref.get("kind") == "parquet_row":
            return DeepEyesEnv._load_dataset_row_from_parquet_ref(dataset_ref)

        return dict(dataset)

    @staticmethod
    def _load_dataset_row_from_parquet_ref(dataset_ref: Dict[str, Any]) -> Dict[str, Any]:
        read_start = time.perf_counter()
        path = Path(str(dataset_ref.get("path", "")).strip()).expanduser()
        row_group = int(dataset_ref.get("row_group", 0))
        row_in_group = int(dataset_ref.get("row_in_group", 0))
        logger.info(
            "DeepEyes Dataset Row Start: dataset_path=%s row_group=%s row_in_group=%s",
            path,
            row_group,
            row_in_group,
        )
        try:
            try:
                import pyarrow.parquet as pq
            except ImportError as exc:
                raise ImportError("DeepEyes parquet row references require pyarrow.") from exc

            if not path.exists():
                raise FileNotFoundError(f"Parquet dataset row path does not exist: {path}")

            parquet_file = pq.ParquetFile(path)
            table = parquet_file.read_row_group(row_group).slice(row_in_group, 1)
            rows = table.to_pylist()
            elapsed_ms = _elapsed_ms(read_start)
            if elapsed_ms >= SLOW_PARQUET_READ_MS:
                logger.warning(
                    "Slow DeepEyes Dataset Row Read: dataset_path=%s row_group=%s row_in_group=%s elapsed_ms=%s rows=%s",
                    path,
                    row_group,
                    row_in_group,
                    elapsed_ms,
                    len(rows),
                )
            else:
                logger.info(
                    "DeepEyes Dataset Row Read Done: dataset_path=%s row_group=%s row_in_group=%s elapsed_ms=%s rows=%s",
                    path,
                    row_group,
                    row_in_group,
                    elapsed_ms,
                    len(rows),
                )
            return dict(rows[0]) if rows else {}
        except Exception:
            logger.exception(
                "DeepEyes Dataset Row Read Failed: dataset_path=%s row_group=%s row_in_group=%s elapsed_ms=%s",
                path,
                row_group,
                row_in_group,
                _elapsed_ms(read_start),
            )
            raise

    @staticmethod
    def _normalize_prompt_messages(value: Any) -> List[Dict[str, Any]]:
        if value is None:
            return []
        if hasattr(value, "tolist"):
            value = value.tolist()
        return list(value) if isinstance(value, list) else []

    @staticmethod
    def _resolve_ground_truth(row: Dict[str, Any]) -> Any:
        reward_model = row.get("reward_model")
        if isinstance(reward_model, dict) and "ground_truth" in reward_model:
            return reward_model.get("ground_truth")
        if "answer" in row:
            return row.get("answer")
        if "golden_answers" in row:
            return row.get("golden_answers")
        extra_info = row.get("extra_info")
        if isinstance(extra_info, dict) and "answer" in extra_info:
            return extra_info.get("answer")
        return ""

    @staticmethod
    def _normalize_ground_truth(raw_answer: Any) -> str:
        value = raw_answer
        if value is None:
            return ""
        if hasattr(value, "tolist"):
            value = value.tolist()
        if isinstance(value, (list, tuple)):
            for item in value:
                item_str = str(item).strip()
                if item_str:
                    return item_str
            return ""
        return str(value).strip()

    @staticmethod
    def _normalize_image_sources(row: Dict[str, Any]) -> List[Any]:
        raw_images = row.get("images")
        if raw_images is None:
            extra_info = row.get("extra_info")
            if isinstance(extra_info, dict) and extra_info.get("image"):
                raw_images = [extra_info["image"]]
            elif row.get("image"):
                raw_images = [row["image"]]

        if raw_images is None:
            return []

        if hasattr(raw_images, "tolist"):
            raw_images = raw_images.tolist()

        if isinstance(raw_images, str):
            raw_images = [raw_images]

        if not isinstance(raw_images, (list, tuple)):
            return []

        image_sources: List[Any] = []
        for item in raw_images:
            if isinstance(item, dict):
                if isinstance(item.get("bytes"), (bytes, bytearray)):
                    image_sources.append(
                        {
                            "bytes": bytes(item["bytes"]),
                            "path": item.get("path"),
                            "src": item.get("src"),
                            "url": item.get("url"),
                            "image": item.get("image"),
                        }
                    )
                    continue
                source = item.get("src") or item.get("url") or item.get("image") or item.get("path")
                if isinstance(source, str) and source.strip():
                    image_sources.append(source.strip())
            else:
                source = str(item).strip()
                if source:
                    image_sources.append(source)
        return image_sources

    @staticmethod
    def _resolve_question(row: Dict[str, Any]) -> str:
        for key in ("question", "problem"):
            value = row.get(key)
            if value is not None and str(value).strip():
                return str(value).strip()

        extra_info = row.get("extra_info")
        if isinstance(extra_info, dict):
            question = extra_info.get("question")
            if question is not None and str(question).strip():
                return str(question).strip()

        prompt = row.get("prompt")
        if hasattr(prompt, "tolist"):
            prompt = prompt.tolist()
        if isinstance(prompt, list):
            for msg in reversed(prompt):
                if not isinstance(msg, dict):
                    continue
                if str(msg.get("role", "")).strip() != "user":
                    continue
                text = DeepEyesEnv._message_content_to_text(msg.get("content"))
                text = text.replace("<image>", " ").replace("<video>", " ")
                cleaned = " ".join(text.split()).strip()
                if cleaned:
                    return cleaned
        return ""

    @staticmethod
    def _message_content_to_text(content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if not isinstance(item, dict):
                    parts.append(str(item))
                    continue
                item_type = item.get("type")
                if item_type == "text":
                    parts.append(str(item.get("text", "")))
                elif item_type == "image_url":
                    parts.append("[image_url]")
                else:
                    parts.append(json.dumps(item, ensure_ascii=False))
            return " ".join(parts)
        if isinstance(content, dict):
            return json.dumps(content, ensure_ascii=False)
        return str(content)

    @staticmethod
    def _reward_result_to_dict(result: Optional[RewardResult]) -> Optional[Dict[str, Any]]:
        if result is None:
            return None
        return {
            "score": result.score,
            "answer_text": result.answer_text,
            "acc_reward": result.acc_reward,
            "format_reward": result.format_reward,
            "tool_reward": result.tool_reward,
            "judge_used": result.judge_used,
            "judge_available": result.judge_available,
            "judge_response": result.judge_response,
            "reward_type": result.reward_type,
        }
