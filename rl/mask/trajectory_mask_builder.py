"""Trajectory Builder for LLM Proxy.

核心设计：
1. 匹配：用字符级匹配（含 <think> 跳过）找到最佳匹配的历史记录
2. 追加：在匹配到的记录基础上，追加新的 context tokens 和 output tokens
"""

import json
import hashlib
import logging
import os
import time
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

from qwen_vl_utils import process_vision_info
from slime.utils.processing_utils import encode_image_for_rollout_engine

logger = logging.getLogger(__name__)


# 用于计算单个 message 增量的基础对话
BASE_CHAT_HISTORY = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "I am a user."},
]


class TrajectoryMaskBuilder:
    """按 session 维护 trajectory 的构造器。

    核心功能：
    - 字符级匹配：处理 <think>...</think> 的可选对齐
    - token 级追加：在匹配的基础上追加新的 tokens
    """

    def __init__(self, tokenizer, processor: Any = None) -> None:
        self.tokenizer = tokenizer
        self.processor = processor
        self.enable_cache_processor_compare = self._read_bool_env(
            "AIEVOBOX_DEBUG_CACHE_PROCESSOR_COMPARE",
            default=False,
        )
        self.generation_tokens, self.generation_prompt = self._init_generation_tokens_and_prompt()
        self.end_tokens, self.end_prompt_str = self._init_end_tokens_and_prompt()

        # session_id -> List[{
        #     "messages_str": str,           # 完整字符串（用于匹配）
        #     "tokens": List[int],
        #     "response_mask": List[int],    # 0=context, 1=generated
        #     "logprobs": List[float],
        #     "image_data": List[str],       # 编码后的图片数据
        # }]
        self.session_data: Dict[str, List[Dict[str, Any]]] = {}
        # session_id -> image_key -> {
        #     "pixel_values": Tensor,
        #     "image_grid_thw": Tensor,
        #     "num_image_tokens": int,
        #     "image_data": str,
        # }
        self.session_image_cache: Dict[str, Dict[str, Dict[str, Any]]] = {}

    def _read_bool_env(self, name: str, default: bool = False) -> bool:
        raw_value = os.environ.get(name)
        if raw_value is None:
            return default

        normalized_value = raw_value.strip().lower()
        if normalized_value in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized_value in {"0", "false", "no", "n", "off"}:
            return False
        return default

    def _init_generation_tokens_and_prompt(self) -> Tuple[List[int], str]:
        """计算 generation_tokens 和 generation_prompt（如 <|im_start|>assistant\n）。

        通过比较 add_generation_prompt=False/True 的差异，
        提取 generation prompt 的字符串，然后直接 tokenize 得到 tokens。
        """
        # 获取字符串
        without_gen_str = self.tokenizer.apply_chat_template(
            BASE_CHAT_HISTORY, add_generation_prompt=False, tokenize=False
        )
        with_gen_str = self.tokenizer.apply_chat_template(
            BASE_CHAT_HISTORY, add_generation_prompt=True, tokenize=False
        )
        generation_prompt = with_gen_str[len(without_gen_str):]

        # 直接 tokenize 得到 tokens
        generation_tokens = self.tokenizer.encode(generation_prompt, add_special_tokens=False)

        return generation_tokens, generation_prompt

    def _init_end_tokens_and_prompt(self) -> Tuple[List[int], str]:
        """计算 end_tokens 和 end_prompt_str（如 <|im_end|>\n）。

        通过比较两个不同assistant回复的字符串，
        找到末尾相同的部分，然后直接 tokenize 得到 tokens。

        注意：只提取 assistant 消息自己的结束部分，消息之间的分隔符
        会在下一个消息加入时自然包含在 new_context_str 中。
        """
        # 创建两个不同的assistant回复
        messages1 = BASE_CHAT_HISTORY + [{"role": "assistant", "content": "UNIQUE_CONTENT_1"}]
        messages2 = BASE_CHAT_HISTORY + [{"role": "assistant", "content": "UNIQUE_CONTENT_2_DIFFERENT"}]

        # 获取字符串
        full_str1 = self.tokenizer.apply_chat_template(messages1, tokenize=False, add_generation_prompt=False)
        full_str2 = self.tokenizer.apply_chat_template(messages2, tokenize=False, add_generation_prompt=False)

        # 找到末尾相同的字符串部分
        end_prompt_str = ""
        for i in range(1, min(len(full_str1), len(full_str2)) + 1):
            if full_str1[-i] == full_str2[-i]:
                end_prompt_str = full_str1[-i] + end_prompt_str
            else:
                break

        # 直接 tokenize 得到 tokens
        end_tokens = self.tokenizer.encode(end_prompt_str, add_special_tokens=False)

        return end_tokens, end_prompt_str

    def _normalize_messages_for_qwen_vision(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert OpenAI image_url blocks to qwen_vl_utils-compatible image blocks.

        qwen_vl_utils expects image content as:
          {"type": "image", "image": "<str url/path/dataurl>"}
        but OpenAI style often uses:
          {"type": "image_url", "image_url": {"url": "..."}}
        """
        normalized: List[Dict[str, Any]] = []
        for msg in messages:
            if not isinstance(msg, dict):
                normalized.append(msg)
                continue

            new_msg = dict(msg)
            content = msg.get("content")
            if not isinstance(content, list):
                normalized.append(new_msg)
                continue

            new_content: List[Any] = []
            for item in content:
                if not isinstance(item, dict):
                    new_content.append(item)
                    continue

                image_url_val = item.get("image_url")
                if item.get("type") == "image_url" or image_url_val is not None:
                    url_value: Any = None
                    if isinstance(image_url_val, dict):
                        url_value = image_url_val.get("url")
                    elif image_url_val is not None:
                        url_value = image_url_val
                    elif "image" in item:
                        url_value = item.get("image")

                    if isinstance(url_value, str):
                        converted = dict(item)
                        converted["type"] = "image"
                        converted["image"] = url_value
                        converted.pop("image_url", None)
                        new_content.append(converted)
                        continue

                new_content.append(item)

            new_msg["content"] = new_content
            normalized.append(new_msg)

        return normalized

    def _get_session_image_cache(self, session_id: str) -> Dict[str, Dict[str, Any]]:
        if session_id not in self.session_image_cache:
            self.session_image_cache[session_id] = {}
        return self.session_image_cache[session_id]

    def _make_image_cache_key(self, image_value: str) -> str:
        return hashlib.sha256(image_value.encode("utf-8")).hexdigest()

    def _extract_ordered_image_keys(self, messages: List[Dict[str, Any]]) -> List[str]:
        image_keys: List[str] = []
        for msg in messages:
            if not isinstance(msg, dict):
                continue

            content = msg.get("content")
            if not isinstance(content, list):
                continue

            for item in content:
                if not isinstance(item, dict):
                    continue
                if item.get("type") != "image":
                    continue

                image_value = item.get("image")
                if isinstance(image_value, str):
                    image_keys.append(self._make_image_cache_key(image_value))

        return image_keys

    def _grid_product(self, grid: Any) -> int:
        if hasattr(grid, "prod"):
            product = grid.prod()
            if hasattr(product, "item"):
                return int(product.item())
            return int(product)

        result = 1
        for value in grid:
            if hasattr(value, "item"):
                value = value.item()
            result *= int(value)
        return result

    def _slice_copy(self, value: Any, start: int, end: int) -> Any:
        sliced = value[start:end]
        if hasattr(sliced, "clone"):
            return sliced.clone()
        if hasattr(sliced, "copy"):
            return sliced.copy()
        return sliced

    def _concat_cached_values(self, values: List[Any]) -> Any:
        if not values:
            return None
        first = values[0]
        module_name = getattr(first.__class__, "__module__", "")
        if module_name.startswith("torch"):
            import torch

            return torch.cat(values, dim=0)

        try:
            import numpy as np

            return np.concatenate(values, axis=0)
        except Exception:
            if len(values) == 1:
                return values[0]
            raise

    def _ensure_images_cached(
        self,
        session_id: str,
        image_keys: List[str],
        images: List[Any],
    ) -> bool:
        image_processor = getattr(self.processor, "image_processor", None)
        image_token = getattr(self.processor, "image_token", None) or getattr(self.tokenizer, "image_token", None)
        merge_size = getattr(image_processor, "merge_size", None)
        if self.processor is None or image_processor is None or image_token is None or merge_size is None:
            return False
        if len(image_keys) != len(images):
            logger.warning(
                "Image key count mismatch: session=%s image_keys=%s images=%s",
                session_id,
                len(image_keys),
                len(images),
            )
            return False

        session_cache = self._get_session_image_cache(session_id)
        missing_keys: List[str] = []
        missing_images: List[Any] = []
        scheduled_keys = set()
        for image_key, image in zip(image_keys, images):
            if image_key in session_cache or image_key in scheduled_keys:
                continue
            scheduled_keys.add(image_key)
            missing_keys.append(image_key)
            missing_images.append(image)

        if not missing_images:
            return True

        image_inputs = self.processor.image_processor(images=missing_images, return_tensors="pt")
        pixel_values = image_inputs["pixel_values"]
        image_grid_thw = image_inputs["image_grid_thw"]
        merge_length = self.processor.image_processor.merge_size ** 2

        cursor = 0
        for idx, (image_key, image) in enumerate(zip(missing_keys, missing_images)):
            grid = image_grid_thw[idx]
            patch_count = self._grid_product(grid)
            session_cache[image_key] = {
                "pixel_values": self._slice_copy(pixel_values, cursor, cursor + patch_count),
                "image_grid_thw": self._slice_copy(image_grid_thw, idx, idx + 1),
                "num_image_tokens": patch_count // merge_length,
                "image_data": encode_image_for_rollout_engine(image),
            }
            cursor += patch_count

        return True

    def _expand_messages_with_cached_images(
        self,
        session_id: str,
        messages_str: str,
        image_keys: List[str],
    ) -> str:
        if not image_keys:
            return messages_str

        image_token = getattr(self.processor, "image_token", None) or getattr(self.tokenizer, "image_token", None)
        if image_token is None:
            raise ValueError("Processor image token is not available.")

        session_cache = self._get_session_image_cache(session_id)
        placeholder = "<|aievo_image_placeholder|>"
        expanded_messages_str = messages_str

        for image_key in image_keys:
            cache_entry = session_cache.get(image_key)
            if cache_entry is None:
                raise KeyError(f"Missing cached image for key={image_key}")
            token_count = int(cache_entry["num_image_tokens"])
            if image_token not in expanded_messages_str:
                raise ValueError("Image token count mismatch while expanding cached images.")
            expanded_messages_str = expanded_messages_str.replace(
                image_token,
                placeholder * token_count,
                1,
            )

        if image_token in expanded_messages_str:
            raise ValueError("Unexpanded image token remains after cached image expansion.")

        return expanded_messages_str.replace(placeholder, image_token)

    def _build_mm_train_inputs_from_cache(
        self,
        session_id: str,
        image_keys: List[str],
    ) -> Optional[Dict[str, Any]]:
        if not image_keys:
            return None

        session_cache = self._get_session_image_cache(session_id)
        pixel_values = []
        image_grid_thw = []
        for image_key in image_keys:
            cache_entry = session_cache.get(image_key)
            if cache_entry is None:
                return None
            pixel_values.append(cache_entry["pixel_values"])
            image_grid_thw.append(cache_entry["image_grid_thw"])

        return {
            "pixel_values": self._concat_cached_values(pixel_values),
            "image_grid_thw": self._concat_cached_values(image_grid_thw),
        }

    def _count_image_tokens_in_input_ids(self, input_ids: List[int]) -> Optional[int]:
        image_token_id = getattr(self.processor, "image_token_id", None) or getattr(self.tokenizer, "image_token_id", None)
        if image_token_id is None:
            image_token = getattr(self.processor, "image_token", None) or getattr(self.tokenizer, "image_token", None)
            if image_token is None:
                return None
            image_token_id = self.tokenizer.convert_tokens_to_ids(image_token)
        return sum(1 for token_id in input_ids if token_id == image_token_id)

    def _count_image_tokens_from_mm_inputs(self, mm_train_inputs: Optional[Dict[str, Any]]) -> Optional[int]:
        if mm_train_inputs is None:
            return 0

        image_grid_thw = mm_train_inputs.get("image_grid_thw")
        if image_grid_thw is None:
            return 0

        image_processor = getattr(self.processor, "image_processor", None)
        merge_size = getattr(image_processor, "merge_size", None)
        if merge_size is None:
            return None

        merge_length = merge_size ** 2
        total = 0
        for grid in image_grid_thw:
            total += self._grid_product(grid) // merge_length
        return total

    def _values_equal(self, left: Any, right: Any) -> bool:
        if left is None or right is None:
            return left is right

        left_module = getattr(left.__class__, "__module__", "")
        right_module = getattr(right.__class__, "__module__", "")
        if left_module.startswith("torch") and right_module.startswith("torch"):
            import torch

            return torch.equal(left, right)

        try:
            import numpy as np

            if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
                return np.array_equal(left, right)
        except Exception:
            pass

        if isinstance(left, list) and isinstance(right, list):
            return left == right

        return left == right

    def _value_shape(self, value: Any) -> Optional[List[int]]:
        if value is None:
            return None
        shape = getattr(value, "shape", None)
        if shape is not None:
            return [int(dim) for dim in shape]
        if isinstance(value, list):
            return [len(value)]
        return None

    def _value_fingerprint(self, value: Any) -> Optional[str]:
        if value is None:
            return None

        module_name = getattr(value.__class__, "__module__", "")
        if module_name.startswith("torch"):
            tensor = value.detach().cpu().contiguous()
            return hashlib.sha256(tensor.numpy().tobytes()).hexdigest()

        try:
            import numpy as np

            if isinstance(value, np.ndarray):
                return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
        except Exception:
            pass

        if isinstance(value, list):
            return hashlib.sha256(json.dumps(value, ensure_ascii=False).encode("utf-8")).hexdigest()

        return hashlib.sha256(repr(value).encode("utf-8")).hexdigest()

    def _first_diff_index(self, left: List[int], right: List[int]) -> Optional[int]:
        for idx, (left_value, right_value) in enumerate(zip(left, right)):
            if left_value != right_value:
                return idx
        if len(left) != len(right):
            return min(len(left), len(right))
        return None

    def _value_debug_summary(self, value: Any) -> Dict[str, Any]:
        summary: Dict[str, Any] = {
            "type": type(value).__name__ if value is not None else None,
            "shape": self._value_shape(value),
            "fingerprint": self._value_fingerprint(value),
        }
        dtype = getattr(value, "dtype", None)
        if dtype is not None:
            summary["dtype"] = str(dtype)
        return summary

    def _write_cache_processor_mismatch_debug_file(
        self,
        session_id: str,
        messages: List[Dict[str, Any]],
        normalized_messages: List[Dict[str, Any]],
        original_messages_str: str,
        cached_messages_str: str,
        image_keys: List[str],
        cached_input_ids: List[int],
        processor_input_ids: List[int],
        cached_mm_train_inputs: Optional[Dict[str, Any]],
        processor_mm_train_inputs: Optional[Dict[str, Any]],
        mismatches: List[str],
    ) -> str:
        debug_root = os.environ.get("AIEVOBOX_ROOT", os.getcwd())
        debug_dir = os.path.join(debug_root, "logs", "trajectory_cache_processor_mismatch")
        os.makedirs(debug_dir, exist_ok=True)

        session_name = session_id or "empty_session"
        debug_file = os.path.join(
            debug_dir,
            f"{session_name}_{time.time_ns()}.json",
        )

        first_diff_index = self._first_diff_index(cached_input_ids, processor_input_ids)
        window_start = max((first_diff_index or 0) - 16, 0)
        window_end = (first_diff_index or 0) + 16

        payload = {
            "session_id": session_id,
            "messages": messages,
            "normalized_messages": normalized_messages,
            "original_messages_str": original_messages_str,
            "cached_messages_str": cached_messages_str,
            "image_keys": image_keys,
            "mismatches": mismatches,
            "cached_input_ids_len": len(cached_input_ids),
            "processor_input_ids_len": len(processor_input_ids),
            "cached_image_tokens": self._count_image_tokens_in_input_ids(cached_input_ids),
            "processor_image_tokens": self._count_image_tokens_in_input_ids(processor_input_ids),
            "first_input_diff_index": first_diff_index,
            "cached_input_ids_window": cached_input_ids[window_start:window_end],
            "processor_input_ids_window": processor_input_ids[window_start:window_end],
            "cached_mm_train_inputs": {
                "pixel_values": self._value_debug_summary(
                    None if cached_mm_train_inputs is None else cached_mm_train_inputs.get("pixel_values")
                ),
                "image_grid_thw": self._value_debug_summary(
                    None if cached_mm_train_inputs is None else cached_mm_train_inputs.get("image_grid_thw")
                ),
            },
            "processor_mm_train_inputs": {
                "pixel_values": self._value_debug_summary(
                    None if processor_mm_train_inputs is None else processor_mm_train_inputs.get("pixel_values")
                ),
                "image_grid_thw": self._value_debug_summary(
                    None if processor_mm_train_inputs is None else processor_mm_train_inputs.get("image_grid_thw")
                ),
            },
        }

        with open(debug_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        return debug_file

    def _validate_cache_against_processor(
        self,
        session_id: str,
        messages: List[Dict[str, Any]],
        normalized_messages: List[Dict[str, Any]],
        original_messages_str: str,
        cached_messages_str: str,
        images: List[Any],
        image_keys: List[str],
        cached_input_ids: List[int],
        cached_mm_train_inputs: Optional[Dict[str, Any]],
    ) -> None:
        proc_out = self.processor(text=original_messages_str, images=images, return_tensors="pt")
        raw_input_ids = proc_out["input_ids"][0]
        processor_input_ids = raw_input_ids.tolist() if hasattr(raw_input_ids, "tolist") else list(raw_input_ids)
        processor_mm_train_inputs = {
            "pixel_values": proc_out.get("pixel_values"),
            "image_grid_thw": proc_out.get("image_grid_thw"),
        }

        mismatches = []
        if cached_input_ids != processor_input_ids:
            mismatches.append("input_ids")

        cached_pixel_values = None if cached_mm_train_inputs is None else cached_mm_train_inputs.get("pixel_values")
        cached_image_grid_thw = None if cached_mm_train_inputs is None else cached_mm_train_inputs.get("image_grid_thw")
        if not self._values_equal(cached_pixel_values, processor_mm_train_inputs.get("pixel_values")):
            mismatches.append("pixel_values")
        if not self._values_equal(cached_image_grid_thw, processor_mm_train_inputs.get("image_grid_thw")):
            mismatches.append("image_grid_thw")

        if mismatches:
            debug_file = self._write_cache_processor_mismatch_debug_file(
                session_id=session_id,
                messages=messages,
                normalized_messages=normalized_messages,
                original_messages_str=original_messages_str,
                cached_messages_str=cached_messages_str,
                image_keys=image_keys,
                cached_input_ids=cached_input_ids,
                processor_input_ids=processor_input_ids,
                cached_mm_train_inputs=cached_mm_train_inputs,
                processor_mm_train_inputs=processor_mm_train_inputs,
                mismatches=mismatches,
            )
            raise RuntimeError(
                f"cache vs processor mismatch: session={session_id} mismatches={mismatches} debug_file={debug_file}"
            )

    def _process_with_image_cache(
        self,
        session_id: str,
        messages: List[Dict[str, Any]],
        normalized_messages: List[Dict[str, Any]],
        messages_str: str,
        images: List[Any],
        image_keys: List[str],
    ) -> Dict[str, Any]:
        if not self._ensure_images_cached(session_id, image_keys, images):
            raise ValueError(f"Failed to populate image cache for session={session_id}")

        expanded_messages_str = self._expand_messages_with_cached_images(
            session_id,
            messages_str,
            image_keys,
        )
        if expanded_messages_str:
            expanded_messages_str = unicodedata.normalize("NFC", expanded_messages_str)

        input_ids = self.tokenizer.encode(expanded_messages_str, add_special_tokens=False)
        session_cache = self._get_session_image_cache(session_id)
        mm_train_inputs = self._build_mm_train_inputs_from_cache(session_id, image_keys)
        if self.enable_cache_processor_compare:
            self._validate_cache_against_processor(
                session_id=session_id,
                messages=messages,
                normalized_messages=normalized_messages,
                original_messages_str=messages_str,
                cached_messages_str=expanded_messages_str,
                images=images,
                image_keys=image_keys,
                cached_input_ids=input_ids,
                cached_mm_train_inputs=mm_train_inputs,
            )

        return {
            "messages_str": expanded_messages_str,
            "input_ids": input_ids,
            "image_data": [session_cache[image_key]["image_data"] for image_key in image_keys],
            "mm_train_inputs": mm_train_inputs,
        }

    def clear_image_cache(self, session_id: str) -> None:
        if session_id in self.session_image_cache:
            del self.session_image_cache[session_id]

    def match_prefix_with_mask(
        self,
        prefix_text: str,
        target_text: str,
    ) -> Optional[int]:
        """
        判断 prefix_text 是否是 target_text 的前缀（prefix_text 中的 <think>...</think> 在 target_text 中可选）。

        如果匹配成功，返回 target_text 中匹配到的前缀长度；否则返回 None。
        """
        prefix_idx = 0
        target_idx = 0

        while prefix_idx < len(prefix_text):
            if prefix_text[prefix_idx:].startswith("<think>"):
                end_tag = "</think>"
                think_start = prefix_idx
                close_pos = prefix_text.find(end_tag, prefix_idx + len("<think>"))
                if close_pos == -1:
                    # 没有闭合标签，当作普通字符处理
                    if target_idx >= len(target_text):
                        return target_idx
                    if prefix_text[prefix_idx] != target_text[target_idx]:
                        return None
                    prefix_idx += 1
                    target_idx += 1
                else:
                    think_end = close_pos + len(end_tag)
                    think_content = prefix_text[think_start:think_end]
                    if target_text[target_idx:].startswith(think_content):
                        # target 中存在 <think> 块
                        target_idx += len(think_content)
                        prefix_idx = think_end
                    else:
                        # target 中没有 <think> 块，跳过
                        prefix_idx = think_end
                        while prefix_idx < len(prefix_text) and prefix_text[prefix_idx] == '\n':
                            prefix_idx += 1
            else:
                if target_idx >= len(target_text):
                    # target_text 已用完，返回匹配长度
                    return target_idx
                if prefix_text[prefix_idx] != target_text[target_idx]:
                    # 字符不匹配
                    return None
                prefix_idx += 1
                target_idx += 1

        return target_idx

    def _find_best_match(
        self,
        session_id: str,
        target_messages_str: str,
    ) -> Tuple[Optional[Dict[str, Any]], int]:
        """找到最佳匹配的历史记录。

        Returns:
            (matched_record, matched_prefix_length)
            如果没有匹配，返回 (None, 0)
        """
        if session_id not in self.session_data:
            return None, 0

        if target_messages_str:
            target_messages_str = unicodedata.normalize("NFC", target_messages_str)
        best_record = None
        best_match_len = 0

        for record in self.session_data[session_id]:
            normalized_record_messages_str = (
                unicodedata.normalize("NFC", record["messages_str"])
                if record["messages_str"]
                else record["messages_str"]
            )
            if normalized_record_messages_str != record["messages_str"]:
                record["messages_str"] = normalized_record_messages_str
            matched_len = self.match_prefix_with_mask(
                record["messages_str"],
                target_messages_str
            )
            if matched_len is not None and matched_len > best_match_len:
                best_record = record
                best_match_len = matched_len

        return best_record, best_match_len

    def _write_no_match_debug_file(
        self,
        session_id: str,
        messages: List[Dict[str, Any]],
        normalized_messages: List[Dict[str, Any]],
        target_messages_str: str,
    ) -> str:
        debug_root = os.environ.get("AIEVOBOX_ROOT", os.getcwd())
        debug_dir = os.path.join(debug_root, "logs", "trajectory_match_miss")
        os.makedirs(debug_dir, exist_ok=True)

        session_name = session_id or "empty_session"
        debug_file = os.path.join(
            debug_dir,
            f"{session_name}_{time.time_ns()}.json",
        )

        payload = {
            "session_id": session_id,
            "messages": messages,
            "normalized_messages": normalized_messages,
            "target_messages_str": target_messages_str,
            "target_messages_str_len": len(target_messages_str),
            "session_records": self.session_data.get(session_id, []),
            "session_records_count": len(self.session_data.get(session_id, [])),
        }

        with open(debug_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        return debug_file

    def _write_mm_mismatch_debug_file(
        self,
        session_id: str,
        messages: List[Dict[str, Any]],
        normalized_messages: List[Dict[str, Any]],
        target_messages_str: str,
        matched_record: Dict[str, Any],
        record_image_tokens: int,
        mm_image_tokens: int,
    ) -> str:
        debug_root = os.environ.get("AIEVOBOX_ROOT", os.getcwd())
        debug_dir = os.path.join(debug_root, "logs", "trajectory_mm_mismatch")
        os.makedirs(debug_dir, exist_ok=True)

        session_name = session_id or "empty_session"
        debug_file = os.path.join(
            debug_dir,
            f"{session_name}_{time.time_ns()}.json",
        )

        payload = {
            "session_id": session_id,
            "messages": messages,
            "normalized_messages": normalized_messages,
            "target_messages_str": target_messages_str,
            "target_messages_str_len": len(target_messages_str),
            "matched_record_messages_str": matched_record.get("messages_str"),
            "matched_record_tokens_len": len(matched_record.get("tokens") or []),
            "record_image_tokens": record_image_tokens,
            "mm_image_tokens": mm_image_tokens,
        }

        with open(debug_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        return debug_file

    def prepare_generate_input(
        self,
        session_id: str,
        messages: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """准备发送给 /generate 的输入。

        通过匹配历史记录，复用已有的 tokens，只 tokenize 新增部分。

        Args:
            session_id: Session ID
            messages: 输入的 messages（不含本次 assistant 回复）

        Returns:
            {
                "input_ids": List[int],        # 准备发送的完整 token 序列
                "messages_str": str,           # 当前 messages 对应的字符串
                "image_data": List[str],       # 编码后的图片数据（发给 /generate）
                "matched_record": Optional[Dict],  # 匹配到的历史记录
                "matched_tokens_count": int,   # 匹配到的 tokens 数量
            }
        """
        normalized_messages = self._normalize_messages_for_qwen_vision(messages)

        # 提取多模态信息
        images, _ = process_vision_info(normalized_messages)
        images = images or []  # 确保 images 不为 None
        image_keys = self._extract_ordered_image_keys(normalized_messages) if images else []

        # 构建当前 messages 的字符串（不含 generation_prompt，用于匹配）
        current_messages_str = self.tokenizer.apply_chat_template(
            normalized_messages,
            add_generation_prompt=False,
            tokenize=False
        )
        if self.processor is not None and images:
            cached_result = self._process_with_image_cache(
                session_id,
                messages,
                normalized_messages,
                current_messages_str,
                images,
                image_keys,
            )
            current_messages_str = cached_result["messages_str"]
            current_input_ids = list(cached_result["input_ids"])
            image_data = list(cached_result["image_data"])
        else:
            current_input_ids = self.tokenizer.encode(current_messages_str, add_special_tokens=False)
            image_data = []
        if current_messages_str:
            current_messages_str = unicodedata.normalize("NFC", current_messages_str)

        # 字符级匹配
        matched_record, matched_prefix_len = self._find_best_match(
            session_id, current_messages_str
        )

        matched_tokens_count = 0  # 匹配到的 tokens 数量

        if matched_record is not None:
            for i in range(len(matched_record["tokens"])):
                decoded = self.tokenizer.decode(matched_record["tokens"][: i + 1])
                if decoded:
                    decoded = unicodedata.normalize("NFC", decoded)
                if len(decoded) <= matched_prefix_len:
                    matched_tokens_count = i + 1
                else:
                    break
            input_ids = list(matched_record["tokens"][:matched_tokens_count])
            new_context_str = current_messages_str[matched_prefix_len:]
            new_context_str += self.generation_prompt
            if new_context_str:
                input_ids.extend(self.tokenizer.encode(new_context_str, add_special_tokens=False))
        else:
            input_ids = list(current_input_ids)
            input_ids.extend(self.generation_tokens)

        # 处理 image_data：复用匹配记录的 image_data，只编码新增图片
        if not image_data:
            if matched_record is not None:
                prev_image_data = list(matched_record.get("image_data") or [])
                if len(images) > len(prev_image_data):
                    new_images = images[len(prev_image_data):]
                    image_data = prev_image_data + [encode_image_for_rollout_engine(img) for img in new_images]
                else:
                    image_data = prev_image_data
            else:
                image_data = [encode_image_for_rollout_engine(img) for img in images]

        return {
            "input_ids": input_ids,
            "messages_str": current_messages_str,
            "image_data": image_data,
            "matched_record": matched_record,
            "matched_tokens_count": matched_tokens_count,
        }

    def save(
        self,
        session_id: str,
        messages_str: str,
        input_ids: List[int],
        output_ids: List[int],
        output_logprobs: List[List],  # [[logprob, token_id, ...], ...]
        image_data: List[str],
        finish_reason: Optional[str] = None,
        matched_record: Optional[Dict[str, Any]] = None,
        matched_tokens_count: int = 0,
        assistant_text: str = "",
    ) -> None:
        """保存一次生成的 trajectory 信息。

        Args:
            session_id: Session ID
            messages_str: 当前 messages 对应的字符串（来自 prepare_generate_input）
            input_ids: 发送给 /generate 的 input_ids（来自 prepare_generate_input）
            output_ids: 本次生成的 token IDs
            output_logprobs: 本次生成的 logprobs
            image_data: 编码后的图片数据
            finish_reason: 生成结束原因
            matched_record: 已匹配的历史记录（来自 prepare_generate_input，避免重复匹配）
            matched_tokens_count: 匹配到的 tokens 数量（来自 prepare_generate_input）
            assistant_text: generate API 返回的解码文本（已 skip_special_tokens）
        """
        if session_id not in self.session_data:
            self.session_data[session_id] = []

        # 如果没有传入 matched_record，重新匹配
        if matched_record is None:
            matched_record, _ = self._find_best_match(
                session_id,
                unicodedata.normalize("NFC", messages_str) if messages_str else messages_str,
            )

        # 在匹配的基础上构建 tokens、response_mask、logprobs
        if matched_record is not None and matched_tokens_count > 0:
            # 复制匹配记录的数据（只复制匹配部分）
            prev_response_mask = list(matched_record["response_mask"][:matched_tokens_count])
            prev_logprobs = list(matched_record["logprobs"][:matched_tokens_count])
        else:
            prev_response_mask = []
            prev_logprobs = []

        # input_ids 已经包含了匹配的 tokens + 新增 context tokens + generation tokens
        # 计算新增的 context 部分（包括 generation_prompt）
        new_context_len = len(input_ids) - matched_tokens_count

        # 构建完整的 tokens 和 mask
        tokens = list(input_ids)  # input_ids 已经是正确的
        response_mask = prev_response_mask + [0] * new_context_len
        logprobs_list = prev_logprobs + [0.0] * new_context_len

        # 追加 output tokens (mask=1)
        # Decode output_ids（不 skip special tokens），以便完整包含格式化字符
        decoded_output = self.tokenizer.decode(output_ids, skip_special_tokens=False)

        # 解析 logprobs
        parsed_logprobs = []
        for item in output_logprobs:
            if isinstance(item, (list, tuple)) and len(item) >= 1:
                logprob = float(item[0]) if item[0] is not None else 0.0
                parsed_logprobs.append(logprob)
            else:
                parsed_logprobs.append(0.0)

        tokens.extend(output_ids)
        response_mask.extend([1] * len(output_ids))
        logprobs_list.extend(parsed_logprobs)

        # 构建 messages_str（通过 append 方式，而不是重新 apply_chat_template）
        # 直接 append decoded_output，它已经包含了所有格式化字符（如 <|im_end|>\n）
        full_messages_str = messages_str + self.generation_prompt + decoded_output
        if full_messages_str:
            full_messages_str = unicodedata.normalize("NFC", full_messages_str)

        # 保存新记录
        new_record = {
            "messages_str": full_messages_str,
            "tokens": tokens,
            "response_mask": response_mask,
            "logprobs": logprobs_list,
            "image_data": image_data,
        }
        self.session_data[session_id].append(new_record)

    def get_training_info(
        self,
        session_id: str,
        messages: List[Dict[str, Any]],
    ) -> Tuple[List[int], List[int], List[str], str, Optional[Dict]]:
        """获取训练所需的 tokens、response_mask、image_data、messages_str 和 mm_train_inputs。

        Args:
            session_id: Session ID
            messages: 完整的 messages（含 assistant 回复）

        Returns:
            (tokens, response_mask, image_data, messages_str, mm_train_inputs)
            mm_train_inputs: processor 输出的视觉张量（pixel_values, image_grid_thw 等），
                             无图片时为 None。
        """
        normalized_messages = self._normalize_messages_for_qwen_vision(messages)

        # 提取多模态信息
        images, _ = process_vision_info(normalized_messages)
        images = images or []  # 确保 images 不为 None
        image_keys = self._extract_ordered_image_keys(normalized_messages) if images else []

        # 构建 messages_str
        messages_str = self.tokenizer.apply_chat_template(
            normalized_messages,
            add_generation_prompt=False,
            tokenize=False
        )

        if self.processor is not None and images:
            cached_result = self._process_with_image_cache(
                session_id,
                messages,
                normalized_messages,
                messages_str,
                images,
                image_keys,
            )
            messages_str = cached_result["messages_str"]
            mm_train_inputs = cached_result["mm_train_inputs"]
        else:
            mm_train_inputs = None
        if messages_str:
            messages_str = unicodedata.normalize("NFC", messages_str)

        # 查找匹配的记录
        matched_record, matched_len = self._find_best_match(session_id, messages_str)
        if matched_record is None:
            has_data = session_id in self.session_data and len(self.session_data[session_id]) > 0
            debug_file = self._write_no_match_debug_file(
                session_id=session_id,
                messages=messages,
                normalized_messages=normalized_messages,
                target_messages_str=messages_str,
            )
            logger.warning(
                "get_training_info failed: session=%s, has_data=%s, matched_len=%s, debug_file=%s",
                session_id,
                has_data,
                matched_len,
                debug_file,
            )
            return [], [], [], "", None

        if mm_train_inputs is not None:
            record_image_tokens = self._count_image_tokens_in_input_ids(list(matched_record["tokens"]))
            mm_image_tokens = self._count_image_tokens_from_mm_inputs(mm_train_inputs)
            if (
                record_image_tokens is not None
                and mm_image_tokens is not None
                and record_image_tokens != mm_image_tokens
            ):
                debug_file = self._write_mm_mismatch_debug_file(
                    session_id=session_id,
                    messages=messages,
                    normalized_messages=normalized_messages,
                    target_messages_str=messages_str,
                    matched_record=matched_record,
                    record_image_tokens=record_image_tokens,
                    mm_image_tokens=mm_image_tokens,
                )
                logger.error(
                    "get_training_info mm mismatch: session=%s record_image_tokens=%s mm_image_tokens=%s debug_file=%s",
                    session_id,
                    record_image_tokens,
                    mm_image_tokens,
                    debug_file,
                )
                return [], [], [], "", None
        return (
            list(matched_record["tokens"]),
            list(matched_record["response_mask"]),
            list(matched_record.get("image_data") or []),
            messages_str,
            mm_train_inputs,
        )

    def query_logprobs(self, session_id: str, messages_str: str) -> List[float]:
        """查询与 messages_str 匹配的 logprobs。"""
        matched_record, _ = self._find_best_match(
            session_id,
            unicodedata.normalize("NFC", messages_str) if messages_str else messages_str,
        )
        if matched_record is None:
            return []
        return list(matched_record["logprobs"])

    def query_latest(self, session_id: str) -> Optional[Dict[str, Any]]:
        """查询 session 最新的记录。"""
        if session_id not in self.session_data or not self.session_data[session_id]:
            return None
        return self.session_data[session_id][-1]

    def clear_session(self, session_id: str) -> None:
        """清除指定 session 的数据。"""
        if session_id in self.session_data:
            del self.session_data[session_id]
        self.clear_image_cache(session_id)
