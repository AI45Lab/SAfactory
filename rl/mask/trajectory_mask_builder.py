"""Trajectory Builder for LLM Proxy.

核心设计：
1. 匹配：用字符级匹配（含 <think> 跳过）找到最佳匹配的历史记录
2. 追加：在匹配到的记录基础上，追加新的 context tokens 和 output tokens
"""

import logging
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

        best_record = None
        best_match_len = 0

        for record in self.session_data[session_id]:
            matched_len = self.match_prefix_with_mask(
                record["messages_str"],
                target_messages_str
            )
            if matched_len is not None and matched_len > best_match_len:
                best_record = record
                best_match_len = matched_len

        return best_record, best_match_len

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
        images, videos = process_vision_info(normalized_messages)
        images = images or []  # 确保 images 不为 None

        # 构建当前 messages 的字符串（不含 generation_prompt，用于匹配）
        current_messages_str = self.tokenizer.apply_chat_template(
            normalized_messages,
            add_generation_prompt=False,
            tokenize=False
        )

        # 多模态处理：如果有 processor 且有 images，用 processor 处理
        proc_out = None
        if self.processor is not None and images:
            proc_out = self.processor(text=current_messages_str, images=images, tokenize=False)
            current_messages_str = self.tokenizer.decode(
                proc_out["input_ids"][0],
                skip_special_tokens=False,
            )

        # 字符级匹配
        matched_record, matched_prefix_len = self._find_best_match(
            session_id, current_messages_str
        )

        matched_tokens_count = 0  # 匹配到的 tokens 数量

        if matched_record is not None:
            # 找到匹配字符串对应的 token 位置
            # matched_prefix_len 是字符级别的匹配长度
            # 我们需要找到对应的 token 边界
            matched_str = matched_record["messages_str"][:matched_prefix_len]

            # Tokenize 匹配部分，找到对应的 token 数量
            # 注意：我们不能简单地 tokenize matched_str，因为可能有上下文依赖
            # 最安全的方式是遍历 tokens 逐个 decode，找到匹配位置
            for i, token in enumerate(matched_record["tokens"]):
                decoded = self.tokenizer.decode(matched_record["tokens"][:i+1])
                if len(decoded) <= matched_prefix_len:
                    matched_tokens_count = i + 1
                else:
                    break

            # 复用匹配部分的 tokens
            input_ids = list(matched_record["tokens"][:matched_tokens_count])

            # 计算新增的 context 字符串（使用字符级匹配长度）
            new_context_str = current_messages_str[matched_prefix_len:]
            new_context_str += self.generation_prompt

            # Tokenize 新增部分并追加
            if new_context_str:
                new_context_tokens = self.tokenizer.encode(new_context_str, add_special_tokens=False)
                input_ids.extend(new_context_tokens)
        else:
            # 没有匹配，完整 tokenize
            if proc_out is not None:
                # 使用 processor 的输出，并添加 generation_prompt
                input_ids = proc_out["input_ids"][0]
                input_ids.extend(self.generation_tokens)
            else:
                input_ids = self.tokenizer.apply_chat_template(
                    normalized_messages,
                    tokenize=True,
                    add_generation_prompt=True
                )

        # 处理 image_data：复用匹配记录的 image_data，只编码新增图片
        image_data: List[str] = []
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
            matched_record, _ = self._find_best_match(session_id, messages_str)

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
    ) -> Tuple[List[int], List[int], List[str], str]:
        """获取训练所需的 tokens、response_mask、image_data 和 messages_str。

        Args:
            session_id: Session ID
            messages: 完整的 messages（含 assistant 回复）

        Returns:
            (tokens, response_mask, image_data, messages_str)
        """
        normalized_messages = self._normalize_messages_for_qwen_vision(messages)

        # 提取多模态信息
        images, videos = process_vision_info(normalized_messages)
        images = images or []  # 确保 images 不为 None

        # 构建 messages_str
        messages_str = self.tokenizer.apply_chat_template(
            normalized_messages,
            add_generation_prompt=False,
            tokenize=False
        )

        # 多模态处理：如果有 processor 且有 images，用 processor 处理
        if self.processor is not None and images:
            proc_out = self.processor(text=messages_str, images=images, tokenize=False)
            messages_str = self.tokenizer.decode(
                proc_out["input_ids"][0],
                skip_special_tokens=False,
            )

        # 查找匹配的记录
        matched_record, matched_len = self._find_best_match(session_id, messages_str)
        if matched_record is None:
            has_data = session_id in self.session_data and len(self.session_data[session_id]) > 0
            logger.warning(f"get_training_info failed: session={session_id}, has_data={has_data}, matched_len={matched_len}")
            return [], [], [], ""
        return (
            list(matched_record["tokens"]),
            list(matched_record["response_mask"]),
            list(matched_record.get("image_data") or []),
            messages_str,
        )

    def query_logprobs(self, session_id: str, messages_str: str) -> List[float]:
        """查询与 messages_str 匹配的 logprobs。"""
        matched_record, _ = self._find_best_match(session_id, messages_str)
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
