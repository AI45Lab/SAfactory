"""Trajectory Builder for LLM Proxy.

核心设计：
1. 匹配：用字符级匹配（含 <think> 跳过）找到最佳匹配的历史记录
2. 追加：在匹配到的记录基础上，追加新的 context tokens 和 output tokens
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from utils.multimodal import (
    encode_image_for_rollout_engine,
    extract_image_urls_from_messages,
    load_pil_images,
)

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
        #     "image_data": List[str],       # 多模态图片（按出现顺序累计，发给 /generate）
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

    def check_if_prefix(
        self,
        prefix_messages_str: str,
        target_messages_str: str,
    ) -> Tuple[bool, int]:
        """
        判断 prefix_messages_str 是否是 target_messages_str 的前缀（prefix_messages_str 中的 <think>...</think> 在 target_messages_str 中可被跳过）。

        如果匹配成功，返回 (True, target_idx)；否则返回 (False, 0)。
        """
        prefix_idx = 0
        target_idx = 0

        while prefix_idx < len(prefix_messages_str):
            if prefix_messages_str[prefix_idx:].startswith("<think>"):
                end_tag = "</think>"
                think_start = prefix_idx
                close_pos = prefix_messages_str.find(end_tag, prefix_idx + len("<think>"))
                if close_pos == -1:
                    # 没有闭合标签，当作普通字符处理
                    if prefix_messages_str[prefix_idx] != target_messages_str[target_idx]:
                        return False, -1
                    prefix_idx += 1
                    target_idx += 1
                else:
                    think_end = close_pos + len(end_tag)
                    think_content = prefix_messages_str[think_start:think_end]
                    if target_messages_str[target_idx:].startswith(think_content):
                        # target 中存在 <think> 块
                        target_idx += len(think_content)
                        prefix_idx = think_end
                    else:
                        # target 中没有 <think> 块，跳过
                        prefix_idx = think_end
                        # 对齐 prefix 和 target 的换行符
                        while prefix_idx < len(prefix_messages_str) and prefix_messages_str[prefix_idx] == '\n':
                            prefix_idx += 1
                        while target_idx < len(target_messages_str) and target_messages_str[target_idx] == '\n':
                            target_idx += 1
            else:
                if prefix_messages_str[prefix_idx] != target_messages_str[target_idx]:
                    return False, -1
                prefix_idx += 1
                target_idx += 1

        return True, target_idx

    def _find_longest_prefix(
        self,
        session_id: str,
        target_messages_str: str,
    ) -> Tuple[Optional[Dict[str, Any]], int]:
        """找到最长前缀匹配的历史记录。

        Returns:
            (matched_record, matched_target_prefix_len)
            如果没有找到前缀匹配，返回 (None, 0)
        """
        if session_id not in self.session_data:
            return None, 0

        longest_matched_record = None
        longest_target_idx = 0

        for record in self.session_data[session_id]:
            matched, matched_target_idx = self.check_if_prefix(
                record["messages_str"],
                target_messages_str
            )
            if not matched:
                continue
            if matched_target_idx > longest_target_idx:
                longest_matched_record = record
                longest_target_idx = matched_target_idx

        return longest_matched_record, longest_target_idx

    def _find_best_match(
        self,
        session_id: str,
        target_messages_str: str,
    ) -> Tuple[Optional[Dict[str, Any]], int, int]:
        """兼容旧接口：基于当前“最长前缀匹配”返回 best match。"""
        matched_record, matched_target_idx = self._find_longest_prefix(session_id, target_messages_str)
        if matched_record is None:
            return None, 0, 0
        matched_record_prefix_len = len(matched_record.get("messages_str", ""))
        return matched_record, matched_record_prefix_len, matched_target_idx

    def prepare_input_tokens(
        self,
        session_id: str,
        messages: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """准备发送给 /generate 的 input_ids。

        通过匹配历史记录，复用已有的 tokens，只 tokenize 新增部分。

        Args:
            session_id: Session ID
            messages: 输入的 messages（不含本次 assistant 回复）

        Returns:
            {
                "input_tokens": List[int],
                "image_data": List[str],  # 多模态 generate 需要的数据（PNG base64，无 data-uri 前缀；按出现顺序累计）
                "messages_str": str,  # 用于匹配/保存的字符串（多模态时为 processor 展开后的 tokens decode）
                "matched_record": Optional[Dict[str, Any]],
                "matched_target_idx": int,
            }
        """
        # 维护多模态状态：image_data 需要按“历史累计”发送给 /generate
        # 对齐 slime/examples/geo3k_vlm_multi_turn：发送 PNG bytes 的 base64 字符串（无 data-uri 前缀）
        image_refs = extract_image_urls_from_messages(messages)
        images = load_pil_images(image_refs) if image_refs else []

        # 构建当前 messages 的字符串
        current_messages_str = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )

        proc_out = None
        # multimodal: current_messages_str 使用 processor 展开后的 token 序列 decode 得到（与 save/query 对齐）
        if self.processor is not None and images:
            proc_out = self.processor(
                text=current_messages_str,
                images=images,
                tokenize=False
            )
            current_messages_str = self.tokenizer.decode(
                proc_out["input_ids"][0].tolist(),
                skip_special_tokens=False,
            )

        matched_record, matched_target_idx = self._find_longest_prefix(
            session_id, current_messages_str
        )

        if matched_record is not None:
            # 找到匹配字符串对应的 token 位置
            # 复用匹配部分的 tokens
            input_tokens = list(matched_record["tokens"])

            # 计算新增的 context 字符串（使用字符级匹配长度）
            new_context_str = current_messages_str[matched_target_idx:]

            # Tokenize 新增部分并追加
            if new_context_str:
                new_context_tokens = self.tokenizer.encode(new_context_str, add_special_tokens=False)
                input_tokens.extend(new_context_tokens)
        else:
            # 没有匹配，完整 tokenize
            if proc_out is not None:
                input_tokens = proc_out["input_ids"][0].tolist()
            else:
                input_tokens = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True
                )

        # 合并多模态数据：尽量复用 matched_record 的 image_data，只编码新增图片。
        image_data: List[str] = []
        if matched_record is not None and matched_record.get("image_data"):
            cached = list(matched_record.get("image_data") or [])
            cached_len = len(cached)
            if not images:
                image_data = cached
            elif len(images) == cached_len:
                image_data = cached
            elif len(images) > cached_len:
                new_images = images[cached_len:]
                image_data = cached + [encode_image_for_rollout_engine(img) for img in new_images]
            else:
                # 兜底：messages 未包含历史图片（或被裁剪），将本次图片视为增量追加
                image_data = cached + [encode_image_for_rollout_engine(img) for img in images]
        else:
            image_data = [encode_image_for_rollout_engine(img) for img in images]

        return {
            "input_tokens": input_tokens,
            "messages_str": current_messages_str,
            "image_data": image_data,
            "matched_record": matched_record,
            "matched_target_idx": matched_target_idx,
        }

    def save(
        self,
        session_id: str,
        messages_str: str,
        input_tokens: List[int],
        output_tokens: List[int],
        output_logprobs: List[List],  # [[logprob, token_id, ...], ...]
        image_data: Optional[List[str]] = None,
        finish_reason: Optional[str] = None,
        matched_record: Optional[Dict[str, Any]] = None,
        matched_tokens_count: int = 0,
        assistant_text: str = "",
    ) -> None:
        """保存一次生成的 trajectory 信息。

        Args:
            session_id: Session ID
            messages_str: 本次 prepare_input_tokens 生成时用于匹配/保存的 messages_str
            input_tokens: 发送给 /generate 的 input token IDs（来自 prepare_input_tokens）
            output_tokens: 本次生成的 token IDs
            output_logprobs: 本次生成的 logprobs
            image_data: 发给 /generate 的 image_data（多模态时按历史累计）
            finish_reason: 生成结束原因
            matched_record: 已匹配的历史记录（来自 prepare_input_tokens，可选）
            matched_tokens_count: 匹配到的 tokens 数量（可选）
            assistant_text: generate API 返回的解码文本（已 skip_special_tokens）
        """
        if session_id not in self.session_data:
            self.session_data[session_id] = []

        # 拼接本次完整 token 序列（补上 assistant 消息结束符，便于下一轮前缀复用）
        input_tokens = list(input_tokens)
        output_tokens = list(output_tokens)

        tokens = input_tokens + output_tokens + list(self.end_tokens)

        # response_mask: 0=context, 1=generated（仅标记真实生成的 output_tokens）
        response_mask = [0] * len(input_tokens) + [1] * len(output_tokens) + [0] * len(self.end_tokens)

        # logprobs: context/end 用 0.0 填充；generated 用 output_logprobs 的第 0 项
        out_lp: List[float] = []
        for item in output_logprobs or []:
            try:
                out_lp.append(float(item[0]))
            except Exception:
                out_lp.append(0.0)
        if len(out_lp) < len(output_tokens):
            out_lp.extend([0.0] * (len(output_tokens) - len(out_lp)))
        elif len(out_lp) > len(output_tokens):
            out_lp = out_lp[: len(output_tokens)]

        logprobs_list = [0.0] * len(input_tokens) + out_lp + [0.0] * len(self.end_tokens)

        full_messages_str = self.tokenizer.decode(tokens, skip_special_tokens=False)

        new_record = {
            "messages_str": full_messages_str,
            "tokens": tokens,
            "response_mask": response_mask,
            "logprobs": logprobs_list,
            "image_data": list(image_data or []),
            "finish_reason": finish_reason,
            "assistant_text": assistant_text,
        }
        self.session_data[session_id].append(new_record)

    def query_tokens(self, session_id: str, messages_str: str) -> Tuple[List[int], List[int]]:
        """查询与 messages_str 匹配的 tokens 和 response_mask。"""
        matched_record, _, matched_len = self._find_best_match(session_id, messages_str)
        if matched_record is None:
            # Debug: 打印是否有数据
            has_data = session_id in self.session_data and len(self.session_data[session_id]) > 0
            logger.warning(f"query_tokens failed: session={session_id}, has_data={has_data}, matched_len={matched_len}")
            return [], []
        return list(matched_record["tokens"]), list(matched_record["response_mask"])

    def query_logprobs(self, session_id: str, messages_str: str) -> List[float]:
        """查询与 messages_str 匹配的 logprobs。"""
        matched_record, _, _ = self._find_best_match(session_id, messages_str)
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
