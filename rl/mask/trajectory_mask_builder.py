from typing import Any, Dict, List, Optional


class TrajectoryMaskBuilder:
    """按 session 维护字符串级 trajectory mask 的构造器。

    核心功能：
    - 将一整段对话序列解码为字符串，并维护一份与之等长的字符级 mask。
    - mask 中通常用 0/1 标记哪些字符参与损失计算。
    - 多轮对话时，尽量复用历史中已经算好的前缀 mask。
    - 对 `<think>...</think>` 片段做“可选前缀匹配”：允许新字符串里缺失或保留这段内容。
    """

    def __init__(self, tokenizer) -> None:
        """初始化构造器。

        Args:
            tokenizer: 负责将 messages 序列解码为字符串的 tokenizer，对外保持透明。
        """
        self.tokenizer = tokenizer
        # session_id -> List[Tuple[messages_str, mask]]
        self.session_masks: dict[str, list[tuple[str, list[int]]]] = {}

    def match_prefix_with_mask(
        self,
        prefix_text: str,
        prefix_mask: List[int],
        target_text: str,
    ) -> Optional[List[int]]:
        """
        判断 prefix_text 是否是 target_text 的前缀（prefix_text 中的 <think>...</think> 在 target_text 中可选）。

        如果匹配成功，返回 target_text 中对应前缀部分的 mask；否则返回 None。

        Args:
            prefix_text: 前缀字符串（可能包含 <think>...</think>）。
            prefix_mask: 与 prefix_text 等长的 mask 列表。
            target_text: 目标字符串。

        Returns:
            匹配成功时返回 target_text 对应前缀的 mask，失败返回 None。
        """

        prefix_idx = 0
        target_idx = 0
        target_mask: List[int] = []
        while prefix_idx < len(prefix_text):
            # 检查是否遇到 <think>
            if prefix_text[prefix_idx:].startswith("<think>"):
                # 找到对应的 </think>
                end_tag = "</think>"
                think_start = prefix_idx
                close_pos = prefix_text.find(end_tag, prefix_idx + len("<think>"))
                if close_pos == -1:
                    # 没有闭合标签，当作普通字符处理
                    if target_idx >= len(target_text) or prefix_text[prefix_idx] != target_text[target_idx]:
                        return None
                    target_mask.append(prefix_mask[prefix_idx])
                    prefix_idx += 1
                    target_idx += 1
                else:
                    think_end = close_pos + len(end_tag)
                    think_content = prefix_text[think_start:think_end]
                    # 检查 target_text 中是否也有这段 <think>...</think>
                    if target_text[target_idx:].startswith(think_content):
                        # 目标字符串中存在，映射 mask
                        for i in range(len(think_content)):
                            target_mask.append(prefix_mask[think_start + i])
                        target_idx += len(think_content)
                        prefix_idx = think_end
                    else:
                        # 否则 target_text 中没有这段，跳过 prefix_text 中的 think 块（不映射）
                        prefix_idx = think_end

                        while prefix_idx < len(prefix_text) and prefix_text[prefix_idx] == '\n':
                            prefix_idx += 1
            else:
                # 普通字符，必须逐字符匹配
                if target_idx >= len(target_text) or prefix_text[prefix_idx] != target_text[target_idx]:
                    return None
                target_mask.append(prefix_mask[prefix_idx])
                prefix_idx += 1
                target_idx += 1
        
        return target_mask

    def save(self, session_id: str, messages: List[Dict[str, Any]], assistant_text: str) -> None:
        """记录一次新的对话轨迹，并更新对应的 mask。

        当前实现假设：
        - `messages` 包含历史消息以及当前用户消息；
        - `assistant_text` 是这次新生成的模型回复（纯文本）。
        """
        if session_id not in self.session_masks:
            self.session_masks[session_id] = []

        # 先只用历史 + 当前用户消息，构造当前上下文字符串
        current_messages_str = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=False,
            tokenize=False
        )

        # 在已有的 session 轨迹中，寻找与当前字符串前缀匹配且长度最长的一条 mask
        current_mask: List[int] = []
        for prev_messages_str, prev_mask in self.session_masks[session_id]:
            matched_mask = self.match_prefix_with_mask(prev_messages_str, prev_mask, current_messages_str)
            if matched_mask is not None and len(matched_mask) > len(current_mask):
                current_mask = matched_mask

        # 把 mask 填充到与当前上下文字符串长度一致，新增部分全部置为 0（仅上下文，不计入损失）
        current_mask = current_mask + [0] * (len(current_messages_str) - len(current_mask))

        # 将本轮 assistant 输出拼接在字符串后面
        full_messages_str = self.tokenizer.apply_chat_template(
            [*messages, {"role": "assistant", "content": assistant_text}],
            add_generation_prompt=False,
            tokenize=False
        )

        # 对新增长的 assistant 文本位置全部置为 1
        current_mask = current_mask + [1] * (len(full_messages_str) - len(current_mask))

        # 记录下这条完整轨迹
        self.session_masks[session_id].append((full_messages_str, current_mask))

    def query(self, session_id: str, messages_str: str) -> List[int]:
        """给定某个 session 下的一段对话字符串，返回与之对齐的 mask。"""
        current_mask: List[int] = []
        for prev_messages_str, prev_mask in self.session_masks[session_id]:
            matched_mask = self.match_prefix_with_mask(prev_messages_str, prev_mask, messages_str)
            if matched_mask is not None and len(matched_mask) > len(current_mask):
                current_mask = matched_mask

        # 对于当前字符串中没有历史 mask 覆盖到的部分，全部填 0
        current_mask = current_mask + [0] * (len(messages_str) - len(current_mask))
        return current_mask
                

    
