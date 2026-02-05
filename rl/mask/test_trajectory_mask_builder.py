#!/usr/bin/env python3
"""
测试 TrajectoryMaskBuilder 的 tokenize 添加过程。

使用真实的 Qwen2.5 tokenizer，mock 生成部分。

Usage:
    # 使用默认 tokenizer (Qwen/Qwen2.5-7B-Instruct)
    python test_trajectory_mask_builder.py

    # 指定 tokenizer 路径
    TOKENIZER_PATH=/path/to/tokenizer python test_trajectory_mask_builder.py
"""

import os
import sys

# Add paths
AIEVOBOX_ROOT = os.environ.get("AIEVOBOX_ROOT", "/root/AIEvoBox")
sys.path.insert(0, AIEVOBOX_ROOT)
sys.path.insert(0, os.path.join(AIEVOBOX_ROOT, "rl"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from typing import List, Tuple
from transformers import AutoTokenizer
from trajectory_mask_builder import TrajectoryMaskBuilder


def get_tokenizer():
    """获取 tokenizer。"""
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "Qwen/Qwen2.5-7B-Instruct")
    print(f"Loading tokenizer from: {tokenizer_path}")
    return AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)


def mock_generate(tokenizer, text: str) -> Tuple[List[int], List[List]]:
    """Mock 生成：将文本转换为 token IDs 和假的 logprobs。"""
    output_ids = tokenizer.encode(text, add_special_tokens=False)
    # Mock logprobs: [[logprob, token_id], ...]
    output_logprobs = [[-0.5, tid] for tid in output_ids]
    return output_ids, output_logprobs


def print_tokens_with_mask(tokenizer, tokens: List[int], response_mask: List[int]):
    """打印 tokens 和对应的 mask，便于调试。"""
    print(f"\n{'='*60}")
    print(f"Total tokens: {len(tokens)}, Mask length: {len(response_mask)}")
    print(f"{'='*60}")

    # 分段显示
    segments = []
    current_mask = response_mask[0] if response_mask else 0
    current_tokens = []

    for i, (tid, m) in enumerate(zip(tokens, response_mask)):
        if m != current_mask:
            segments.append((current_tokens, current_mask))
            current_tokens = [tid]
            current_mask = m
        else:
            current_tokens.append(tid)
    if current_tokens:
        segments.append((current_tokens, current_mask))

    for tids, mask in segments:
        text = tokenizer.decode(tids)
        mask_str = "CONTEXT" if mask == 0 else "GENERATED"
        print(f"\n[{mask_str}] ({len(tids)} tokens)")
        print(f"  Text: {repr(text)}")
        print(f"  IDs: {tids[:10]}{'...' if len(tids) > 10 else ''}")


def test_single_turn():
    """测试单轮对话。"""
    print("\n" + "="*60)
    print("TEST: Single Turn")
    print("="*60)

    tokenizer = get_tokenizer()
    builder = TrajectoryMaskBuilder(tokenizer)

    session_id = "test_single"
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2+2?"},
    ]

    # Step 1: prepare_input_ids
    input_ids, matched_record, matched_prefix_len, _ = builder.prepare_input_ids(session_id, messages)
    print(f"\n[prepare_input_ids]")
    print(f"  input_ids length: {len(input_ids)}")
    print(f"  matched_record: {matched_record}")
    print(f"  matched_prefix_len: {matched_prefix_len}")

    # Step 2: Mock generate
    assistant_text = "The answer is 4."
    output_ids, output_logprobs = mock_generate(tokenizer, assistant_text)
    print(f"\n[mock_generate]")
    print(f"  assistant_text: {assistant_text}")
    print(f"  output_ids: {output_ids}")

    # Step 3: save
    saved_text = builder.save(
        session_id, messages, input_ids, output_ids, output_logprobs,
        finish_reason="stop", matched_record=matched_record
    )
    print(f"\n[save]")
    print(f"  saved_text: {saved_text}")

    # Step 4: 验证
    record = builder.query_latest(session_id)
    tokens = record["tokens"]
    response_mask = record["response_mask"]
    logprobs = record["logprobs"]

    print_tokens_with_mask(tokenizer, tokens, response_mask)

    # 验证 mask
    num_generated = sum(response_mask)
    print(f"\n[验证]")
    print(f"  Generated tokens (mask=1): {num_generated}")
    print(f"  Context tokens (mask=0): {len(response_mask) - num_generated}")

    # 验证 input_ids 是 tokens 的前缀
    assert tokens[:len(input_ids)] == input_ids, "input_ids should be prefix of tokens"
    print("  ✓ input_ids is prefix of tokens")

    # 验证 output_ids 在 tokens 中
    assert tokens[len(input_ids):len(input_ids)+len(output_ids)] == output_ids, "output_ids should follow input_ids"
    print("  ✓ output_ids follows input_ids")

    return True


def test_multi_turn():
    """测试多轮对话。"""
    print("\n" + "="*60)
    print("TEST: Multi Turn")
    print("="*60)

    tokenizer = get_tokenizer()
    builder = TrajectoryMaskBuilder(tokenizer)

    session_id = "test_multi"

    # ===== 第一轮 =====
    print("\n--- Turn 1 ---")
    messages1 = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ]

    input_ids1, matched1, _, _ = builder.prepare_input_ids(session_id, messages1)
    print(f"Turn 1: input_ids length = {len(input_ids1)}, matched = {matched1 is not None}")

    assistant1 = "Hi! How can I help you?"
    output_ids1, output_logprobs1 = mock_generate(tokenizer, assistant1)

    builder.save(session_id, messages1, input_ids1, output_ids1, output_logprobs1,
                 finish_reason="stop", matched_record=matched1)

    record1 = builder.query_latest(session_id)
    print(f"Turn 1: total tokens = {len(record1['tokens'])}")
    print_tokens_with_mask(tokenizer, record1["tokens"], record1["response_mask"])

    # ===== 第二轮 =====
    print("\n--- Turn 2 ---")
    messages2 = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": assistant1},
        {"role": "user", "content": "What is 2+2?"},
    ]

    input_ids2, matched2, matched_len2, _ = builder.prepare_input_ids(session_id, messages2)
    print(f"Turn 2: input_ids length = {len(input_ids2)}, matched = {matched2 is not None}")
    print(f"Turn 2: matched_prefix_len = {matched_len2}")

    # 验证复用了历史 tokens
    assert matched2 is not None, "Should match Turn 1 record"
    print("  ✓ Matched Turn 1 record")

    assistant2 = "The answer is 4."
    output_ids2, output_logprobs2 = mock_generate(tokenizer, assistant2)

    builder.save(session_id, messages2, input_ids2, output_ids2, output_logprobs2,
                 finish_reason="stop", matched_record=matched2)

    record2 = builder.query_latest(session_id)
    print(f"Turn 2: total tokens = {len(record2['tokens'])}")
    print_tokens_with_mask(tokenizer, record2["tokens"], record2["response_mask"])

    # 验证 Turn 1 的 tokens 是 Turn 2 的前缀
    assert record2["tokens"][:len(record1["tokens"])] == record1["tokens"], \
        "Turn 1 tokens should be prefix of Turn 2 tokens"
    print("  ✓ Turn 1 tokens is prefix of Turn 2 tokens")

    # 验证 mask 正确累加
    turn1_generated = sum(record1["response_mask"])
    turn2_generated = sum(record2["response_mask"])
    print(f"  Turn 1 generated: {turn1_generated}, Turn 2 generated: {turn2_generated}")
    assert turn2_generated > turn1_generated, "Turn 2 should have more generated tokens"
    print("  ✓ Generated tokens accumulated correctly")

    return True


def test_think_skip():
    """测试 <think> 跳过匹配。"""
    print("\n" + "="*60)
    print("TEST: <think> Skip Matching")
    print("="*60)

    tokenizer = get_tokenizer()
    builder = TrajectoryMaskBuilder(tokenizer)

    session_id = "test_think"

    # ===== 第一轮：生成包含 <think> 的内容 =====
    print("\n--- Turn 1: Generate with <think> ---")
    messages1 = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2+2?"},
    ]

    input_ids1, matched1, _, _ = builder.prepare_input_ids(session_id, messages1)

    # 生成包含 <think> 的回复
    assistant1_with_think = "<think>Let me calculate: 2+2=4</think>The answer is 4."
    output_ids1, output_logprobs1 = mock_generate(tokenizer, assistant1_with_think)

    builder.save(session_id, messages1, input_ids1, output_ids1, output_logprobs1,
                 finish_reason="stop", matched_record=matched1)

    record1 = builder.query_latest(session_id)
    print(f"Turn 1: total tokens = {len(record1['tokens'])}")
    print(f"Turn 1: messages_str preview: {record1['messages_str'][-100:]}")

    # ===== 第二轮：上层省略了 <think> 内容 =====
    print("\n--- Turn 2: Messages without <think> ---")
    # 上层省略了 <think> 块
    assistant1_without_think = "The answer is 4."
    messages2 = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": assistant1_without_think},  # 省略了 <think>
        {"role": "user", "content": "What about 3+3?"},
    ]

    input_ids2, matched2, matched_len2, _ = builder.prepare_input_ids(session_id, messages2)
    print(f"Turn 2: input_ids length = {len(input_ids2)}, matched = {matched2 is not None}")
    print(f"Turn 2: matched_prefix_len = {matched_len2}")

    # 验证仍然能匹配（<think> 跳过）
    if matched2 is not None:
        print("  ✓ Matched despite <think> being omitted")
    else:
        print("  ✗ Failed to match (check <think> skip logic)")
        return False

    assistant2 = "The answer is 6."
    output_ids2, output_logprobs2 = mock_generate(tokenizer, assistant2)

    builder.save(session_id, messages2, input_ids2, output_ids2, output_logprobs2,
                 finish_reason="stop", matched_record=matched2)

    record2 = builder.query_latest(session_id)
    print(f"Turn 2: total tokens = {len(record2['tokens'])}")
    print_tokens_with_mask(tokenizer, record2["tokens"], record2["response_mask"])

    return True


def test_consistency():
    """测试 input_ids 与保存的 tokens 的一致性。"""
    print("\n" + "="*60)
    print("TEST: Consistency Check")
    print("="*60)

    tokenizer = get_tokenizer()
    builder = TrajectoryMaskBuilder(tokenizer)

    session_id = "test_consistency"

    for turn in range(3):
        print(f"\n--- Turn {turn + 1} ---")

        # 构建 messages
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
        ]
        for i in range(turn + 1):
            messages.append({"role": "user", "content": f"Question {i+1}"})
            if i < turn:
                messages.append({"role": "assistant", "content": f"Answer {i+1}"})

        # prepare_input_ids
        input_ids, matched, _, _ = builder.prepare_input_ids(session_id, messages)
        print(f"  input_ids length: {len(input_ids)}")

        # mock generate
        output_ids, output_logprobs = mock_generate(tokenizer, f"Answer {turn + 1}")

        # save
        builder.save(session_id, messages, input_ids, output_ids, output_logprobs,
                     finish_reason="stop", matched_record=matched)

        # 验证一致性
        record = builder.query_latest(session_id)
        saved_tokens = record["tokens"]

        # input_ids 应该是 saved_tokens 的前缀（减去 output 部分）
        prefix_len = len(input_ids)
        assert saved_tokens[:prefix_len] == input_ids, f"Turn {turn+1}: input_ids mismatch"
        print(f"  ✓ input_ids matches saved tokens prefix")

        # output_ids 应该紧跟在 input_ids 后面
        output_start = prefix_len
        output_end = output_start + len(output_ids)
        assert saved_tokens[output_start:output_end] == output_ids, f"Turn {turn+1}: output_ids mismatch"
        print(f"  ✓ output_ids matches saved tokens")

    return True


def main():
    print("="*60)
    print("TrajectoryMaskBuilder Tests")
    print("="*60)

    tests = [
        ("Single Turn", test_single_turn),
        ("Multi Turn", test_multi_turn),
        ("<think> Skip", test_think_skip),
        ("Consistency", test_consistency),
    ]

    results = []
    for name, test_fn in tests:
        try:
            passed = test_fn()
            results.append((name, passed))
        except Exception as e:
            import traceback
            print(f"\n[ERROR] {name}: {e}")
            traceback.print_exc()
            results.append((name, False))

    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {name}: {status}")

    all_passed = all(p for _, p in results)
    print(f"\nOverall: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
