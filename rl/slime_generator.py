import asyncio
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List
import copy

# Add rl directory to path for utils import
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from utils import get_env, AggType, MetricsRecorder

import aiohttp
import requests
from transformers import AutoTokenizer
from transformers import AutoProcessor, PreTrainedTokenizerBase, ProcessorMixin

from slime.utils.async_utils import run
from slime.utils.types import Sample

from utils.multimodal import (
    extract_image_urls_from_messages,
    has_image_in_messages,
    load_pil_images,
    normalize_messages,
    split_messages_prompt_and_assistant,
)

__all__ = ["generate_rollout"]

logger = logging.getLogger(__name__)

# Global variables for evaluation
TOKENIZER = None
PROCESSOR = None
START_ROLLOUT = True

# LLM Proxy URL for getting trajectory masks (constructed from host and port)
_llm_proxy_host = get_env("LLM_PROXY_HOST")
_llm_proxy_port = get_env("LLM_PROXY_PORT")
LLM_PROXY_URL = f"http://{_llm_proxy_host}:{_llm_proxy_port}"


def decode_tokens_with_mask_debug(tokenizer, token_ids, loss_mask):
    """
    token_ids: 全部 tokens
    loss_mask: 从第一个 1 开始的 mask（已去掉尾部 0）

    返回 [(text, 0/1), ...] 合并连续相同 mask 的段
    """
    if not token_ids:
        return []

    # 前面没有 mask 覆盖的部分，mask 都是 0
    prefix_len = len(token_ids) - len(loss_mask)
    full_mask = [0] * prefix_len + loss_mask

    # 合并连续相同 mask 的段
    segments = []
    current_tokens = [token_ids[0]]
    current_mask = full_mask[0]

    for tid, mask in zip(token_ids[1:], full_mask[1:]):
        if mask == current_mask:
            current_tokens.append(tid)
        else:
            segments.append((tokenizer.decode(current_tokens), current_mask))
            current_tokens = [tid]
            current_mask = mask

    if current_tokens:
        segments.append((tokenizer.decode(current_tokens), current_mask))

    return segments


def write_debug_to_file(
    tokenizer,
    rollout_id: int,
    record: Dict,
    oai_messages: List[Dict],
    token_ids: List[int],
    loss_mask: List[int],
    response_length: int,
):
    """将训练数据的调试信息写入文件。"""
    debug_dir = os.path.join(get_env("AIEVOBOX_ROOT"), "logs")
    os.makedirs(debug_dir, exist_ok=True)

    debug_segments = decode_tokens_with_mask_debug(tokenizer, token_ids, loss_mask)

    # 解析 messages 中的 JSON content
    oai_messages_parsed = copy.deepcopy(oai_messages)
    for msg in oai_messages_parsed:
        if isinstance(msg.get("content"), str):
            try:
                msg["content"] = json.loads(msg["content"])
            except json.JSONDecodeError:
                pass

    debug_file = os.path.join(debug_dir, f"train_{rollout_id}.log")
    with open(debug_file, "a+", encoding="utf-8") as f:
        f.write(json.dumps({
            "messages": oai_messages_parsed,
            "debug_segments": debug_segments,
            "index": record["instance_id"],
            "prompt": record["uid"],
            "tokens": token_ids,
            "response_length": response_length,
            "reward": record["reward"],
            "status": (
                "completed"
                if "finish_reason" not in record["extra_info"]
                or record["extra_info"]["finish_reason"] != "length"
                else "truncated"
            ),
            "loss_mask": loss_mask,
            "metadata": record["extra_info"],
        }, ensure_ascii=False) + "\n")

def query_trajectory(
    session_id: str,
    messages_str: str,
    max_retries: int = 10,
    timeout: int = 60,
) -> tuple[List[int], List[int]]:
    """Query tokens and response_mask from LLM Proxy."""
    url = f"{LLM_PROXY_URL}/get_trajectory_mask"
    payload = {"session_id": session_id, "messages_str": messages_str}

    last_error = None
    for attempt in range(max_retries):
        try:
            resp = requests.post(url, json=payload, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            last_error = e
            delay = min(2 ** attempt, 30)
            logger.warning(f"[query_trajectory] attempt {attempt+1}/{max_retries} failed: {e}, retry in {delay}s")
            time.sleep(delay)
            continue

        tokens = data.get("tokens")
        response_mask = data.get("response_mask")
        if isinstance(tokens, list) and isinstance(response_mask, list):
            return [int(t) for t in tokens], [int(m) for m in response_mask]

        raise KeyError("Missing `tokens`/`response_mask` in response")

    raise RuntimeError(f"[query_trajectory] All {max_retries} retries exhausted. Last error: {last_error}")


def build_loss_mask_from_response_mask(
    tokens: List[int],
    response_mask: List[int],
) -> tuple[List[int], List[int], int]:
    """Convert full-length token-level response_mask to (token_ids, loss_mask, response_length).

    Returns:
        - token_ids: 完整的 tokens（未截取）
        - loss_mask: 从第一个 mask=1 的位置开始截取到末尾
        - response_length: loss_mask 的长度

    注意：token_ids 和 loss_mask 的长度不同，loss_mask 对应 token_ids 的后半部分。
    """
    # 边界情况：空输入
    if not tokens or not response_mask:
        return list(tokens or []), [], 0

    # 验证长度一致性
    if len(tokens) != len(response_mask):
        raise ValueError(f"Tokens and response_mask have different lengths: {len(tokens)} != {len(response_mask)}")

    # 找到第一个生成 token 的位置（mask=1）
    try:
        first_generated_idx = response_mask.index(1)
    except ValueError:
        # 没有生成内容（全是 mask=0）
        return list(tokens), [], 0

    # 从第一个生成 token 开始截取 loss_mask
    loss_mask = response_mask[first_generated_idx:]
    response_length = len(loss_mask)

    return tokens, loss_mask, response_length


def group_by_instance_id(results: List[Dict]) -> List[List[Dict]]:
    """按 instance_id 将样本分组。

    Args:
        results: 样本列表，每个样本必须包含 instance_id

    Returns:
        分组后的样本列表 List[List[Dict]]
    """
    if not results:
        return []

    groups = {}
    for item in results:
        instance_id = item.get("instance_id")
        if instance_id is None:
            raise ValueError("instance_id must be in item")
        if instance_id not in groups:
            groups[instance_id] = []
        groups[instance_id].append(item)

    return list(groups.values())


async def get_rollout_data(api_base_url: str) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    start_time = time.time()
    async with aiohttp.ClientSession() as session:
        while True:
            async with session.post(
                f"{api_base_url}/get_rollout_data", json={}, timeout=aiohttp.ClientTimeout(total=120)
            ) as response:
                response.raise_for_status()
                resp_json = await response.json()
                if resp_json["success"]:
                    break
            await asyncio.sleep(3)
            if time.time() - start_time > 30:
                print("rollout data is not ready, have been waiting for 30 seconds")
                # Reset start_time to continue waiting or handle timeout differently
                start_time = time.time()  # Or raise an exception, or return empty list

        data = resp_json["data"]
        meta_info = {}
        if isinstance(data, list):
            if "data" in data[0]:
                data = [item["data"] for item in data]
        elif isinstance(data, dict):
            if "data" in data:
                meta_info = data["meta_info"]
                data = data["data"]
        print(f"Meta info: {meta_info}")
        required_keys = {"uid", "instance_id", "messages", "reward", "extra_info"}
        for item in data:
            if not required_keys.issubset(item.keys()):
                raise ValueError(f"Missing required keys in response item: {item}")

        return data, meta_info


def start_rollout(api_base_url: str, args, metadata):
    url = f"{api_base_url}/start_rollout"
    print(f"metadata: {metadata}")
    finished_groups_instance_id_list = [item for sublist in metadata.values() for item in sublist]
    restart_training = os.environ.get("SLIME_ROLLBUF_RESTART_TRAINING", "True").strip().lower() == "true"
    payload = {
        "num_process": str(getattr(args, "rollout_num_process", 100)),
        "num_epoch": str(args.num_epoch or 3),
        "remote_engine_url": f"http://{args.sglang_router_ip}:{args.sglang_router_port}",
        "remote_buffer_url": args.rollout_buffer_url,
        "task_type": args.rollout_task_type,
        "input_file": args.prompt_data,
        "num_repeat_per_sample": int(args.n_samples_per_prompt),
        "max_tokens": int(args.rollout_max_response_len),
        "sampling_params": {
            "max_tokens": int(args.rollout_max_response_len),
            "temperature": args.rollout_temperature,
            "top_p": args.rollout_top_p,
        },
        "tokenizer_path": args.hf_checkpoint,
        "skip_instance_ids": finished_groups_instance_id_list,
        "restart_training": restart_training,
    }
    print("start rollout with payload: ", payload)

    while True:
        try:
            resp = requests.post(url, json=payload, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            print(f"[start_rollout] Success: {data}")
            return data
        except Exception as e:
            print(f"[start_rollout] Failed to send rollout config: {e}")


def filter_by_weight_version(data_buffer, current_version: int, off_by_n: int = 0):
    """根据权重版本过滤 buffer 中的数据。

    过滤掉那些权重版本与当前版本差距超过 off_by_n 的样本。

    Args:
        data_buffer: 数据 buffer
        current_version: 当前权重版本（通常是 rollout_id）
        off_by_n: 允许的最大权重差，默认为 0（只保留当前版本的数据）
    """
    buffer_length = data_buffer.get_buffer_length()
    if buffer_length == 0:
        return

    # 获取所有样本
    all_samples = data_buffer.get_samples(buffer_length)

    # 过滤样本
    filtered_samples = []
    for sample_group in all_samples:
        filtered_group = []
        len_sample_group = len(sample_group)
        for sample in sample_group:
            metadata = getattr(sample, "metadata", None) or {}
            sample_version = metadata.get("weight_version", 0)
            try:
                sample_version = int(sample_version)
            except (ValueError, TypeError):
                sample_version = 0

            # 检查权重版本差距是否在允许范围内
            if current_version - sample_version <= off_by_n:
                filtered_group.append(sample)
            else:
                logger.debug(
                    f"Filtered out sample with weight_version={sample_version}, "
                    f"current_version={current_version}, off_by_n={off_by_n}"
                )

        if filtered_group and len(filtered_group) == len_sample_group:
            filtered_samples.append(filtered_group)

    if filtered_samples:
        data_buffer.add_samples(filtered_samples)

async def generate_rollout_async(args, rollout_id: int, data_buffer, evaluation: bool = False) -> Dict[str, Any]:
    if evaluation:
        raise NotImplementedError("Evaluation rollout is not implemented")

    metrics = MetricsRecorder()
    print("rollout_id: ", rollout_id)

    # 根据weight_version过滤已完成的数据
    off_by_n = int(get_env("RL_OFF_BY_N"))
    filter_by_weight_version(data_buffer, current_version=rollout_id, off_by_n=off_by_n)
    data_number_to_fetch = (args.rollout_batch_size - data_buffer.get_buffer_length()) * args.n_samples_per_prompt
    print(f"INFO: buffer length: {data_buffer.get_buffer_length()}, data_number_to_fetch: {data_number_to_fetch}")
    if data_number_to_fetch <= 0:
        print(
            f"❕buffer length: {data_buffer.get_buffer_length()}, buffer has enough data, return {args.rollout_batch_size} prompts"
        )
        final_return_results = data_buffer.get_samples(args.rollout_batch_size)
        # Record used metrics
        for group in final_return_results:
            for sample in group:
                metrics.record("used/reward", float(sample.reward), AggType.MEAN)
                metrics.record("used/response_length", float(sample.response_length), AggType.MEAN)
                meta = getattr(sample, "metadata", {}) or {}
                metrics.record("used/weight_version", float(meta.get("weight_version", 0)), AggType.MEAN)
        metrics.record("used/count", float(sum(len(g) for g in final_return_results)), AggType.SUM)
        metrics.push(step=rollout_id)
        return final_return_results
    base_url = args.rollout_buffer_url
    tokenizer = TOKENIZER
    retry_times = 0
    all_meta_info = []

    # 需要的 group 数量
    need_groups = data_number_to_fetch // args.n_samples_per_prompt
    valid_groups = []

    if args.fetch_trajectory_retry_times == -1:
        print(
            f"⚠️  [get_rollout_data] Fetch trajectory retry times set to -1, will retry indefinitely until sufficient data is collected"
        )

    # 持续获取数据，直到有足够的符合版本要求的 groups
    while len(valid_groups) < need_groups and (args.fetch_trajectory_retry_times == -1 or retry_times < args.fetch_trajectory_retry_times):
        try:
            # 按实际需要的数量获取（还差多少就获取多少）
            remaining_groups = need_groups - len(valid_groups)
            fetch_sample_count = remaining_groups * args.n_samples_per_prompt
            print(f"need sample count: fetch_sample_count: {fetch_sample_count}")
            raw_results = []

            while len(raw_results) < fetch_sample_count:
                await asyncio.sleep(5)
                data, meta_info = await get_rollout_data(api_base_url=base_url)
                raw_results.extend(data)
                if meta_info:
                    all_meta_info.append(meta_info)
                print(f"get rollout data with length: {len(raw_results)}")

            # 从 extra_info 中获取 weight_version，记录 fetched metrics
            for record in raw_results:
                extra_info = record.get("extra_info") or {}
                record["weight_version"] = extra_info.get("weight_version", 0)
                metrics.record("fetched/reward", float(record.get("reward", 0)), AggType.MEAN)
                metrics.record("fetched/weight_version", float(record["weight_version"]), AggType.MEAN)
            metrics.record("fetched/count", float(len(raw_results)), AggType.SUM)

            # 按 instance_id 分组
            grouped_results = group_by_instance_id(raw_results)

            # 按 group 过滤：group 中所有 sample 都必须符合版本要求
            for group in grouped_results:
                rewards = [record.get("reward") for record in group]
                if len(set(rewards)) == 1:
                    logger.info(
                        f"Filtered out group with rewards={rewards}, "
                        f"current_version={rollout_id}"
                    )
                    continue
                if all(rollout_id - record.get("weight_version", 0) <= off_by_n for record in group):
                    valid_groups.append(group)
                else:
                    # 记录被过滤的 group 信息
                    versions = [record.get("weight_version", 0) for record in group]
                    logger.info(
                        f"Filtered out group with weight_versions={versions}, "
                        f"current_version={rollout_id}, off_by_n={off_by_n}"
                    )

            print(f"✅ Valid groups collected: {len(valid_groups)}/{need_groups}")

            # 如果已经有足够的 valid groups，退出循环
            if len(valid_groups) >= need_groups:
                break

        except Exception as err:
            print(f"[get_rollout_data] Failed to get rollout data: {err}, retry times: {retry_times}")
            retry_times += 1

    # 使用所有符合版本要求的 valid_groups
    results = valid_groups

    if len(all_meta_info) > 0 and "finished_groups" in all_meta_info[0]:
        finished_groups_instance_id_list = []
        for item in all_meta_info:
            finished_groups_instance_id_list.extend(item["finished_groups"])

        data_buffer.update_metadata({str(rollout_id): finished_groups_instance_id_list})

    print("finally get rollout data with length: ", len(results))
    sample_results = []

    for group_record in results:
        group_results = []
        for record in group_record:
            oai_messages = normalize_messages(record["messages"])
            session_id = record["extra_info"].get("session_id", "")

            # Convert messages to string (query key for llm_proxy trajectory lookup).
            # For VLM we must use processor-expanded string; otherwise different images can collide.
            messages_str = tokenizer.apply_chat_template(
                oai_messages,
                add_generation_prompt=False,
                tokenize=False
            )
            global PROCESSOR
            if PROCESSOR is not None and has_image_in_messages(oai_messages):
                prompt_text = tokenizer.apply_chat_template(
                    oai_messages,
                    add_generation_prompt=False,
                    tokenize=False,
                )
                image_refs = extract_image_urls_from_messages(oai_messages)
                images = load_pil_images(image_refs) if image_refs else None
                proc_out = PROCESSOR(
                    text=prompt_text,
                    images=images,
                    padding=True,
                    return_tensors="pt",
                )
                messages_str = tokenizer.decode(
                    proc_out["input_ids"][0].tolist(),
                    skip_special_tokens=False,
                )

            # Get tokens + response_mask from LLM Proxy
            tokens, response_mask = query_trajectory(session_id, messages_str)
            token_ids, loss_mask, response_length = build_loss_mask_from_response_mask(tokens, response_mask)

            # 写入调试文件
            write_debug_to_file(tokenizer, rollout_id, record, oai_messages, token_ids, loss_mask, response_length)

            # 构建 metadata（从 extra_info 复制，已包含 weight_version）
            metadata = dict(record["extra_info"])
            sample = Sample(
                index=record["instance_id"],
                prompt=record["uid"],
                tokens=token_ids,
                response_length=response_length,
                reward=record["reward"],
                status=(
                    Sample.Status.COMPLETED
                    if "finish_reason" not in record["extra_info"]
                    or record["extra_info"]["finish_reason"] != "length"
                    else Sample.Status.TRUNCATED
                ),
                loss_mask=loss_mask,
                metadata=metadata,
            )

            # VLM training: attach processed multimodal tensors if there are images in the prompt
            # (prompt = messages without the final assistant response).
            prompt_messages, _ = split_messages_prompt_and_assistant(oai_messages)
            if PROCESSOR is not None and has_image_in_messages(prompt_messages):
                prompt_text = tokenizer.apply_chat_template(
                    prompt_messages,
                    add_generation_prompt=True,
                    tokenize=False,
                )
                image_refs = extract_image_urls_from_messages(prompt_messages)
                images = load_pil_images(image_refs) if image_refs else None
                proc_out = PROCESSOR(
                    text=prompt_text,
                    images=images,
                    padding=True,
                    return_tensors="pt",
                )
                mm_train_inputs = {
                    k: v for k, v in proc_out.items() if k not in ["input_ids", "attention_mask"]
                } or None
                sample.multimodal_train_inputs = mm_train_inputs

            group_results.append(sample)
        sample_results.append(group_results)

    data_buffer.add_samples(sample_results)
    final_return_results = data_buffer.get_samples(args.rollout_batch_size)

    # Record used metrics
    for group in final_return_results:
        for sample in group:
            metrics.record("used/reward", float(sample.reward), AggType.MEAN)
            metrics.record("used/response_length", float(sample.response_length), AggType.MEAN)
            meta = getattr(sample, "metadata", {}) or {}
            metrics.record("used/weight_version", float(meta.get("weight_version", 0)), AggType.MEAN)
    metrics.record("used/count", float(sum(len(g) for g in final_return_results)), AggType.SUM)
    metrics.push(step=rollout_id)

    return final_return_results


def generate_rollout(args, rollout_id, data_buffer, evaluation=False):
    """Generate rollout for both training and evaluation."""
    global START_ROLLOUT, TOKENIZER
    if START_ROLLOUT:
        metadata = data_buffer.get_metadata()
        start_inform = start_rollout(args.rollout_buffer_url, args, metadata)
        print(f"start rollout with payload: {start_inform}")
        print(f"start rollout id: {rollout_id}")
        START_ROLLOUT = False

    if TOKENIZER is None:
        TOKENIZER = AutoTokenizer.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
    global PROCESSOR
    if PROCESSOR is None:
        try:
            proc = AutoProcessor.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
            if isinstance(proc, PreTrainedTokenizerBase) or not isinstance(proc, ProcessorMixin):
                proc = None
            PROCESSOR = proc
        except Exception:
            PROCESSOR = None

    return run(generate_rollout_async(args, rollout_id, data_buffer, evaluation))
