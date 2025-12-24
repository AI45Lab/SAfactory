import asyncio
import json
import logging
import os
import time
from typing import Any, Dict, List

import aiohttp
import requests
import wandb
from transformers import AutoTokenizer

from slime.utils.async_utils import run
from slime.utils.mask_utils import MultiTurnLossMaskGenerator
from slime.utils.types import Sample

__all__ = ["generate_rollout"]

logger = logging.getLogger(__name__)

# Global variables for evaluation
TOKENIZER = None
START_ROLLOUT = True

# LLM Proxy URL for getting trajectory masks
LLM_PROXY_URL = os.environ.get("LLM_PROXY_URL", "http://127.0.0.1:8890")


def get_mask_ranges_from_server(
    session_id: str,
    messages_str: str,
    max_retries: int = 10,
    timeout: int = 60,
) -> List[tuple]:
    """Get character-level mask_ranges from LLM Proxy.

    Args:
        session_id: Session ID
        messages_str: Full conversation string
        max_retries: Max retry attempts
        timeout: Request timeout in seconds

    Returns:
        mask_ranges: List of (start, end) tuples where mask=1
    """
    url = f"{LLM_PROXY_URL}/get_trajectory_mask"
    payload = {"session_id": session_id, "messages_str": messages_str}

    last_error = None
    for attempt in range(max_retries):
        try:
            resp = requests.post(url, json=payload, timeout=timeout)
            resp.raise_for_status()
            return resp.json()["mask_ranges"]
        except Exception as e:
            last_error = e
            delay = min(2 ** attempt, 30)
            logger.warning(f"[get_mask_ranges] attempt {attempt+1}/{max_retries} failed: {e}, retry in {delay}s")
            time.sleep(delay)

    raise RuntimeError(f"[get_mask_ranges] All {max_retries} retries exhausted. Last error: {last_error}")


def tokenize_with_char_mask_ranges(tokenizer, messages_str: str, mask_ranges: List[tuple]) -> tuple:
    """Tokenize with character-level mask_ranges.

    Args:
        tokenizer: Tokenizer instance
        messages_str: Full conversation string
        mask_ranges: List of (start, end) tuples where mask=1

    Returns:
        token_ids: Full token sequence
        loss_mask: Mask from first 1 to end (length = response_length)
        response_length: Token count from first 1 to end
    """
    if not mask_ranges:
        token_ids = tokenizer.encode(messages_str, add_special_tokens=False)
        return token_ids, [], 0

    # Split string by mask_ranges
    segments = []
    prev_end = 0
    for start, end in sorted(mask_ranges):
        if start > prev_end:
            segments.append((messages_str[prev_end:start], False))
        segments.append((messages_str[start:end], True))
        prev_end = end
    if prev_end < len(messages_str):
        segments.append((messages_str[prev_end:], False))

    # Tokenize segments and merge
    all_token_ids = []
    all_loss_mask = []

    for text, is_masked in segments:
        if not text:
            continue
        segment_tokens = tokenizer.encode(text, add_special_tokens=False)
        all_token_ids.extend(segment_tokens)
        if is_masked:
            all_loss_mask.extend([1] * len(segment_tokens))
        else:
            all_loss_mask.extend([0] * len(segment_tokens))

    # response_length = from first 1 to end
    if 1 in all_loss_mask:
        first_one_idx = all_loss_mask.index(1)
        response_length = len(all_loss_mask) - first_one_idx
        truncated_loss_mask = all_loss_mask[-response_length:]
    else:
        response_length = 0
        truncated_loss_mask = []

    return all_token_ids, truncated_loss_mask, response_length


_MASK_SEGMENTS_LOG_PATH: str | None = None


def _get_mask_segments_log_path() -> str:
    global _MASK_SEGMENTS_LOG_PATH
    if _MASK_SEGMENTS_LOG_PATH:
        return _MASK_SEGMENTS_LOG_PATH

    env_path = os.environ.get("MASK_SEGMENTS_LOG_PATH")
    if env_path:
        _MASK_SEGMENTS_LOG_PATH = env_path
        log_dir = os.path.dirname(env_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        return _MASK_SEGMENTS_LOG_PATH

    log_dir = os.environ.get("MASK_SEGMENTS_LOG_DIR")
    if not log_dir:
        aievobox_root = os.environ.get("AIEVOBOX_ROOT", "/root/AIEvoBox")
        log_dir = os.path.join(aievobox_root, "logs")
    os.makedirs(log_dir, exist_ok=True)

    _MASK_SEGMENTS_LOG_PATH = os.path.join(log_dir, f"mask_segments_{os.getpid()}.jsonl")
    return _MASK_SEGMENTS_LOG_PATH


def _sanitize_and_merge_mask_ranges(mask_ranges: List[tuple], text_len: int) -> List[tuple[int, int]]:
    cleaned: List[tuple[int, int]] = []
    for r in mask_ranges or []:
        try:
            start = int(r[0])
            end = int(r[1])
        except Exception:
            continue
        if end <= start:
            continue
        if start < 0:
            start = 0
        if end < 0:
            continue
        if start > text_len:
            continue
        if end > text_len:
            end = text_len
        if end <= start:
            continue
        cleaned.append((start, end))

    if not cleaned:
        return []

    cleaned.sort(key=lambda x: (x[0], x[1]))
    merged: List[list[int]] = []
    for start, end in cleaned:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(s, e) for s, e in merged]


def split_messages_str_by_mask_ranges(messages_str: str, mask_ranges: List[tuple]) -> List[tuple[str, int]]:
    """Split messages_str into labeled segments according to mask_ranges."""
    text_len = len(messages_str)
    merged_ranges = _sanitize_and_merge_mask_ranges(mask_ranges, text_len=text_len)
    if not merged_ranges:
        return [(messages_str, 0)]

    segments: List[tuple[str, int]] = []
    prev_end = 0
    for start, end in merged_ranges:
        if start > prev_end:
            segments.append((messages_str[prev_end:start], 0))
        segments.append((messages_str[start:end], 1))
        prev_end = end
    if prev_end < text_len:
        segments.append((messages_str[prev_end:], 0))
    return segments


def log_mask_segments_to_file(
    *,
    session_id: str,
    messages_str: str,
    mask_ranges: List[tuple],
    instance_id: str | None = None,
    uid: str | None = None,
) -> None:
    """Append mask split info to a local jsonl file (not wandb)."""
    try:
        path = _get_mask_segments_log_path()
        segments = split_messages_str_by_mask_ranges(messages_str, mask_ranges)
        record = {
            "ts": time.time(),
            "session_id": session_id,
            "instance_id": instance_id,
            "uid": uid,
            "segments": [(s, m) for s, m in segments],
        }
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"Failed to log mask segments: {e}")


def select_rollout_data(args, results, need_length):
    """
    Select the most recent groups when there are too many samples.
    Groups all samples by instance_id, sorts groups by timestamp.

    Args:
        args: Arguments containing configuration
        results: List of rollout data items with timestamps

    Returns:
        Selected samples from the newest groups based on timestamp cutoff
    """
    if not results:
        return results

    # Group samples by instance_id
    groups = {}
    for item in results:
        assert "instance_id" in item, "instance_id must be in item"
        instance_id = item["instance_id"]
        if instance_id not in groups:
            groups[instance_id] = []
        groups[instance_id].append(item)

    print(f"📊 Total groups: {len(groups)}, total samples: {len(results)}")

    # If we don't have too many samples, return all
    assert need_length < len(results), "need_length must be smaller than results length"

    # Get timestamp for each group (use the latest timestamp in the group)
    def get_group_timestamp(group_items):
        timestamps = []
        for item in group_items:
            if "timestamp" in item:
                timestamps.append(float(item["timestamp"]))
            elif "extra_info" in item and "timestamp" in item["extra_info"]:
                timestamps.append(float(item["extra_info"]["timestamp"]))
        return max(timestamps) if timestamps else 0

    # Create list of (group_id, timestamp, samples) and sort by timestamp
    group_data = []
    for group_id, group_items in groups.items():
        group_timestamp = get_group_timestamp(group_items)
        group_data.append((group_id, group_timestamp, group_items))

    # Sort groups by timestamp (newest first)
    group_data.sort(key=lambda x: x[1], reverse=True)

    selected_groups = group_data[:need_length]

    # Flatten selected groups back to sample list
    selected_results = []
    for group_id, timestamp, group_items in selected_groups:
        selected_results.append(group_items)

    # Statistics for monitoring
    if selected_groups:
        newest_ts = selected_groups[0][1]
        oldest_ts = selected_groups[-1][1]
        print(
            f"📈 Selected {len(selected_groups)} groups with {len(selected_results)*args.n_samples_per_prompt} samples"
        )
        print(f"📈 Group timestamp range: {oldest_ts:.2f} to {newest_ts:.2f}")
        print(f"📈 Time span: {newest_ts - oldest_ts:.2f} seconds")

    return selected_results


def log_raw_info(args, all_meta_info, rollout_id):
    final_meta_info = {}
    if all_meta_info:
        final_meta_info = {
            "total_samples": sum(meta["total_samples"] for meta in all_meta_info if "total_samples" in meta)
        }

        total_samples = final_meta_info["total_samples"]
        if total_samples > 0:
            weighted_reward_sum = sum(
                meta["avg_reward"] * meta["total_samples"]
                for meta in all_meta_info
                if "avg_reward" in meta and "total_samples" in meta
            )

            final_meta_info["avg_reward"] = weighted_reward_sum / total_samples

            if hasattr(args, "use_wandb") and args.use_wandb:
                log_dict = {
                    "rollout/total_samples": final_meta_info["total_samples"],
                    "rollout/avg_reward": final_meta_info["avg_reward"],
                }
                try:
                    # 当前版本：使用 rollout 对应的“train step”近似
                    step = (
                        rollout_id
                        if not args.wandb_always_use_train_step
                        else rollout_id * args.rollout_batch_size * args.n_samples_per_prompt // args.global_batch_size
                    )

                    # 计算被使用数据的平均 age（step - 生成时版本）
                    # Buffer Server 在 meta_info 中提供 avg_weight_version（按样本数加权的权重版本）
                    weighted_version_sum = sum(
                        meta.get("avg_weight_version", 0.0) * meta["total_samples"]
                        for meta in all_meta_info
                        if "total_samples" in meta
                    )
                    if total_samples > 0:
                        avg_weight_version = weighted_version_sum / total_samples
                        # 好像第0个step是1，第一个step还是1
                        avg_data_age = max(step - avg_weight_version, 0)
                        final_meta_info["avg_weight_version"] = avg_weight_version
                        final_meta_info["avg_data_age"] = avg_data_age
                        log_dict["rollout/avg_data_age"] = avg_data_age

                    if args.use_wandb:
                        log_dict["rollout/step"] = step
                        wandb.log(log_dict, step=step)

                    if args.use_tensorboard:
                        from slime.utils.tensorboard_utils import _TensorboardAdapter

                        tb = _TensorboardAdapter(args)
                        tb.log(data=log_dict, step=step)
                    print(f"no filter rollout log {rollout_id}: {log_dict}")
                except Exception as e:
                    print(f"Failed to log to wandb: {e}")
                    print(f"no filter rollout log {rollout_id}: {final_meta_info}")
            else:
                print(f"no filter rollout log {rollout_id}: {final_meta_info}")


def _compute_log_step(args, rollout_id: int) -> int:
    """Compute the logging step used for wandb/tensorboard."""
    try:
        always_use_train_step = bool(getattr(args, "wandb_always_use_train_step", False))
        if not always_use_train_step:
            return int(rollout_id)
        rollout_batch_size = int(getattr(args, "rollout_batch_size"))
        n_samples_per_prompt = int(getattr(args, "n_samples_per_prompt"))
        global_batch_size = int(getattr(args, "global_batch_size"))
        return int(rollout_id * rollout_batch_size * n_samples_per_prompt // global_batch_size)
    except Exception:
        return int(rollout_id)


_WANDB_STEP_METRICS_DEFINED: set[str] = set()


def _maybe_define_wandb_step_metric(prefix: str) -> None:
    """Configure W&B so `{prefix}/*` uses `{prefix}/step` as the default x-axis."""
    if prefix in _WANDB_STEP_METRICS_DEFINED:
        return
    try:
        # Only define metrics after wandb.init(); otherwise this may error.
        if getattr(wandb, "run", None) is None:
            return
        wandb.define_metric(f"{prefix}/step")
        wandb.define_metric(f"{prefix}/*", step_metric=f"{prefix}/step")
        _WANDB_STEP_METRICS_DEFINED.add(prefix)
    except Exception as e:
        print(f"Failed to define wandb step metric for {prefix}: {e}")


def log_used_batch_info(args, batch_samples, rollout_id: int, prefix: str = "rollout_used") -> None:
    """Log metrics computed from the actual batch returned to the trainer."""
    try:
        step = _compute_log_step(args, rollout_id)
        use_wandb = bool(getattr(args, "use_wandb", False))
        use_tensorboard = bool(getattr(args, "use_tensorboard", False))

        # batch_samples is typically List[List[Sample]] (grouped by prompt)
        flat_samples = []
        if isinstance(batch_samples, list):
            for group in batch_samples:
                if isinstance(group, list):
                    flat_samples.extend(group)
                else:
                    flat_samples.append(group)

        num_prompts = len(batch_samples) if isinstance(batch_samples, list) else 0
        total_samples = len(flat_samples)
        if total_samples <= 0:
            return

        norm_rewards = []
        raw_rewards = []
        response_lengths = []
        weight_versions = []
        for sample in flat_samples:
            try:
                r = getattr(sample, "reward", None)
                if r is not None:
                    norm_rewards.append(float(r))
            except Exception:
                pass
            try:
                rl = getattr(sample, "response_length", None)
                if rl is not None:
                    response_lengths.append(float(rl))
            except Exception:
                pass
            try:
                meta = getattr(sample, "metadata", None)
                if isinstance(meta, dict):
                    rr = meta.get("raw_reward", None)
                    if rr is not None:
                        raw_rewards.append(float(rr))
                    wv = meta.get("weight_version", None)
                    if wv is not None:
                        weight_versions.append(float(wv))
            except Exception:
                pass

        log_dict = {
            f"{prefix}/step": step,
            f"{prefix}/num_prompts": num_prompts,
            f"{prefix}/total_samples": total_samples,
        }
        if raw_rewards:
            log_dict[f"{prefix}/avg_reward"] = sum(raw_rewards) / len(raw_rewards)
        elif norm_rewards:
            # Fallback when raw_reward is not available.
            log_dict[f"{prefix}/avg_reward"] = sum(norm_rewards) / len(norm_rewards)
        if norm_rewards:
            log_dict[f"{prefix}/avg_reward_norm"] = sum(norm_rewards) / len(norm_rewards)
        if response_lengths:
            log_dict[f"{prefix}/avg_response_length"] = sum(response_lengths) / len(response_lengths)
        if weight_versions:
            avg_weight_version = sum(weight_versions) / len(weight_versions)
            log_dict[f"{prefix}/avg_weight_version"] = avg_weight_version
            log_dict[f"{prefix}/avg_data_age"] = (step + 1) - avg_weight_version

        if use_wandb:
            _maybe_define_wandb_step_metric(prefix)
            wandb.log(log_dict, step=step)
        if use_tensorboard:
            from slime.utils.tensorboard_utils import _TensorboardAdapter

            tb = _TensorboardAdapter(args)
            tb.log(data=log_dict, step=step)
        print(f"used-batch rollout log {rollout_id}: {log_dict}")
    except Exception as e:
        print(f"Failed to log used-batch metrics: {e}")


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
    restart_training = os.environ.get("ROLLBUF_RESTART_TRAINING", "True").strip().lower() == "true"
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


async def generate_rollout_async(args, rollout_id: int, data_buffer, evaluation: bool = False) -> Dict[str, Any]:

    global START_ROLLOUT
    if evaluation:
        raise NotImplementedError("Evaluation rollout is not implemented")

    if START_ROLLOUT:
        metadata = data_buffer.get_metadata()
        start_inform = start_rollout(args.rollout_buffer_url, args, metadata)
        print(f"start rollout with payload: {start_inform}")
        print(f"start rollout id: {rollout_id}")
        START_ROLLOUT = False

    data_number_to_fetch = args.rollout_batch_size * args.n_samples_per_prompt - data_buffer.get_buffer_length()
    if data_number_to_fetch <= 0:
        print(
            f"❕buffer length: {data_buffer.get_buffer_length()}, buffer has enough data, return {args.rollout_batch_size} prompts"
        )
        final_return_results = data_buffer.get_samples(args.rollout_batch_size)
        log_used_batch_info(args, final_return_results, rollout_id)
        return final_return_results
    assert (
        data_number_to_fetch % args.n_samples_per_prompt == 0
    ), "data_number_to_fetch must be a multiple of n_samples_per_prompt"
    print(f"INFO: buffer length: {data_buffer.get_buffer_length()}, data_number_to_fetch: {data_number_to_fetch}")
    base_url = args.rollout_buffer_url
    tokenizer = AutoTokenizer.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
    retry_times = 0
    results = []
    all_meta_info = []

    if args.fetch_trajectory_retry_times == -1:
        print(
            f"⚠️  [get_rollout_data] Fetch trajectory retry times set to -1, will retry indefinitely until sufficient data is collected"
        )
    while args.fetch_trajectory_retry_times == -1 or retry_times < args.fetch_trajectory_retry_times:
        try:
            while len(results) < data_number_to_fetch:
                time.sleep(5)
                data, meta_info = await get_rollout_data(api_base_url=base_url)
                results.extend(data)
                if meta_info:
                    all_meta_info.append(meta_info)
                print(f"get rollout data with length: {len(results)}")
            break
        except Exception as err:
            print(f"[get_rollout_data] Failed to get rollout data: {err}, retry times: {retry_times}")
            retry_times += 1

    log_raw_info(args, all_meta_info, rollout_id)

    # Apply group-based data selection if there are too many samples
    results = select_rollout_data(args, results, data_number_to_fetch // args.n_samples_per_prompt)

    if len(all_meta_info) > 0 and "finished_groups" in all_meta_info[0]:
        finished_groups_instance_id_list = []
        for item in all_meta_info:
            finished_groups_instance_id_list.extend(item["finished_groups"])

        data_buffer.update_metadata({str(rollout_id): finished_groups_instance_id_list})

    print("finally get rollout data with length: ", len(results))
    sample_results = []

    for i, group_record in enumerate(results):
        group_results = []
        for record in group_record:
            oai_messages = record["messages"]
            session_id = record["extra_info"].get("session_id", "")

            # Convert messages to string
            messages_str = tokenizer.apply_chat_template(
                oai_messages,
                add_generation_prompt=False,
                tokenize=False
            )

            # Get mask from LLM Proxy
            mask_ranges = get_mask_ranges_from_server(session_id, messages_str)
            log_mask_segments_to_file(
                session_id=session_id,
                messages_str=messages_str,
                mask_ranges=mask_ranges,
                instance_id=str(record.get("instance_id", "")),
                uid=str(record.get("uid", "")),
            )

            # Tokenize with mask ranges
            token_ids, loss_mask, response_length = tokenize_with_char_mask_ranges(
                tokenizer, messages_str, mask_ranges
                )

            group_results.append(
                Sample(
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
                    metadata={**record["extra_info"], "raw_reward": record.get("raw_reward")},
                )
            )
        sample_results.append(group_results)

    data_buffer.add_samples(sample_results)
    final_return_results = data_buffer.get_samples(args.rollout_batch_size)  # type: ignore
    log_used_batch_info(args, final_return_results, rollout_id)

    return final_return_results


def generate_rollout(args, rollout_id, data_buffer, evaluation=False):
    """Generate rollout for both training and evaluation."""
    return run(generate_rollout_async(args, rollout_id, data_buffer, evaluation))
