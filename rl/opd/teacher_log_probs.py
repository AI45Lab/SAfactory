import asyncio
import logging
import os
from typing import Any, Dict, List, Optional, Sequence

import aiohttp

from slime.utils.types import Sample

logger = logging.getLogger(__name__)


def _to_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_token_logprobs(response_json: Dict[str, Any]) -> List[float]:
    """Parse teacher token log-probs from SGLang-compatible response payload."""
    meta_info = response_json.get("meta_info") or {}
    raw = meta_info.get("input_token_logprobs")
    if not isinstance(raw, list):
        return []

    parsed: List[float] = []
    for item in raw:
        value: Optional[float] = None
        if isinstance(item, (list, tuple)) and item:
            value = _to_float(item[0])
        elif isinstance(item, dict):
            # Be defensive for different server response schemas.
            value = _to_float(item.get("logprob"))
        else:
            value = _to_float(item)
        if value is not None:
            parsed.append(value)
    return parsed


async def _fetch_teacher_log_probs(
    session: aiohttp.ClientSession,
    teacher_url: str,
    token_ids: Sequence[int],
    response_length: int,
) -> List[float]:
    if response_length <= 0:
        return []

    payload = {
        "input_ids": list(token_ids),
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 0,
            "skip_special_tokens": False,
        },
        "return_logprob": True,
        "logprob_start_len": 0,
    }

    async with session.post(teacher_url, json=payload) as response:
        response.raise_for_status()
        response_json = await response.json()

    token_logprobs = _extract_token_logprobs(response_json)
    if not token_logprobs:
        return []
    # Align with generated response span.
    return token_logprobs[-response_length:]


async def attach_teacher_log_probs(args: Any, sample_groups: List[List[Sample]]) -> List[List[Sample]]:
    use_opd = bool(getattr(args, "use_opd", False))
    opd_type = str(getattr(args, "opd_type", "")).lower()
    if not use_opd or opd_type != "sglang":
        return sample_groups

    teacher_url = (
        getattr(args, "rm_url", None)
        or os.environ.get("TEACHER_URL")
        or os.environ.get("RM_URL")
    )
    if not teacher_url:
        logger.warning("OPD is enabled but TEACHER_URL is not configured. Skip teacher scoring.")
        for group in sample_groups:
            for sample in group:
                sample.teacher_log_probs = None
        return sample_groups

    request_timeout_s = int(os.environ.get("OPD_TEACHER_TIMEOUT_SECONDS", "60"))
    max_concurrency = int(os.environ.get("OPD_TEACHER_MAX_CONCURRENCY", "16"))
    semaphore = asyncio.Semaphore(max(1, max_concurrency))
    timeout = aiohttp.ClientTimeout(total=request_timeout_s)

    total_count = 0
    attached_count = 0

    async with aiohttp.ClientSession(timeout=timeout) as session:
        async def _score_sample(sample: Sample) -> None:
            nonlocal total_count, attached_count
            total_count += 1
            sample.teacher_log_probs = None
            try:
                response_length = int(getattr(sample, "response_length", 0))
                token_ids = getattr(sample, "tokens", [])
                if response_length <= 0 or not token_ids:
                    return

                async with semaphore:
                    teacher_log_probs = await _fetch_teacher_log_probs(
                        session=session,
                        teacher_url=teacher_url,
                        token_ids=token_ids,
                        response_length=response_length,
                    )

                if len(teacher_log_probs) == response_length:
                    sample.teacher_log_probs = teacher_log_probs
                    attached_count += 1
            except Exception as exc:
                logger.debug("Attach teacher log-probs failed: %s", exc, exc_info=True)

        tasks = [_score_sample(sample) for group in sample_groups for sample in group]
        if tasks:
            await asyncio.gather(*tasks)

    logger.info(
        "OPD teacher scoring done: attached=%s total=%s teacher_url=%s",
        attached_count,
        total_count,
        teacher_url,
    )
    return sample_groups
