import asyncio
import os
from typing import List, Dict, Optional
import logging
import aiohttp
import threading

logger = logging.getLogger(__name__)

# 从环境变量读取重试配置，支持模型权重更新期间的长时间等待
# -1 表示无限重试
DEFAULT_MAX_RETRIES = int(os.environ.get("LLM_MAX_RETRIES", "-1"))  # 默认无限重试
DEFAULT_RETRY_BACKOFF = float(os.environ.get("LLM_RETRY_BACKOFF", "2.0"))  # 初始退避 2 秒
DEFAULT_MAX_RETRY_DELAY = float(os.environ.get("LLM_MAX_RETRY_DELAY", "30.0"))  # 最大退避 30 秒


class LLM:
    """
    基础 LLM 客户端封装，负责直接调用 OpenAI 兼容的 /chat/completions 接口。
    支持长时间重试以应对模型权重更新等场景。
    """

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        temperature: float = 1,
        llm_proxy: str = "",
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_backoff: float = DEFAULT_RETRY_BACKOFF,
        max_retry_delay: float = DEFAULT_MAX_RETRY_DELAY,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.llm_proxy = llm_proxy
        self.max_retries = int(max_retries)  # -1 表示无限重试
        self.retry_backoff = float(retry_backoff)
        self.max_retry_delay = float(max_retry_delay)

    async def generate(self, messages: List[dict]) -> str:
        """
        基于 OpenAI messages 列表调用 chat.completions（异步接口，内置失败重试）。

        messages 必须是形如 [{"role": "...", "content": "..."}] 的标准结构。

        重试策略：指数退避，最大延迟封顶，适应模型权重更新场景。
        """
        attempt = 0
        url = f"{self.base_url}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }

        last_error = None
        # max_retries == -1 表示无限重试
        while self.max_retries == -1 or attempt <= self.max_retries:
            # 每次重试创建新的 session，避免连接状态问题
            timeout = aiohttp.ClientTimeout(total=300)
            try:
                async with aiohttp.ClientSession(timeout=timeout, trust_env=True) as session:
                    logger.info(f"[LLM] POST {url} (thread={threading.current_thread().name}, attempt={attempt})")
                    async with session.post(url, json=payload, headers=headers, proxy=self.llm_proxy) as resp:
                        logger.info(f"[LLM] POST {url} got response status={resp.status}")
                        resp.raise_for_status()
                        data = await resp.json()
                    logger.info(f"[LLM] POST {url} returned")
                    content = data["choices"][0]["message"]["content"]
                    if not content:
                        raise RuntimeError("LLM returned empty content")
                    return content
            except Exception as e:
                last_error = e
                attempt += 1
                # 无限重试模式下不检查 max_retries
                if self.max_retries != -1 and attempt > self.max_retries:
                    break
                # 指数退避，但不超过 max_retry_delay
                delay = min(self.retry_backoff * (2 ** (attempt - 1)), self.max_retry_delay)
                # 检测可能是模型更新的错误（503、连接拒绝等）
                error_str = str(e).lower()
                retry_info = "∞" if self.max_retries == -1 else f"{attempt}/{self.max_retries}"
                if any(hint in error_str for hint in ["503", "service unavailable", "connection refused", "temporarily unavailable"]):
                    print(f"[LLM] Model may be updating, waiting {delay:.1f}s before retry (attempt={retry_info}): {e}")
                else:
                    print(f"[LLM] generate failed (attempt={retry_info}), retry in {delay:.1f}s: {e}")
                await asyncio.sleep(delay)

        raise RuntimeError(f"[LLM] All {self.max_retries} retries exhausted. Last error: {last_error}")
