import io
import re
import json
import textwrap
from typing import List, Dict, Any, Optional, Tuple

import gymnasium as gym
import requests
from PIL import Image, ImageDraw, ImageFont

from core.types.base import (
    ResetOutput,
    StepOutput,
    RenderOutput,
    PromptOutput,
    TextContent,
    OpenAIMessage,
    MessageContent,
)
from core.env.base_env import BaseEnv
from core.env.env_register import register_env


def _normalize_answer(s: str) -> str:
    s = s.lower()
    # Remove punctuation and extra spaces, basic normalization
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _em_check(prediction: str, gold: List[str]) -> bool:
    p = _normalize_answer(prediction)
    for g in gold:
        if _normalize_answer(g) == p:
            return True
    return False


def _serper_search(
    api_key: str,
    query: str,
    topk: int = 3,
    timeout: int = 30,
    gl: str = "us",
    hl: str = "en",
    proxy: Optional[str] = None,
) -> List[Dict[str, str]]:
    """Query Google Serper API and return a list of {title, snippet, link}.
    Keeps behavior simple and uses only snippet (no page fetching).
    """
    if not api_key:
        return []
    url = "https://google.serper.dev/search"
    payload = {
        "q": query,
        "num": max(1, min(topk, 10)),
        "gl": gl,
        "hl": hl,
    }
    headers = {
        "Content-Type": "application/json",
        "X-API-KEY": api_key,
    }
    try:
        proxies = None
        if proxy:
            proxies = {"http": proxy, "https": proxy}
        r = requests.post(url, headers=headers, json=payload, timeout=timeout, proxies=proxies)
        r.raise_for_status()
        data = r.json()
        items = data.get("organic", [])
        out = []
        for it in items:
            out.append({
                "title": it.get("title", "No title."),
                "snippet": it.get("snippet", "No snippet available."),
                "link": it.get("link", ""),
            })
        return out
    except Exception:
        return []


def _format_results_as_docs(results: List[Dict[str, str]]) -> str:
    out = []
    for i, it in enumerate(results, 1):
        title = it.get("title", "No title.")
        desc = it.get("snippet", it.get("description", "No snippet available."))
        url = it.get("link", it.get("url", ""))
        block = f'Doc {i}(Title: {title}) {desc}\n{url}'.strip()
        out.append(block)
    return "\n".join(out)


def _extract_tag(text: str, tag: str) -> Optional[str]:
    pattern = rf"<{tag}>(.*?)</{tag}>"
    m = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else None


ACTION_RE = re.compile(r"<(search|answer)>(.*?)</\1>", re.DOTALL | re.IGNORECASE)


@register_env("search")
class SearchEnv(BaseEnv):
    """
    A minimal search environment inspired by Search-R1:
    - The agent should iterate with <think>...</think> and <search>...</search>.
    - The env returns <information>...</information> after a search.
    - The episode ends when the agent outputs <answer>...</answer> or max_turns is reached.
    - Reward: 1.0 for exact-match (EM) with any ground truth answer, else 0.0.
    """

    def __init__(
        self,
        question: str,
        ground_truth: List[str] | str,
        max_turns: int = 3,
        topk: int = 3,
        google_api_key: Optional[str] = None,
        gl: str = "us",
        hl: str = "en",
        proxy: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.question = question
        if isinstance(ground_truth, str):
            self.ground_truth = [ground_truth]
        else:
            self.ground_truth = list(ground_truth)
        self.max_turns = max(1, max_turns)
        self.topk = max(1, topk)
        self.google_api_key = google_api_key
        self.gl = gl
        self.hl = hl
        self.proxy = proxy

        # Spaces kept simple
        self.action_space = gym.spaces.Text(max_length=20000)
        self.observation_space = gym.spaces.Dict({
            "question": gym.spaces.Text(max_length=10000),
            "history": gym.spaces.Text(max_length=200000),
        })

        # runtime state
        self.turn = 0
        self.history_blocks: List[str] = []  # store <information>...</information> blocks
        self.last_action: str = ""
        self.last_reward: float = 0.0

    def reset(self, seed: Optional[int] = None) -> ResetOutput:
        self.done = False
        self.turn = 0
        self.history_blocks = []
        self.last_action = ""
        self.last_reward = 0.0
        obs = {
            "question": self.question,
            "history": ""  # empty initially
        }
        info: Dict[str, Any] = {}
        return ResetOutput(observation=obs, info=info)

    def step(self, action: str) -> StepOutput:
        if self.done:
            return StepOutput(
                observation=self._make_observation(),
                reward=0.0,
                terminated=True,
                truncated=False,
                info={}
            )

        self.last_action = str(action or "").strip()
        reward = 0.0
        terminated = False
        truncated = False

        m = ACTION_RE.search(self.last_action)
        if m:
            action_tag = m.group(1).lower()
            content = m.group(2).strip()
        else:
            action_tag = None
            content = None

        if action_tag == "search" and content:
            if not self.google_api_key:
                docs = (
                    "Serper API key not provided. Please provide 'google_api_key' "
                    "when creating the environment."
                )
                results = []
            else:
                results = _serper_search(
                    api_key=self.google_api_key,
                    query=content,
                    topk=self.topk,
                    gl=self.gl,
                    hl=self.hl,
                    proxy=self.proxy,
                )
                docs = _format_results_as_docs(results) if results else "No result."
            info_block = f"<information>\n{docs}\n</information>"
            self.history_blocks.append(info_block)
            reward = 0.0
            self.turn += 1
            if self.turn >= self.max_turns:
                truncated = True
                self.done = True
        elif action_tag == "answer" and content is not None:
            reward = 1.0 if _em_check(content, self.ground_truth) else 0.0
            terminated = True
            self.done = True
        else:
            # Invalid action format: provide gentle hint via history
            hint = (
                "<information>\n"
                "Invalid action. Use <search>query</search> to retrieve, "
                "or finish with <answer>final</answer>.\n"
                "</information>"
            )
            self.history_blocks.append(hint)
            reward = 0.0
            self.turn += 1
            if self.turn >= self.max_turns:
                truncated = True
                self.done = True

        return StepOutput(
            observation=self._make_observation(),
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info={}
        )

    def _make_observation(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "history": "\n\n".join(self.history_blocks)
        }

    def get_task_prompt(self) -> PromptOutput:
        system_text = (
            "You are a helpful search assistant. Always think step by step inside "
            "<think>...</think>. Use <search>...</search> to query the web. "
            "I will return retrieved text inside <information>...</information>. "
            "When you have enough information, finalize with <answer>...</answer>. "
            "Only use these tags: think, search, information, answer."
        )
        system_message = OpenAIMessage(
            role="system",
            content=[MessageContent(root=TextContent(text=system_text))],
        )

        history_text = self._make_observation()["history"]
        user_parts = [
            f"Question: {self.question}",
        ]
        if history_text:
            user_parts.append("Context so far:")
            user_parts.append(history_text)

        user_parts.append(
            "Respond using exactly one of the following actions: "
            "<search>your query</search> OR <answer>your final answer</answer>."
        )
        user_message = OpenAIMessage(
            role="user",
            content=[MessageContent(root=TextContent(text="\n\n".join(user_parts)))],
        )

        return PromptOutput(system_message=system_message, user_message=user_message)

    def render(self) -> RenderOutput:
        # Render a very simple log image with the last action and progress
        width, height = 900, 500
        img = Image.new("RGB", (width, height), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)

        header = f"SearchEnv | Turn {self.turn}/{self.max_turns}"
        draw.text((20, 20), header, fill=(0, 0, 0))

        last_action = (self.last_action[:800] + "...") if len(self.last_action) > 800 else self.last_action
        wrapped = textwrap.fill(last_action, width=100)
        y = 60
        draw.text((20, y), "Last Action:", fill=(0, 0, 0))
        y += 24
        draw.text((20, y), wrapped, fill=(0, 0, 0))

        # Draw the last information block if any
        if self.history_blocks:
            latest_info = self.history_blocks[-1]
            latest_info_text = (latest_info[:800] + "...") if len(latest_info) > 800 else latest_info
            wrapped_info = textwrap.fill(latest_info_text, width=100)
            y += 24 * (1 + wrapped.count("\n")) + 10
            draw.text((20, y), "Latest Info:", fill=(0, 0, 0))
            y += 24
            draw.text((20, y), wrapped_info, fill=(0, 0, 0))

        # Convert to bytes
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        data = buf.getvalue()
        buf.close()
        return RenderOutput(image_data=data, step=self.turn)

    def close(self) -> None:
        super().close()
