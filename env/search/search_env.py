import io
import re
import json
import textwrap
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import gymnasium as gym
import requests
from PIL import Image, ImageDraw

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


def _resolve_dataset_path(path: str) -> Path:
    """Resolve dataset path, supporting relative paths."""
    candidate = Path(path)
    if candidate.exists():
        return candidate

    # Try relative to project root and current working directory
    fallback_roots = [
        Path.cwd(),
        Path(__file__).resolve().parents[2],
    ]
    for root in fallback_roots:
        resolved = (root / path).resolve()
        if resolved.exists():
            return resolved
    raise FileNotFoundError(f"Search dataset not found at '{path}'")


def _load_search_dataset(path: str) -> List[Dict[str, Any]]:
    """Load dataset entries with question and ground truth answers."""
    dataset_path = _resolve_dataset_path(path)
    suffix = dataset_path.suffix.lower()
    entries: List[Dict[str, Any]] = []

    if suffix == ".jsonl":
        with dataset_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entries.append(json.loads(line))
    elif suffix == ".json":
        with dataset_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            entries = data
        elif isinstance(data, dict):
            # Prefer common keys, default to values list
            for key in ("data", "examples", "items"):
                if key in data and isinstance(data[key], list):
                    entries = data[key]
                    break
            else:
                # Treat dict values as entries if iterable
                entries = list(data.values())
        else:
            raise ValueError(f"Unsupported JSON structure in dataset '{dataset_path}'")
    else:
        raise ValueError(f"Unsupported dataset format '{dataset_path.suffix}'. Use .json or .jsonl.")

    normalized: List[Dict[str, Any]] = []
    for idx, raw in enumerate(entries):
        if not isinstance(raw, dict):
            raise ValueError(f"Dataset entry #{idx} is not an object: {repr(raw)}")
        question = raw.get("question") or raw.get("query")
        if not question or not isinstance(question, str):
            raise ValueError(f"Dataset entry #{idx} missing 'question' text")

        answers = raw.get("ground_truth") or raw.get("answers") or raw.get("answer")
        if answers is None:
            raise ValueError(f"Dataset entry #{idx} missing ground truth answers")
        if isinstance(answers, str):
            answers_list = [answers]
        elif isinstance(answers, list):
            answers_list = [str(a) for a in answers if a is not None]
        else:
            raise ValueError(f"Dataset entry #{idx} has unsupported answers type: {type(answers)}")

        answers_list = [ans.strip() for ans in answers_list if ans.strip()]
        if not answers_list:
            raise ValueError(f"Dataset entry #{idx} has empty answers list")

        normalized.append({
            "question": question.strip(),
            "ground_truth": answers_list,
            "metadata": {k: v for k, v in raw.items() if k not in {"question", "query", "ground_truth", "answers", "answer"}},
        })

    if not normalized:
        raise ValueError(f"No valid entries found in dataset '{dataset_path}'")

    return normalized


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
        dataset_path: str,
        question_index: Optional[int] = None,
        max_turns: int = 3,
        topk: int = 3,
        google_api_key: Optional[str] = None,
        gl: str = "us",
        hl: str = "en",
        proxy: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.dataset_path = dataset_path
        self.dataset = _load_search_dataset(dataset_path)
        self.dataset_resolved_path = str(_resolve_dataset_path(dataset_path))
        self.dataset_size = len(self.dataset)

        if question_index is None:
            question_index = 0
        if not (0 <= question_index < self.dataset_size):
            raise ValueError(
                f"question_index {question_index} out of range for dataset "
                f"with {self.dataset_size} entries"
            )

        self.default_question_index = question_index
        self.active_question_index = question_index
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
        self.current_question: str = ""
        self.current_ground_truth: List[str] = []
        self.current_metadata: Dict[str, Any] = {}
        self.reset_count = 0

    def _select_question(self, options: Optional[Dict[str, Any]] = None) -> None:
        idx = self.default_question_index
        if options:
            candidate = options.get("question_index", options.get("question_id"))
            if candidate is not None:
                if not isinstance(candidate, int):
                    raise ValueError("question_index in reset options must be an integer")
                if not (0 <= candidate < self.dataset_size):
                    raise ValueError(
                        f"question_index {candidate} out of range for dataset "
                        f"with {self.dataset_size} entries"
                    )
                idx = candidate
        self.active_question_index = idx
        entry = self.dataset[idx]
        self.current_question = entry["question"]
        self.current_ground_truth = entry["ground_truth"]
        self.current_metadata = entry.get("metadata", {})

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> ResetOutput:
        self.done = False
        self.turn = 0
        self.history_blocks = []
        self.last_action = ""
        self.last_reward = 0.0
        self.reset_count += 1
        self._select_question(options)
        obs = {
            "question": self.current_question,
            "history": ""  # empty initially
        }
        info: Dict[str, Any] = {
            "question_index": self.active_question_index,
            "dataset_path": self.dataset_resolved_path,
        }
        if self.current_metadata:
            info["metadata"] = self.current_metadata
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
            reward = 1.0 if _em_check(content, self.current_ground_truth) else 0.0
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
            "question": self.current_question,
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
            f"Question: {self.current_question}",
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

        header = (
            f"SearchEnv | Turn {self.turn}/{self.max_turns} "
            f"| Q#{self.active_question_index + 1}/{self.dataset_size}"
        )
        draw.text((20, 20), header, fill=(0, 0, 0))

        last_action = (self.last_action[:800] + "...") if len(self.last_action) > 800 else self.last_action
        wrapped = textwrap.fill(last_action, width=100)
        y = 60
        draw.text((20, y), "Question:", fill=(0, 0, 0))
        y += 24
        question_wrapped = textwrap.fill(self.current_question, width=100)
        draw.text((20, y), question_wrapped, fill=(0, 0, 0))
        y += 24 * (1 + question_wrapped.count("\n")) + 10

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
