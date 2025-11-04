import io
import os
import re
import json
import logging
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

# 设置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ANSI 颜色码
class Colors:
    """ANSI color codes for terminal output"""
    RESET = '\033[0m'
    BOLD = '\033[1m'
    # 前景色
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    # 背景色
    BG_BLUE = '\033[44m'
    BG_GREEN = '\033[42m'
    BG_YELLOW = '\033[43m'
    BG_RED = '\033[41m'


def _normalize_answer(s: str) -> str:
    s = s.lower()
    # Remove punctuation and extra spaces, basic normalization
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _em_check(prediction: str, gold: List[str]) -> bool:
    p = _normalize_answer(prediction)
    logger.debug(f"Checking answer: normalized prediction='{p}'")
    for g in gold:
        normalized_gold = _normalize_answer(g)
        if normalized_gold == p:
            logger.debug(f"Match found with ground truth: '{g}'")
            return True
    logger.debug(f"No match found among {len(gold)} ground truth answers")
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
        logger.warning("Serper API key not provided, returning empty results")
        return []

    logger.info(f"Initiating Serper search - Query: '{query}', TopK: {topk}, GL: {gl}, HL: {hl}")

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

    logger.debug(f"Search payload: {payload}")

    try:
        proxies = None

        if proxy:
            # 优先使用显式传入的proxy参数
            proxies = {"http": proxy, "https": proxy}
            logger.info(f"Using explicit proxy: {proxy}")
        else:
            # 检查环境变量中的代理设置
            http_proxy = os.environ.get('HTTP_PROXY') or os.environ.get('http_proxy')
            https_proxy = os.environ.get('HTTPS_PROXY') or os.environ.get('https_proxy')

            if http_proxy or https_proxy:
                proxies = {}
                if http_proxy:
                    proxies['http'] = http_proxy
                    logger.info(f"Using HTTP_PROXY from environment: {http_proxy}")
                if https_proxy:
                    proxies['https'] = https_proxy
                    logger.info(f"Using HTTPS_PROXY from environment: {https_proxy}")
            else:
                logger.debug("No proxy configured (neither explicit nor from environment)")

        logger.info(f"Sending request to Serper API...")
        r = requests.post(url, headers=headers, json=payload, timeout=timeout, proxies=proxies)
        r.raise_for_status()

        data = r.json()
        items = data.get("organic", [])
        logger.info(f"Received {len(items)} search results from Serper API")

        out = []
        for idx, it in enumerate(items):
            result_item = {
                "title": it.get("title", "No title."),
                "snippet": it.get("snippet", "No snippet available."),
                "link": it.get("link", ""),
            }
            out.append(result_item)
            logger.debug(f"Result {idx + 1}: {result_item['title'][:50]}...")

        logger.info(f"Successfully processed {len(out)} search results")
        return out

    except requests.exceptions.RequestException as e:
        logger.error(f"Request error during Serper search: {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error during Serper search: {e}")
        return []


def _resolve_dataset_path(path: str) -> Path:
    """Resolve dataset path, supporting relative paths."""
    logger.debug(f"Resolving dataset path: {path}")
    candidate = Path(path)
    if candidate.exists():
        logger.debug(f"Found dataset at direct path: {candidate}")
        return candidate

    # Try relative to project root and current working directory
    fallback_roots = [
        Path.cwd(),
        Path(__file__).resolve().parents[2],
    ]
    logger.debug(f"Trying fallback roots: {[str(r) for r in fallback_roots]}")

    for root in fallback_roots:
        resolved = (root / path).resolve()
        logger.debug(f"Checking: {resolved}")
        if resolved.exists():
            logger.info(f"Found dataset at resolved path: {resolved}")
            return resolved

    logger.error(f"Dataset not found at any of the tried paths for: {path}")
    raise FileNotFoundError(f"Search dataset not found at '{path}'")


def _load_search_dataset(path: str) -> List[Dict[str, Any]]:
    """Load dataset entries with question and ground truth answers."""
    logger.info(f"Loading search dataset from: {path}")
    dataset_path = _resolve_dataset_path(path)
    suffix = dataset_path.suffix.lower()
    entries: List[Dict[str, Any]] = []

    logger.debug(f"Dataset format: {suffix}")

    if suffix == ".jsonl":
        logger.info(f"Reading JSONL file: {dataset_path}")
        with dataset_path.open("r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                entries.append(json.loads(line))
                logger.debug(f"Loaded entry {line_num} from JSONL")
    elif suffix == ".json":
        logger.info(f"Reading JSON file: {dataset_path}")
        with dataset_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            entries = data
            logger.debug(f"JSON contains list with {len(entries)} entries")
        elif isinstance(data, dict):
            # Prefer common keys, default to values list
            for key in ("data", "examples", "items"):
                if key in data and isinstance(data[key], list):
                    entries = data[key]
                    logger.debug(f"Using key '{key}' from JSON dict, found {len(entries)} entries")
                    break
            else:
                # Treat dict values as entries if iterable
                entries = list(data.values())
                logger.debug(f"Using dict values as entries, found {len(entries)} entries")
        else:
            raise ValueError(f"Unsupported JSON structure in dataset '{dataset_path}'")
    else:
        raise ValueError(f"Unsupported dataset format '{dataset_path.suffix}'. Use .json or .jsonl.")

    logger.info(f"Loaded {len(entries)} raw entries, normalizing...")

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

        logger.debug(f"Normalized entry {idx}: Question length={len(question)}, Answers count={len(answers_list)}")

    if not normalized:
        raise ValueError(f"No valid entries found in dataset '{dataset_path}'")

    logger.info(f"Successfully loaded and normalized {len(normalized)} dataset entries")
    return normalized


def _format_results_as_docs(results: List[Dict[str, str]]) -> str:
    logger.debug(f"Formatting {len(results)} search results as documents")
    out = []
    for i, it in enumerate(results, 1):
        title = it.get("title", "No title.")
        desc = it.get("snippet", it.get("description", "No snippet available."))
        url = it.get("link", it.get("url", ""))
        block = f'Doc {i}(Title: {title}) {desc}\n{url}'.strip()
        out.append(block)
        logger.debug(f"Formatted Doc {i}: {title[:50]}...")
    formatted_text = "\n".join(out)
    logger.debug(f"Total formatted text length: {len(formatted_text)} characters")
    return formatted_text


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
        ** kwargs
    ) -> None:
        super().__init__(**kwargs)
        logger.info("Initializing SearchEnv environment")
        logger.info(f"Parameters: dataset_path={dataset_path}, question_index={question_index}, "
                    f"max_turns={max_turns}, topk={topk}, gl={gl}, hl={hl}")

        self.dataset_path = dataset_path
        self.dataset = _load_search_dataset(dataset_path)
        self.dataset_resolved_path = str(_resolve_dataset_path(dataset_path))
        self.dataset_size = len(self.dataset)
        logger.info(f"Dataset loaded successfully with {self.dataset_size} entries")

        if question_index is None:
            question_index = 0
            logger.debug("question_index not provided, defaulting to 0")
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

        # Log API key status
        if self.google_api_key:
            logger.info("Google Serper API key provided")
        else:
            logger.warning("No Google Serper API key provided - search functionality will be limited")

        # Log proxy configuration
        if self.proxy:
            logger.info(f"Explicit proxy configured: {self.proxy}")
        else:
            # Check for environment proxy settings
            http_proxy = os.environ.get('HTTP_PROXY') or os.environ.get('http_proxy')
            https_proxy = os.environ.get('HTTPS_PROXY') or os.environ.get('https_proxy')
            if http_proxy or https_proxy:
                logger.info("Environment proxy detected:")
                if http_proxy:
                    logger.info(f"  HTTP_PROXY: {http_proxy}")
                if https_proxy:
                    logger.info(f"  HTTPS_PROXY: {https_proxy}")
            else:
                logger.debug("No proxy configuration detected")

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

        logger.info("SearchEnv environment initialized successfully")

    def _select_question(self, options: Optional[Dict[str, Any]] = None) -> None:
        idx = self.default_question_index
        if options:
            candidate = options.get("question_index", options.get("question_id"))
            if candidate is not None:
                logger.debug(f"Question index override from options: {candidate}")
                if not isinstance(candidate, int):
                    raise ValueError("question_index in reset options must be an integer")
                if not (0 <= candidate < self.dataset_size):
                    raise ValueError(
                        f"question_index {candidate} out of range for dataset "
                        f"with {self.dataset_size} entries"
                    )
                idx = candidate
        else:
            logger.debug(f"Using default question index: {idx}")

        self.active_question_index = idx
        entry = self.dataset[idx]
        self.current_question = entry["question"]
        self.current_ground_truth = entry["ground_truth"]
        self.current_metadata = entry.get("metadata", {})

        logger.info(f"Question #{idx} selected: {len(self.current_question)} chars, "
                    f"{len(self.current_ground_truth)} ground truth answers")

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> ResetOutput:
        logger.info(f"Resetting environment (reset #{self.reset_count + 1})")
        self.done = False
        self.turn = 0
        self.history_blocks = []
        self.last_action = ""
        self.last_reward = 0.0
        self.reset_count += 1
        self._select_question(options)

        logger.info(f"Selected question #{self.active_question_index}: '{self.current_question[:100]}...'")
        logger.debug(f"Ground truth answers: {self.current_ground_truth}")

        # 彩色日志：显示新问题
        print(f"\n{Colors.YELLOW}{Colors.BOLD}{'='*80}{Colors.RESET}")
        print(f"{Colors.YELLOW}{Colors.BOLD}🎯 NEW SEARCH TASK (Reset #{self.reset_count}):{Colors.RESET}")
        print(f"{Colors.YELLOW}Question #{self.active_question_index + 1}: {self.current_question}{Colors.RESET}")
        print(f"{Colors.YELLOW}Max turns: {self.max_turns}{Colors.RESET}")
        print(f"{Colors.YELLOW}{Colors.BOLD}{'='*80}{Colors.RESET}\n")

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
            logger.debug(f"Metadata included: {list(self.current_metadata.keys())}")

        logger.info(f"Environment reset complete - Ready for turn 1/{self.max_turns}")
        return ResetOutput(observation=obs, info=info)

    def step(self, action: str) -> StepOutput:
        if self.done:
            logger.warning("Step called on already completed episode, returning done state")
            return StepOutput(
                observation=self._make_observation(),
                reward=0.0,
                terminated=True,
                truncated=False,
                info={}
            )

        self.last_action = str(action or "").strip()
        logger.info(f"Turn {self.turn + 1}/{self.max_turns}: Processing action")
        logger.debug(f"Raw action (first 500 chars): {self.last_action[:500]}...")

        reward = 0.0
        terminated = False
        truncated = False

        m = ACTION_RE.search(self.last_action)
        if m:
            action_tag = m.group(1).lower()
            content = m.group(2).strip()
            logger.info(f"Detected action tag: <{action_tag}> with content length: {len(content)}")
        else:
            action_tag = None
            content = None
            logger.warning("No valid action tag detected in the action")

        if action_tag == "search" and content:
            logger.info(f"Processing SEARCH action with query: '{content[:100]}...'")

            # 彩色日志：显示搜索查询
            print(f"\n{Colors.CYAN}{Colors.BOLD}🔍 SEARCH QUERY (Turn {self.turn + 1}/{self.max_turns}):{Colors.RESET}")
            print(f"{Colors.CYAN}Query: {content}{Colors.RESET}")
            print(f"{Colors.CYAN}{'='*80}{Colors.RESET}")

            if not self.google_api_key:
                docs = (
                    "Serper API key not provided. Please provide 'google_api_key' "
                    "when creating the environment."
                )
                results = []
                logger.warning("Search attempted without API key")

                # 彩色日志：显示API密钥缺失警告
                print(f"\n{Colors.YELLOW}{Colors.BOLD}⚠️ SEARCH FAILED - No API Key{Colors.RESET}")
                print(f"{Colors.YELLOW}Serper API key not provided. Search functionality is disabled.{Colors.RESET}")
                print(f"{Colors.YELLOW}{'='*80}{Colors.RESET}")
            else:
                logger.info("Calling Serper API for search...")
                results = _serper_search(
                    api_key=self.google_api_key,
                    query=content,
                    topk=self.topk,
                    gl=self.gl,
                    hl=self.hl,
                    proxy=self.proxy,
                )
                docs = _format_results_as_docs(results) if results else "No result."
                logger.info(f"Search completed, {len(results)} results returned")

                # 彩色日志：显示完整搜索结果
                print(f"\n{Colors.BLUE}{Colors.BOLD}📄 SEARCH RESULTS:{Colors.RESET}")
                if results:
                    for idx, result in enumerate(results, 1):
                        print(f"{Colors.BLUE}[Result {idx}]{Colors.RESET}")
                        print(f"  {Colors.BOLD}Title:{Colors.RESET} {result.get('title', 'N/A')}")
                        print(f"  {Colors.BOLD}Snippet:{Colors.RESET} {result.get('snippet', 'N/A')}")
                        print(f"  {Colors.BOLD}Link:{Colors.RESET} {result.get('link', 'N/A')}")
                        if idx < len(results):
                            print(f"{Colors.BLUE}  {'-'*40}{Colors.RESET}")
                else:
                    print(f"{Colors.YELLOW}No results found.{Colors.RESET}")
                print(f"{Colors.BLUE}{'='*80}{Colors.RESET}")

            info_block = f"<information>\n{docs}\n</information>"
            self.history_blocks.append(info_block)
            logger.debug(f"Added information block #{len(self.history_blocks)} to history")

            reward = 0.0
            self.turn += 1
            if self.turn >= self.max_turns:
                truncated = True
                self.done = True
                logger.info(f"Max turns ({self.max_turns}) reached - episode truncated")

                # 彩色日志：显示达到最大回合数
                print(f"\n{Colors.RED}{Colors.BOLD}🛑 MAX TURNS REACHED - Episode Truncated (after search){Colors.RESET}\n")

        elif action_tag == "answer" and content is not None:
            logger.info(f"Processing ANSWER action: '{content[:100]}...'")

            # 彩色日志：显示最终答案
            print(f"\n{Colors.MAGENTA}{Colors.BOLD}✅ FINAL ANSWER (Turn {self.turn + 1}/{self.max_turns}):{Colors.RESET}")
            print(f"{Colors.MAGENTA}Answer: {content}{Colors.RESET}")
            print(f"{Colors.MAGENTA}{'='*80}{Colors.RESET}")

            is_correct = _em_check(content, self.current_ground_truth)
            reward = 1.0 if is_correct else 0.0
            terminated = True
            self.done = True

            # 彩色日志：显示评估结果
            print(f"\n{Colors.BOLD}📊 ANSWER EVALUATION:{Colors.RESET}")
            if is_correct:
                print(f"{Colors.GREEN}{Colors.BOLD}✓ CORRECT ANSWER! (Reward: 1.0){Colors.RESET}")
                print(f"{Colors.GREEN}Your answer '{content}' matches the expected answer(s).{Colors.RESET}")
            else:
                print(f"{Colors.RED}{Colors.BOLD}✗ INCORRECT ANSWER (Reward: 0.0){Colors.RESET}")
                print(f"{Colors.RED}Your answer: {content}{Colors.RESET}")
                print(f"{Colors.YELLOW}Expected answers: {', '.join(self.current_ground_truth)}{Colors.RESET}")
            print(f"{Colors.MAGENTA}{'='*80}{Colors.RESET}\n")

            logger.info(f"Answer evaluation: {'CORRECT' if is_correct else 'INCORRECT'} (reward={reward})")
            logger.debug(f"Given answer: '{content}', Expected: {self.current_ground_truth}")

        else:
            logger.warning("Invalid action format received")

            # 彩色日志：显示无效动作警告
            print(f"\n{Colors.RED}{Colors.BOLD}⚠️ INVALID ACTION (Turn {self.turn + 1}/{self.max_turns}):{Colors.RESET}")
            print(f"{Colors.RED}Your action: {self.last_action[:100]}{'...' if len(self.last_action) > 100 else ''}{Colors.RESET}")
            print(f"{Colors.YELLOW}Hint: Use <search>query</search> to search, or <answer>your answer</answer> to answer.{Colors.RESET}")
            print(f"{Colors.RED}{'='*80}{Colors.RESET}\n")

            # Invalid action format: provide gentle hint via history
            hint = (
                "<information>\n"
                "Invalid action. Use <search>query</search> to retrieve, "
                "or finish with <answer>final</answer>.\n"
                "</information>"
            )
            self.history_blocks.append(hint)
            logger.debug("Added invalid action hint to history")

            reward = 0.0
            self.turn += 1
            if self.turn >= self.max_turns:
                truncated = True
                self.done = True
                logger.info(f"Max turns ({self.max_turns}) reached after invalid action - episode truncated")

                # 彩色日志：显示达到最大回合数
                print(f"{Colors.RED}{Colors.BOLD}🛑 MAX TURNS REACHED - Episode Truncated{Colors.RESET}\n")

        logger.info(f"Step complete - Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}, Done: {self.done}")

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
        logger.info("Closing SearchEnv environment")
        logger.debug(f"Final stats - Total resets: {self.reset_count}, Last turn: {self.turn}")
        super().close()
        logger.info("SearchEnv environment closed successfully")
