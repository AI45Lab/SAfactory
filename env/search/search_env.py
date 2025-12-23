import io
import json
import re
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import logging
import gymnasium as gym
import matplotlib.pyplot as plt
import requests
import yaml
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from functools import wraps
import time

from core.env.base_env import BaseEnv
from core.env.env_register import register_env
from core.types.base import RenderOutput, ResetOutput, StepOutput

logger = logging.getLogger(__name__)

# 全局数据集缓存，避免重复加载
_dataset_cache: Dict[str, "pd.DataFrame"] = {}


def _get_cached_dataset(dataset_path: str) -> "pd.DataFrame":
    """获取缓存的数据集，避免重复加载 parquet 文件"""
    import pandas as pd

    if dataset_path not in _dataset_cache:
        logger.info(f"Loading dataset from {dataset_path}")
        _dataset_cache[dataset_path] = pd.read_parquet(dataset_path)
        logger.info(f"Dataset loaded: {len(_dataset_cache[dataset_path])} rows")
    return _dataset_cache[dataset_path]


@register_env("search")
class SearchEnv(BaseEnv):
    """
    一个基于 Web 搜索的问答环境。

    - observation_space: 当前问题 + 多轮对话历史（压缩为文本）
    - action_space: LLM 的完整回复字符串（包含 <think>、<tool_use> 等）
    - get_task_prompt(): 返回 system + user，两者共同编码当前对话历史
    - step(action): 解析 LLM 回复中的 <tool_use name="search"> 调用，执行搜索，
      并将 <tool_result> 形式的结果追加到对话历史（作为 user 消息）
    - render(): 将最近几轮对话绘制成简单图片，方便可视化
    """

    def __init__(
        self,
        dataset_path: Optional[str] = None,
        dataset_index: Optional[int] = None,
        config_path: Optional[str] = None,
        env_id: str = "",
        env_name: str = "",
    ) -> None:
        """
        Args:
            dataset_path: parquet 数据集路径，必须提供，用于从数据集中读取 question / ground_truth
            dataset_index: 使用数据集中的哪一行（问题索引）
        其他环境参数（top_k、search_api_url、judger 配置等）统一从配置文件中加载，
        不再通过构造函数显式传入。
        """
        super().__init__(env_id=env_id, env_name=env_name)

        # ---- 从配置文件加载通用参数 ----
        if config_path is None:
            # 默认使用与本文件同目录下的 search_env_runtime.yaml
            default_path = Path(__file__).with_name("search_env_runtime.yaml")
            config_path = str(default_path)

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
        except FileNotFoundError:
            cfg = {}

        # ---- 解析数据集路径 ----
        if dataset_path is None:
            # 优先读取顶层 dataset_path，兼容旧版 dataset.path
            dataset_path = cfg.get("dataset_path")
            if dataset_path is None:
                ds_cfg = cfg.get("dataset", {}) or {}
                dataset_path = ds_cfg.get("path")
        if dataset_path is None:
            raise ValueError("SearchEnv requires dataset_path (either argument or config.dataset_path), but none was provided.")

        # ---- 从数据集获取 question / ground_truth ----
        # 使用缓存，避免重复加载 parquet 文件
        df = _get_cached_dataset(str(dataset_path))
        idx = int(dataset_index) if dataset_index is not None else 0
        if idx < 0 or idx >= len(df):
            raise IndexError(f"dataset_index {idx} out of range for dataset of size {len(df)}")
        row = df.iloc[idx]

        question = row.get("question")
        answers = row.get("golden_answers")
        if question is None:
            raise ValueError("SearchEnv requires a question column in dataset, but got None.")

        ground_truth: List[str]
        if answers is None:
            ground_truth = []
        else:
            if isinstance(answers, (list, tuple)):
                seq = answers
            elif hasattr(answers, "tolist"):
                seq = answers.tolist()
            else:
                seq = [answers]
            ground_truth = [str(a) for a in seq]

        self.question: str = str(question)
        # 归一化 ground truth，统一为字符串列表
        self.ground_truth: List[str] = [str(x) for x in ground_truth]

        self.top_k: int = int(cfg.get("top_k", 5))

        # 检索服务配置
        self.search_api_url: str = str(cfg.get("search_api_url", "http://100.99.186.41:8000/retrieve"))
        self.search_timeout: float = float(cfg.get("search_timeout", 10.0))

        # LLM 判别器配置（可选），统一从配置中读取
        judge_cfg = cfg.get("judge", {}) or {}
        judge_api_key = judge_cfg.get("api_key") or ""
        judge_base_url = judge_cfg.get("base_url") or ""
        judge_model = judge_cfg.get("model") or ""
        judge_temperature = float(judge_cfg.get("temperature", 0.0))

        self._judge_client: Optional[OpenAI] = None
        self._judge_model: Optional[str] = None
        self._judge_temperature: float = float(judge_temperature)
        if judge_base_url:
            self._judge_client = OpenAI(
                api_key=str(judge_api_key),
                base_url=str(judge_base_url),
            )
            self._judge_model = str(judge_model)

        # 环境状态
        self.step_count: int = 0
        self.total_search_calls: int = 0
        self.final_answer: Optional[str] = None

        # 对话历史：直接维护为 OpenAI messages 格式
        # List[ChatCompletionMessageParam]，元素形如 {"role": "...", "content": "..."}
        self.messages: List[ChatCompletionMessageParam] = []

        # 定义 action/observation space（仅用于规范）
        self.action_space = gym.spaces.Text(max_length=8000)
        # 当前环境不依赖 observation 内容，这里给一个空的 Dict 空间
        self.observation_space = gym.spaces.Dict({})

    def format_validate(self, action: str, is_final_action: bool = False) -> bool:
        """
        验证 action 格式是否符合预期。
        """
        if is_final_action:
            # 对于最终答案，不能包含工具调用
            if "<tool_use>" in action:
                return False
        # 任何情况下都不能包含工具结果
        if "<tool_result>" in action:
            return False
        return True

    # ------------------------------------------------------------------
    # 环境核心接口
    # ------------------------------------------------------------------
    def reset(self, seed: Optional[int] = None) -> ResetOutput:
        """重置环境：清空对话历史，只保留 system + 首条 user 问题"""
        self.step_count = 0
        self.done = False
        self.total_search_calls = 0
        self.final_answer = None

        system_text = """You are an intelligent assistant.
Your goal is to answer the user's question as accurately and concisely as possible.
You MUST first think in <think>...</think> whenever you receive new information.
If, after thinking, you find that you need external knowledge, you may call the search tool exactly as:
  <tool_use name="search">{"query": "real_query"}</tool_use>
The environment will then send you another user message containing the search results, formatted as:
  <tool_result>{"result": [{"title": "title text", "content": "short summary"}, ...]}</tool_result>
You may use the search tool multiple times across the conversation, but at most once per message.
When you have enough information, stop calling tools and reply with the final answer in natural language, without any special tags.
"""
        # 初始化完整 messages：system + 首条 user（问题）
        self.messages = [
            {"role": "system", "content": system_text},  # type: ignore[typeddict-item]
            {
                "role": "user",
                "content": f"Question: {self.question}",
            },  # type: ignore[typeddict-item]
        ]

        # 当前环境不依赖 observation，返回空 dict
        observation: Dict[str, Any] = {}
        info: Dict[str, Any] = {
            "question": self.question,
            "ground_truth": list(self.ground_truth),
            "step": self.step_count,
            "total_search_calls": self.total_search_calls,
            "conversation": list(self.messages),
        }

        return ResetOutput(observation=observation, info=info)

    def step(self, action: str) -> StepOutput:
        """
        接收 LLM 的回复：
        - 追加到对话历史（assistant）
        - 若包含 <tool_use name="search"> 调用，则执行一次搜索并追加 <tool_result> 消息
        - 若不包含工具调用，则视为已经给出最终答案，直接终止 episode
        - 奖励：query 步（成功解析工具调用）给 1；answer 步若配置了 LLM 判别器则根据正确性给 0/1
        """
        self.step_count += 1

        reward, terminated, truncated, extra_info = self._process_action(action or "")
        # 当前环境不需要向 Agent 返回新的 observation 内容，这里统一为空 dict
        observation: Dict[str, Any] = {}
        info: Dict[str, Any] = {
            "question": self.question,
            "step": self.step_count,
            "total_search_calls": self.total_search_calls,
            "num_messages": len(self.messages),
            "final_answer": self.final_answer,
            "judger": extra_info.get("judger"),
            "conversation": list(self.messages),
        }

        return StepOutput(
            observation=observation,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )

    def get_task_prompt(self) -> List[ChatCompletionMessageParam]:
        """
        返回当前任务提示的 messages 列表（直接返回内部维护的 self.messages）。
        """
        return list(self.messages)

    def render(self) -> RenderOutput:
        """
        将最近几轮对话渲染成简单图片，满足可视化接口需求。
        """
        # 仅取最近若干条消息，避免图片过长
        tail_k = 8
        recent = self.messages[-tail_k:] if self.messages else []

        lines: List[str] = []
        for i, msg in enumerate(recent, 1):
            role = msg.get("role", "?")
            content = msg.get("content", "")
            # 截断长内容
            snippet = content[:180] + ("..." if len(content) > 180 else "")
            lines.append(f"[{i}] {role}: {snippet}")

        if not lines:
            lines = ["(no messages yet)"]

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.axis("off")
        ax.text(
            0.01,
            0.99,
            "\n\n".join(lines),
            va="top",
            ha="left",
            wrap=True,
            fontsize=10,
        )
        ax.set_title(f"SearchEnv Conversation | step={self.step_count}", fontsize=12)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
        buf.seek(0)
        image_data = buf.read()
        buf.close()
        plt.close(fig)

        return RenderOutput(step=self.step_count, image_data=image_data)

    def close(self) -> None:
        """当前环境无额外资源需要清理，占位实现。"""
        pass

    # ------------------------------------------------------------------
    # 内部工具方法
    # ------------------------------------------------------------------
    def _process_action(self, assistant_msg: str) -> Tuple[float, bool, bool, Dict[str, Any]]:
        """
        处理当前 assistant 的消息（query 或 answer），并返回：
        - reward: query 步为 1；answer 步为 0/1
        - terminated / truncated: episode 是否结束
        - extra_info: 额外信息（例如 judger 结果）
        """
        msg = (assistant_msg or "").strip()
        # 记录 assistant 消息到对话历史
        self.messages.append(
            {"role": "assistant", "content": msg}  # type: ignore[typeddict-item]
        )

        reward: float = 0.0
        judge_result: Optional[Dict[str, Any]] = None

        # 尝试解析单个 search 工具调用
        query = self._extract_search_query(msg)
        is_final_action = False
        llm_as_a_judge_score: float = 0.0
        # case 1: 存在工具调用 -> 执行一次搜索，并继续对话（不终止）
        if query:
            results = self._run_web_search(query)
            logger.info(
                "[SearchEnv] search query: %s",
                query
            )
            self.total_search_calls += 1
            payload = {"result": results}
            self.messages.append(
                {"role": "user", "content": f"<tool_result>{json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}</tool_result>"}  # type: ignore[typeddict-item]
            )

            terminated = False
            truncated = False

        # case 2: 无工具调用 -> 视为已经给出最终答案，调用 LLM 判别器打分
        else:
            is_final_action = True
            self.final_answer = re.sub(r"<think>.*?</think>", "", msg, flags=re.DOTALL).strip()
            logger.info(
                "[SearchEnv] final answer: %s",
                self.final_answer
            )
            terminated = True
            truncated = False
            self.done = True

            # 若配置了判别器则用 LLM 打 0/1 分
            try:
                judge_result = self._llm_judge(
                    question=self.question, answer=self.final_answer
                )
                logger.debug(
                    "[SearchEnv] llm judge ground truth: %s",
                    " or ".join(self.ground_truth)
                )
                logger.debug(
                    "[SearchEnv] llm judge score=%f", 
                    judge_result.get("score", 0.0) or 0.0
                )
                llm_as_a_judge_score = judge_result.get("score", 0.0) or 0.0
            except Exception as e:
                logger.error(
                    "[SearchEnv] llm judge error: %s", 
                    e
                )
                judge_result = {"error": f"{type(e).__name__}: {e}"}
                
        # apply format validate
        if not self.format_validate(msg, is_final_action):
            reward = 0.0
        else:
            reward = max(0.1, llm_as_a_judge_score)
        extra_info: Dict[str, Any] = {"judger": judge_result}
        return reward, terminated, truncated, extra_info
    
    def _parse_judge_response(self, raw: str) -> dict:
        """用正则提取 score 和 explanation"""
        score_match = re.search(r'"score"\s*:\s*(\d+)', raw)
        expl_match = re.search(r'"explanation"\s*:\s*"(.+?)"\s*,\s*"score"', raw, re.DOTALL)

        if score_match and expl_match:
            return {
                "explanation": expl_match.group(1),
                "score": int(score_match.group(1)),
            }
        raise ValueError(f"No valid JSON with 'explanation' and 'score' found in: {raw}")


    def retry(max_retries: int = 10, sleep: int = 1, exceptions: tuple = (Exception,)):
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                last_exception = None
                for i in range(max_retries):
                    try:
                        return func(*args, **kwargs)
                    except exceptions as e:
                        logger.error(f"Error in {func.__name__}, retry {i+1} of {max_retries}: {type(e).__name__}")
                        last_exception = e
                        if i < max_retries - 1 and sleep:
                            time.sleep(sleep)
                raise last_exception
            return wrapper
        return decorator

    @retry(max_retries=5, sleep=1, exceptions=(Exception,))
    def _llm_judge(self, question: str, answer: str) -> Dict[str, Any]:
        """
        使用配置好的 LLM 对 (question, answer) 进行打分。

        返回：
        {
          "score": 0 或 1,
          "explanation": str,
          "raw_response": str
        }
        """
        if self._judge_client is None or self._judge_model is None:
            raise RuntimeError("LLM judger is not configured for this SearchEnv.")

        prompt_system = """You are a strict evaluator for question-answer pairs.
Given a user question, a list of ground-truth answers, and a candidate answer, you must decide whether the candidate answer is correct (1) with respect to at least one ground-truth answer, or incorrect (0).
Respond ONLY with a JSON object of the form:
{"explanation": "<short explanation>", "score": 0 or 1}"""

        # 将 ground truth 列表格式化到 prompt 中
        gt_lines = [f"- {gt}" for gt in (self.ground_truth or [])]
        gt_block = "\n".join(gt_lines) if gt_lines else "(none provided)"

        prompt_user = f"""Question:
{question}

Ground-truth answers (list):
{gt_block}

Candidate Answer:
{answer}

Now produce the JSON object."""
        response = self._judge_client.chat.completions.create(
            model=self._judge_model,
            messages=[
                {"role": "system", "content": prompt_system},
                {"role": "user", "content": prompt_user},
            ],
            temperature=self._judge_temperature
        )

        raw = (response.choices[0].message.content or "").strip()
        
        obj = self._parse_judge_response(raw)

        return {
            "score": float(obj.get("score", 0.0)),
            "explanation": obj.get("explanation", ""),
            "raw_response": raw,
        }

    @staticmethod
    def _extract_search_query(text: str) -> Optional[str]:
        """
        从 LLM 回复中提取 <tool_use name="search"> 的 query 字符串。

        期望格式：
        <tool_use name="search">{"query": "real_query"}</tool_use>
        """
        if not text:
            return None

        pattern = re.compile(
            r'<tool_use\s+name\s*=\s*["\']search["\']\s*>(.*?)</tool_use>',
            re.IGNORECASE | re.DOTALL,
        )
        m_block = pattern.search(text)
        if not m_block:
            return None

        try:
            data = json.loads(m_block.group(1).strip())
            q = data.get("query", "")
            return q.strip() or None if isinstance(q, str) else None
        except (json.JSONDecodeError, AttributeError):
            return None


    def _run_web_search(self, query: str) -> List[Dict[str, str]]:
        """
        调用外部检索服务执行搜索。

        接口格式（POST JSON）：
          url: self.search_api_url
          body: {
            "queries": [query],
            "topk": self.top_k,
            "return_scores": false
          }

        预期返回示例：
        {
          "result": [[{"id": "...", "contents": "...."}, ...]]
        }

        我们将每个条目转换为：
          {"title": <从 contents 或 id 提取>, "content": <主体内容>}

        若 HTTP/网络错误发生，直接抛出异常（let it crash）。
        """
        query = query.strip()
        if not query:
            return []

        payload = {
            "queries": [query],
            "topk": int(self.top_k),
            "return_scores": False,
        }
        resp = requests.post(
            self.search_api_url,
            json=payload,
            timeout=self.search_timeout,
        )
        resp.raise_for_status()
        data = resp.json()

        results: List[Dict[str, str]] = []

        # data["result"] 预期为二维列表：[[{id, contents}, ...]]
        raw_lists = data.get("result") or []
        first_list = raw_lists[0] if raw_lists else []

        for item in first_list:
            if not isinstance(item, dict):
                continue
            contents = str(item.get("contents") or "").strip()
            if not contents:
                continue

            lines = contents.splitlines()
            first_line = lines[0].strip() if lines else ""

            # 尝试从第一行中提取引号内标题，例如 `"Machine learning"`
            m = re.search(r'"([^"]+)"', first_line)
            if m:
                title = m.group(1).strip()
            else:
                title = first_line or str(item.get("id") or "result")

            # 内容主体：去掉首行，保留剩余文本；若没有剩余，则用原 contents
            body = "\n".join(lines[1:]).strip() if len(lines) > 1 else contents
            results.append({"title": title, "content": body})

            if len(results) >= self.top_k:
                break

        # 截断到 top_k（可能为 0，表示未检索到结果）
        return results[: self.top_k]
