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
from openai.types.chat import ChatCompletionMessageParam

from core.env.base_env import BaseEnv
from core.env.env_register import register_env
from core.types.base import RenderOutput, ResetOutput, StepOutput

logger = logging.getLogger(__name__)


@register_env("search")
class SearchEnv(BaseEnv):
    """
    一个基于 Web 搜索的问答环境。

    - observation_space: 当前问题 + 多轮对话历史（压缩为文本）
    - action_space: LLM 的完整回复字符串（包含 <function_call> 等）
    - get_task_prompt(): 返回 system + user，两者共同编码当前对话历史
    - step(action): 解析 LLM 回复中的 <function_call> 调用，执行搜索，
      并将 <function_result> 形式的结果追加到对话历史（作为 user 消息）
    - render(): 将最近几轮对话绘制成简单图片，方便可视化
    """

    def __init__(
        self,
        dataset: Dict[str, Any],
        config_path: Optional[str] = None,
        env_id: str = "",
        env_name: str = "",
    ) -> None:
        """
        Args:
            dataset: 数据集行数据，包含 question 和 golden_answers 字段
            config_path: 运行时配置文件路径（默认使用同目录下的 search_env_runtime.yaml）
        """
        super().__init__(env_id=env_id, env_name=env_name)

        # ---- 从运行时配置文件加载参数 ----
        if config_path is None:
            config_path = str(Path(__file__).with_name("search_env_runtime.yaml"))

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                runtime_cfg = yaml.safe_load(f) or {}
        except FileNotFoundError:
            logger.warning(f"Runtime config not found: {config_path}, using defaults")
            runtime_cfg = {}

        top_k = runtime_cfg.get("top_k")
        search_api_url = runtime_cfg.get("search_api_url")
        search_timeout = runtime_cfg.get("search_timeout")
        judge_url = runtime_cfg.get("judge_url")
        judge_timeout = runtime_cfg.get("judge_timeout")

        # ---- 获取 question / ground_truth ----
        question = dataset.get("question")
        answers = dataset.get("golden_answers")

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

        # 使用传入参数，提供默认值
        self.top_k: int = int(top_k if top_k is not None else 5)

        # 检索服务配置
        self.search_api_url: str = str(search_api_url if search_api_url is not None else "http://100.99.186.41:8000/retrieve")
        # timeout: -1 means no limit (None in requests)
        _search_timeout = float(search_timeout if search_timeout is not None else 10.0)
        self.search_timeout: Optional[float] = None if _search_timeout < 0 else _search_timeout

        # Judge API URL and timeout
        self._judge_url: Optional[str] = judge_url
        _judge_timeout = float(judge_timeout if judge_timeout is not None else 30.0)
        self.judge_timeout: Optional[float] = None if _judge_timeout < 0 else _judge_timeout

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

    # ------------------------------------------------------------------
    # 环境核心接口
    # ------------------------------------------------------------------
    def reset(self, seed: Optional[int] = None) -> ResetOutput:
        """重置环境：清空对话历史，只保留 system + 首条 user 问题"""
        self.step_count = 0
        self.done = False
        self.total_search_calls = 0
        self.final_answer = None

        system_text = """# Tools
You may call one or more functions to assist with the user query.
You are provided with function signatures within <function>...</function> XML tags:
<function>
{"name": "search", "description": "Search external knowledge to answer user questions", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "Search query string"}}, "required": ["query"]}}
</function>

For each function call, return a json object with function name and arguments within <function_call></function_call> XML tags:
<function_call>
{"name": <function-name>, "arguments": <args-json-object>}
</function_call>"""
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
        - 若包含 <function_call> 调用，则执行一次搜索并追加 <function_result> 消息
        - 若不包含工具调用，则视为已经给出最终答案，直接终止 episode
        - 奖励：answer 步根据 judge API 返回的正确性给 0/1
        """
        self.step_count += 1

        reward, terminated, truncated, extra_info = self._process_action(action or "")
        if (terminated or truncated) and self.step_count <= 1:
            reward = 0.0
        # 当前环境不需要向 Agent 返回新的 observation 内容，这里统一为空 dict
        observation: Dict[str, Any] = {}
        info: Dict[str, Any] = {
            "question": self.question,
            "step": self.step_count,
            "total_search_calls": self.total_search_calls,
            "num_messages": len(self.messages),
            "final_answer": self.final_answer,
            "judger": extra_info.get("judger", {}),
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

        msg = re.sub(r"<think>.*?</think>", "", msg, flags=re.DOTALL).strip()

        reward: float = 0.0
        # 尝试解析单个 search 工具调用
        query = self._extract_search_query(msg)

        # case 1: 存在工具调用 -> 执行一次搜索，并继续对话（不终止）
        if query:
            if "<function_result>" in msg or "</function_result>" in msg:
                terminated = True
                truncated = False
                self.done = True
                reward = 0.0
                return reward, terminated, truncated, {}

            results = self._run_web_search(query)
            logger.info(
                "[SearchEnv] search query: %s",
                query
            )
            self.total_search_calls += 1
            payload = {"result": results}
            self.messages.append(
                {"role": "user", "content": f"<function_result>\n{json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}\n</function_result>"}  # type: ignore[typeddict-item]
            )
            terminated = False
            truncated = False
            return reward, terminated, truncated, {}


        # case 2: 无工具调用 -> 视为已经给出最终答案，调用 LLM 判别器打分
        else:
            
            self.final_answer = msg
            logger.info(
                "[SearchEnv] final answer: %s",
                self.final_answer
            )
            terminated = True
            truncated = False
            self.done = True

            if "<function_call>" in msg or "</function_call>" in msg:
                reward = 0.0
                return reward, terminated, truncated, {}

            # 调用 judge API 打分
            judge_result = self._judge_answer(
                question=self.question, answer=self.final_answer
            )
            reward = max(0.1, judge_result.get("score"))
            return reward, terminated, truncated, {"judger": judge_result}
    
    def _judge_answer(self, question: str, answer: str) -> Dict[str, Any]:
        """
        调用 judge API 对 (question, answer) 进行打分。

        返回：
        {
          "score": 0.0 或 1.0,
          "explanation": str,
        }
        """
        if self._judge_url is None:
            raise RuntimeError("Judge URL is not configured for this SearchEnv.")

        payload = {
            "question": question,
            "answer": answer,
            "ground_truth": self.ground_truth,
        }
        resp = requests.post(self._judge_url, json=payload, timeout=self.judge_timeout)
        resp.raise_for_status()
        data = resp.json()

        return {
            "score": 1.0 if data.get("correct") == "yes" else 0.0,
            "explanation": data.get("reasoning", ""),
        }

    @staticmethod
    def _extract_search_query(text: str) -> Optional[str]:
        """
        从 LLM 回复中提取 <function_call> 的 search query 字符串。

        期望格式：
        <function_call>
        {"name": "search", "arguments": {"query": "real_query"}}
        </function_call>
        """
        if not text:
            return None

        pattern = re.compile(r'<function_call>(.*?)</function_call>', re.DOTALL)
        m_block = pattern.search(text)
        if not m_block:
            return None

        try:
            data = json.loads(m_block.group(1).strip())
            if isinstance(data, dict) and data.get("name") != "search":
                return None
            args = data.get("arguments", {})
            if isinstance(args, dict):
                q = args.get("query", "")
                if not isinstance(q, str):
                    return None
                return q.strip() or None
            else:
                return None
        except (json.JSONDecodeError, AttributeError, TypeError):
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
