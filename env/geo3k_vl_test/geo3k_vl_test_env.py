import io
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import matplotlib.pyplot as plt
import yaml
from openai.types.chat import ChatCompletionMessageParam

from core.env.base_env import BaseEnv
from core.env.env_register import register_env
from core.types.base import RenderOutput, ResetOutput, StepOutput
from env.geo3k_vl_test.math_utils import extract_answer as extract_boxed_answer
from env.geo3k_vl_test.math_utils import grade_answer_verl

logger = logging.getLogger(__name__)

TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
SUPPORTED_TOOL_NAMES = {"calc_score", "calc_geo3k_reward"}


@register_env("geo3k_vl_test")
class Geo3kVLTestEnv(BaseEnv):
    """
    A VL multi-turn test env adapted for AIEvoBox.

    Data contract (dataset row):
    - problem: question text
    - answer: ground truth answer
    - images: list[str] image urls (typically data urls)
    """

    def __init__(
        self,
        dataset: Dict[str, Any],
        config_path: Optional[str] = None,
        env_id: str = "",
        env_name: str = "",
    ) -> None:
        super().__init__(env_id=env_id, env_name=env_name, dataset=dataset)

        if config_path is None:
            config_path = str(Path(__file__).with_name("geo3k_vl_test_env_runtime.yaml"))
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                runtime_cfg = yaml.safe_load(f) or {}
        except FileNotFoundError:
            logger.warning("Runtime config not found: %s, using defaults", config_path)
            runtime_cfg = {}

        self.max_turns: int = int(runtime_cfg.get("max_turns", 3))
        self.max_images: int = int(runtime_cfg.get("max_images", 1))
        self.echo_images_on_feedback: bool = bool(runtime_cfg.get("echo_images_on_feedback", False))

        question = dataset.get("problem", dataset.get("question"))
        if question is None:
            raise ValueError("Geo3kVLTestEnv requires dataset.problem (or dataset.question).")
        self.question: str = str(question)

        self.ground_truth: str = self._normalize_ground_truth(
            dataset.get("answer", dataset.get("golden_answers"))
        )
        if not self.ground_truth:
            logger.warning("Ground truth is empty for env_id=%s", env_id)

        # Explicitly ignore `preprocessed_images`; we only use `images`.
        self.image_urls: List[str] = self._normalize_image_urls(dataset.get("images"))

        self.step_count: int = 0
        self.total_tool_calls: int = 0
        self.final_answer: Optional[str] = None
        self.latest_boxed_answer: Optional[str] = None
        self.last_tool_score: Optional[float] = None
        self.messages: List[ChatCompletionMessageParam] = []
        self.tool_calls: List[Dict[str, Any]] = []

        self.action_space = gym.spaces.Text(max_length=12000)
        self.observation_space = gym.spaces.Dict({})

    def reset(self, seed: Optional[int] = None) -> ResetOutput:
        self.step_count = 0
        self.total_tool_calls = 0
        self.final_answer = None
        self.latest_boxed_answer = None
        self.last_tool_score = None
        self.tool_calls = []
        self.done = False

        user_content: List[Dict[str, Any]] = [
            {"type": "text", "text": self.question},
        ]

        if self.max_images > 0:
            for image_url in self.image_urls[: self.max_images]:
                user_content.append({"type": "image_url", "image_url": {"url": image_url}})

        self.messages = [
            {"role": "user", "content": user_content},  # type: ignore[typeddict-item]
        ]

        info = {
            "question": self.question,
            "ground_truth": self.ground_truth,
            "image_count": len(self.image_urls),
            "step": self.step_count,
            "max_turns": self.max_turns,
            "total_tool_calls": self.total_tool_calls,
        }
        return ResetOutput(observation={}, info=info)

    def step(self, action: str) -> StepOutput:
        self.step_count += 1
        reward, terminated, truncated, extra_info = self._process_action(action or "")

        # Hard turn cap at env level.
        if not (terminated or truncated) and self.step_count >= self.max_turns:
            if self.latest_boxed_answer:
                reward = self._score_answer(self.latest_boxed_answer)
                reward_source = "latest_boxed_answer"
            else:
                reward = 0.0
                reward_source = "missing_boxed_answer"
            self.final_answer = self.latest_boxed_answer
            terminated = True
            self.done = True
            extra_info = dict(extra_info)
            extra_info["final_score"] = reward
            extra_info["reward_source"] = reward_source

        info = {
            "question": self.question,
            "step": self.step_count,
            "max_turns": self.max_turns,
            "total_tool_calls": self.total_tool_calls,
            "last_tool_score": self.last_tool_score,
            "final_answer": self.final_answer,
            "tool_calls": list(self.tool_calls),
            "extra": extra_info,
        }
        return StepOutput(
            observation={},
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )

    def get_task_prompt(self) -> List[ChatCompletionMessageParam]:
        return list(self.messages)

    def render(self) -> RenderOutput:
        tail_k = 8
        recent = self.messages[-tail_k:] if self.messages else []
        lines: List[str] = []

        for i, msg in enumerate(recent, 1):
            role = msg.get("role", "?")
            content = self._message_content_to_text(msg.get("content"))
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
        ax.set_title(f"Geo3kVLTestEnv | step={self.step_count}", fontsize=12)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
        buf.seek(0)
        image_data = buf.read()
        buf.close()
        plt.close(fig)

        return RenderOutput(step=self.step_count, image_data=image_data)

    def close(self) -> None:
        return

    def _process_action(self, assistant_msg: str) -> Tuple[float, bool, bool, Dict[str, Any]]:
        msg = (assistant_msg or "").strip()
        self.messages.append({"role": "assistant", "content": msg})  # type: ignore[typeddict-item]
        msg_wo_think = re.sub(r"<think>.*?</think>", "", msg, flags=re.DOTALL).strip()
        boxed_answer = extract_boxed_answer(msg_wo_think)
        if boxed_answer is not None:
            boxed_answer = boxed_answer.strip()
            if boxed_answer:
                self.latest_boxed_answer = boxed_answer

        tool_call = self._extract_tool_call(msg_wo_think)
        if tool_call is not None:
            return self._handle_tool_call(tool_call)

        self.final_answer = self.latest_boxed_answer or msg_wo_think
        if self.latest_boxed_answer:
            score = self._score_answer(self.latest_boxed_answer)
            reward_source = "latest_boxed_answer"
        else:
            score = 0.0
            reward_source = "missing_boxed_answer"
        self.done = True
        return score, True, False, {"final_score": score, "reward_source": reward_source}

    def _handle_tool_call(self, tool_call: Dict[str, Any]) -> Tuple[float, bool, bool, Dict[str, Any]]:
        name = str(tool_call.get("name", "")).strip()
        arguments = tool_call.get("arguments", {})

        if name not in SUPPORTED_TOOL_NAMES:
            self._append_tool_feedback(
                (
                    f"Tool `{name}` is not supported. "
                    "Use `<tool_call>{\"name\":\"calc_score\",\"arguments\":{\"answer\":\"...\"}}</tool_call>`."
                )
            )
            return 0.0, False, False, {"tool_executed": False, "unsupported_tool": name}

        if not isinstance(arguments, dict):
            self._append_tool_feedback("Tool arguments must be a JSON object.")
            return 0.0, False, False, {"tool_executed": False, "bad_arguments": True}

        raw_answer = arguments.get("answer")
        parsed_answer = "" if raw_answer is None else str(raw_answer).strip()
        if not parsed_answer:
            self._append_tool_feedback("Tool call detected but no `answer` was provided.")
            return 0.0, False, False, {"tool_executed": False, "answer_missing": True}

        score = self._score_answer(parsed_answer)
        self.last_tool_score = score
        self.total_tool_calls += 1

        tool_record = {"name": name, "answer": parsed_answer, "score": score}
        self.tool_calls.append(tool_record)

        self._append_tool_feedback(self._build_tool_feedback(score, parsed_answer))

        return 0.0, False, False, {"tool_executed": True, "tool_score": score}

    def _append_tool_feedback(self, text: str) -> None:
        if self.echo_images_on_feedback and self.max_images > 0 and self.image_urls:
            content: List[Dict[str, Any]] = [{"type": "text", "text": text}]
            for image_url in self.image_urls[: self.max_images]:
                content.append({"type": "image_url", "image_url": {"url": image_url}})
            self.messages.append({"role": "user", "content": content})  # type: ignore[typeddict-item]
        else:
            self.messages.append({"role": "user", "content": text})  # type: ignore[typeddict-item]

    def _extract_tool_call(self, text: str) -> Optional[Dict[str, Any]]:
        matches = list(TOOL_CALL_RE.finditer(text))
        if not matches:
            return None

        raw_json = matches[-1].group(1).strip()
        try:
            payload = json.loads(raw_json)
        except json.JSONDecodeError:
            return None

        name = payload.get("name") or payload.get("function", {}).get("name")
        arguments = payload.get("arguments") or payload.get("function", {}).get("arguments") or {}
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                return None

        if not name:
            return None
        return {"name": name, "arguments": arguments}

    def _score_answer(self, answer: str) -> float:
        if not self.ground_truth:
            return 0.0

        answer = answer.strip()
        candidates = [answer]
        if "\\boxed" not in answer:
            candidates.append(f"\\boxed{{{answer}}}")

        for candidate in candidates:
            if grade_answer_verl(candidate, self.ground_truth):
                return 1.0
        return 0.0

    def _build_tool_feedback(self, score: float, parsed_answer: str) -> str:
        turn_idx = self.step_count - 1  # zero-based
        last_warning_turn = None
        if self.max_turns is not None:
            if self.max_turns >= 2:
                last_warning_turn = self.max_turns - 2
            else:
                last_warning_turn = self.max_turns - 1
        is_final_turn = last_warning_turn is not None and turn_idx >= last_warning_turn

        if score == 1.0:
            return (
                f"calc_score result: {score}. Parsed answer '{parsed_answer}' matches the reference. "
                "You can now stop reasoning and provide the final solution in \\boxed{}."
            )
        if is_final_turn:
            return (
                f"calc_score result: {score}. Parsed answer '{parsed_answer}' does not match the reference. "
                "Your answer is wrong. You may need to reason in a different way. Don't repeat your answer unless necessary. "
                "Since you only have one chance to answer, don't call tool again. "
                "You should provide your final answer in the form Answer: \\boxed{$Answer} where $Answer is your final answer to this problem."
            )
        return (
            f"calc_score result: {score}. Parsed answer '{parsed_answer}' does not match the reference. "
            "Your answer is wrong. You may need to reason in a different way. Don't repeat your answer unless necessary."
        )

    @staticmethod
    def _normalize_ground_truth(raw_answer: Any) -> str:
        if raw_answer is None:
            return ""

        value = raw_answer
        if hasattr(value, "tolist"):
            value = value.tolist()

        if isinstance(value, (list, tuple)):
            for item in value:
                s = str(item).strip()
                if s:
                    return s
            return ""

        s = str(value).strip()
        if not s:
            return ""

        if s.startswith("[") and s.endswith("]"):
            try:
                obj = json.loads(s)
                if isinstance(obj, list):
                    for item in obj:
                        t = str(item).strip()
                        if t:
                            return t
                    return ""
            except json.JSONDecodeError:
                return s

        return s

    @staticmethod
    def _normalize_image_urls(raw_images: Any) -> List[str]:
        if raw_images is None:
            return []

        value = raw_images
        if hasattr(value, "tolist"):
            value = value.tolist()

        if isinstance(value, str):
            s = value.strip()
            if not s:
                return []
            if s.startswith("[") and s.endswith("]"):
                value = json.loads(s)
            else:
                value = [s]

        if not isinstance(value, (list, tuple)):
            return []

        urls: List[str] = []
        for item in value:
            if isinstance(item, str):
                s = item.strip()
                if s:
                    urls.append(s)
                continue
            if isinstance(item, dict):
                src = item.get("src") or item.get("url")
                if isinstance(src, str) and src.strip():
                    urls.append(src.strip())
                continue
            s = str(item).strip()
            if s:
                urls.append(s)

        return urls

    @staticmethod
    def _message_content_to_text(content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if not isinstance(item, dict):
                    parts.append(str(item))
                    continue
                item_type = item.get("type")
                if item_type == "text":
                    parts.append(str(item.get("text", "")))
                elif item_type == "image_url":
                    parts.append("[image_url]")
                else:
                    parts.append(json.dumps(item, ensure_ascii=False))
            return " ".join(parts)
        if isinstance(content, dict):
            return json.dumps(content, ensure_ascii=False)
        return str(content)
