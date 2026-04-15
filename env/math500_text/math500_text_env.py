import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import gymnasium as gym
import yaml
from openai.types.chat import ChatCompletionMessageParam

from core.env.base_env import BaseEnv
from core.env.env_register import register_env
from core.types.base import RenderOutput, ResetOutput, StepOutput


def _norm_text(s: str) -> str:
    return str(s or "").strip()


def _extract_boxed_text(s: str) -> str:
    idx = s.rfind("\\boxed")
    if idx < 0:
        return ""
    start = s.find("{", idx)
    if start < 0:
        return ""
    depth = 0
    for i in range(start, len(s)):
        ch = s[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return s[start + 1 : i].strip()
    return ""


def _normalize_answer_text(s: str) -> str:
    s = _norm_text(s)
    if not s:
        return ""
    boxed = _extract_boxed_text(s)
    if boxed:
        s = boxed
    s = s.replace(" ", "")
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\!", "")
    s = s.replace("tfrac", "frac").replace("dfrac", "frac")
    s = re.sub(r"\s+", "", s)
    return s.lower()


def score_math500_answer(pred: str, ground_truth: str) -> float:
    pred_norm = _normalize_answer_text(pred)
    gt_norm = _normalize_answer_text(ground_truth)
    if not pred_norm or not gt_norm:
        return 0.0
    return 1.0 if pred_norm == gt_norm else 0.0


@register_env("math500_text")
class Math500TextEnv(BaseEnv):
    def __init__(
        self,
        dataset: Dict[str, Any],
        config_path: Optional[str] = None,
        env_id: str = "",
        env_name: str = "",
    ) -> None:
        super().__init__(env_id=env_id, env_name=env_name, dataset=dataset)

        self.runtime_cfg: Dict[str, Any] = {}
        if config_path is None:
            config_path = str(Path(__file__).with_name("math500_text_env_runtime.yaml"))
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                self.runtime_cfg = yaml.safe_load(f) or {}
        except Exception:
            self.runtime_cfg = {}

        self.max_turns = int(self.runtime_cfg.get("max_turns", 1) or 1)
        self.system_prompt = str(
            self.runtime_cfg.get(
                "system_prompt",
                "You are a math expert. First reason step by step, then provide the final answer in exactly this format on the last line: Answer: \\boxed{...}",
            )
        )

        self.problem = _norm_text(dataset.get("problem", ""))
        self.answer = _norm_text(dataset.get("answer", ""))
        self.uid = _norm_text(dataset.get("unique_id", dataset.get("id", "")))

        if not self.problem or not self.answer:
            raise ValueError("Math500TextEnv requires dataset fields: problem, answer")

        self.turn = 0
        self.messages: List[ChatCompletionMessageParam] = []

        self.action_space = gym.spaces.Text(max_length=8192)
        self.observation_space = gym.spaces.Dict({})

    def reset(self, seed: Optional[int] = None) -> ResetOutput:
        self.done = False
        self.turn = 0
        self.messages = [
            {"role": "system", "content": self.system_prompt},  # type: ignore[typeddict-item]
            {"role": "user", "content": f"Problem: {self.problem}"},  # type: ignore[typeddict-item]
        ]
        return ResetOutput(
            observation={},
            info={
                "uid": self.uid,
                "turn": self.turn,
                "max_turns": self.max_turns,
                "ground_truth": self.answer,
            },
        )

    def step(self, action: str) -> StepOutput:
        if self.done:
            return StepOutput(
                observation={},
                reward=0.0,
                terminated=True,
                truncated=False,
                info={"uid": self.uid, "reason": "already_done"},
            )

        self.turn += 1
        action = _norm_text(action)
        self.messages.append({"role": "assistant", "content": action})  # type: ignore[typeddict-item]

        reward = score_math500_answer(action, self.answer)
        terminated = bool(reward >= 1.0) or self.turn >= self.max_turns
        truncated = bool(self.turn >= self.max_turns and reward < 1.0)
        self.done = terminated

        return StepOutput(
            observation={},
            reward=float(reward),
            terminated=terminated,
            truncated=truncated,
            info={
                "uid": self.uid,
                "turn": self.turn,
                "max_turns": self.max_turns,
                "ground_truth": self.answer,
                "pred_answer": action,
            },
        )

    def get_task_prompt(self) -> List[ChatCompletionMessageParam]:
        return list(self.messages)

    def render(self) -> RenderOutput:
        summary = [
            f"uid={self.uid}",
            f"turn={self.turn}/{self.max_turns}",
            f"problem={self.problem[:240]}",
        ]
        return RenderOutput(step=self.turn, text_list=summary)

    def close(self) -> None:
        return None

    def is_done(self) -> bool:
        return bool(self.done)

    def health(self) -> bool:
        return True
