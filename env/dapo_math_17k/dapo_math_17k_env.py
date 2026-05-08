from pathlib import Path
from typing import Any, Dict, List, Optional

import gymnasium as gym
import yaml
from openai.types.chat import ChatCompletionMessageParam

from core.env.base_env import BaseEnv
from core.env.env_register import register_env
from core.types.base import RenderOutput, ResetOutput, StepOutput
from env.math500_text.math500_text_env import score_math500_answer


def _norm_text(s: Any) -> str:
    return str(s or "").strip()


def _normalize_prompt(raw_prompt: Any) -> List[ChatCompletionMessageParam]:
    if isinstance(raw_prompt, str):
        return [{"role": "user", "content": raw_prompt}]  # type: ignore[typeddict-item]

    if isinstance(raw_prompt, list):
        messages: List[ChatCompletionMessageParam] = []
        for item in raw_prompt:
            if not isinstance(item, dict):
                continue
            role = _norm_text(item.get("role")) or "user"
            content = _norm_text(item.get("content"))
            if content:
                messages.append({"role": role, "content": content})  # type: ignore[typeddict-item]
        return messages

    return []


@register_env("dapo_math_17k")
class DapoMath17KEnv(BaseEnv):
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
            config_path = str(Path(__file__).with_name("dapo_math_17k_env_runtime.yaml"))
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                self.runtime_cfg = yaml.safe_load(f) or {}
        except Exception:
            self.runtime_cfg = {}

        self.max_turns = int(self.runtime_cfg.get("max_turns", 1) or 1)
        self.messages = _normalize_prompt(dataset.get("prompt"))
        self.answer = _norm_text(dataset.get("label", dataset.get("answer", "")))
        self.uid = _norm_text(dataset.get("unique_id", dataset.get("id", "")))

        if not self.messages or not self.answer:
            raise ValueError("DapoMath17KEnv requires dataset fields: prompt, label")

        self.problem = self.messages[-1]["content"]
        self.turn = 0

        self.action_space = gym.spaces.Text(max_length=8192)
        self.observation_space = gym.spaces.Dict({})

    def reset(self, seed: Optional[int] = None) -> ResetOutput:
        self.done = False
        self.turn = 0
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
