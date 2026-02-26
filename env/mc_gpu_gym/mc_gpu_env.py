from __future__ import annotations

import io
import json
import re
import base64
from pathlib import Path
import sys

# Ensure project root is on PYTHONPATH when run via relative launcher
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from typing import Any, Dict, List

import numpy as np
from PIL import Image
from openai.types.chat import ChatCompletionMessageParam

from core.types.base import ResetOutput, StepOutput, RenderOutput
from core.env.base_env import BaseEnv
from core.env.env_register import register_env
from env.mc_gpu_gym.mc_simulator import MCSimulator


@register_env("mc_gpu_gym")
class MCGPUGym(BaseEnv):
    """
    GPU 版 Minecraft 环境。

    - 仅保留与 MCSimulator 兼容的最小接口
    - 支持 LLM 文本动作与标准动作字符串
    """

    def __init__(
        self,
        env_config: Any = None,
        env_id: str = "",
        env_name: str = "",
        dataset: Any = None,
        **kwargs: Any,
    ):
        super().__init__(env_id, env_name, dataset)
        self.env_config = env_config
        self.dataset = dataset
        self.messages: List[ChatCompletionMessageParam] = []
        self.current_step = 0
        self.last_obs: Dict[str, Any] | None = None
        self.simulator = self._init_simulator(env_config, dataset, **kwargs)

    def _init_simulator(self, env_config: Any, dataset: Any, **kwargs: Any) -> MCSimulator:
        """根据传入配置创建 GPU 模拟器"""
        sim_kwargs: Dict[str, Any] = {}
        base_dir = Path(__file__).resolve().parent  # env/mc_gpu_gym
        # 仓库根目录应是 AIEvoBox
        repo_root = Path(__file__).resolve().parents[2]

        # 场景/任务数据：优先使用 dataset，其次使用 env_config
        if isinstance(dataset, (dict, list)):
            sim_kwargs["config"] = dataset
        elif isinstance(dataset, str) and dataset:
            sim_kwargs["data_path"] = dataset

        if isinstance(env_config, (dict, list)):
            sim_kwargs.setdefault("config", env_config)
        elif isinstance(env_config, str) and env_config:
            cfg_path = Path(env_config)
            if cfg_path.suffix.lower() in {".json", ".jsonl"}:
                sim_kwargs.setdefault("data_path", str(cfg_path))
            else:
                sim_kwargs.setdefault("config_path", str(cfg_path))

        # 透传可选参数
        for key in ("output_dir", "display_port", "working_dir", "mc_root", "data_path", "config_path"):
            if key in kwargs and kwargs[key] is not None:
                sim_kwargs[key] = kwargs[key]

        # 将相对路径转换为以仓库根目录为基准的绝对路径，避免启动目录不同导致找不到文件
        for path_key in ("working_dir", "mc_root", "output_dir", "data_path", "config_path"):
            val = sim_kwargs.get(path_key)
            if isinstance(val, str) and not Path(val).is_absolute():
                sim_kwargs[path_key] = str((repo_root / val).resolve())

        return MCSimulator(**sim_kwargs)

    def _normalize_action(self, action_input: Any) -> Any:
        """将多种动作格式统一为 MCSimulator 可接受的格式"""
        # 字典格式：优先提取激活的标准动作键
        if isinstance(action_input, dict):
            for name in MCSimulator.STANDARD_ACTIONS:
                if action_input.get(name):
                    return name
            return action_input

        # 列表格式：通常来自解析后的 JSON
        if isinstance(action_input, list):
            for item in action_input:
                if isinstance(item, dict) and "action" in item:
                    return item["action"]
        # 字符串格式：可能带有 <answer> 包裹或 JSON 片段
        if isinstance(action_input, str):
            answer_match = re.search(r"<answer>(.*?)</answer>", action_input, re.S)
            candidate = answer_match.group(1) if answer_match else action_input

            json_match = re.search(r"\[.*\]", candidate, re.S)
            if json_match:
                try:
                    parsed = json.loads(json_match.group(0))
                    if isinstance(parsed, list) and parsed:
                        first = parsed[0]
                        if isinstance(first, dict) and "action" in first:
                            return first["action"]
                except Exception:
                    pass

            candidate = candidate.strip()
            if candidate:
                return candidate

        return action_input

    def _extract_instruction(self) -> str:
        """尝试从配置中读取任务描述"""
        for source in (self.env_config, self.dataset):
            if isinstance(source, dict):
                for key in ("instructions", "instruction", "text", "task", "goal"):
                    if source.get(key):
                        return str(source[key])
        return ""

    def step(self, action: Any) -> StepOutput:
        self.messages.append({"role": "assistant", "content": str(action)})
        normalized_action = self._normalize_action(action)

        obs, reward, terminated, truncated, info = self.simulator.step(normalized_action)
        self.current_step += 1
        self.last_obs = obs
        self._persist_last_obs()

        return StepOutput(
            observation=obs,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )

    def reset(self, seed: int | None = None) -> ResetOutput:
        del seed  # 当前模拟器未使用随机种子
        obs, info = self.simulator.reset()
        self.last_obs = obs
        self.current_step = 0
        self._persist_last_obs()
        return ResetOutput(observation=obs, info=info)

    def close(self) -> None:
        self.simulator.close()

    def _build_system_prompt(self) -> str:
        actions = "\n".join(
            [
                # "- wait: no-op/idle 1 tick.",
                "- walk_forward: move forward.",
                "- walk_backward: move backward.",
                "- move_left: strafe left (A).",
                "- move_right: strafe right (D).",
                "- sprint: hold sprint modifier.",
                "- sneak: hold sneak/crouch modifier.",
                "- jump: jump once.",
                "- use: right-click interact/place/use.",
                "- attack: left-click attack/break.",
                "- turn_right: turn camera right (coarse).",
                "- turn_left: turn camera left (coarse).",
                "- look_up: pitch up (coarse).",
                "- look_down: pitch down (coarse).",
                "- look_down-left: pitch down + yaw left (coarse).",
                "- look_up-right: pitch up + yaw right (coarse).",
                "- inventory: open/close inventory.",
            ]
        )
        instruction = self._extract_instruction()
        if not instruction:
            instruction = "Control the Minecraft agent to accomplish the task."
        return (
            f"{instruction}\n\n"
            "Allowed actions (choose EXACT string):\n"
            f"{actions}\n\n"
            "Output exactly ONE action as a JSON list of length 1, e.g. "
            '[{\"action\": \"walk_forward\"}]. Do not add extra text. '
            "Yaw/pitch are NOT used in this GPU env; omit them."
        )

    def get_task_prompt(self) -> List[ChatCompletionMessageParam]:
        """生成包含动作空间与图像占位的 prompt"""
        if not self.messages:
            sys_prompt_text = self._build_system_prompt()
            self.messages = [{"role": "system", "content": sys_prompt_text}]

            # 如果已有观察，附带图像占位
            user_content: List[Dict[str, Any]] = [
                {"type": "text", "text": "Based on the current view, choose the next action."}
            ]

            # 仅当观测中存在图像且格式正确时附带图像
            if isinstance(self.last_obs, dict):
                img = self.last_obs.get("image")
                pov_path = self.last_obs.get("pov")
                if isinstance(img, np.ndarray):
                    try:
                        render_output = self.render()
                        base64_str = render_output.image_base64 or ""
                        if base64_str and not base64_str.startswith("data:"):
                            base64_str = f"data:image/png;base64,{base64_str}"
                        if base64_str:
                            user_content.append(
                                {"type": "image_url", "image_url": {"url": base64_str}}
                            )
                    except Exception:
                        # 图像缺失或格式异常时，仅返回文本，避免 500
                        ...
                elif isinstance(pov_path, str) and pov_path:
                    try:
                        p = Path(pov_path)
                        if p.is_file():
                            data = p.read_bytes()
                            b64 = base64.b64encode(data).decode("ascii")
                            user_content.append(
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/png;base64,{b64}"},
                                }
                            )
                    except Exception:
                        ...

            self.messages.append({"role": "user", "content": user_content})

        return self.messages

    def render(self) -> RenderOutput:
        if self.last_obs is None:
            raise RuntimeError("No observation available. Call reset() first.")

        image_array = self.last_obs.get("image") if isinstance(self.last_obs, dict) else None
        pov_path = self.last_obs.get("pov") if isinstance(self.last_obs, dict) else None

        image_data = None
        # 优先使用 numpy 图像
        if isinstance(image_array, np.ndarray):
            if image_array.dtype != np.uint8:
                image_array = image_array.astype(np.uint8)
            img = Image.fromarray(image_array, "RGB")
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            image_data = buffer.getvalue()
            buffer.close()
        # 其次尝试从文件路径读取
        elif isinstance(pov_path, str) and pov_path:
            p = Path(pov_path)
            if p.is_file():
                image_data = p.read_bytes()

        if image_data is None:
            raise ValueError("Observation does not contain an image.")

        return RenderOutput(step=self.current_step, image_data=image_data)

    def _persist_last_obs(self) -> None:
        """将最近一次观测保存到 logs/obs 目录，便于排查。"""
        if self.last_obs is None:
            return

        try:
            log_dir = Path(__file__).parent / "logs" / "obs"
            log_dir.mkdir(parents=True, exist_ok=True)

            def _to_jsonable(x: Any):
                if isinstance(x, np.ndarray):
                    return x.tolist()
                if isinstance(x, np.generic):
                    return x.item()
                if isinstance(x, dict):
                    return {k: _to_jsonable(v) for k, v in x.items()}
                if isinstance(x, (list, tuple)):
                    return [_to_jsonable(v) for v in x]
                return x

            fname = f"{self.env_id or 'env'}_step{self.current_step}.json"
            path = log_dir / fname
            with path.open("w", encoding="utf-8") as f:
                json.dump(_to_jsonable(self.last_obs), f, ensure_ascii=False)
        except Exception:
            # 日志保存失败不影响主流程
            ...
