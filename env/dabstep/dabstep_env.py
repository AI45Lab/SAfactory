# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import os
import re
import subprocess
import time
import sys
import pandas as pd
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
import matplotlib
matplotlib.use("Agg")  # 重要：无显示环境也能渲染
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib import font_manager as fm
from io import BytesIO
import glob
import base64
from pathlib import Path
from matplotlib.font_manager import FontProperties
import matplotlib.colors as mcolors
import textwrap

_MONO_FONTS = ["DejaVu Sans Mono", "Consolas", "Courier New", "monospace"]
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[mKHFJ]")

import gymnasium as gym
import numpy as np

from core.env.base_env import BaseEnv
from core.env.env_register import register_env
from core.types.base import (
    ResetOutput,
    StepOutput,
    RenderOutput,
    PromptOutput,
    TextContent,
    OpenAIMessage,
    MessageContent,
)

# --- ensure vendored packages (agent_core) take precedence ---
_AGENT_CORE = Path(__file__).resolve().parent / "agent_core"
if _AGENT_CORE.exists():
    sys.path.insert(0, str(_AGENT_CORE))

def _fmt_safe(s: str) -> str:
    """让包含 { } 的字符串在被上游用 .format 或 f-string 包裹时也不会报错。"""
    if not s:
        return ""
    return s.replace("{", "{{").replace("}", "}}")

# -------------------------
# 默认上下文文件（可通过参数覆盖）
# -------------------------
DEFAULT_CONTEXT_FILES: List[str] = [
    "acquirer_countries.csv",
    "fees.json",
    "manual.md",
    "merchant_category_codes.csv",
    "merchant_data.json",
    "payments.csv",
]


@register_env("dabstepgym")
class DABStepEnv(BaseEnv):
    """
    DABStep 环境（TradingGym 风格）：
    - reset() -> ResetOutput
    - step(action: str) -> StepOutput   # action 为 LLM 原样字符串（JSON / 'submit k=v'）
    - get_task_prompt() -> PromptOutput
    - render() -> RenderOutput

    运行逻辑：
    1) reset 选出 task_id（HF / runner_module / 本地 tasks）
    2) step 解析 'submit'，调用 runner（优先函数，其次 CLI）
    3) split=dev 优先读取 runner 写出的 dev_metrics.json（官方评分），
       若不存在则本地近似算一次但不落盘；default 无本地分。
    4) data/context/* 自动校验/下载。
    """

    ## need to add more tasks to run
    def __init__(self, ** kwargs):
        super().__init__()
        cfg = kwargs or {}

        # --- 运行根与产物 ---
        self.root = Path(cfg.get("root", ".")).resolve()
        self.artifacts_dir = Path(
            cfg.get("artifacts_dir", self.root / "env/dabstep/artifacts")
        ).resolve()
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

        # 兼容 orchestrator 传下来的无关参数，直接忽略
        for k in ("env_num", "env_image", "env_name"):
            cfg.pop(k, None)

        # 分片/随机相关（支持用环境变量控制）
        self.num_shards     = int(cfg.get("num_shards", int(os.getenv("NUM_SHARDS", "1"))))
        self.shard_index    = int(cfg.get("shard_index", int(os.getenv("SHARD_INDEX", "0"))))
        self.tasks_shuffle  = bool(int(cfg.get("tasks_shuffle", int(os.getenv("TASKS_SHUFFLE", "0")))))
        self.seed           = int(cfg.get("seed", int(os.getenv("SEED", "42"))))

        # --- agent (OpenAI-compatible) ---
        self.agent_base_url    = cfg.get("agent_base_url", os.getenv("AGENT_BASE_URL") or os.getenv("OPENAI_API_BASE"))
        self.agent_api_key     = cfg.get("agent_api_key",  os.getenv("AGENT_API_KEY")  or os.getenv("OPENAI_API_KEY") or "EMPTY")
        self.agent_temperature = float(cfg.get("agent_temperature", os.getenv("AGENT_TEMPERATURE") or 0.0))

        # 模型名：支持 env_params.model_id / agent_model，以及环境变量 AGENT_MODEL / MODEL_NAME
        self.model_id = str(
            cfg.get("model_id",
            cfg.get("agent_model",
            os.getenv("AGENT_MODEL") or os.getenv("MODEL_NAME") or "gpt-4o-mini")))

        # --- runner（优先模块，兜底脚本） ---
        self.runner_module = cfg.get("runner_module")     # e.g. "env.bench_env.dabstep.dapsteb_runner"
        self.runner_script = cfg.get("runner_script")     # e.g. "dapsteb_runner.py"

        self._fn_load_ds = None
        self._fn_solve = None
        self._fn_score = None
        if self.runner_module:
            try:
                import importlib

                mod = importlib.import_module(self.runner_module)
                self._fn_load_ds = getattr(mod, "load_dabstep_dataset", None)
                self._fn_solve = getattr(mod, "solve_task_with_agent", None)
                self._fn_score = getattr(mod, "score_dev_all", None)
            except Exception:
                # 保持可运行，具体异常在实际调用时再暴露
                self._fn_load_ds = self._fn_load_ds or None
                self._fn_solve = self._fn_solve or None
                self._fn_score = self._fn_score or None

        # --- 运行参数 ---
        self.split = str(cfg.get("split", os.getenv("DABSTEP_SPLIT", "default")))
        self.data_dir = str(cfg.get("data_dir", cfg.get("root", ".")))# 仅用于定位 pack 根
        self.model_id = str(cfg.get("model_id", os.getenv("MODEL_NAME", "gpt-4o-mini")))
        self.max_steps = int(cfg.get("max_steps", int(os.getenv("DABSTEP_MAX_STEPS", "10"))))
        self.timeout = int(cfg.get("timeout", int(os.getenv("DABSTEP_TIMEOUT", "60"))))
        self.limit = int(cfg.get("limit", 0))  # >0 仅缓存前 N 个任务
        # 渲染尺寸与比例
        self.render_df_preview = bool(int(cfg.get("render_df_preview",
                                        int(os.getenv("RENDER_DF_PREVIEW", "0")))))

        # 列宽（字符数，用于硬换行）
        self.render_th_w   = int(cfg.get("render_th_w",   int(os.getenv("RENDER_TH_W",   "78"))))
        self.render_code_w = int(cfg.get("render_code_w", int(os.getenv("RENDER_CODE_W", "68"))))
        self.render_obs_w  = int(cfg.get("render_obs_w",  int(os.getenv("RENDER_OBS_W",  "72"))))
        self.render_col_weights = tuple(
            float(x) for x in str(
                cfg.get("render_col_weights", os.getenv("RENDER_COL_WEIGHTS", "1.05,0.95,1.10"))
            ).split(",")
        )

        # 行数上限（0或负数=不截断）
        self.render_max_thought = int(cfg.get("render_max_thought", int(os.getenv("RENDER_MAX_THOUGHT", "0"))))
        self.render_max_code    = int(cfg.get("render_max_code",    int(os.getenv("RENDER_MAX_CODE",    "0"))))
        self.render_max_obs     = int(cfg.get("render_max_obs",     int(os.getenv("RENDER_MAX_OBS",     "0"))))

        # 行高参数（越大越“高/疏”）
        self.render_header_h    = float(cfg.get("render_header_h",   float(os.getenv("RENDER_HEADER_H",   "0.52"))))
        self.render_base_row_h  = float(cfg.get("render_base_row_h", float(os.getenv("RENDER_BASE_ROW_H", "0.56"))))
        self.render_per_line_h  = float(cfg.get("render_per_line_h", float(os.getenv("RENDER_PER_LINE_H", "0.075"))))

        # 画布宽度(英寸)与 DPI；height_scale>1 就按比例“变高”；fig_scale>1 等比“放大整张图”(宽高都放大)
        self.render_fig_w_in    = float(cfg.get("render_fig_w_in",    float(os.getenv("RENDER_FIG_W_IN",    "18"))))
        self.render_dpi         = int(cfg.get("render_dpi",           int(os.getenv("RENDER_DPI",           "180"))))
        self.render_height_scale= float(cfg.get("render_height_scale",float(os.getenv("RENDER_HEIGHT_SCALE","1.0"))))
        self.render_fig_scale   = float(cfg.get("render_fig_scale",   float(os.getenv("RENDER_FIG_SCALE",   "1.0"))))
        self.render_font_scale  = float(cfg.get("render_font_scale",  float(os.getenv("RENDER_FONT_SCALE",  "1.0"))))

        # —— 全局裁剪与尺寸上限 —— 
        self.render_max_rows = int(cfg.get("render_max_rows",
                                int(os.getenv("RENDER_MAX_ROWS", "6"))))         # 最多展示多少步（0=不限制）
        self.render_tail_rows = int(cfg.get("render_tail_rows",
                                int(os.getenv("RENDER_TAIL_ROWS", "1"))))        # 末尾保留几步
        self.render_max_fig_h_in = float(cfg.get("render_max_fig_h_in",
                                    float(os.getenv("RENDER_MAX_FIG_H_IN", "14"))))  # 图片最大高度(英寸，0=不限制)
        # （可选）总行数上限：超过则在最后一行后截断并标注
        self.render_max_total_lines = int(cfg.get("render_max_total_lines",
                                    int(os.getenv("RENDER_MAX_TOTAL_LINES", "0"))))


        # —— 列配色（可覆写：RENDER_COL_COLORS="#2f6df6,#8b5cf6,#0ea5a4"）
        # 渲染：列配色（默认很淡的蓝/紫/绿）
        self.render_col_colors = tuple(
            s.strip() for s in str(
                cfg.get("render_col_colors",
                        os.getenv("RENDER_COL_COLORS", "#eef2f7,#f4f0ff,#eaf7f0"))
            ).split(",")
        )

        # 任务与上下文目录（离线
        self.context_dir = Path(cfg.get("context_dir", "pack/context"))
        self.tasks_dir   = Path(cfg.get("tasks_dir",   "pack/tasks"))
        self.required_context_files: List[str] = list(
            cfg.get("required_context_files", DEFAULT_CONTEXT_FILES)
        )

        # 任务来源：tasks 列表 > tasks_file > 本地 JSONL
        self.tasks: Optional[List[str]] = list(cfg.get("tasks", [])) or self._load_tasks_from_file(
            cfg.get("tasks_file")
        )

        # --- 空间参数 ---
        spaces_cfg = cfg.get("spaces", {}) or {}
        self.task_id_max_len = int(spaces_cfg.get("task_id_max_len", 256))
        self.status_max_len = int(spaces_cfg.get("status_max_len", 16))
        self.action_max_len = int(spaces_cfg.get("action_max_len", 8192))
        score_low, score_high = spaces_cfg.get("score_range", (-1.0, 1.0))
        self.score_low, self.score_high = float(score_low), float(score_high)

        # --- Gym spaces ---
        self.action_space = gym.spaces.Text(max_length=self.action_max_len)
        self.observation_space = gym.spaces.Dict(
            {
                "task_id": gym.spaces.Text(min_length=1, max_length=self.task_id_max_len),
                "status": gym.spaces.Text(min_length=1, max_length=self.status_max_len),
                "last_score": gym.spaces.Box(
                    low=self.score_low, high=self.score_high, shape=(), dtype=np.float32
                ),  # -1.0 表示无分
                "has_score": gym.spaces.Discrete(2),  # 0/1
            }
        )

        # --- 运行态缓存 ---
        self._cursor = -1
        self._task_ids: Optional[List[str]] = None
        self._id2task: Optional[Dict[str, Dict[str, Any]]] = None
        self._last_obs: Dict[str, Any] = {
            "task_id": "",
            "status": "idle",
            "last_score": -1.0,
            "has_score": 0,
        }

        # ---- render 用到的状态缓存 ----
        self._render_step = 0
        self._last_task_full: Optional[Dict[str, Any]] = None   # reset/step 时缓存完整任务
        self._last_run_info: Dict[str, Any] = {}                # step 后缓存本次求解信息

        # 预检查数据（缺失则下载）
        self._ensure_context_ready()

    # -------------------------
    # 生命周期
    # -------------------------
    def reset(self, seed: Optional[int] = None) -> ResetOutput:
        self._ensure_task_cache()
        if not self._task_ids:
            raise RuntimeError("No tasks available. Put JSONL under pack/tasks or provide tasks_file/tasks.")

        self._cursor = (self._cursor + 1) % len(self._task_ids)
        tid = self._task_ids[self._cursor]
        self._last_obs = {"task_id": tid, "status": "idle", "last_score": -1.0, "has_score": 0}
        self._last_task_full = self._fetch_single_task(tid, self.split)
        return ResetOutput(observation=self._last_obs, info={"split": self.split})

    def step(self, action: str) -> StepOutput:
        super().step(action=action)
        parsed = self._parse_submit_action(action, default_task_id=self._last_obs["task_id"])
        safe = action.replace(self.agent_api_key, "***") if self.agent_api_key else action
        print(f"[DABStepEnv] action <- {safe!r}")

        if parsed.get("type") != "submit":
            return StepOutput(
                observation=self._last_obs,
                reward=0.0,
                terminated=False,
                truncated=False,
                info={"error": "Invalid action. Expect JSON {\"type\":\"submit\",...} or 'submit key=value'."},
            )

        tid = str(parsed["task_id"])
        params = dict(parsed.get("params") or {})
        model_id = str(
            params.pop("model_id",
            params.pop("agent_model",
            params.pop("model", self.model_id)))
        )
        max_steps = int(params.pop("max_steps", self.max_steps))
        timeout   = int(params.pop("timeout", self.timeout))
        split     = str(params.pop("split", self.split))
        # add more args
        agent_base_url = str(
            params.pop("agent_base_url",
            params.pop("base_url",
            params.pop("openai_api_base",
                    os.getenv("AGENT_BASE_URL", os.getenv("OPENAI_API_BASE", "")))))
        )

        agent_api_key = str(
            params.pop("agent_api_key",
            params.pop("api_key",
            params.pop("openai_api_key",
                    os.getenv("AGENT_API_KEY", os.getenv("OPENAI_API_KEY", "EMPTY")))))
        )

        agent_temp_raw = params.pop("agent_temperature", params.pop("temperature", None))  # optional
        agent_temp     = None if agent_temp_raw is None else float(agent_temp_raw)

        if agent_base_url:
            self.agent_base_url = agent_base_url
        if agent_api_key:
            self.agent_api_key = agent_api_key
        if agent_temp is not None:
            self.agent_temperature = float(agent_temp)

        self._ensure_context_ready()

        # 任务与求解
        task   = self._fetch_single_task(tid, split)
        result = self._solve_one(task, model_id, max_steps, timeout, data_dir=self._pack_root(), split=split)

        solved_tid = str(result.get("task_id") or tid)
        if solved_tid != tid:
            task = self._fetch_single_task(solved_tid, split)

        # info/metrics
        info: Dict[str, Any] = {"split": split, "resolved_task_id": solved_tid}
        sub_path_str = result.get("_submission_path")
        if sub_path_str and Path(sub_path_str).exists():
            info["submission_path"] = sub_path_str

        reward = 0.0
        if split == "dev":
            mpath_str = result.get("_metrics_path")
            metrics = None
            if mpath_str and Path(mpath_str).exists():
                info["dev_metrics_path"] = mpath_str
                try:
                    metrics = json.loads(Path(mpath_str).read_text(encoding="utf-8"))
                except Exception:
                    metrics = None
            if metrics is None:
                metrics = self._score_dev_one([task], [result])
            reward = float(metrics.get("accuracy") or 0.0)
            info["dev_metrics"] = metrics
            info["score"] = reward
            info["scorer"] = metrics.get("scorer")

        # 缓存运行信息
        self._last_run_info = {
            "agent_answer":    str(result.get("agent_answer", "")),
            "status":          str(result.get("status", "")),
            "submission_path": result.get("_submission_path") or info.get("submission_path"),
            "metrics_path":    result.get("_metrics_path") or info.get("dev_metrics_path"),
            "metrics":         info.get("dev_metrics"),
            "split":           split,
            "task_id":         solved_tid,
            "trace":           result.get("trace"),
            "log_path":        result.get("_log_path"),
            "artifact_dir":    result.get("_artifact_dir"),
        }
        self._last_task_full = self._fetch_single_task(solved_tid, split)

        # 更新观测
        self._last_obs = {
            "task_id": solved_tid,
            "status": "done",
            "last_score": reward if split == "dev" else -1.0,
            "has_score": 1 if split == "dev" else 0,
        }

        # ★ 生成并落盘 OpenAI 风格对话
        try:
            msgs = self._build_openai_messages()
            art_dir = self._last_run_info.get("artifact_dir")
            if art_dir:
                art_dir_p = Path(art_dir)
                art_dir_p.mkdir(parents=True, exist_ok=True)
                out_json = art_dir_p / "openai_messages.json"
                out_json.write_text(json.dumps(msgs, ensure_ascii=False, indent=2), encoding="utf-8")
                self._last_run_info["openai_messages_path"] = str(out_json)
                info["openai_messages_path"] = str(out_json)  # ★ 暴露给调用方
        except Exception:
            pass

        return StepOutput(
            observation=self._last_obs,
            reward=reward,
            terminated=True,
            truncated=False,
            info=info,
        )


    # -------------------------
    # PUI 提示与渲染
    # -------------------------
    def _build_openai_messages(self) -> List[Dict[str, str]]:
        tid   = self._last_obs.get("task_id", "") or ""
        split = self._last_run_info.get("split", self.split)

        task     = self._last_task_full or {}
        question = self._truncate(task.get("question", ""), 2000)
        #context  = self._truncate(task.get("context", ""), 2000)
        ans_fmt  = self._truncate(task.get("answer_format", ""), 800)
        gt       = self._truncate(task.get("answer", ""), 400) if split == "dev" else None

        agent_answer = self._truncate(self._last_run_info.get("agent_answer", ""), 400)
        metrics = self._last_run_info.get("metrics") or {}
        score   = metrics.get("accuracy")
        scorer  = metrics.get("scorer")

        ctx_dir = str(self._context_path().resolve())
        try:
            available = ", ".join(sorted([p.name for p in self._context_path().iterdir() if p.is_file()]))
        except Exception:
            available = ""

        messages: List[Dict[str, str]] = []
        # 与你示例一致：先 system，再 user（包含 Context+Question）
        messages.append({"role": "system", "content": f"DABStep (task_id={tid}, split={split})."})
        user_block = f"Context:\nData files at:\n{ctx_dir}"
        if available:
            user_block += f"\nAvailable: {available}"
        user_block += f"\n\nQuestion:\n{question or '(empty)'}"
        if ans_fmt:
            user_block += f"\n\nAnswer format:\n{ans_fmt}"
        messages.append({"role": "user", "content": user_block})

        # 拼入逐步 trace（assistant 思考/代码 → user 执行反馈）
        trace = self._last_run_info.get("trace") or []
        for ev in trace:
            a = (ev.get("assistant_message") or "").strip()
            if a:
                messages.append({"role": "assistant", "content": a})
            if ev.get("code"):
                messages.append({"role": "assistant", "content": f"```python\n{ev['code']}\n```"})
            ex = ev.get("exec")
            if isinstance(ex, dict):
                ok = "succeeded" if ex.get("success") else "failed"
                obs = f"Execution {ok}.\nOutput:\n{ex.get('output','')}"
                if ex.get("error"):
                    obs += f"\nError: {ex['error']}"
                messages.append({"role": "user", "content": obs})
            if ev.get("final_answer"):
                messages.append({"role": "assistant", "content": f"FINAL ANSWER: {ev['final_answer']}"})

        # 没有 trace 也至少给最终答案
        if not trace and agent_answer:
            messages.append({"role": "assistant", "content": f"FINAL ANSWER: {agent_answer}"})

        # dev 评分摘要
        if split == "dev" and metrics:
            messages.append({"role": "assistant", "content": f"[dev score: {float(score):.3f} via {scorer}]"})
            if gt is not None:
                messages.append({"role": "assistant", "content": f"[ground_truth: {gt}]"})

        return messages


    def get_task_prompt(self) -> PromptOutput:
        tid = self._last_obs.get("task_id", "")

        # —— 纯文案（先写正常字符串，再做 format-safe）——
        schema_txt = (
            "You are in the DABStep environment.\n"
            "Reply with JSON ONLY (no code blocks, no prose). Use this schema:\n"
            '{"type":"submit","task_id":"<id>","params":{'
            '"model_id":"...","max_steps":10,"timeout":60,'
            '"agent_base_url":"http://host:port/v1","agent_api_key":"EMPTY","agent_temperature":0.0'
            "}}\n"
            # 这里明确：split 可不传，环境会用 YAML 里的 split（如 dev）
            "You may omit \"split\"; the environment will use its configured split.\n"
            f'Defaults if omitted: model_id="{self.model_id}", '
            f'max_steps={self.max_steps}, timeout={self.timeout}, '
            f'agent_base_url="{self.agent_base_url or ""}", agent_api_key="***", '
            f'agent_temperature={self.agent_temperature}.\n'
            "Aliases supported: base_url/api_key/temperature.\n"
            # 注意：下面这条只是示例，不一定要用；我们仍然做 format-safe
            'Minimal example: {"type":"submit","task_id":"<id>","params":{}}\n'
        )
        # 关键：做成 format-safe
        schema_txt = _fmt_safe(schema_txt)

        example_user = (
            f"Current task_id: {tid}\n"
            f'Example JSON: {{"type":"submit","task_id":"{tid}","params":{{}}}}'
        )
        example_user = _fmt_safe(example_user)

        system = OpenAIMessage(
            role="system",
            content=[MessageContent(root=TextContent(text=schema_txt))]
        )
        user = OpenAIMessage(
            role="user",
            content=[MessageContent(root=TextContent(text=example_user))]
        )
        return PromptOutput(system_message=system, user_message=user)

    
    def _set_mono_font(self) -> None:
        try:
            chosen = None
            for fam in _MONO_FONTS:
                try:
                    fm.findfont(fam, fallback_to_default=False)
                    chosen = fam; break
                except Exception:
                    continue
            if chosen is None:
                chosen = "DejaVu Sans Mono"
            plt.rcParams["font.family"] = chosen
            plt.rcParams["font.monospace"] = [chosen]
            plt.rcParams["axes.unicode_minus"] = False
            self._mono_font = chosen          # ★ 记录，后面每格都用
        except Exception:
            self._mono_font = "DejaVu Sans Mono"

    @staticmethod
    def _tint(hex_color: str, factor: float = 0.90) -> str:
        """
        把颜色往白色提亮，factor∈(0,1]，越小越淡。输入可以是 #rrggbb。
        """
        rgb = np.array(mcolors.to_rgb(hex_color))
        out = 1.0 - (1.0 - rgb) * factor
        return mcolors.to_hex(np.clip(out, 0, 1))

    def _wrap_to_ax(self, ax, text: str, fontsize: float,
                    mono: bool = True, pad_rel: float = 0.04,
                    max_lines: Optional[int] = None) -> str:
        """
        按“实际像素宽度”换行：
        - 先用 renderer 取到这个 Axes 的像素宽度（减去左右 padding）；
        - 用 renderer 估一个等宽字符的像素宽度；
        - 计算能容纳的最大字符数，然后 textwrap.wrap。
        """
        if not text:
            return ""
        fig = ax.figure
        renderer = fig.canvas.get_renderer()
        bbox = ax.get_window_extent(renderer=renderer)
        usable_px = max(10.0, bbox.width * (1.0 - 2.0 * pad_rel))

        fp = FontProperties(family=(self._mono_font if mono else None), size=fontsize)
        w_px, _, _ = renderer.get_text_width_height_descent("M" * 16, fp, ismath=False)
        char_px = max(5.0, w_px / 16.0)  # 单字符像素宽

        width_chars = max(1, int(usable_px // char_px))

        lines: List[str] = []
        for ln in str(text).splitlines():
            wrapped = textwrap.wrap(
                ln, width=width_chars,
                break_long_words=True, drop_whitespace=False
            )
            lines += (wrapped or [""])
        if max_lines and max_lines > 0 and len(lines) > max_lines:
            lines = lines[:max_lines] + ["…(truncated)"]
        return "\n".join(lines)

    @staticmethod
    def _strip_ansi(s: str) -> str:
        return _ANSI_RE.sub("", s or "")

    @staticmethod
    def _hard_wrap(text: str, width: int) -> str:
        if not text:
            return ""
        lines = []
        for ln in text.splitlines():
            for i in range(0, len(ln), width):
                lines.append(ln[i:i+width])
        return "\n".join(lines)

    @staticmethod
    def _wrap_mono_keep_spaces(s: str, width: int, max_lines: int) -> str:
        s = DABStepEnv._strip_ansi(s)
        if not s:
            return ""
        body = DABStepEnv._hard_wrap(s, width)
        lines = body.splitlines()
        # 0 或负数 → 不截断；正数才截断
        if max_lines and max_lines > 0 and len(lines) > max_lines:
            lines = lines[:max_lines] + ["…(truncated)"]
        return "\n".join(lines)
    
    @staticmethod
    def _hex_to_rgb(h: str):
        c = c.strip()
        if c.startswith("#"): c = c[1:]
        if len(c) == 3: c = "".join(ch*2 for ch in c)
        r = int(c[0:2], 16) / 255.0
        g = int(c[2:4], 16) / 255.0
        b = int(c[4:6], 16) / 255.0
        return (r, g, b)

    @staticmethod
    def _rgb_to_hex(rgb):
        r, g, b = rgb
        return "#{:02x}{:02x}{:02x}".format(int(round(r*255)), int(round(g*255)), int(round(b*255)))

    @classmethod
    def _tint(self, base: str, mix: float = 0.9) -> str:
        """与白色按比例混合，mix 越大越浅；出错时原样返回。"""
        try:
            r, g, b = self._hex_to_rgb(str(base))
            r = r + (1 - r) * mix
            g = g + (1 - g) * mix
            b = b + (1 - b) * mix
            return self._rgb_to_hex((r, g, b))
        except Exception:
            return str(base)

    def _trace_to_rows(self, max_thought=None, max_code=None, max_obs=None):
        TH_W, CODE_W, OBS_W = self.render_th_w, self.render_code_w, self.render_obs_w
        if max_thought is None: max_thought = self.render_max_thought
        if max_code   is None: max_code   = self.render_max_code
        if max_obs    is None: max_obs    = self.render_max_obs

        trace = self._last_run_info.get("trace") or []
        rows = []
        for ev in trace:
            thought = (ev.get("assistant_message") or "").strip()
            code    = (ev.get("code") or "").strip()
            ex      = ev.get("exec") or {}
            status  = "OK" if isinstance(ex, dict) and ex.get("success") else ("FAILED" if ex else "")
            obs_txt = ""
            if isinstance(ex, dict):
                out = (ex.get("output") or "").strip()
                err = (ex.get("error") or "").strip()
                if err:
                    out = (out + ("\n" if out and err else "") + "Error: " + err).strip()
                obs_txt = (f"[{status}] " if status else "") + out
            if code:
                code = "\n".join(f"{i+1:>3}  {ln}" for i, ln in enumerate(code.splitlines()))
            rows.append({
                "thought": self._wrap_mono_keep_spaces(thought, TH_W,   max_thought),
                "code":    self._wrap_mono_keep_spaces(code,    CODE_W, max_code),
                "obs":     self._wrap_mono_keep_spaces(obs_txt, OBS_W,  max_obs),
            })
        return rows
    
    def _apply_row_caps(self, rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """按 render_max_rows / render_tail_rows 裁剪步骤行数。"""
        mr = getattr(self, "render_max_rows", 0)
        tail = max(0, getattr(self, "render_tail_rows", 0))
        if mr and len(rows) > mr:
            head = mr - (1 if tail > 0 else 0) - tail
            head = max(0, head)
            kept = rows[:head]
            hidden = len(rows) - head - tail
            kept.append({"thought": f"… ({hidden} steps hidden)", "code": "", "obs": ""})
            if tail > 0:
                kept += rows[-tail:]
            rows = kept
        return rows


    def render(self) -> RenderOutput:
        # 1) 字体与步号
        self._set_mono_font()
        self._render_step += 1
        # 每列颜色（头部/边框/斑马底色）
        col_theme = []
        for base in self.render_col_colors:
            col_theme.append({
                "head_bg": self._tint(base, 0.88),
                "edge":    self._tint(base, 0.70),
                "even_bg": self._tint(base, 0.95),
                "odd_bg":  self._tint(base, 0.92),
            })

        # 2) 元信息
        tid   = self._last_obs.get("task_id", "") or ""
        split = self._last_run_info.get("split", self.split)
        metrics = self._last_run_info.get("metrics") or {}
        score   = metrics.get("accuracy"); scorer = metrics.get("scorer")

        # 3) 折 trace → 三列文本
        rows = self._trace_to_rows()  # 宽度/行数限制用 __init__ 里的 render_* 配置
        rows = self._apply_row_caps(rows)
        if not rows:
            task  = self._last_task_full or {}
            question = str(task.get("question",""))
            aa = str(self._last_run_info.get("agent_answer",""))
            rows = [{
                "thought": self._wrap_mono_keep_spaces(question, self.render_th_w,  self.render_max_thought),
                "code":    "",
                "obs":     self._wrap_mono_keep_spaces(aa,       self.render_obs_w, self.render_max_obs),
            }]

        # 4) 行高自适应
        def _nlines(s: str) -> int: return (s.count("\n") + 1) if s else 1
        row_lines = [max(_nlines(r["thought"]), _nlines(r["code"]), _nlines(r["obs"])) for r in rows]

        # 5) （可选）底部 DF 预览
        df, df_path = None, None
        if getattr(self, "render_df_preview", False):
            try:
                cands = []
                adir = self._last_run_info.get("artifact_dir")
                if adir and os.path.isdir(adir):
                    cands += glob.glob(os.path.join(adir, "*.csv"))
                cands += glob.glob(str(self._context_path() / "*.csv"))
                for p in cands:
                    try:
                        dfx = pd.read_csv(p)
                        df  = dfx.head(10).copy()
                        df  = df.iloc[:, :8]
                        df.columns = [str(c)[:24] for c in df.columns]
                        df_path = p
                        break
                    except Exception:
                        continue
            except Exception:
                pass

        # 6) 画布尺寸/字号（全部用配置项）
        header_h   = float(self.render_header_h)
        base_row_h = float(self.render_base_row_h)
        per_line_h = float(self.render_per_line_h)
        main_hs    = [base_row_h + per_line_h*(L-1) for L in row_lines]
        extra_df_h = 1.6 if df is not None else 0.0

        fig_w = float(self.render_fig_w_in) * float(self.render_fig_scale)
        fig_h = (1.6 + header_h + sum(main_hs) + extra_df_h) \
                * float(self.render_height_scale) * float(self.render_fig_scale)

        fig = plt.figure(figsize=(fig_w, fig_h), dpi=int(self.render_dpi))
        fig.patch.set_facecolor("#ffffff")
        plt.subplots_adjust(top=0.965, bottom=0.065)

        fs = float(self.render_font_scale or 1.0)
        title_fs, header_fs, body_fs, footer_fs = 14*fs, 11*fs, 9*fs, 8*fs

        # 估计高度
        fig_w = float(self.render_fig_w_in) * float(self.render_fig_scale)
        fig_h = (1.6 + header_h + sum(main_hs) + extra_df_h) \
                * float(self.render_height_scale) * float(self.render_fig_scale)

        # —— 最大高度钳制（超限时整体收紧，避免巨图）
        max_h = float(getattr(self, "render_max_fig_h_in", 0.0))
        if max_h and fig_h > max_h:
            scale = max_h / fig_h
            fig_h = max_h
            # 收紧行距与字号（保底值防止过小）
            per_line_h *= scale
            title_fs   = max(10, title_fs * (scale ** 0.9))
            header_fs  = max( 8, header_fs * (scale ** 0.9))
            body_fs    = max( 7, body_fs   * (scale ** 0.9))

        # 7) 标题
        title = f"DABStep • Task #{tid} (split={split})"
        if score is not None:
            try:
                title += f"    |    dev score: {float(score):.3f} via {scorer}"
            except Exception:
                title += f"    |    dev score: {score} via {scorer}"
        fig.suptitle(title, x=0.02, ha="left", y=0.988, fontsize=title_fs, fontweight="bold")

        # [ADD] 若你没定义过字号缩放，放这里
        header_fs = int(11 * getattr(self, "render_font_scale", 1.0))
        body_fs   = int( 9 * getattr(self, "render_font_scale", 1.0))

        # [ADD] 每列颜色（头部/边框/斑马底色）
        col_theme = []
        for base in self.render_col_colors:
            theme = {
                "head_bg": self._tint(base, 0.82),  # 头部浅底
                "edge":    self._tint(base, 0.55),  # 边框稍深
                "even_bg": self._tint(base, 0.94),  # 行斑马（偶数）
                "odd_bg":  self._tint(base, 0.90),  # 行斑马（奇数）
            }
            col_theme.append(theme)

        # 8) 三列网格
        height_ratios = [header_h] + main_hs + ([extra_df_h] if df is not None else [])
        gs = fig.add_gridspec(
            nrows=len(height_ratios),
            ncols=3,
            width_ratios=list(self.render_col_weights),
            height_ratios=height_ratios,
            hspace=0.18, wspace=0.06
        )

        # 9) 统一的单元格绘制：自动换行 + 圆角裁剪（所有列都生效）
        def draw_cell(ax, title, body, bg="#f9fafb", edge="#d9dee3", mono=True):
            ax.set_xticks([]); ax.set_yticks([]); ax.set_frame_on(False)
            ax.set_xlim(0,1); ax.set_ylim(0,1)

            box = FancyBboxPatch(
                (0,0),1,1, boxstyle="round,pad=0.012,rounding_size=0.02",
                fc=bg, ec=edge, lw=1.0
            )
            ax.add_patch(box)

            # 标题（柔和）
            if title:
                ax.text(0.02, 0.95, title, ha="left", va="top",
                        fontsize=header_fs, fontweight="bold",
                        transform=ax.transAxes, clip_on=True)

            # 正文：按像素宽度包裹后再画，且裁剪到圆角框
            if body:
                wrapped = self._wrap_to_ax(
                    ax, body, fontsize=body_fs, mono=mono,
                    pad_rel=0.04,  # 与 draw 的左右内边距一致
                    max_lines=None if mono else None
                )
                kw_font = {"fontfamily": getattr(self, "_mono_font", None)} if mono else {}
                t = ax.text(0.02, 0.90, wrapped, ha="left", va="top",
                            fontsize=body_fs, linespacing=1.35,
                            transform=ax.transAxes, clip_on=True, **kw_font)
                t.set_clip_path(box.get_path(), box.get_transform())

        # 表头
        ax_h1 = fig.add_subplot(gs[0,0]); ax_h2 = fig.add_subplot(gs[0,1]); ax_h3 = fig.add_subplot(gs[0,2])
        heads = ("Thought", "Code", "Observation / Output")
        for col, (ax, txt) in enumerate(((ax_h1, heads[0]), (ax_h2, heads[1]), (ax_h3, heads[2]))):
            th = col_theme[col]
            draw_cell(ax, txt, "", bg=th["head_bg"], edge=th["edge"], mono=False)

        # 每步一行（按列主题上色）
        for i, r in enumerate(rows, start=1):
            th0, th1, th2 = col_theme[0], col_theme[1], col_theme[2]
            bg0 = th0["even_bg"] if (i % 2 == 0) else th0["odd_bg"]
            bg1 = th1["even_bg"] if (i % 2 == 0) else th1["odd_bg"]
            bg2 = th2["even_bg"] if (i % 2 == 0) else th2["odd_bg"]

            ax_t = fig.add_subplot(gs[i,0]); draw_cell(ax_t, None, r["thought"], bg=bg0, edge=th0["edge"], mono=True)
            ax_c = fig.add_subplot(gs[i,1]); draw_cell(ax_c, None, r["code"] or "(no code)", bg=bg1, edge=th1["edge"], mono=True)
            ax_o = fig.add_subplot(gs[i,2]); draw_cell(ax_o, None, r["obs"]  or "(no output)", bg=bg2, edge=th2["edge"], mono=True)

        # （可选）底部 DataFrame
        if df is not None:
            ax_df = fig.add_subplot(gs[len(height_ratios)-1, :])
            ax_df.set_xticks([]); ax_df.set_yticks([]); ax_df.set_frame_on(False)
            ax_df.set_xlim(0,1); ax_df.set_ylim(0,1)
            box = FancyBboxPatch((0,0),1,1, boxstyle="round,pad=0.012,rounding_size=0.02",
                                fc="#fffdf7", ec="#e0c080", lw=1.1)
            ax_df.add_patch(box)
            ax_df.text(0.02, 0.95, "DataFrame Preview", ha="left", va="top",
                    fontsize=header_fs, fontweight="bold", transform=ax_df.transAxes)
            sub = ax_df.inset_axes([0.02, 0.10, 0.96, 0.78]); sub.axis("off")
            nrows, ncols = df.shape
            zebra_even, zebra_odd, header_colour = "#ffffff", "#f6f8fa", "#e9edf5"
            cell_colours  = [[zebra_even if (r % 2 == 0) else zebra_odd for _ in range(ncols)] for r in range(nrows)]
            tbl = sub.table(cellText=df.values, colLabels=df.columns.tolist(),
                            cellColours=cell_colours, colColours=[header_colour]*ncols, loc="center")
            tbl.auto_set_font_size(False); tbl.set_fontsize(body_fs); tbl.scale(1.0, 1.08)
            if df_path:
                ax_df.text(0.02, 0.07, f"source: {Path(df_path).name}", ha="left", va="bottom",
                        fontsize=footer_fs, color="#666", transform=ax_df.transAxes)

        # 10) 页脚
        ctx_dir = str(self._context_path().resolve())
        try:
            available = ", ".join(sorted([p.name for p in self._context_path().iterdir() if p.is_file()]))
        except Exception:
            available = ""
        footer = f"context: {ctx_dir}" + (f" | files: {available}" if available else "")
        fig.text(0.02, 0.012, self._wrap_mono_keep_spaces(footer, 160, 2),
                ha="left", va="bottom", fontsize=footer_fs, color="#555")

        # —— 导出到内存
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
        img_bytes = buf.getvalue()
        buf.close()

        # —— 自行落盘到 artifacts（不依赖 orchestrator）
        art_dir = self._last_run_info.get("artifact_dir") or str(self.artifacts_dir)
        out_dir = Path(art_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"render_step_{self._render_step:02d}.png"
        out_file.write_bytes(img_bytes)

        # —— 可选：如果你希望“也复制”到一个统一目录（比如 visualize\bench_env），
        # 可通过环境变量控制（不改 base_eval）：
        vis_dir = os.getenv("VISUAL_SAVE_PATH")
        if vis_dir:
            Path(vis_dir).mkdir(parents=True, exist_ok=True)
            # 目标文件名可带上 task_id 便于区分
            tid = self._last_obs.get("task_id", "unknown")
            vis_file = Path(vis_dir) / f"{tid}_step_{self._render_step:02d}.png"
            try:
                # 复制
                with open(vis_file, "wb") as f:
                    f.write(img_bytes)
            except Exception:
                pass  # 复制失败不影响主流程

        # —— 返回时把三种字段都带上，最大化兼容
        return RenderOutput(
            step=self._render_step,
            image_data=img_bytes,
            image_base64=base64.b64encode(img_bytes).decode("utf-8"),
            image_path=str(out_file),
            text_content=None,
            text_dict=None
        )


    # ---------- 内部：任务/执行 ----------
    def _ensure_task_cache(self):
        if self._task_ids is not None:
            return

        # 1) 基础任务列表
        if self.tasks:
            # 传进来的是 ID 列表 → 只记录 task_id，不构建 _id2task（保持 None，后面按需加载完整任务）
            tasks_objs = [{"task_id": str(t)} for t in self.tasks]
            looks_full = False
        else:
            # 从本地 jsonl 读取的通常是完整对象
            tasks_objs = self._load_tasks()
            looks_full = True

        # 2) 随机化（可选）
        if getattr(self, "tasks_shuffle", False):
            import random
            rnd = random.Random(getattr(self, "seed", 42))
            rnd.shuffle(tasks_objs)

        # 3) 分片（可选）
        ns = max(1, int(getattr(self, "num_shards", 1)))
        si = max(0, int(getattr(self, "shard_index", 0)))
        if ns > 1:
            tasks_objs = [t for i, t in enumerate(tasks_objs) if (i % ns) == si]

        # 4) limit（仍保留）
        lim = int(getattr(self, "limit", 0))
        if lim > 0:
            tasks_objs = tasks_objs[:lim]

        # 5) 建立缓存
        self._task_ids = [str(t["task_id"]) for t in tasks_objs]
        # 只有“看起来是完整任务对象”时才建 _id2task；纯 ID 模式下保持 None，后续 _fetch_single_task 再加载
        self._id2task = {str(t["task_id"]): t for t in tasks_objs} if looks_full else None

    def _load_tasks(self) -> List[Dict[str, Any]]:
        # 1) 本地 JSONL（离线首选）
        tdir = self._tasks_path()
        f = tdir / f"{self.split}_tasks.jsonl"
        if f.exists():
            rows = []
            for line in f.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rows.append(json.loads(line))
            return rows

        # 2) runner 模块（可选）
        if self._fn_load_ds:
            try:
                return list(self._fn_load_ds(split=self.split, limit=None, data_dir=str(self._pack_root())))
            except TypeError:
                # 某些实现没有 data_dir 形参
                return list(self._fn_load_ds(split=self.split, limit=None))
            except Exception as e:
                raise RuntimeError(f"runner_module.load_dabstep_dataset failed: {e}")

        raise FileNotFoundError(f"No local tasks at {f}. Please put JSONL under pack/tasks.")

    def _fetch_single_task(self, task_id: str, split: str) -> Dict[str, Any]:
        if self._id2task and task_id in self._id2task:
            t = self._id2task[task_id]
            # 没有关键字段 → 触发回载（适配 self.tasks 传纯 ID 的情况）
            if any(k in t for k in ("question", "context", "answer_format", "answer")):
                return t
            else:
                self._id2task = None  # 强制重建

        tasks = self._load_tasks()
        self._id2task = {str(x["task_id"]): x for x in tasks}
        return self._id2task.get(task_id, {"task_id": task_id, "question": "", "context": "", "answer_format": ""})

    def _solve_one(self, task: Dict[str, Any], model_id: str, max_steps: int, timeout: int,
                   data_dir: str, split: str) -> Dict[str, Any]:
        # 模块函数优先
        if self._fn_solve:
            try:
                r = self._fn_solve(task, model_id=model_id, max_steps=max_steps, timeout=timeout, data_dir=data_dir)
                result_dict = {
                    "task_id": str(r.task_id),
                    "agent_answer": str(r.agent_answer),
                    "status": r.status,
                    "_raw": r,
                    "trace": getattr(r, "trace", None),  # ★ 保存 trace
                }
                # 记录到 _last_run_info（render 会用）
                self._last_run_info["trace"] = result_dict["trace"]
                out_dir = self._episode_outdir(str(task.get("task_id", "unknown")))
                out_dir.mkdir(parents=True, exist_ok=True)
                result_dict["_artifact_dir"] = str(out_dir)
                return result_dict
            except Exception as e:
                raise RuntimeError(f"runner_module.solve_task_with_agent failed: {e}")

        # CLI 兜底：在本 env 中创建一次性产物目录
        if not self.runner_script:
            raise RuntimeError("No runner_module or runner_script configured.")

        out_dir = self._episode_outdir(str(task.get("task_id", "unknown")))
        out_dir.mkdir(parents=True, exist_ok=True)
        sub_path     = out_dir / "submission.jsonl"
        metrics_path = out_dir / "dev_metrics.json"

        # 把 pack 根作为 data_dir 传下去（runner 仍可用 data_dir/context/...）
        runner_path = Path(self.runner_script)
        if not runner_path.is_absolute():
            runner_path = (Path(self.root) / runner_path).resolve()
        pack_root = self._pack_root().resolve()

        selected_tid = str(task.get("task_id", "unknown"))

        cmd = [
            sys.executable, str(runner_path),
            "--model-id",  model_id,
            "--max-steps", str(max_steps),
            "--timeout",   str(timeout),
            "--split",     str(split),
            "--data-dir",  str(pack_root),
            "--dev-metrics-out", str(metrics_path),
            "--out",       str(sub_path),
            "--temperature", str(self.agent_temperature),
            "--only-task-id", selected_tid,
        ]

        if self.agent_base_url:
            cmd += ["--base-url", self.agent_base_url, "--agent-base-url", self.agent_base_url]
        if self.agent_api_key:
            cmd += ["--api-key", self.agent_api_key, "--agent-api-key", self.agent_api_key]
        if self.agent_temperature is not None:
            t = str(self.agent_temperature)
            cmd += ["--temperature", t, "--agent-temperature", t]

        child_env = os.environ.copy()
        # 注入两套环境变量，兼容新老 runner 行为
        if self.agent_base_url:
            child_env["AGENT_BASE_URL"]   = str(self.agent_base_url)
            child_env["OPENAI_API_BASE"]  = str(self.agent_base_url)
        if self.agent_api_key:
            child_env["AGENT_API_KEY"]    = str(self.agent_api_key)
            child_env["OPENAI_API_KEY"]   = str(self.agent_api_key)
        child_env["AGENT_TEMPERATURE"] = str(self.agent_temperature)

        print(
        f"[DABStepEnv] launch runner: base_url={self.agent_base_url!r}, "
        f"api_key=***{self.agent_api_key[-4:] if self.agent_api_key else ''}, "
        f"cwd={out_dir}"
        )

        proc = subprocess.run(cmd, cwd=str(out_dir), check=False, env=child_env, capture_output=True, text=True)
        if proc.returncode != 0:
            log_tail = self._tail_file(str(out_dir / "dabstep_run.log"), n=80) or ""
            stderr_t = (proc.stderr or "").strip()
            stdout_t = (proc.stdout or "").strip()
            raise RuntimeError(
                "Runner process failed "
                f"(rc={proc.returncode}).\nSTDERR:\n{stderr_t}\n\nSTDOUT:\n{stdout_t}\n\nLOG TAIL:\n{log_tail}"
            )

        if not sub_path.exists():
            raise RuntimeError(f"Runner did not produce submission file: {sub_path}")
        
        trace_path = out_dir / "trace.jsonl"              # ★ 与 runner 写的是同名
        trace = None
        if trace_path.exists():
            trace = [json.loads(x) for x in trace_path.read_text(encoding="utf-8").splitlines() if x.strip()]

        line = sub_path.read_text(encoding="utf-8").splitlines()[0]
        obj  = json.loads(line)
        return {
            "task_id": str(obj.get("task_id")),
            "agent_answer": str(obj.get("agent_answer")),
            "status": "success",
            "trace": trace,
            "_artifact_dir":   str(out_dir),
            "_submission_path": str(sub_path),
            "_metrics_path":    str(metrics_path),
            "_log_path":        str(out_dir / "dabstep_run.log"),
        }

    # ---------- 评分（近似兜底） ----------
    def _score_dev_one(self, tasks: List[Dict[str, Any]], results: List[Dict[str, Any]]) -> Dict[str, Any]:
        if self._fn_score:
            try:
                adapted: List[Any] = []
                for it in results:
                    if isinstance(it, dict) and it.get("_raw") is not None:
                        adapted.append(it["_raw"])
                    elif hasattr(it, "task_id") and hasattr(it, "agent_answer"):
                        adapted.append(it)
                    elif isinstance(it, dict):
                        adapted.append(SimpleNamespace(
                            task_id=str(it.get("task_id","")),
                            agent_answer=str(it.get("agent_answer","")),
                        ))
                    else:
                        raise TypeError(f"Unsupported result item type: {type(it)}")
                return self._fn_score(tasks, adapted)
            except Exception:
                pass

        if not tasks or not results:
            return {"accuracy": 0.0, "details": [], "scorer": "approx"}

        t = tasks[0]; r = results[0]
        gt = str(t.get("answer", "")).strip()
        pred = str((r.get("agent_answer") if isinstance(r, dict) else getattr(r, "agent_answer","")) or "").strip()
        if not gt:
            return {"accuracy": 0.0, "details":[{"task_id": t.get("task_id"), "score": False}], "scorer":"approx"}

        ok = self._approx_equal(gt, pred)
        return {"accuracy": 1.0 if ok else 0.0,
                "details":[{"task_id": t.get("task_id"), "score": bool(ok), "pred": pred, "gt": gt}],
                "scorer":"approx"}

    @staticmethod
    def _approx_equal(gt: str, pred: str) -> bool:
        def _try_float(s: str):
            try: return float(s)
            except Exception: return None
        def _norm_str(s: str) -> str:
            return re.sub(r"\s+", "", str(s)).strip().lower()
        def _norm_ans(s: str):
            s = str(s).strip()
            if "," in s:
                items = [x.strip() for x in s.split(",") if x.strip()]
                floats, all_float = [], True
                for it in items:
                    v = _try_float(it)
                    if v is None: all_float = False; break
                    floats.append(round(v, 8))
                return ("list_float", tuple(floats)) if all_float else ("list_str", tuple(_norm_str(it) for it in items))
            v = _try_float(s)
            return ("float", round(v, 8)) if v is not None else ("str", _norm_str(s))
        def _close(a: float, b: float, tol: float = 1e-6) -> bool:
            return abs(a - b) <= tol
        ta, tb = _norm_ans(gt), _norm_ans(pred)
        if ta[0] != tb[0]:
            return _norm_str(gt) == _norm_str(pred)
        if ta[0] == "float": return _close(ta[1], tb[1])
        if ta[0] == "list_float":
            if len(ta[1]) != len(tb[1]): return False
            return all(_close(x, y) for x, y in zip(ta[1], tb[1]))
        return ta[1] == tb[1]

    # -------------------------
    # 内部：动作解析与产物
    # -------------------------
    def _parse_submit_action(self, s: str, default_task_id: str) -> Dict[str, Any]:
        s = (s or "").strip()
        try:
            obj = json.loads(s)
            if isinstance(obj, dict) and obj.get("type") == "submit":
                if not obj.get("task_id"):
                    obj["task_id"] = default_task_id
                return obj
        except Exception:
            pass
        m = re.match(r"(?i)submit\s+(.*)$", s)
        if m:
            params: Dict[str, Any] = {}
            for k, v in re.findall(r'([\w\-]+)\s*=\s*(".*?"|\'.*?\'|[^\s]+)', m.group(1)):
                if v and (v[0] in "\"'") and v[-1] == v[0]:
                    v = v[1:-1]
                params[k] = v
            task_id = str(params.pop("task_id", default_task_id))
            return {"type": "submit", "task_id": task_id, "params": params}
        if s.lower().startswith("submit"):
            return {"type": "submit", "task_id": default_task_id, "params": {}}
        return {}

    def _episode_outdir(self, task_id: str) -> Path:
        ts = time.strftime("%Y%m%d_%H%M%S")
        return self.artifacts_dir / f"dabstep_{ts}_{task_id}"

    @staticmethod
    def _load_tasks_from_file(path: Optional[str]) -> Optional[List[str]]:
        if not path: return None
        p = Path(path)
        if not p.exists(): return None
        return [line.strip() for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]

    def _context_path(self) -> Path:
        base = Path(self.data_dir)
        ctx = self.context_dir if self.context_dir.is_absolute() else (base / self.context_dir)
        ctx.mkdir(parents=True, exist_ok=True)
        return ctx

    def _tasks_path(self) -> Path:
        base = Path(self.data_dir)
        tdir = self.tasks_dir if self.tasks_dir.is_absolute() else (base / self.tasks_dir)
        tdir.mkdir(parents=True, exist_ok=True)
        return tdir

    def _pack_root(self) -> Path:
        # pack 根目录 = context 的上一级
        return self._context_path().parent

    def _ensure_context_ready(self):
        ctx = self._context_path()
        missing = [name for name in self.required_context_files if not (ctx / name).exists()]
        if missing:
            raise FileNotFoundError(f"Missing offline context files under {ctx}: {missing}")

    def _truncate(self, s: Optional[str], limit: int = 1200) -> str:
        if not s: return ""
        s = str(s)
        return s if len(s) <= limit else (s[:limit] + f"... [truncated, {len(s)-limit} more chars]")

    def _tail_file(self, path: Optional[str], n: int = 40) -> Optional[str]:
        if not path: return None
        try:
            p = Path(path)
            if not p.exists(): return None
            lines = p.read_text(encoding="utf-8", errors="ignore").splitlines()
            tail = lines[-n:] if len(lines) > n else lines
            return "\n".join(tail)
        except Exception:
            return None

    def close(self) -> None:
        super().close()
