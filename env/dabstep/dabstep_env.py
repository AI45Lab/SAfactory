# -*- coding: utf-8 -*-
from __future__ import annotations

"""
DABStepEnv: Environment = runner (monolithic)
- reset(): Task selection, state clearing, creation of artifacts subdirectory.
- get_task_prompt(): First round system/user prompts (including data directory, available files, loading examples, and task specifications).
- step(action:str): Receives LLM text -> Extracts FINAL ANSWER or ```python``` code -> Execute/Record/Score.
- render(): Three-column progressive view (Thought | Code | Observation) + bottom CSV preview.
- Output: artifacts/dabstep_YYYYmmdd_HHMMSS_<taskid>/{trace.jsonl, dev_metrics.json?, env.log, ...}

No separate runner.py needed. The external orchestrator only needs to:
- reset()
- Pass get_task_prompt() to LLM
- Feed LLM response to step()
- Optionally call render() to save a "process snapshot" PNG.
"""

import os, re, io, sys, json, glob, time, textwrap, logging, shutil
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

from io import BytesIO
import base64
import hashlib

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib import font_manager as fm, colors as mcolors
from matplotlib.font_manager import FontProperties

import gymnasium as gym

# AIEvoBox core
from openai.types.chat import ChatCompletionMessageParam
from core.types.base import (
    ResetOutput, StepOutput, RenderOutput,
    PromptOutput, OpenAIMessage, MessageContent, TextContent
)
from core.env.base_env import BaseEnv
from core.env.env_register import register_env


# ========== logging ==========
def _make_logger(log_path: Path) -> logging.Logger:
    log = logging.getLogger(f"DABStepEnv[{id(log_path)}]")
    log.setLevel(logging.INFO)
    # avoid duplicated handlers
    if not any(isinstance(h, logging.FileHandler) and getattr(h, 'baseFilename', '') == str(log_path) for h in log.handlers):
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(str(log_path), mode="w", encoding="utf-8")
        sh = logging.StreamHandler(sys.stdout)
        fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        fh.setFormatter(fmt); sh.setFormatter(fmt)
        log.addHandler(fh); log.addHandler(sh)
    return log

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
# ========== helpers ==========
def _now_tag() -> str:
    import datetime as _dt
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")

def _safe_filename(x: Any) -> str:
    s = re.sub(r"[^0-9A-Za-z_.-]+", "_", str(x).strip())
    return s[:64] or "task"

def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def _try_import_official():
    try:
        # Check if the official evaluation package is installed
        from dabstep_benchmark.utils import evaluate as official_evaluate
        return True, official_evaluate
    except ImportError:
        # reminder if the library is missing
        print("[DABStepEnv] Notice: Official evaluation library not detected. To use scoring, please run: pip install git+https://huggingface.co/spaces/adyen/DABstep.git@main")
        return False, None
    except Exception as e:
        print(f"[DABStepEnv] Unexpected error importing evaluation library: {e}")
        return False, None
    
def _set_mono_font(self) -> None:
    try:
        chosen = None
        for fam in getattr(self, "render_mono_fonts", ["JetBrains Mono", "Cascadia Mono", "Consolas", "Menlo", "DejaVu Sans Mono"]):
            try:
                fm.findfont(fam, fallback_to_default=False)
                chosen = fam
                break
            except Exception:
                continue
        if chosen is None:
            chosen = "DejaVu Sans Mono"
        plt.rcParams["font.family"] = chosen
        plt.rcParams["font.monospace"] = [chosen]
        plt.rcParams["axes.unicode_minus"] = False
        self._mono_font = chosen
    except Exception:
        self._mono_font = "DejaVu Sans Mono"

def _strip_ansi(s: str) -> str:
    return _ANSI_RE.sub("", s or "")

def _hard_wrap(text: str, width: int) -> str:
    if not text:
        return ""
    lines = []
    for ln in str(text).splitlines():
        for i in range(0, len(ln), max(1, width)):
            lines.append(ln[i:i+width])
    return "\n".join(lines)

def _wrap_mono_keep_spaces(s: str, width: int, max_lines: int) -> str:
    s = _strip_ansi(s)
    if not s:
        return ""
    body = _hard_wrap(s, width)
    lines = body.splitlines()
    if max_lines and max_lines > 0 and len(lines) > max_lines:
        lines = lines[:max_lines] + ["…(truncated)"]
    return "\n".join(lines)

def _hex_to_rgb(h: str):
    c = str(h or "").strip()
    if c.startswith("#"):
        c = c[1:]
    if len(c) == 3:
        c = "".join(ch * 2 for ch in c)
    r = int(c[0:2], 16) / 255.0
    g = int(c[2:4], 16) / 255.0
    b = int(c[4:6], 16) / 255.0
    return (r, g, b)

def _rgb_to_hex(rgb):
    r, g, b = rgb
    return "#{:02x}{:02x}{:02x}".format(int(round(r * 255)), int(round(g * 255)), int(round(b * 255)))

def _tint(base: str, mix: float = 0.9) -> str:
    try:
        r, g, b = _hex_to_rgb(str(base))
        r = r + (1 - r) * mix
        g = g + (1 - g) * mix
        b = b + (1 - b) * mix
        return _rgb_to_hex((r, g, b))
    except Exception:
        return str(base)

def _wrap_to_ax(self, ax, text: str, fontsize: float,
                mono: bool = True, pad_rel: float = 0.04,
                max_lines: Optional[int] = None) -> str:
    if not text:
        return ""
    fig = ax.figure
    try:
        renderer = fig.canvas.get_renderer()
        if renderer is None:
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
    except Exception:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()

    bbox = ax.get_window_extent(renderer=renderer)
    usable_px = max(10.0, bbox.width * (1.0 - 2.0 * pad_rel))

    fp = FontProperties(family=(self._mono_font if mono else None), size=fontsize)
    w_px, _, _ = renderer.get_text_width_height_descent("M" * 16, fp, ismath=False)
    char_px = max(5.0, w_px / 16.0)

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

def _trace_to_rows(self, max_thought=None, max_code=None, max_obs=None):
    TH_W, CODE_W, OBS_W = self.render_th_w, self.render_code_w, self.render_obs_w
    if max_thought is None: max_thought = self.render_max_thought
    if max_code   is None: max_code   = self.render_max_code
    if max_obs    is None: max_obs    = self.render_max_obs

    trace = (self._trace or self._last_run_info.get("trace") or [])
    rows = []
    for ev in trace:
        full_msg = (ev.get("assistant_message") or "").strip()
        code = (ev.get("code") or "").strip()
        
        # Separate thought from code
        thought = full_msg
        if code:
            # Remove code blocks; what remains is the thought process
            thought = re.sub(r'```python.*?```', '', full_msg, flags=re.DOTALL | re.IGNORECASE)
            thought = re.sub(r'```(?:code|py)?.*?```', '', thought, flags=re.DOTALL | re.IGNORECASE)
            thought = thought.strip()
        
        ex = ev.get("exec") or {}
        status = "OK" if isinstance(ex, dict) and ex.get("success") else ("FAILED" if ex else "")
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
    """Truncate step rows based on render_max_rows / render_tail_rows constraints."""
    mr = int(getattr(self, "render_max_rows", 0) or 0)
    tail = max(0, int(getattr(self, "render_tail_rows", 0) or 0))
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

# ========== env ==========
@register_env("dabstepgym")
class DABStepEnv(BaseEnv):
    """
    DABStep Gymnasium Environment: A stateful sandboxed runner for data analysis tasks.
    """
    def __init__(self, data_dir: str, context_dir: str = "context",
             tasks_dir: str = "tasks",
             artifacts_dir: str = "env/dabstep/artifacts",
             split: str = "default", max_steps: int = 10, timeout: int = 60,
             limit: int = 0, only_task_id: str = "", env_id: str = "",
             shard_index: int = 0, num_shards: int = 1,
             env_name: str = "dabstepgym", **kwargs):
        super().__init__(env_id=env_id, env_name=env_name)

        self._data_root   = Path(data_dir).resolve()
        self._context_rel = Path(context_dir)
        self._tasks_rel   = Path(tasks_dir)
        self._art_root    = Path(artifacts_dir).resolve()

        self._art_root.mkdir(parents=True, exist_ok=True)
        self._logger = _make_logger(self._art_root / "env_bootstrap.log")
        self._ensure_data_ready()

        # Operational Parameters
        self.split        = str(split).strip().lower()
        self.max_steps    = int(max_steps)
        self.timeout      = int(timeout)
        self.limit        = int(limit)
        self.only_task_id = str(only_task_id).strip()

        # Episode state
        self._tasks: List[Dict[str, Any]] = []
        self._task_idx = -1
        self._task: Dict[str, Any] = {}
        self._task_id: str = ""
        self._question: str = ""
        self._answer_format: str = ""
        self._agent_answer: str = ""
        self._metrics: Dict[str, Any] = {}
        self._trace: List[Dict[str, Any]] = []
        self._exec_globals: Dict[str, Any] = {}
        self._step_i = 0
        self._artifact_dir: Path = self._art_root
        
        # Conversation state (system + interaction history)
        self._conversation_history: List[Dict[str, str]] = []

        self._last_obs: Dict[str, Any] = {}
        self._last_run_info: Dict[str, Any] = {}

        # Rendering Configuration
        self._render_step = 0
        self.render_mono_fonts = ["JetBrains Mono", "Cascadia Mono", "Consolas", "Menlo", "DejaVu Sans Mono"]

        # UI Colors (Blue for Thought, Green for Code, Yellow for Observation)
        self.render_col_colors = ["#90caf9", "#a5d6a7", "#fff59d"]
        self.render_col_weights = [1.0, 1.0, 1.0]

        # Pre-wrap width
        self.render_th_w   = 55
        self.render_code_w = 55
        self.render_obs_w  = 55

        # Max lines per column
        self.render_max_thought = 25
        self.render_max_code    = 30
        self.render_max_obs     = 25

        # Cropping for max displayed steps
        self.render_max_rows = 0
        self.render_tail_rows = 0

        # Canvas and font sizes
        self.render_header_h     = 0.45
        self.render_base_row_h   = 2.2
        self.render_per_line_h   = 0.18
        self.render_fig_w_in     = 20
        self.render_fig_scale    = 1.0
        self.render_height_scale = 1.1
        self.render_dpi          = 150
        self.render_font_scale   = 0.95
        self.render_max_fig_h_in = 0.0
        
        # Added: Question display area height
        self.render_question_h = 1.2

        # Whether to show the bottom DataFrame preview
        self.render_df_preview = kwargs.get(
            "render_df_preview",
            (os.getenv("DABSTEP_RENDER_DF_PREVIEW", "0").lower() in ("1", "true", "yes"))
        )

        self.obs_scratchpad_last_n = 3
        self._seen_code_sigs: set[str] = set()
        self._seen_out_sigs: set[str] = set()
        self._last_out_sig: str = ""

        self.error_tips_enabled = True
        self._err_sig_counts: dict[str, int] = {}

        self.limit = int(limit)
        self.only_task_id = str(only_task_id).strip()
        self.shard_index = int(shard_index)  
        self.num_shards = int(num_shards)    

        # spaces
        self.observation_space = gym.spaces.Dict({
            "stdout": gym.spaces.Text(max_length=5_000_000),
            "error":  gym.spaces.Text(max_length=5_000_000),
            "step_idx": gym.spaces.Box(low=0, high=10_000, shape=(), dtype=np.int32),
        })
        self.action_space = gym.spaces.Text(max_length=5_000_000)

        # Task loading
        self._load_tasks()
        self._has_official, self._official_eval = _try_import_official()

        self._logger.info(f"DABStepEnv successfully initialized! Current data root: {self._data_root} | split={self.split} | limit={self.limit}")
        
    # --- bind module-level helpers as methods on this class ---
    _set_mono_font = _set_mono_font
    _wrap_to_ax = _wrap_to_ax
    _apply_row_caps = _apply_row_caps

    # helpers that don't need self
    _wrap_mono_keep_spaces = staticmethod(_wrap_mono_keep_spaces)
    _strip_ansi = staticmethod(_strip_ansi)
    _hard_wrap = staticmethod(_hard_wrap)
    _hex_to_rgb = staticmethod(_hex_to_rgb)
    _rgb_to_hex = staticmethod(_rgb_to_hex)
    _tint = staticmethod(_tint)
    _trace_to_rows = _trace_to_rows

    def _ensure_data_ready(self):
        context_path = self._data_root / self._context_rel
        tasks_path = self._data_root / self._tasks_rel
        context_ok = context_path.exists() and any(context_path.iterdir())
        tasks_ok = tasks_path.exists() and any(tasks_path.glob("*.jsonl"))

        if context_ok and tasks_ok:
            return

        self._logger.info("[Auto-Loader] Local dataset not found, downloading...")
        try:
            # 1. download context file（csv/json）
            if not context_ok:
                from huggingface_hub import snapshot_download
                tmp_dir = self._data_root.parent / f"tmp_download_{_now_tag()}"
                snapshot_download(
                    repo_id="adyen/DABstep",
                    repo_type="dataset",
                    local_dir=tmp_dir,
                    allow_patterns=["data/context/**"],  
                    ignore_patterns=[".git*"],
                    local_dir_use_symlinks=False, 
                )
                source_inner = tmp_dir / "data" / "context"
                if not source_inner.exists():
                    raise RuntimeError(f"[Auto-Loader] data/context/ not found after download. Actual contents: {list(tmp_dir.rglob('*'))[:10]}")
                context_path.mkdir(parents=True, exist_ok=True)
                for item in source_inner.iterdir():
                    dest = context_path / item.name
                    if dest.exists():
                        dest.unlink() if dest.is_file() else shutil.rmtree(dest)
                    shutil.move(str(item), str(dest))
                shutil.rmtree(tmp_dir)
                self._logger.info(f"[Auto-Loader] Context files are prepared: {context_path}")

            # 2. Downloading tasks and exporting to JSONL using the Hugging Face Datasets API.
            if not tasks_ok:
                from datasets import load_dataset
                import json as _json
                tasks_path.mkdir(parents=True, exist_ok=True)
                for split in ["default", "dev"]:
                    out_file = tasks_path / f"{split}_tasks.jsonl"
                    self._logger.info(f"[Auto-Loader] Now exporting tasks for split={split}...")
                    ds = load_dataset("adyen/DABstep", name="tasks", split=split)
                    with open(out_file, "w", encoding="utf-8") as f:
                        for row in ds:
                            f.write(_json.dumps(row, ensure_ascii=False) + "\n")
                    self._logger.info(f"[Auto-Loader] {split}_tasks.jsonl: {len(ds)} items")

            self._logger.info("[Auto-Loader] All data is ready.")

        except Exception as e:
            self._logger.error(f"[Auto-Loader] Fail: {e}")
            raise RuntimeError(
                f"Failed to fetch data automatically. Please manually run:\n"
                f"  pip install datasets huggingface_hub\n"
                f"  and place context/ and tasks/ into {self._data_root}"
            ) from e

    def _context_path(self) -> Path:
        return self._data_root / self._context_rel

    def _tasks_path(self) -> Path:
        return self._data_root / self._tasks_rel / f"{self.split}_tasks.jsonl"

    # ---------- data loading ----------
    def _load_tasks(self) -> None:
        tf = self._tasks_path()
        if not tf.exists():
            raise FileNotFoundError(
                f"Tasks file not found: {tf}\n"
                f"Please place {self.split}_tasks.jsonl into the {self._tasks_rel}/ subdirectory of your data directory."
            )
        rows = _read_jsonl(tf)
        
        # deal only_task_id
        if self.only_task_id:
            rows = [r for r in rows if str(r.get("task_id")) == self.only_task_id]
            if not rows:
                raise FileNotFoundError(f"only_task_id={self.only_task_id} not found in {tf}")
        
        # Sharding logic
        if self.num_shards > 1:
            # Ensure valid shard index
            if self.shard_index < 0 or self.shard_index >= self.num_shards:
                raise ValueError(
                    f"Invalid shard_index={self.shard_index}, must be in [0, {self.num_shards-1}]"
                )
            # Shard by index (ensures uniform task distribution)
            rows = [r for i, r in enumerate(rows) if i % self.num_shards == self.shard_index]
            self._logger.info(
                f"Shard {self.shard_index}/{self.num_shards}: "
                f"loaded {len(rows)} tasks from total"
            )
        
        # Handle task limits
        self._tasks = rows[: self.limit] if (self.limit and self.limit > 0) else rows
        
        if not self._tasks:
            raise RuntimeError(f"No tasks loaded from {tf}")
        
        self._task_idx = -1

    def _next_task(self) -> Dict[str, Any]:
        self._task_idx = (self._task_idx + 1) % len(self._tasks)
        return self._tasks[self._task_idx]

    # ---------- code execution (stateful) ----------
    def _exec_code(self, code: str, is_new_task: bool = False) -> Dict[str, Any]:
        import json as _json, math as _math, statistics as _statistics
        import itertools as _itertools, datetime as _datetime, csv as _csv, re as _re, os as _os

        if is_new_task or not self._exec_globals:
            self._exec_globals = {
                "pd": pd, "pandas": pd,
                "json": _json, "math": _math, "statistics": _statistics,
                "itertools": _itertools, "datetime": _datetime, "re": _re, "csv": _csv, "os": _os,
            }
            try:
                import numpy as _np
                self._exec_globals.update({"np": _np, "numpy": _np})
            except Exception:
                pass

        ctx_dir = str(self._context_path())
        art_dir = str(self._artifact_dir)
        exec_locals = {
            "data_dir": str(self._data_root),
            "context_dir": ctx_dir,
            "artifact_dir": art_dir,
        }

        out_lines: List[str] = []
        def _print(*args, **kwargs):
            s = " ".join(str(a) for a in args)
            out_lines.append(s)
            print(s)
        self._exec_globals["print"] = _print

        try:
            exec(code, self._exec_globals, exec_locals)
            self._exec_globals.update(exec_locals)
            return {"success": True, "output": "\n".join(out_lines), "error": ""}
        except Exception as e:
            return {"success": False, "output": "\n".join(out_lines), "error": f"{type(e).__name__}: {e}"}
        
    @staticmethod
    def _sig(s: str) -> str:
        return hashlib.sha1((s or "").encode("utf-8", "ignore")).hexdigest()

    def _build_scratchpad(self, tail_steps: List[Dict[str, Any]], budget: int = 9000) -> str:
        """Compress recent steps into a readable scratchpad, managing length budget to prevent token explosion."""
        parts: List[str] = []
        for ev in tail_steps:
            step = ev.get("step")
            full_msg = (ev.get("assistant_message") or "")
            code = (ev.get("code") or "")
            
            # Separate thought and code
            thought = full_msg
            if code:
                thought = re.sub(r'```python.*?```', '', full_msg, flags=re.DOTALL | re.IGNORECASE)
                thought = re.sub(r'```(?:code|py)?.*?```', '', thought, flags=re.DOTALL | re.IGNORECASE)
                thought = thought.strip()
            
            ex = ev.get("exec") or {}
            out = (ex.get("output") or "")
            err = (ex.get("error") or "")
            
            piece = []
            if thought:
                piece.append(f"[Step {step}] THOUGHT:\n{thought[:800]}")
            if code:
                piece.append("CODE:\n```python\n" + code[:2000] + "\n```")
            if out:
                piece.append("STDOUT:\n" + out[:2000])
            if err:
                piece.append("STDERR:\n" + err[:1000])
            parts.append("\n".join(piece))
        
        s = "\n\n".join(parts)
        return (s if len(s) <= budget else (s[:budget] + "\n…(truncated)"))
    
    def _coach_from_error(self, err: str, code: str) -> tuple[str, str]:
        """
        Generates an "Error Category + Fix Checklist" based on common exceptions.
        Returns: (error_category, fix_checklist_text)
        """
        e = err or ""
        cat = "RuntimeError"
        tips: list[str] = []

        # NameError
        m = re.search(r"NameError:\s*name '(.+?)' is not defined", e)
        if m:
            cat = "NameError"
            var = m.group(1)
            tips += [
                f"Variable '{var}' is undefined: either define it within the current code block or reuse valid environment variables (Available: 'context_dir', 'data_dir', 'artifact_dir').",
                "Do not rely on local variables defined in previous blocks; each code submission must be self-contained and independently runnable.",
                "If an import is missing, explicitly include 'import pandas as pd', 'import os', etc., at the top of your code block."
            ]
            return cat, "\n- " + "\n- ".join(tips)

        # KeyError
        m = re.search(r"KeyError:\s*'(.+?)'", e)
        if m:
            cat = "KeyError"
            col = m.group(1)
            tips += [
                f"Column '{col}' not found: run 'print(df.columns.tolist())' first to verify actual column names; check for case sensitivity and leading/trailing spaces.",
                "Use 'df.rename(columns={{...}})' to standardize column names, or perform aggregations using the verified column labels.",
            ]
            return cat, "\n- " + "\n- ".join(tips)

        # AttributeError
        m = re.search(r"AttributeError:\s*'(.+?)' object has no attribute '(.+?)'", e)
        if m:
            cat = "AttributeError"
            obj_t, attr = m.groups()
            tips += [
                f"'{obj_t}' object has no attribute '{attr}': verify the object type and check the pandas API documentation.",
                "Common DataFrame patterns: use 'df[\"col\"]' for selection, 'df.groupby([...]).agg(...)' for aggregation, and 'df.merge(...)' for joins.",
                "Ensure you are not treating a pandas 'Series' as a 'DataFrame'; use 'df = df.reset_index()' to convert it if necessary.",
            ]
            return cat, "\n- " + "\n- ".join(tips)

        # FileNotFoundError
        if "FileNotFoundError" in e or "No such file or directory" in e:
            cat = "FileNotFoundError"
            tips += [
                f"Avoid using bare relative paths; use 'os.path.join(context_dir, \"file.csv\")' to build full, reliable paths.",
                "Run 'print(os.listdir(context_dir))' first to verify the actual file names available in the context directory.",
            ]
            return cat, "\n- " + "\n- ".join(tips)

        # TypeError / ValueError / others
        if "TypeError" in e:
            cat = "TypeError"
            tips += ["Print `type(x)` or `df.dtypes` to ensure data types involved in calculations are correct."]
        elif "ValueError" in e:
            cat = "ValueError"
            tips += ["Check passed arguments and dimensions; run `print(...)` to verify the shape/content of aggregation results."]
        elif "ZeroDivisionError" in e:
            cat = "ZeroDivisionError"
            tips += ["Potential division by zero: apply safeguards like `den or 1` or filter zero-values before aggregation."]
        elif "IndexError" in e:
            cat = "IndexError"
            tips += ["Out-of-bounds access: use `print(len(x))` or `print(df.shape)` before indexing."]
        elif "ImportError" in e or "ModuleNotFoundError" in e:
            cat = "ImportError"
            tips += ["Only use allowed built-in libraries (pandas, numpy, json); do not introduce external dependencies."]
        elif "SyntaxError" in e:
            cat = "SyntaxError"
            tips += ["Fix syntax: ensure the code block is self-contained and avoid mixing markdown formatting within the code."]

        if not tips:
            tips = ["Read STDERR to isolate the issue; rewrite a smaller, runnable snippet to approach the solution incrementally."]
        return cat, "\n- " + "\n- ".join(tips)

    # ---------- parsing ----------
    _CODE_PATTS = [
        re.compile(r"```python\s+(.*?)```", re.DOTALL | re.IGNORECASE),
        re.compile(r"```(?:code|py)?\s+(.*?)```", re.DOTALL | re.IGNORECASE),
    ]
    def _extract_code(self, txt: str) -> Optional[str]:
        s = str(txt or "")
        for p in self._CODE_PATTS:
            m = p.search(s)
            if m:
                code = m.group(1)
                return code.strip().strip("`")
        return None

    def _extract_final(self, txt: str) -> Optional[str]:
        m = re.search(r"FINAL ANSWER:\s*(.+?)(?:\n|$)", str(txt or ""), re.IGNORECASE)
        return m.group(1).strip() if m else None

    def _debug_paths(self):
        try:
            self._logger.info("data_root=%s", str(self._data_root))
            self._logger.info("context_path=%s", str(self._context_path()))
            self._logger.info("tasks_path=%s", str(self._tasks_path()))
            self._logger.info("artifacts_root=%s", str(self._art_root))
        except Exception as e:
            self._logger.warning("path debug failed: %s", e)

    # ---------- reset / prompt / step ----------
    def reset(self, seed: Optional[int] = None) -> ResetOutput:
        self._render_step = 0
        self._task        = self._next_task()
        self._task_id     = str(self._task.get("task_id", self._task_idx))
        self._question    = str(self._task.get("question", "") or "")
        self._answer_format = str(self._task.get("guidelines", "") or self._task.get("answer_format", "") or "")

        # Clean state
        self._agent_answer = ""
        self._metrics     = {}
        self._trace       = []
        self._exec_globals= {}
        self._step_i      = 0
        self.done         = False
        self._seen_code_sigs.clear()
        self._seen_out_sigs.clear()
        self._last_out_sig = ""
        
        # clean history
        self._conversation_history = []

        # Artifacts directory (one subdirectory per task)
        tag = _now_tag()
        self._artifact_dir = self._art_root / f"dabstep_{tag}_{_safe_filename(self._task_id)}"
        self._artifact_dir.mkdir(parents=True, exist_ok=True)

        # Switch logger to the current episode
        self._logger = _make_logger(self._artifact_dir / "env.log")
        self._logger.info("===== New episode | task_id=%s | split=%s =====", self._task_id, self.split)

        # Initial observation
        obs = {"stdout": "", "error": "", "step_idx": 0}
        info = {
            "task_id": self._task_id,
            "split": self.split,
            "artifact_dir": str(self._artifact_dir),
            "agent_answer": "",
            "metrics": {},
            "trace": [],
            "conversation_history": [],
        }
        # Rendering cache
        self._last_obs = {"task_id": self._task_id}
        self._last_run_info = {
            "split": self.split,
            "artifact_dir": str(self._artifact_dir),
            "metrics": {},
            "trace": [],
            "agent_answer": "",
            "question": self._question,
        }

        return ResetOutput(observation=obs, info=info)

    def get_task_prompt(self) -> List[ChatCompletionMessageParam]:
        """
        Returns task prompts.
        - First call (_step_i == 0): Returns the initial system + user messages.
        - Subsequent calls: Formats the complete conversation history into a single large string.
        """
        ctx = str(self._context_path().resolve())
        try:
            files = sorted([p.name for p in self._context_path().iterdir() if p.is_file()])
        except Exception:
            files = []

        # ===== First Call: Initialization =====
        if self._step_i == 0:
            demo = (
                "import pandas as pd, os, json\n"
                f"context_dir = r'{ctx}'\n"
                "df = pd.read_csv(os.path.join(context_dir, 'payments.csv'))\n"
                "print(df.head())\n"
                "with open(os.path.join(context_dir, 'fees.json'), 'r', encoding='utf-8') as f:\n"
                "    fees = json.load(f)\n"
                "print(type(fees))\n"
            )

            sys_text = (
                "You are a data-analysis assistant. Solve problems step by step using this workflow:\n\n"
                "**Your response format for EACH step:**\n"
                "1) **THOUGHT**: First, explain what you plan to do and why.\n"
                "2) **CODE**: Then write ONE self-contained ```python``` block.\n"
                "3) **REFLECTION**: After seeing execution results, decide the next step.\n"
                "4) **FINAL ANSWER**: When ready, output DIRECTLY as plain text (NOT in code):\n"
                "   FINAL ANSWER: <value>\n\n"
                
                f"Data files are located in: {ctx}\n\n"
                
                "**CRITICAL RULES:**\n"
                "- ALWAYS use print() to display results - expressions alone won't show output!\n"
                "- Every code block must be self-contained (import everything, read files fresh).\n"
                "- When you have the answer, DO NOT put it in print() or code!\n"
                "- Instead, output directly as text AFTER the code block:\n"
                "  CORRECT: ```python\\nprint(result)\\n```\\nFINAL ANSWER: NL\n"
                "  WRONG: ```python\\nprint('FINAL ANSWER:', result)\\n```\n"
            )
            
            usr_text = (
                f"Context files available: {', '.join(files)}\n\n"
                f"**Question:**\n{self._question}\n\n"
                f"**Answer format** (if specified):\n{self._answer_format or '(none specified)'}\n\n"
                "Start by:\n"
                "1) Explaining your approach\n"
                "2) Exploring the data schema\n"
                "3) Computing the answer step by step\n\n"
                "**Code template:**\n```python\n" + demo + "```\n"
            )
            
            # initialize message histroy
            self._conversation_history = [
                {"role": "system", "content": sys_text},
                {"role": "user", "content": usr_text}
            ]
            
            return self._conversation_history
        
        # returns formatted messages containing the full history
        else:            
            return self._conversation_history

    def step(self, action: str) -> StepOutput:
        """
        - Maintains full conversation history.
        - Detects and prevents duplicate code execution.
        - Prints full conversation_history for debugging purposes.
        """
        self._step_i += 1
        terminated = truncated = False

        msg = (action or "").strip()
        event: Dict[str, Any] = {"step": self._step_i, "assistant_message": msg}
        
        # Record assistant message into conversation history
        self._conversation_history.append({"role": "assistant", "content": msg})

        stdout, error = "", ""
        code = None
        final = self._extract_final(msg)

        if final:
            self._agent_answer = final
            event["final_answer"] = final
            terminated = True
            self._logger.info("FINAL ANSWER: %s", final)
            try:
                self._metrics = self._score_current_task(final) if self.split == "dev" else {}
            except Exception as e:
                self._logger.warning("scoring failed: %s: %s", type(e).__name__, e)
                self._metrics = {}
        else:
            code = self._extract_code(msg)
            if code:
                # check duplicated code
                code_sig = hashlib.md5(code.encode("utf-8")).hexdigest()
                if code_sig in self._seen_code_sigs:
                    self._logger.warning("DUPLICATE CODE DETECTED at step %d", self._step_i)
                    stdout = ""
                    error = "DUPLICATE_CODE_DETECTED: You just submitted identical code that was already executed!"
                    event["code"] = code
                    event["exec"] = {"success": False, "output": "", "error": error}
                else:
                    # Proceed with execution
                    self._seen_code_sigs.add(code_sig)
                    self._logger.info("Executing code (step %d)...", self._step_i)
                    ex = self._exec_code(code, is_new_task=(self._step_i == 1))
                    stdout, error = ex.get("output", ""), ex.get("error", "")
                    event["code"] = code
                    event["exec"] = {"success": bool(ex.get("success")), "output": stdout, "error": error}
                    if error:  self._logger.info("Exec ERROR: %s", error)
                    if stdout: self._logger.info("Exec OUTPUT:\n%s", stdout)
            else:
                stdout = "No code block found. Please provide:\n1) Your THOUGHT about what to do\n2) A ```python``` code block\n3) Or output FINAL ANSWER: <...> if ready."
                self._logger.info("No code / no final answer at step %d.", self._step_i)

            if self._step_i >= self.max_steps:
                truncated = True
                self._logger.info("Truncated due to max_steps=%d", self.max_steps)

        # Save trace.jsonl to disk
        self._trace.append(event)
        try:
            tpath = self._artifact_dir / "trace.jsonl"
            with tpath.open("w", encoding="utf-8") as tf:
                for ev in self._trace:
                    row = dict(ev); row.setdefault("task_id", self._task_id)
                    tf.write(json.dumps(row, ensure_ascii=False) + "\n")
        except Exception as e:
            self._logger.warning("write trace.jsonl failed: %s", e)

        if terminated and self.split == "dev":
            try:
                (self._artifact_dir / "dev_metrics.json").write_text(
                    json.dumps(self._metrics, ensure_ascii=False, indent=2), encoding="utf-8"
                )
            except Exception as e:
                self._logger.warning("write dev_metrics.json failed: %s", e)

        # Not terminated: Constructing feedback with context
        if not terminated and not truncated:
            recent_trace = self._trace[-self.obs_scratchpad_last_n:] if len(self._trace) > 0 else []
            #scratchpad = self._build_scratchpad(recent_trace, budget=4000)
            
            hints: List[str] = []
            is_duplicate_output = False
            
            # Detect duplicate output
            try:
                last_sig = getattr(self, "_last_out_sig", "")
                cur_sig = hashlib.md5(((stdout or "") + "|" + (error or "")).encode("utf-8")).hexdigest()
                if last_sig and cur_sig == last_sig:
                    is_duplicate_output = True
                self._last_out_sig = cur_sig
            except Exception:
                pass
            
            is_duplicate_code = "DUPLICATE_CODE_DETECTED" in error

            answer_detected = False
            detected_answer = ""
            
            if stdout and not error:
                patterns = [
                    r"(?:issuing country|country|answer).*?is[:\s]+([A-Z]{2,3})",
                    r"(?:highest|maximum).*?[:\s]+([A-Z]{2,3})\s+with",
                    r"FINAL ANSWER[:\s]+(.+?)(?:\n|$)",
                ]
                for pattern in patterns:
                    match = re.search(pattern, stdout, re.IGNORECASE)
                    if match:
                        detected_answer = match.group(1).strip().upper()
                        answer_detected = True
                        break
            
            # Hint priority: Answer detection > Repetition detection > Error tips
            if answer_detected:
                hints.insert(0,
                    f"DETECTED ANSWER: You have computed the answer as '{detected_answer}'.\n"
                    f"Now you MUST output exactly:\n"
                    f"FINAL ANSWER: {detected_answer}\n"
                    f"Stop all additional analysis."
                )
            elif is_duplicate_code:
                hints.append(
                    "CRITICAL: DUPLICATE CODE BLOCKED\n"
                    "You submitted the EXACT same code. Change your approach:\n"
                    "- Don't preview data again\n"
                    "- Compute the actual answer\n"
                    "- Read the question and compute the required metric"
                )
            elif is_duplicate_output:
                hints.append(
                    "WARNING: REPEATING OUTPUTS\n"
                    "Same output as last step means:\n"
                    "- Stop exploring - you've seen this data\n"
                    "- Move to computation phase\n"
                    "- Calculate the answer"
                )
            
            if error and not is_duplicate_code and self.error_tips_enabled:
                err_cat, fix_list = self._coach_from_error(error, code or "")
                hints.append(f"**Error Category**: {err_cat}\n**Fix Checklist**:{fix_list}")

            def _clip(s: str, n=2500): 
                s = (s or "").strip()
                return s if len(s) <= n else (s[:n] + "\n…(truncated)")

            user_prompt = ""
            
            if is_duplicate_code or is_duplicate_output:
                user_prompt += "\n" + "="*50 + "\nREPETITION DETECTED - CHANGE YOUR APPROACH!\n" + "="*50 + "\n\n"
            
            if len(self._trace) > 1:
                last_step = self._trace[-2] if len(self._trace) >= 2 else self._trace[-1]
                last_thought = (last_step.get("assistant_message") or "")[:500]
                last_code = (last_step.get("code") or "")[:800]
                last_out = ""
                if last_step.get("exec"):
                    last_out = (last_step["exec"].get("output") or "")[:500]
                
                user_prompt += (
                    f"Your Last Step (Step {last_step.get('step', '?')}):\n"
                    f"Thought: {last_thought}\n"
                    f"Code: ```python\n{last_code}\n```\n"
                    f"Output: {last_out}\n\n"
                )
            
            user_prompt += (
                f"**Current Step ({self._step_i}) Execution Result:**\n"
                f"STDOUT: {_clip(stdout, 1500) if stdout else '(empty)'}\n"
                f"STDERR: {_clip(error, 1500) if error else '(none)'}\n\n"
            )
            
            if hints:
                user_prompt += "**Important Guidance:**\n" + "\n\n".join(hints) + "\n\n"

            user_prompt += (
                "**What to do next:**\n\n"
                "If you have the answer to the original question:\n"
                "→ Output exactly: FINAL ANSWER: <value>\n"
                "→ Do NOT continue with additional analysis\n\n"
                "If you need more information:\n"
                "→ **THOUGHT**: What you'll do next\n"
                "→ **CODE**: ```python\n# your code\n```\n"
            )

            self._conversation_history.append({"role": "user", "content": user_prompt})
            
            self._logger.info("=== User feedback prompt constructed (%d chars) ===", len(user_prompt))
            
            # Print full conversation history for debugging
            self._logger.info("="*80)
            self._logger.info("CURRENT CONVERSATION HISTORY (Step %d):", self._step_i)
            self._logger.info("="*80)
            for i, msg in enumerate(self._conversation_history):
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                # Clip lengthy content
                preview = content[:300] + ("..." if len(content) > 300 else "")
                self._logger.info("[Message %d] Role: %s | Length: %d chars", i, role, len(content))
                self._logger.info("Preview: %s", preview)
                self._logger.info("-"*80)
            self._logger.info("="*80)
            
            stdout = user_prompt

        # Update rendering cache
        self._last_run_info.update({
            "split": self.split,
            "artifact_dir": str(self._artifact_dir),
            "metrics": dict(self._metrics),
            "trace": list(self._trace),
            "agent_answer": self._agent_answer,
            "question": self._question,  
        })

        # obseravtion and info
        obs = {"stdout": stdout, "error": error, "step_idx": self._step_i}
        info = {
            "task_id": self._task_id,
            "split": self.split,
            "artifact_dir": str(self._artifact_dir),
            "agent_answer": self._agent_answer,
            "metrics": self._metrics,
            "trace": list(self._trace),
            "conversation_history": list(self._conversation_history),
        }

        # reward
        reward = 0.0
        if terminated and self.split == "dev":
            try:
                reward = float(self._metrics.get("accuracy", 0.0) or 0.0)
            except Exception:
                reward = 0.0

        self.done = terminated or truncated
        return StepOutput(
            observation=obs,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )

    # ---------- (dev) Single-task evaluation ----------
    def _score_current_task(self, pred: str) -> Dict[str, Any]:
        has_official = self._has_official
        official_eval = self._official_eval

        gt = {
            "task_id": str(self._task_id),
            "answer":  str(self._task.get("answer", "")),
            "level":   str(self._task.get("level", "")),
        }
        if not gt["answer"]:
            return {"accuracy": 0.0, "total": 0, "correct": 0, "details": [], "scorer": "no_gt"}

        if has_official and official_eval is not None:
            try:
                agent_df = pd.DataFrame([{"task_id": gt["task_id"], "agent_answer": str(pred)}])
                tasks_df = pd.DataFrame([gt])[["task_id", "answer", "level"]]
                out = official_eval(agent_answers=agent_df, tasks_with_gt=tasks_df)
                if hasattr(out, "to_dict") and ("score" in getattr(out, "columns", [])):
                    score = bool(out.iloc[0]["score"])
                elif isinstance(out, dict) and "per_question" in out:
                    score = bool(pd.DataFrame(out["per_question"]).iloc[0]["score"])
                elif isinstance(out, list) and len(out) > 0:
                    score = bool(out[0]) if not isinstance(out[0], dict) else bool(out[0].get("score", False))
                else:
                    score = False
                return {"accuracy": 1.0 if score else 0.0, "total": 1, "correct": int(score),
                        "details": [{"task_id": gt["task_id"], "score": bool(score)}],
                        "scorer": "official_single"}
            except Exception as e:
                self._logger.warning("official evaluate failed: %s", e)

        # Approximate scoring
        def _try_float(x):
            try: return float(x)
            except Exception: return None
        def _norm(s: str):
            return re.sub(r"\s+", "", s).strip().lower()
        def _canon(ans: str):
            s = str(ans).strip()
            if "," in s:
                items = [x.strip() for x in s.split(",") if x.strip() != ""]
                floats, all_f = [], True
                for it in items:
                    v = _try_float(it)
                    if v is None:
                        all_f = False; break
                    floats.append(round(float(v), 8))
                if all_f: return ("list_float", tuple(floats))
                return ("list_str", tuple(_norm(it) for it in items))
            v = _try_float(s)
            if v is not None: return ("float", round(float(v), 8))
            return ("str", _norm(s))
        def _eq(a, b, tol=1e-6): return abs(a - b) <= tol

        ta, tb = _canon(gt["answer"]), _canon(str(pred))
        if ta[0] == tb[0]:
            if ta[0] == "float": ok = _eq(ta[1], tb[1])
            elif ta[0] == "list_float":
                ok = (len(ta[1]) == len(tb[1])) and all(_eq(x, y) for x, y in zip(ta[1], tb[1]))
            else:
                ok = (ta[1] == tb[1])
        else:
            ok = (_norm(gt["answer"]) == _norm(str(pred)))

        return {"accuracy": 1.0 if ok else 0.0, "total": 1, "correct": int(ok),
                "details": [{"task_id": gt["task_id"], "score": bool(ok)}],
                "scorer": "approx_single"}

    # ---------- dataframe preview for render ----------
    def _pick_dataframe_for_preview(self) -> Tuple[Optional[pd.DataFrame], Optional[Path]]:
        cands: List[Path] = []
        adir = Path(self._last_run_info.get("artifact_dir", "") or "")
        if adir.exists():
            cands += [Path(p) for p in glob.glob(str(adir / "*.csv"))]
        cands += [Path(p) for p in glob.glob(str(self._context_path() / "*.csv"))]
        for p in cands:
            try:
                dfx = pd.read_csv(p)
                df  = dfx.head(10).copy()
                df  = df.iloc[:, :8]
                df.columns = [str(c)[:24] for c in df.columns]
                return df, p
            except Exception:
                continue
        return None, None

    # ---------- render ----------
    def render(self) -> RenderOutput:
        self._set_mono_font()
        self._render_step += 1

        # Tri-column color scheme
        col_theme = []
        for base in self.render_col_colors:
            col_theme.append({
                "head_bg": self._tint(base, 0.82),
                "edge":    self._tint(base, 0.55),
                "even_bg": self._tint(base, 0.94),
                "odd_bg":  self._tint(base, 0.90),
            })

        # Metadata
        tid   = self._last_obs.get("task_id", "") or ""
        split = self._last_run_info.get("split", self.split)
        metrics = self._last_run_info.get("metrics") or {}
        score   = metrics.get("accuracy"); scorer = metrics.get("scorer")
        question = self._last_run_info.get("question", "") or ""

        # trace → row
        rows = self._trace_to_rows()
        rows = self._apply_row_caps(rows)
        if not rows:
            task  = getattr(self, "_task", {}) or {}
            question_fallback = str(task.get("question",""))
            aa = str(self._last_run_info.get("agent_answer",""))
            rows = [{
                "thought": self._wrap_mono_keep_spaces("Question:\n" + question_fallback, self.render_th_w,  self.render_max_thought),
                "code":    "",
                "obs":     self._wrap_mono_keep_spaces(aa,                      self.render_obs_w, self.render_max_obs),
            }]

        # Row height estimation
        row_lines = []
        for r in rows:
            # Calculate line count after wrapping
            th_wrapped = self._wrap_mono_keep_spaces(r["thought"], self.render_th_w, None)
            code_wrapped = self._wrap_mono_keep_spaces(r["code"], self.render_code_w, None)
            obs_wrapped = self._wrap_mono_keep_spaces(r["obs"], self.render_obs_w, None)
            
            max_lines = max(
                th_wrapped.count("\n") + 1 if th_wrapped else 1,
                code_wrapped.count("\n") + 1 if code_wrapped else 1,
                obs_wrapped.count("\n") + 1 if obs_wrapped else 1
            )
            row_lines.append(max_lines)

        # DF preview
        df, df_path = None, None
        if getattr(self, "render_df_preview", True):
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

        # Canvas dimensions / Font size
        header_h   = float(self.render_header_h)
        base_row_h = float(self.render_base_row_h)
        per_line_h = float(self.render_per_line_h)
        question_h = float(self.render_question_h) 
        
        main_hs    = [base_row_h + per_line_h*(L-1) for L in row_lines]
        extra_df_h = 1.6 if df is not None else 0.0

        fig_w_in = float(self.render_fig_w_in) * float(self.render_fig_scale)
        # Add question area height
        fig_h_in = (1.6 + question_h + header_h + sum(main_hs) + extra_df_h) \
                * float(self.render_height_scale) * float(self.render_fig_scale)

        # Maximum height clamping
        max_h = float(getattr(self, "render_max_fig_h_in", 0.0))
        if max_h and fig_h_in > max_h:
            scale = max_h / fig_h_in
            fig_h_in = max_h
            per_line_h *= scale

        fig = plt.figure(figsize=(fig_w_in, fig_h_in), dpi=int(self.render_dpi))
        fig.patch.set_facecolor("#ffffff")
        plt.subplots_adjust(top=0.965, bottom=0.065)

        fs = float(self.render_font_scale or 1.0)
        title_fs, header_fs, body_fs, footer_fs = int(13*fs), int(10*fs), int(8*fs), int(7*fs)

        # Title
        title = f"DABStep • Task #{tid} (split={split})"
        if score is not None:
            try:
                title += f"    |    dev score: {float(score):.3f} via {scorer}"
            except Exception:
                title += f"    |    dev score: {score} via {scorer}"
        fig.suptitle(title, x=0.02, ha="left", y=0.988, fontsize=title_fs, fontweight="bold")

        # Grid layout: Question row + Header row + Main data rows + (Optional) DF row
        height_ratios = [question_h, header_h] + main_hs + ([extra_df_h] if df is not None else [])
        gs = fig.add_gridspec(
            nrows=len(height_ratios),
            ncols=3,
            width_ratios=list(self.render_col_weights),
            height_ratios=height_ratios,
            hspace=0.25,
            wspace=0.08
        )

        def draw_cell(ax, title, body, bg="#f9fafb", edge="#d9dee3", mono=True):
            ax.set_xticks([]); ax.set_yticks([]); ax.set_frame_on(False)
            ax.set_xlim(0,1); ax.set_ylim(0,1)
            box = FancyBboxPatch((0,0),1,1, boxstyle="round,pad=0.015,rounding_size=0.02",
                                fc=bg, ec=edge, lw=1.0)
            ax.add_patch(box)
            
            if title and body:
                ax.text(0.03, 0.94, title, ha="left", va="top",
                        fontsize=header_fs, fontweight="bold", 
                        transform=ax.transAxes, clip_on=True)
                
                wrapped = self._wrap_to_ax(ax, body, fontsize=body_fs, mono=mono, 
                                        pad_rel=0.05, max_lines=None)
                kw_font = {"fontfamily": getattr(self, "_mono_font", None)} if mono else {}
                t = ax.text(0.03, 0.72, wrapped, ha="left", va="top",
                            fontsize=body_fs, linespacing=1.5,
                            transform=ax.transAxes, clip_on=True, **kw_font)
                t.set_clip_path(box.get_path(), box.get_transform())
            
            elif title:
                ax.text(0.03, 0.50, title, ha="left", va="center",
                        fontsize=header_fs, fontweight="bold", 
                        transform=ax.transAxes, clip_on=True)
            
            elif body:
                wrapped = self._wrap_to_ax(ax, body, fontsize=body_fs, mono=mono, 
                                        pad_rel=0.05, max_lines=None)
                kw_font = {"fontfamily": getattr(self, "_mono_font", None)} if mono else {}
                t = ax.text(0.03, 0.96, wrapped, ha="left", va="top",
                            fontsize=body_fs, linespacing=1.5,
                            transform=ax.transAxes, clip_on=True, **kw_font)
                t.set_clip_path(box.get_path(), box.get_transform())

        # Question Section (Spans across all three columns)
        ax_q = fig.add_subplot(gs[0, :])
        draw_cell(ax_q, "Question", question, bg="#e3f2fd", edge="#90caf9", mono=False)

        # Table Headers
        ax_h1 = fig.add_subplot(gs[1,0]); ax_h2 = fig.add_subplot(gs[1,1]); ax_h3 = fig.add_subplot(gs[1,2])
        for col, (ax, txt) in enumerate(((ax_h1, "Thought"), (ax_h2, "Code"), (ax_h3, "Observation / Output"))):
            th = col_theme[col]
            draw_cell(ax, txt, "", bg=th["head_bg"], edge=th["edge"], mono=False)

        # Step Rows (Index starts from 2: 0 is question, 1 is header)
        for i, r in enumerate(rows, start=2):
            th0, th1, th2 = col_theme[0], col_theme[1], col_theme[2]
            # Zebra striping logic based on data row index (i-2)
            bg0 = th0["even_bg"] if ((i-2) % 2 == 0) else th0["odd_bg"]
            bg1 = th1["even_bg"] if ((i-2) % 2 == 0) else th1["odd_bg"]
            bg2 = th2["even_bg"] if ((i-2) % 2 == 0) else th2["odd_bg"]
            ax_t = fig.add_subplot(gs[i,0]); draw_cell(ax_t, None, r["thought"], bg=bg0, edge=th0["edge"], mono=True)
            ax_c = fig.add_subplot(gs[i,1]); draw_cell(ax_c, None, r["code"] or "(no code)", bg=bg1, edge=th1["edge"], mono=True)
            ax_o = fig.add_subplot(gs[i,2]); draw_cell(ax_o, None, r["obs"]  or "(no output)", bg=bg2, edge=th2["edge"], mono=True)

        # DF Table
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

        # Footer Section
        ctx_dir = str(self._context_path().resolve())
        try:
            available = ", ".join(sorted([p.name for p in self._context_path().iterdir() if p.is_file()]))
        except Exception:
            available = ""
        footer = f"context: {ctx_dir}" + (f" | files: {available}" if available else "")
        fig.text(0.02, 0.012, self._wrap_mono_keep_spaces(footer, 160, 2),
                ha="left", va="bottom", fontsize=footer_fs, color="#555")

        # Output Generation
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
        img_bytes = buf.getvalue()
        buf.close()

        # Required: Persist to artifacts directory
        out_dir = Path(self._last_run_info.get("artifact_dir") or str(self._artifact_dir))
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"render_step_{self._render_step:02d}.png"
        out_file.write_bytes(img_bytes)

        #  Optional: Copy to unified visualization directory if environment variable is set
        vis_dir = os.getenv("VISUAL_SAVE_PATH")
        if vis_dir:
            Path(vis_dir).mkdir(parents=True, exist_ok=True)
            tid_for_name = self._last_obs.get("task_id", "unknown")
            (Path(vis_dir) / f"{tid_for_name}_step_{self._render_step:02d}.png").write_bytes(img_bytes)

        return RenderOutput(step=self._render_step,
                            image_data=img_bytes,
                            image_path=str(out_file))

    # ---------- close ----------
    def close(self) -> None:
        try:
            plt.close('all')
        except Exception:
            pass