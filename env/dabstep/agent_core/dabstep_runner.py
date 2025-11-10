#!/usr/bin/env python3
"""
DABstep multi-step agent using ReAct pattern
- Reasoning: LLM thinks about the problem
- Acting: LLM generates and executes code
- Observing: Check results and iterate
"""
import json
import logging
import os
import sys
import argparse
import re
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
#from datasets import load_dataset
import logging
import dabstep_benchmark

# --- prefer local agent_core (this dir) for vendored packages ---
_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))


# ---------------- logging ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

def _reconfigure_logging(log_file: str):
    """Rewire logging to also write to the given file."""
    for h in list(logging.root.handlers):
        logging.root.removeHandler(h)
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, mode="w", encoding="utf-8"),
        ],
    )
    logging.getLogger(__name__).info("Log file -> %s", os.path.abspath(log_file))

_HAS_OFFICIAL = False
official_evaluate = None
try:
    from dabstep_benchmark.utils import evaluate as official_evaluate
    _HAS_OFFICIAL = True
    logger.info("Using evaluator from dabstep_benchmark (vendored/site-packages depending on sys.path)")
except Exception as e:
    logger.warning("Cannot import dabstep_benchmark.utils.evaluate: %r", e)

import inspect
logger.info("Evaluator module file: %s", inspect.getfile(dabstep_benchmark))


# ============================================================================
# 数据结构
# ============================================================================
@dataclass
class TaskResult:
    task_id: str
    agent_answer: str
    status: str
    error_msg: Optional[str] = None
    steps_used: int = 0
    trace: Optional[List[Dict[str, Any]]] = None   # ★ 新增



# ============================================================================
# 工具函数
# ============================================================================
def prepare_data_dir(data_dir: str) -> Path:
    p = Path(data_dir).resolve()
    p.mkdir(parents=True, exist_ok=True)
    logger.info("Data directory: %s", p)
    return p

def _pack_paths(data_dir: str) -> Dict[str, Path]:
    base = Path(data_dir).resolve()
    pack = base / "pack"
    tasks_dir = pack / "tasks"
    context_dir = pack / "context"
    # 容错：若 pack/context 不存在，回退到 base/context
    if not context_dir.exists() and (base / "context").exists():
        context_dir = base / "context"
    if not tasks_dir.exists() and (base / "tasks").exists():   # 新增：tasks 兜底
        tasks_dir = base / "tasks"
    return {"pack": pack, "tasks_dir": tasks_dir, "context_dir": context_dir}

def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def load_dabstep_dataset(split: str = "default", limit: Optional[int] = None, data_dir: str = ".") -> list:
    """
    OFFLINE: read tasks from <data_dir>/pack/tasks/{split}_tasks.jsonl
    """
    paths = _pack_paths(data_dir)
    tasks_file = paths["tasks_dir"] / f"{split}_tasks.jsonl"
    if not tasks_file.exists():
        raise FileNotFoundError(
            f"Tasks file not found: {tasks_file}\n"
            f"请先将离线任务放到此处，或使用你的下载脚本生成 {split}_tasks.jsonl。"
        )
    rows = _read_jsonl(tasks_file)
    if limit and limit > 0:
        rows = rows[:limit]
        logger.info("Smoke mode: using first %d rows from %s", limit, tasks_file)
    else:
        logger.info("Loaded %d rows from %s", len(rows), tasks_file)
    return rows

def extract_code_from_response(response_text: str) -> Optional[str]:
    m = re.search(r"```python\n(.*?)\n```", response_text, re.DOTALL)
    if m:
        return m.group(1)
    m = re.search(r"```(?:code|python)?\n(.*?)\n```", response_text, re.DOTALL)
    if m:
        return m.group(1)
    return None

def extract_final_answer(text: str) -> Optional[str]:
    m = re.search(r"FINAL ANSWER:\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    m = re.search(r"answer:\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return None

def _env_float(name: str, default: float) -> float:
    v = os.getenv(name, None)
    try:
        return float(v) if v not in (None, "", "None", "null", "NaN") else default
    except (TypeError, ValueError):
        return default
    
def _fmt_temp(x):
    try:
        return f"{float(x):.2f}"
    except (TypeError, ValueError):
        return "unset"
    
_exec_globals_cache = {}
def execute_code(code: str, data_dir: str, step: int, is_new_task: bool = False) -> Dict[str, Any]:
    """
    执行 Python 代码并返回结果；跨 step 复用变量。
    data_dir: 传 env 的 data_dir；文件实际在 data_dir/pack/context 下。
    """
    import json as _json
    import math as _math
    import statistics as _statistics
    import itertools as _itertools
    import datetime as _datetime
    import csv as _csv
    import re as _re
    import os as _os
    global _exec_globals_cache
    if is_new_task:
        _exec_globals_cache = {}
    try:
        if not _exec_globals_cache:
            _exec_globals_cache = {
                "pd": pd, "pandas": pd,
                "json": _json, "math": _math, "statistics": _statistics,
                "itertools": _itertools, "datetime": _datetime, "re": _re, "csv": _csv, "os": _os,
            }
            try:
                import numpy as np
                _exec_globals_cache["np"] = np
                _exec_globals_cache["numpy"] = np
            except Exception:
                pass

        # 传递 context_dir 给用户代码使用
        paths = _pack_paths(data_dir)
        context_dir = str(paths["context_dir"])
        exec_locals = {"data_dir": data_dir, "context_dir": context_dir}

        outputs: List[str] = []
        def custom_print(*args, **kwargs):
            s = " ".join(str(a) for a in args)
            outputs.append(s)
            print(s)
        _exec_globals_cache["print"] = custom_print

        logger.debug("Step %d - Executing code:\n%s", step, code)
        exec(code, _exec_globals_cache, exec_locals)
        _exec_globals_cache.update(exec_locals)
        return {"success": True, "output": "\n".join(outputs), "locals": exec_locals, "error": None}
    except Exception as e:
        logger.warning("Step %d - Code execution failed: %s: %s", step, type(e).__name__, e)
        return {"success": False, "output": "", "locals": {}, "error": f"{type(e).__name__}: {e}"}

def make_client(base_url: str | None, api_key: str | None) -> OpenAI:
    base = (base_url
            or os.getenv("AGENT_BASE_URL")
            or os.getenv("OPENAI_API_BASE")
            or "https://api.openai.com/v1").strip()
    key  = (api_key
            or os.getenv("AGENT_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or "EMPTY").strip()
    return OpenAI(base_url=base, api_key=key)

def solve_task_with_agent(
    task: Dict[str, Any],
    model_id: str,
    client: OpenAI,
    temperature: float = 0.0,
    max_steps: int = 10,
    timeout: int = 60,
    data_dir: str = "./data",
    trace: Optional[List[Dict[str, Any]]] = None
) -> TaskResult:
    """
    多步 agent：思考 -> 执行代码 -> 观察 -> 重复
    - 使用传入的 client 和 temperature，不再在函数内部重读环境覆盖
    """
    trace = [] if trace is None else list(trace)

    task_id = task.get("task_id", "unknown")
    question = task.get("question", "")
    context = task.get("context", "")
    answer_format = task.get("answer_format", "")

    # 解析 context_dir（优先 pack/context）
    paths = _pack_paths(data_dir)
    context_dir = str(paths["context_dir"])

    # 兜底：保证 temperature 为 float
    try:
        temperature = 0.0 if temperature is None else float(temperature)
    except (TypeError, ValueError):
        temperature = 0.0

    logger.info(f"Solving task: {task_id}")

    try:
        # 初始化对话
        messages = [
            {
                "role": "system",
                "content": f"""You are a data analysis expert. You will solve problems step by step.

                        For each step:
                        1. Think about what needs to be done
                        2. Write Python code to do it (wrap in ```python ... ```)
                        3. I will execute it and show you the results
                        4. Continue until you can provide the FINAL ANSWER

                        **IMPORTANT - Data Files Location:**
                        The data files are located in: {os.path.abspath(context_dir)}

                        Available data files:
                        - payments.csv (payment transaction data)
                        - fees.json (fee information)
                        - merchant_data.json (merchant information)
                        - merchant_category_codes.csv (MCC codes)
                        - acquirer_countries.csv (country data)
                        - manual.md (reference manual)

                        **How to load files - ALWAYS use this pattern:**
                        ```python
                        import pandas as pd
                        import os
                        import json

                        context_dir = r'{os.path.abspath(context_dir)}'
                        # Load CSV files
                        df_payments = pd.read_csv(os.path.join(context_dir, 'payments.csv'))
                        df_codes = pd.read_csv(os.path.join(context_dir, 'merchant_category_codes.csv'))

                        # Load JSON files
                        with open(os.path.join(context_dir, 'fees.json'), 'r', encoding='utf-8') as f:
                            fees = json.load(f)
                        ```

                        Common libraries available: pandas, numpy, json, math, statistics, datetime, re

                        When you have the final answer, write:
                        FINAL ANSWER: <your answer here>

                        ...
                        Answer format rules:
                        - Lists: comma-separated, NO spaces (e.g., "1,2,3")
                        - Numbers: follow the precision specified
                        - Strings: no extra prefixes

                        **IMPORTANT - Output:**
                        Always use `print()` to display results so I can see them.
                        For example:
                        - Use `print(df.columns)` not just `df.columns`
                        - Use `print(df.head())` not just `df.head()`
                        - Use `print(result)` for final results
                        """
            },
            {
                "role": "user",
                "content": f"""Please solve this step-by-step:

                        Context:
                        {context}

                        Question:
                        {question}

                        Answer format requirement:
                        {answer_format}

                        Start by analyzing the problem and writing code to explore the data."""
            }
        ]

        # 多步循环
        for step in range(1, max_steps + 1):
            logger.info(f"  Step {step}/{max_steps}...")

            # 调用 LLM
            try:
                resp = client.chat.completions.create(
                    model=model_id,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=2000,
                    timeout=timeout,  # openai>=1.x 一般可用；如需兼容旧版可换 request_timeout
                )
            except Exception as e:
                logger.error(f"  LLM call failed: {e}")
                return TaskResult(
                    task_id=task_id,
                    agent_answer="",
                    status="failed",
                    error_msg=f"LLM error: {str(e)}",
                    steps_used=step,
                    trace=trace,
                )

            response_text = resp.choices[0].message.content
            logger.info(f"  LLM Response (Step {step}):\n{response_text}\n")
            event = {"step": step, "assistant_message": response_text}

            # 检查是否有最终答案
            final_answer = extract_final_answer(response_text)
            if final_answer:
                event["final_answer"] = final_answer
                trace.append(event)
                logger.info(f"✓ Task {task_id} solved in {step} steps")
                logger.info(f"  Answer: {final_answer[:60]}...")
                return TaskResult(
                    task_id=task_id,
                    agent_answer=final_answer,
                    status="success",
                    steps_used=step,
                    trace=trace
                )

            # 提取代码并执行
            code = extract_code_from_response(response_text)
            if code:
                exec_result = execute_code(code, data_dir, step, is_new_task=(step == 1))

                event["code"] = code
                event["exec"] = {
                    "success": bool(exec_result.get("success")),
                    "output":  exec_result.get("output") or "",
                    "error":   exec_result.get("error") or "",
                }
                trace.append(event)

                messages.append({"role": "assistant", "content": response_text})

                if exec_result['success']:
                    result_message = f"""Code executed successfully.
                                        Output:
                                        {exec_result['output']}
                                        Continue with the next step or provide the FINAL ANSWER when ready."""
                else:
                    result_message = f"""Code execution failed:
                                        Error: {exec_result['error']}
                                        Please fix the code and try again."""

                messages.append({"role": "user", "content": result_message})
                logger.info(f"  Execution result: {'success' if exec_result['success'] else 'failed'}\n  Output:\n{exec_result['output']}\n")
            else:
                # 没有代码，也没有最终答案 → 继续提示
                trace.append(event)
                logger.warning(f"  No code or final answer in step {step}")
                messages.append({"role": "assistant", "content": response_text})
                messages.append({"role": "user", "content": "Please write Python code to solve this or provide the FINAL ANSWER."})

        # 超过最大步数
        logger.warning(f"Task {task_id} exceeded max steps ({max_steps})")
        return TaskResult(
            task_id=task_id,
            agent_answer="",
            status="failed",
            error_msg=f"Exceeded maximum steps ({max_steps})",
            steps_used=max_steps,
            trace=trace,
        )

    except Exception as e:
        logger.error(f"Task {task_id} failed: {type(e).__name__}: {e}")
        return TaskResult(
            task_id=task_id,
            agent_answer="",
            status="failed",
            error_msg=str(e),
            steps_used=0,
            trace=trace,
        )

# ===== approximate scorer fallback (dev only) =====
def _try_parse_float(s: str):
    try:
        return float(s)
    except Exception:
        return None

def _normalize_str(s: str) -> str:
    import re
    return re.sub(r"\s+", "", str(s)).strip().lower()

def _normalize_answer(ans: str):
    """
    近似规则（用于本地兜底）：
    - 逗号分隔 → 列表；能转 float 就当数值列表；否则当字符串列表（去空白/小写）
    - 单值 → 能转 float 就当数值；否则当规范化字符串
    """
    s = str(ans).strip()
    if "," in s:
        items = [x.strip() for x in s.split(",") if x.strip() != ""]
        floats, all_float = [], True
        for it in items:
            v = _try_parse_float(it)
            if v is None:
                all_float = False
                break
            floats.append(v)
        if all_float:
            return ("list_float", tuple(round(v, 8) for v in floats))
        return ("list_str", tuple(_normalize_str(it) for it in items))
    v = _try_parse_float(s)
    if v is not None:
        return ("float", round(v, 8))
    return ("str", _normalize_str(s))

def _equal_with_tolerance(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(a - b) <= tol

def _score_one_dev_approx(gt: str, pred: str) -> bool:
    ta, tb = _normalize_answer(gt), _normalize_answer(pred)
    if ta[0] != tb[0]:
        # 类型不同时退回到纯字符串等价（去空白/小写）
        return _normalize_str(gt) == _normalize_str(pred)
    kind = ta[0]
    if kind == "float":
        return _equal_with_tolerance(ta[1], tb[1])
    if kind == "list_float":
        if len(ta[1]) != len(tb[1]):
            return False
        return all(_equal_with_tolerance(x, y) for x, y in zip(ta[1], tb[1]))
    if kind in ("str", "list_str"):
        return ta[1] == tb[1]
    return False

def _approx_score_dev_all(tasks: list, results: list[TaskResult]) -> dict:
    """dev split 近似打分（不上榜，仅兜底）。"""
    gt = {str(t["task_id"]): str(t.get("answer", "")) for t in tasks if "answer" in t}
    rows, correct, total = [], 0, 0
    for r in results:
        tid = str(r.task_id)
        if tid not in gt:
            continue
        total += 1
        ok = _score_one_dev_approx(gt[tid], str(r.agent_answer))
        correct += 1 if ok else 0
        rows.append({"task_id": tid, "score": bool(ok), "pred": str(r.agent_answer), "gt": gt[tid]})
    acc = (correct / total) if total else 0.0
    return {"accuracy": acc, "total": total, "correct": correct, "details": rows}

def score_dev_all(tasks: list, results: list[TaskResult]) -> dict:
    import pandas as pd

    # --- standardized GT---
    tasks_df = pd.DataFrame(tasks)
    if "task_id" not in tasks_df.columns or "answer" not in tasks_df.columns:
        m = _approx_score_dev_all(tasks, results); m["scorer"] = "approx"; return m
    if "level" not in tasks_df.columns:
        tasks_df["level"] = ""  # 若你的离线 dev jsonl 没 level，这样兜一下
    tasks_df = tasks_df[["task_id", "answer", "level"]].copy()
    tasks_df["task_id"] = tasks_df["task_id"].astype(str)
    tasks_df["answer"]  = tasks_df["answer"].astype(str)
    tasks_df["level"]   = tasks_df["level"].astype(str)

    # --- collect prediction---
    rows = []
    for r in results:
        tid   = str(getattr(r, "task_id", ""))
        ans   = str(getattr(r, "agent_answer", "") or "")
        okrun = (getattr(r, "status", "") == "success") and (ans.strip() != "")
        rows.append({"task_id": tid, "agent_answer": ans if okrun else None, "answered": bool(okrat:=okrun)})

    if not rows:
        return {"accuracy": 0.0, "total": len(tasks_df), "correct": 0,
                "details": [{"task_id": str(tid), "score": False} for tid in tasks_df["task_id"]],
                "scorer": "approx"}

    pred_df = pd.DataFrame(rows).drop_duplicates(subset=["task_id"], keep="last")
    df = tasks_df.merge(pred_df, on="task_id", how="left")

    df["score"] = False

    # answered
    mask = df["answered"] == True
    if mask.any():
        try:
            if _HAS_OFFICIAL and official_evaluate is not None:
                # 仅对“已作答子集”做官方打分，避免 KeyError: Task ID not found
                sub_agent = df.loc[mask, ["task_id", "agent_answer"]].copy()
                sub_tasks = df.loc[mask, ["task_id", "answer", "level"]].copy()

                out = official_evaluate(agent_answers=sub_agent, tasks_with_gt=sub_tasks)

                # 兼容返回多形态
                if hasattr(out, "to_dict") and ("score" in getattr(out, "columns", [])):
                    scored = out[["task_id", "score"]].copy()
                elif isinstance(out, dict) and "per_question" in out:
                    scored = pd.DataFrame(out["per_question"])[["task_id", "score"]]
                elif isinstance(out, list):
                    if len(out) > 0 and isinstance(out[0], dict) and {"task_id","score"} <= out[0].keys():
                        scored = pd.DataFrame(out)[["task_id","score"]]
                    else:
                        # 退化为与 sub_agent 顺序一致的布尔列表
                        scored = sub_agent[["task_id"]].copy()
                        scored["score"] = [bool(x) for x in out]
                else:
                    raise RuntimeError(f"Unexpected official_evaluate return type: {type(out)}")

                scored["task_id"] = scored["task_id"].astype(str)
                df = df.drop(columns=["score"]).merge(scored, on="task_id", how="left")
                df["score"] = df["score"].fillna(False).astype(bool)
                scorer_tag = "official+unanswered_zero"
            else:
                # 没有官方 → 回退近似（仅对已作答子集）
                sub_tasks = df.loc[mask, ["task_id", "answer"]].rename(columns={"answer":"answer"})
                sub_results = []
                for _, row in df.loc[mask, ["task_id", "agent_answer"]].iterrows():
                    sub_results.append(TaskResult(task_id=str(row["task_id"]), agent_answer=str(row["agent_answer"]),
                                                  status="success"))
                tmp = _approx_score_dev_all(sub_tasks.to_dict("records"), sub_results)
                scored = pd.DataFrame(tmp["details"])[["task_id","score"]]
                scored["task_id"] = scored["task_id"].astype(str)
                df = df.drop(columns=["score"]).merge(scored, on="task_id", how="left")
                df["score"] = df["score"].fillna(False).astype(bool)
                scorer_tag = "approx+unanswered_zero"
        except Exception as e:
            logger.warning("official_evaluate failed, falling back to approx: %s: %s", type(e).__name__, e)
            sub_tasks = df.loc[mask, ["task_id", "answer"]].rename(columns={"answer":"answer"})
            sub_results = []
            for _, row in df.loc[mask, ["task_id", "agent_answer"]].iterrows():
                sub_results.append(TaskResult(task_id=str(row["task_id"]), agent_answer=str(row["agent_answer"]),
                                              status="success"))
            tmp = _approx_score_dev_all(sub_tasks.to_dict("records"), sub_results)
            scored = pd.DataFrame(tmp["details"])[["task_id","score"]]
            scored["task_id"] = scored["task_id"].astype(str)
            df = df.drop(columns=["score"]).merge(scored, on="task_id", how="left")
            df["score"] = df["score"].fillna(False).astype(bool)
            scorer_tag = "approx_fallback+unanswered_zero"
    else:
        scorer_tag = "no_answer_all_zero"

    total   = len(df)
    correct = int(df["score"].sum())
    acc     = float(correct) / total if total else 0.0
    details = [{"task_id": str(row.task_id), "score": bool(row.score)} for row in df.itertuples(index=False)]

    return {"accuracy": acc, "total": total, "correct": correct, "details": details, "scorer": scorer_tag}

# ============================================================================
# main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description=(
            "DABStep multi-step agent with ReAct pattern.\n"
            "Supports offline 'pack' layout: set --data-dir to the directory that "
            "contains 'pack/context' and 'pack/tasks'."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples (offline, Windows):
            python dabstep_runner.py --split dev --limit 1 ^
                --data-dir .\\env\\bench_env\\dabstep ^
                --out .\\env\\bench_env\\dabstep\\artifacts\\submission.jsonl ^
                --dev-metrics-out .\\env\\bench_env\\dabstep\\artifacts\\dev_metrics.json
            """,
    )

    parser.add_argument("--model-id", type=str, default="gpt-4o-mini")
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--limit", type=int, default=0, help="0 = all")
    parser.add_argument("--out", type=str, default="submission.jsonl")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--split", type=str, choices=["default", "dev"], default="default")
    parser.add_argument("--dev-metrics-out", type=str, default="dev_metrics.json")
    parser.add_argument("--log-file", type=str, default="")
    parser.add_argument("--base-url", type=str, default="", help="OpenAI-compatible base URL")
    parser.add_argument("--api-key",  type=str, default="", help="API key (optional)")
    parser.add_argument("--temperature", type=float, default=None, help="Sampling temperature (overrides default)")

    parser.add_argument("--agent-base-url", type=str, default=None, help="Alias of --base-url")
    parser.add_argument("--agent-api-key",  type=str, default=None, help="Alias of --api-key")
    parser.add_argument("--agent-temperature", type=float, default=None, help="Alias of --temperature")
    # ... 原有 args 定义后面补一个
    parser.add_argument("--only-task-id", type=str, default="",
                        help="If set, solve only the given task_id from the dataset")

    args = parser.parse_args()
    if args.agent_base_url is not None:
        args.base_url = args.agent_base_url
    if args.agent_api_key is not None:
        args.api_key = args.agent_api_key
    if args.agent_temperature is not None:
        args.temperature = args.agent_temperature
    # 计算“数值温度”：优先 CLI，其次环境变量 AGENT_TEMPERATURE，否则 0.0
    temp_num = args.temperature if args.temperature is not None else _env_float("AGENT_TEMPERATURE", 0.0)

    # 构造客户端（支持内网/外网）
    client = make_client(args.base_url or None, args.api_key or None)

    # 这里用字符串打印温度，避免 None 导致 %.2f TypeError
    logger.info(
        "LLM base=%s | model=%s | temp=%s",
        (args.base_url or os.getenv("AGENT_BASE_URL") or os.getenv("OPENAI_API_BASE") or "https://api.openai.com/v1"),
        args.model_id,
        _fmt_temp(temp_num),
    )

    _log_path = args.log_file.strip() or str(Path(args.out).with_name("dabstep_run.log"))
    _reconfigure_logging(_log_path)

    logger.info("=" * 70)
    logger.info("DABstep Multi-Step Agent (ReAct Pattern)")
    logger.info("=" * 70)
    logger.info(f"Model: {args.model_id}")
    logger.info(f"Max steps per task: {args.max_steps}")
    logger.info(f"Number of tasks: {args.limit if args.limit > 0 else 'all'}")
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Timeout: {args.timeout}s")
    logger.info(f"Output file: {args.out}")

    output_path = Path(args.out).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 准备数据目录
    prepare_data_dir(args.data_dir)

    # 读取任务（离线）
    tasks = load_dabstep_dataset(
        split=args.split,
        limit=(args.limit if args.limit > 0 else None),
        data_dir=args.data_dir,
    )

    if args.only_task_id:
        tasks = [t for t in tasks if str(t.get("task_id")) == str(args.only_task_id)]
        if not tasks:
            raise FileNotFoundError(f"No task with task_id={args.only_task_id} in this split.")
    if args.limit and args.limit > 0:
        tasks = tasks[: args.limit]

    logger.info("Loaded %d tasks (split=%s)", len(tasks), args.split)

    results: List[TaskResult] = []
    merged_trace_lines: List[str] = []

    # 逐个求解
    for idx, task in enumerate(tasks, start=1):
        logger.info(f"\n[{idx}/{len(tasks)}] Processing task {task.get('task_id', 'unknown')}...")
        result = solve_task_with_agent(
            task,
            model_id=args.model_id,
            client=client,
            temperature=temp_num,           # ★ 使用 main 里算好的数值温度
            max_steps=args.max_steps,
            timeout=args.timeout,
            data_dir=args.data_dir,
        )
        results.append(result)

        # 写出本任务 trace 到与 submission 同目录（即 out_dir）
        try:
            out_dir = Path(args.out).resolve().parent
            out_dir.mkdir(parents=True, exist_ok=True)
            trace_path = out_dir / "trace.jsonl"
            with trace_path.open("w", encoding="utf-8") as tf:
                for ev in (result.trace or []):
                    ev2 = dict(ev)
                    ev2.setdefault("task_id", str(result.task_id))
                    tf.write(json.dumps(ev2, ensure_ascii=False) + "\n")
            logger.info("Trace written -> %s", trace_path)
        except Exception as e:
            logger.warning("Failed to write trace.jsonl: %s", e)


        # 写出本任务 trace 到与 submission 同目录（即 out_dir）
        #try:
        #    tpath = Path(args.out).with_name(f"trace_{result.task_id}.jsonl")
        #    tpath.parent.mkdir(parents=True, exist_ok=True)
        #    with tpath.open("w", encoding="utf-8") as tf:
        #        for ev in (result.trace or []):
        #            ev2 = dict(ev)
        #            ev2.setdefault("task_id", str(result.task_id))
        #            tf.write(json.dumps(ev2, ensure_ascii=False) + "\n")
        #    logger.info("Trace written -> %s", tpath.resolve())
        #except Exception as e:
        #    logger.warning("Failed to write trace.jsonl: %s", e)


        # 每个任务单独 trace：trace_<taskid>.jsonl（放在 --out 同目录）
        #try:
        #    per_task_path = output_path.with_name(f"trace_{result.task_id}.jsonl")
        #    with per_task_path.open("w", encoding="utf-8") as tf:
        #        for ev in (result.trace or []):
        #            ev2 = dict(ev)
        #            ev2.setdefault("task_id", str(result.task_id))
        #            tf.write(json.dumps(ev2, ensure_ascii=False) + "\n")
        #    logger.info("Per-task trace written -> %s", per_task_path)
        #except Exception as e:
        #    logger.warning("Failed to write per-task trace for %s: %s", result.task_id, e)

        # 累积到合并 trace
        for ev in (result.trace or []):
            merged_trace_lines.append(
                json.dumps({**ev, "task_id": str(result.task_id)}, ensure_ascii=False)
            )

        success_count = sum(1 for r in results if r.status == "success")
        logger.info("Progress: %d/%d", success_count, len(results))

    # 写 submission.jsonl
    with output_path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps({"task_id": str(r.task_id), "agent_answer": str(r.agent_answer)},
                               ensure_ascii=False) + "\n")

    # dev 本地评分
    if args.split == "dev":
        try:
            metrics = score_dev_all(tasks, results)
            mpath = Path(args.dev_metrics_out).resolve()
            mpath.parent.mkdir(parents=True, exist_ok=True)
            mpath.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
            logger.info("[DEV] Local accuracy: %.3f (%d/%d) -> %s",
                        metrics.get("accuracy", 0.0), metrics.get("correct", 0),
                        metrics.get("total", 0), mpath)
        except Exception as e:
            logger.warning("[DEV] Local scoring failed: %s", e)

    if args.split == "default":
        logger.info("Default split has no ground-truth → skipping evaluate(). Upload submission.jsonl on the official page to get the score.")

    logger.info("\n" + "=" * 70)
    logger.info("✓ Generated %s", output_path)
    logger.info("Split: %s  |  Output file: %s", args.split, args.out)

    total = len(results)
    success = sum(1 for r in results if r.status == "success")
    failed = total - success
    avg_steps = (sum(r.steps_used for r in results if r.status == "success") / success) if success else 0.0
    logger.info("Total: %d | Success: %d | Failed: %d | Avg steps(success): %.1f",
                total, success, failed, avg_steps)

    if failed > 0:
        logger.warning("\nFailed tasks:")
        for r in results:
            if r.status != "success":
                logger.warning("  - %s: %s", r.task_id, r.error_msg)

    logger.info("\nCheck dabstep_run.log for details.")


if __name__ == "__main__":
    main()