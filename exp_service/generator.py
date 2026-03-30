"""
ExpGenerator: 使用 LLM 从轨迹批次中生成 Experience。

按 eval_mode + env_type 路由到专用 prompt：

  env_type 含 "os" + eval_mode="safety"  → generate_os_safety.md
  env_type 含 "os" + eval_mode="standard" → generate_os_standard.md
  其他 env_type + eval_mode="safety"      → generate_safety_lesson.md（失败轨迹）
                                           → generate_exp.md（成功轨迹）
  其他 env_type + eval_mode="standard"    → generate_exp.md
  失败轨迹（positive=False）              → generate_failure_exp.md / generate_safety_lesson.md
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional

from .config import GeneratorConfig, LLMConfig
from .trajectory import TrajectoryRecord

log = logging.getLogger("exp_service.generator")

_PROMPTS_DIR = Path(__file__).parent / "prompts"

# env_type 关键词 → OS 类环境
_OS_ENV_KEYWORDS = ("os_gym", "desktop", "os", "ubuntu", "windows", "computer_use")


# ------------------------------------------------------------------ #
# 数据模型
# ------------------------------------------------------------------ #

@dataclass
class ExpDraft:
    """LLM 生成的经验草稿，尚未写入文件系统"""
    name: str
    env_type: str
    description: str
    trigger: str
    content: str
    eval_mode: str = "standard"
    is_safety_lesson: bool = False      # 来自安全失败轨迹的教训
    is_failure_lesson: bool = False     # 来自任务失败轨迹（非安全）
    source_sessions: List[str] = field(default_factory=list)
    estimated_success_rate: float = 0.5


# ------------------------------------------------------------------ #
# LLM 调用封装
# ------------------------------------------------------------------ #

LLMGenerateFn = Callable[[List[Dict[str, Any]]], Awaitable[str]]


def make_openai_llm(cfg: LLMConfig) -> LLMGenerateFn:
    """根据 LLMConfig 构建 OpenAI 兼容的异步 LLM 调用函数。"""
    import openai

    client = openai.AsyncOpenAI(
        api_key=cfg.api_key,
        base_url=cfg.base_url,
        timeout=cfg.timeout_s,
    )
    semaphore = asyncio.Semaphore(cfg.max_concurrency)

    async def _generate(messages: List[Dict[str, Any]]) -> str:
        async with semaphore:
            resp = await client.chat.completions.create(
                model=cfg.model,
                messages=messages,
                temperature=cfg.temperature,
                max_tokens=cfg.max_tokens,
            )
            return resp.choices[0].message.content or ""

    return _generate


# ------------------------------------------------------------------ #
# Prompt 路由逻辑
# ------------------------------------------------------------------ #

def _is_os_env(env_type: str) -> bool:
    t = env_type.lower()
    return any(kw in t for kw in _OS_ENV_KEYWORDS)


def _select_prompt(
    env_type: str,
    eval_mode: str,
    positive: bool,
) -> str:
    """
    根据环境类型、评分模式和轨迹正负性选择 prompt 文件名。

    返回文件名（不含路径）。
    """
    if not positive:
        # 失败/风险轨迹
        if eval_mode == "safety":
            return "generate_safety_lesson.md"
        return "generate_failure_exp.md"

    # 成功/正向轨迹
    if _is_os_env(env_type):
        if eval_mode == "safety":
            return "generate_os_safety.md"
        return "generate_os_standard.md"

    return "generate_exp.md"


# ------------------------------------------------------------------ #
# ExpGenerator
# ------------------------------------------------------------------ #

class ExpGenerator:
    """
    从轨迹批次中生成 ExpDraft。

    Args:
        llm_generate:  异步 LLM 调用函数
        cfg:           GeneratorConfig
    """

    def __init__(self, llm_generate: LLMGenerateFn, cfg: Optional[GeneratorConfig] = None):
        self._llm = llm_generate
        self._cfg = cfg or GeneratorConfig()
        self._prompt_cache: Dict[str, str] = {}

    def _load_prompt(self, filename: str) -> str:
        if filename not in self._prompt_cache:
            p = _PROMPTS_DIR / filename
            if p.exists():
                self._prompt_cache[filename] = p.read_text(encoding="utf-8")
            else:
                log.warning("prompt file not found: %s", p)
                self._prompt_cache[filename] = ""
        return self._prompt_cache[filename]

    async def generate_from_success(
        self, trajectories: List[TrajectoryRecord]
    ) -> List[ExpDraft]:
        """
        从成功/正向轨迹中提取经验。
        按 (env_type, eval_mode) 分组后分别处理，使用专用 prompt。
        """
        if not trajectories:
            return []

        # 分组：(env_type, eval_mode) → List[TrajectoryRecord]
        groups: Dict[tuple, List[TrajectoryRecord]] = {}
        for t in trajectories:
            key = (_infer_group_env_type(t.env_type), t.eval_mode)
            groups.setdefault(key, []).append(t)

        all_drafts: List[ExpDraft] = []
        for (env_type, eval_mode), group in groups.items():
            prompt_file = _select_prompt(env_type, eval_mode, positive=True)
            batches = _batch(group, self._cfg.batch_size)
            for batch in batches:
                drafts = await self._process_batch(
                    batch, env_type=env_type, eval_mode=eval_mode,
                    prompt_file=prompt_file, positive=True,
                )
                all_drafts.extend(drafts)

        log.info(
            "generate_from_success: %d trajectories -> %d drafts",
            len(trajectories), len(all_drafts),
        )
        return all_drafts

    async def generate_from_failure(
        self, trajectories: List[TrajectoryRecord]
    ) -> List[ExpDraft]:
        """
        从失败/风险轨迹中提取教训。
        safety 模式下区分风险触发（score<=1）与任务失败。
        """
        if not trajectories:
            return []

        groups: Dict[tuple, List[TrajectoryRecord]] = {}
        for t in trajectories:
            key = (_infer_group_env_type(t.env_type), t.eval_mode)
            groups.setdefault(key, []).append(t)

        all_drafts: List[ExpDraft] = []
        for (env_type, eval_mode), group in groups.items():
            prompt_file = _select_prompt(env_type, eval_mode, positive=False)
            batches = _batch(group, self._cfg.batch_size)
            for batch in batches:
                drafts = await self._process_batch(
                    batch, env_type=env_type, eval_mode=eval_mode,
                    prompt_file=prompt_file, positive=False,
                )
                all_drafts.extend(drafts)

        log.info(
            "generate_from_failure: %d trajectories -> %d drafts",
            len(trajectories), len(all_drafts),
        )
        return all_drafts

    # ---------------------------------------------------------------- #

    async def _process_batch(
        self,
        batch: List[TrajectoryRecord],
        env_type: str,
        eval_mode: str,
        prompt_file: str,
        positive: bool,
    ) -> List[ExpDraft]:
        traj_text = "\n\n---\n\n".join(t.to_text(max_turns=15) for t in batch)
        tmpl = self._load_prompt(prompt_file)
        if not tmpl:
            log.warning("empty prompt template: %s", prompt_file)
            return []

        prompt_text = (
            tmpl
            .replace("{trajectories}", traj_text)
            .replace("{env_type}", env_type)
            .replace("{max_exps}", str(self._cfg.max_exps_per_batch))
        )
        messages = [{"role": "user", "content": prompt_text}]
        try:
            response = await self._llm(messages)
        except Exception as e:
            log.error(
                "LLM call failed (env=%s eval=%s prompt=%s): %s",
                env_type, eval_mode, prompt_file, e,
            )
            return []

        source_sessions = [t.session_id for t in batch]
        is_safety_lesson = (eval_mode == "safety" and not positive)
        is_failure_lesson = (not positive and not is_safety_lesson)

        drafts = _parse_exp_json(
            response,
            env_type=env_type,
            eval_mode=eval_mode,
            source_sessions=source_sessions,
            is_safety_lesson=is_safety_lesson,
            is_failure_lesson=is_failure_lesson,
            positive=positive,
        )
        log.debug(
            "batch (%s/%s/%s): %d traj -> %d drafts",
            env_type, eval_mode, prompt_file, len(batch), len(drafts),
        )
        return drafts


# ------------------------------------------------------------------ #
# 工具函数
# ------------------------------------------------------------------ #

def _load_prompt(filename: str) -> str:
    """模块级辅助函数，供 updater.py 等外部模块使用。"""
    p = _PROMPTS_DIR / filename
    if p.exists():
        return p.read_text(encoding="utf-8")
    log.warning("prompt file not found: %s", p)
    return ""


def _infer_group_env_type(env_type: str) -> str:
    """将细粒度 env_type 归并到粗粒度组（用于 prompt 路由）。"""
    t = env_type.lower()
    if any(kw in t for kw in _OS_ENV_KEYWORDS):
        return "os_gym"
    return env_type


def _batch(items: List[Any], size: int) -> List[List[Any]]:
    return [items[i: i + size] for i in range(0, len(items), size)]


def _parse_exp_json(
    text: str,
    env_type: str,
    eval_mode: str,
    source_sessions: List[str],
    is_safety_lesson: bool = False,
    is_failure_lesson: bool = False,
    positive: bool = True,
) -> List[ExpDraft]:
    """从 LLM 响应中解析 ExpDraft 列表。"""
    raw = _extract_json_block(text)
    if not raw:
        log.warning("no JSON found in LLM response (len=%d)", len(text))
        return []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        log.warning("JSON parse error: %s", e)
        return []

    if data.get("no_extraction"):
        log.debug("LLM decided no_extraction: %s", data.get("reason", ""))
        return []

    # 支持 "experiences" 或旧版 "skills" 键
    items = data.get("experiences") or data.get("skills") or []
    drafts = []
    for item in items:
        name = _normalize_name(item.get("name", ""))
        if not name:
            continue
        drafts.append(ExpDraft(
            name=name,
            env_type=env_type,
            description=str(item.get("description", "")).strip(),
            trigger=str(item.get("trigger", "")).strip(),
            content=str(item.get("content", "")).strip(),
            eval_mode=eval_mode,
            is_safety_lesson=is_safety_lesson,
            is_failure_lesson=is_failure_lesson,
            source_sessions=source_sessions,
            estimated_success_rate=0.3 if not positive else 0.6,
        ))
    return drafts


def _extract_json_block(text: str) -> Optional[str]:
    m = re.search(r"```json\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    m = re.search(r"```\s*([\s\S]*?)\s*```", text)
    if m:
        candidate = m.group(1).strip()
        if candidate.startswith("{"):
            return candidate
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        return m.group(0)
    return None


def _normalize_name(name: str) -> str:
    n = name.lower()
    n = re.sub(r"[\s\-]+", "_", n)
    n = re.sub(r"[^a-z0-9_]", "", n)
    n = re.sub(r"_+", "_", n).strip("_")
    return n or ""
