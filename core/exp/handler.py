"""
EpisodeHandler: episode prompt 处理钩子。

内置实现：
  NullEpisodeHandler            -- 默认空实现
  BaseExpEpisodeHandler         -- experience handler 基类，封装固定生命周期
  TemplateExpEpisodeHandler     -- success_rate 排序注入
  UCBExpEpisodeHandler          -- UCB 经验选择与反馈回写
  ContextualUCBExpEpisodeHandler -- prompt-aware contextual UCB
  EmbeddingExpEpisodeHandler    -- 预留实现，当前回退到 template
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, runtime_checkable

import yaml

log = logging.getLogger("exp.handler")

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
_TASK_PATTERNS = (
    re.compile(r"You are asked to complete the following task:\s*(.+?)(?:\n\s*\n|\n##|\Z)", re.IGNORECASE | re.DOTALL),
    re.compile(r"Task:\s*(.+?)(?:\n\s*\n|\n##|\Z)", re.IGNORECASE | re.DOTALL),
)


@dataclass(frozen=True)
class _CandidateEntry:
    env_type: str
    name: str
    path: str
    trigger: str
    success_rate: float
    eval_mode: str

    @property
    def text_profile(self) -> str:
        return f"{self.name} {self.trigger}".strip()


@dataclass(frozen=True)
class _SelectedExperience:
    env_type: str
    name: str
    eval_mode: str


@dataclass(frozen=True)
class _ContextualSelectionState:
    selected: _SelectedExperience
    context_vector: List[float]


@runtime_checkable
class EpisodeHandler(Protocol):
    """
    Episode prompt 处理钩子。

    Interactor 在每个 step 调用 handle()，在 episode 结束后调用 on_episode_end()。
    """

    async def handle(
        self,
        env_name: str,
        env_id: str,
        step_i: int,
        prompt: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]: ...

    async def on_episode_end(
        self,
        env_name: str,
        env_id: str,
        total_reward: float,
        info: Optional[Dict[str, Any]] = None,
    ) -> None: ...


class NullEpisodeHandler:
    """默认空实现：不修改 prompt，也不记录 episode 结果。"""

    async def handle(
        self,
        env_name: str,
        env_id: str,
        step_i: int,
        prompt: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        return prompt

    async def on_episode_end(
        self,
        env_name: str,
        env_id: str,
        total_reward: float,
        info: Optional[Dict[str, Any]] = None,
    ) -> None:
        return None


class BaseExpEpisodeHandler:
    """
    experience handler 基类。

    固定职责：
    - step_i == 1 时读取 index / SKILL.md，缓存注入内容
    - step_i > 1 时复用缓存
    - episode 结束后交由子类处理反馈回写
    """

    MODE = "base"

    def __init__(
        self,
        exp_dir: str,
        enabled: bool = False,
        top_k: int = 3,
        embedding_model: Optional[str] = None,
    ):
        self._enabled = bool(enabled)
        self._exp_dir = Path(exp_dir)
        self._top_k = max(1, int(top_k))
        self._embedding_model = embedding_model
        self._cache: Dict[str, Optional[str]] = {}
        self._feedback_state: Dict[str, Any] = {}
        self._lock = asyncio.Lock()
        log.info(
            "%s ready: enabled=%s dir=%s top_k=%d mode=%s",
            self.__class__.__name__,
            self._enabled,
            exp_dir,
            self._top_k,
            self.MODE,
        )

    async def handle(
        self,
        env_name: str,
        env_id: str,
        step_i: int,
        prompt: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if not self._enabled:
            return prompt
        if step_i == 1:
            await self._ensure_cached(env_name, env_id, prompt)
        ctx = self._cache.get(env_id)
        return _inject(prompt, ctx) if ctx else prompt

    async def on_episode_end(
        self,
        env_name: str,
        env_id: str,
        total_reward: float,
        info: Optional[Dict[str, Any]] = None,
    ) -> None:
        feedback_state = self._feedback_state.pop(env_id, None)
        self._cache.pop(env_id, None)

        if not self._enabled or feedback_state is None:
            return
        await self.record_feedback(
            env_name=env_name,
            env_id=env_id,
            feedback_state=feedback_state,
            total_reward=total_reward,
            info=info,
        )

    async def _ensure_cached(
        self,
        env_name: str,
        env_id: str,
        prompt: List[Dict[str, Any]],
    ) -> None:
        if env_id in self._cache:
            return
        async with self._lock:
            if env_id in self._cache:
                return
            ctx, feedback_state = self._retrieve_sync(env_name, prompt)
            self._cache[env_id] = ctx
            self._feedback_state[env_id] = feedback_state

    def _retrieve_sync(
        self,
        env_name: str,
        prompt: List[Dict[str, Any]],
    ) -> tuple[Optional[str], Any]:
        index_path = self._exp_dir / "index.yaml"
        if not index_path.exists():
            log.debug("exp index not found: %s", index_path)
            return None, None

        try:
            index = yaml.safe_load(index_path.read_text(encoding="utf-8")) or {}
        except Exception as e:
            log.error("failed to read experience index: %s", e)
            return None, None

        candidates = self._match_candidates(env_name, index)
        if not candidates:
            log.debug("no experience matched for env_name=%s", env_name)
            return None, None

        selected_entries = self.select_entries(
            env_name=env_name,
            prompt=prompt,
            candidates=candidates,
        )
        if not selected_entries:
            return None, None

        blocks: List[str] = []
        valid_entries: List[_CandidateEntry] = []
        for entry in selected_entries:
            exp_md = self._exp_dir / entry.path
            if not exp_md.exists():
                log.warning("experience file not found: %s", exp_md)
                continue

            try:
                content = exp_md.read_text(encoding="utf-8").strip()
            except Exception as e:
                log.warning("failed to read experience file %s: %s", exp_md, e)
                continue

            valid_entries.append(entry)
            header = f"**Experience: {entry.name}**"
            if entry.trigger:
                header += f"\n*Apply when*: {entry.trigger}"
            blocks.append(f"{header}\n\n{content}")

        if not blocks:
            return None, None

        feedback_state = self.build_feedback_state(
            env_name=env_name,
            prompt=prompt,
            entries=valid_entries,
        )
        ctx = "## Relevant Experiences\n\n" + "\n\n---\n\n".join(blocks)
        log.info(
            "%s: injecting %d experience(s) for env_name=%s",
            self.__class__.__name__,
            len(blocks),
            env_name,
        )
        return ctx, feedback_state

    def _match_candidates(self, env_name: str, index: Dict[str, Any]) -> List[_CandidateEntry]:
        envs: Dict[str, list] = index.get("environments") or {}
        candidates: List[_CandidateEntry] = []

        for env_type, entries in envs.items():
            if env_type not in env_name and env_name not in env_type:
                continue
            for entry in entries or []:
                candidates.append(
                    _CandidateEntry(
                        env_type=str(env_type),
                        name=str(entry.get("name", "")).strip(),
                        path=str(
                            entry.get("path")
                            or f"{env_type}/{entry.get('name', '')}/SKILL.md"
                        ).strip(),
                        trigger=str(entry.get("trigger", "")).strip(),
                        success_rate=float(entry.get("success_rate", 0.0) or 0.0),
                        eval_mode=str(entry.get("eval_mode", "standard")).strip().lower() or "standard",
                    )
                )
        return [candidate for candidate in candidates if candidate.name and candidate.path]

    def select_entries(
        self,
        env_name: str,
        prompt: List[Dict[str, Any]],
        candidates: List[_CandidateEntry],
    ) -> List[_CandidateEntry]:
        raise NotImplementedError

    def build_feedback_state(
        self,
        env_name: str,
        prompt: List[Dict[str, Any]],
        entries: List[_CandidateEntry],
    ) -> Any:
        return None

    async def record_feedback(
        self,
        env_name: str,
        env_id: str,
        feedback_state: Any,
        total_reward: float,
        info: Optional[Dict[str, Any]] = None,
    ) -> None:
        return None

    def _record_usage_sync(
        self,
        selected: _SelectedExperience,
        success: bool,
        risk_triggered: bool,
    ) -> None:
        try:
            from exp_service.exp_bank import ExpBank

            ExpBank(str(self._exp_dir)).record_usage(
                env_type=selected.env_type,
                name=selected.name,
                success=success,
                risk_triggered=risk_triggered,
            )
            log.info(
                "%s: updated %s/%s usage success=%s risk_triggered=%s",
                self.__class__.__name__,
                selected.env_type,
                selected.name,
                success,
                risk_triggered,
            )
        except Exception as e:
            log.error(
                "%s: failed to update %s/%s: %s",
                self.__class__.__name__,
                selected.env_type,
                selected.name,
                e,
            )


_HANDLER_REGISTRY: Dict[str, type[BaseExpEpisodeHandler]] = {}


def register_exp_handler(mode: str) -> Callable[[type[BaseExpEpisodeHandler]], type[BaseExpEpisodeHandler]]:
    normalized_mode = str(mode).strip().lower()

    def _decorator(cls: type[BaseExpEpisodeHandler]) -> type[BaseExpEpisodeHandler]:
        _HANDLER_REGISTRY[normalized_mode] = cls
        cls.MODE = normalized_mode
        return cls

    return _decorator


@register_exp_handler("template")
class TemplateExpEpisodeHandler(BaseExpEpisodeHandler):
    """按 success_rate 降序取 top_k 条经验。"""

    def select_entries(
        self,
        env_name: str,
        prompt: List[Dict[str, Any]],
        candidates: List[_CandidateEntry],
    ) -> List[_CandidateEntry]:
        del env_name, prompt
        return sorted(
            candidates,
            key=lambda entry: (entry.success_rate, entry.name),
            reverse=True,
        )[: self._top_k]


@register_exp_handler("ucb")
class UCBExpEpisodeHandler(BaseExpEpisodeHandler):
    """每个 episode 只选 1 条经验，并在结束后回写 bandit 反馈。"""

    def select_entries(
        self,
        env_name: str,
        prompt: List[Dict[str, Any]],
        candidates: List[_CandidateEntry],
    ) -> List[_CandidateEntry]:
        del env_name, prompt
        return self._rank_ucb_candidates(candidates)[:1]

    def build_feedback_state(
        self,
        env_name: str,
        prompt: List[Dict[str, Any]],
        entries: List[_CandidateEntry],
    ) -> Optional[_SelectedExperience]:
        del env_name, prompt
        if not entries:
            return None
        entry = entries[0]
        return _SelectedExperience(
            env_type=entry.env_type,
            name=entry.name,
            eval_mode=entry.eval_mode,
        )

    async def record_feedback(
        self,
        env_name: str,
        env_id: str,
        feedback_state: _SelectedExperience,
        total_reward: float,
        info: Optional[Dict[str, Any]] = None,
    ) -> None:
        del env_name, env_id
        success, risk_triggered = _derive_outcome(
            total_reward=total_reward,
            eval_mode=feedback_state.eval_mode,
            info=info,
        )
        self._record_usage_sync(
            selected=feedback_state,
            success=success,
            risk_triggered=risk_triggered,
        )

    def _rank_ucb_candidates(self, candidates: List[_CandidateEntry]) -> List[_CandidateEntry]:
        metas: List[tuple[_CandidateEntry, Dict[str, Any], float]] = []
        total_usage = 0

        for entry in candidates:
            meta = self._read_meta(entry)
            total_usage += int(meta.get("usage_count", 0))
            metas.append((entry, meta, 0.0))

        ranked: List[tuple[_CandidateEntry, Dict[str, Any], float]] = []
        for entry, meta, _ in metas:
            score = _ucb_score(meta=meta, total_usage=total_usage)
            ranked.append((entry, meta, score))

        ranked.sort(
            key=lambda item: (
                -int(int(item[1].get("usage_count", 0)) == 0),
                -item[2],
                -item[0].success_rate,
                item[0].name,
            )
        )
        return [entry for entry, _, _ in ranked]

    def _read_meta(self, entry: _CandidateEntry) -> Dict[str, Any]:
        meta_path = self._exp_dir / entry.env_type / entry.name / "meta.yaml"
        raw: Dict[str, Any] = {}

        if meta_path.exists():
            try:
                raw = yaml.safe_load(meta_path.read_text(encoding="utf-8")) or {}
            except Exception as e:
                log.warning("failed to read meta %s: %s", meta_path, e)

        return {
            "usage_count": int(raw.get("usage_count", 0) or 0),
            "success_count": int(raw.get("success_count", 0) or 0),
            "risk_triggered_count": int(raw.get("risk_triggered_count", 0) or 0),
            "eval_mode": str(raw.get("eval_mode", entry.eval_mode)).strip().lower() or "standard",
        }


@register_exp_handler("contextual_ucb")
class ContextualUCBExpEpisodeHandler(BaseExpEpisodeHandler):
    """
    prompt-aware contextual UCB。

    选择分数由两部分组成：
    - diagonal LinUCB 估计项：利用历史 prompt context 学习经验收益
    - lexical prior：当前 task prompt 与 experience(name/trigger) 的文本相关性
    """

    FEATURE_DIM = 64
    ALPHA = 1.0
    PRIOR_WEIGHT = 0.35

    def select_entries(
        self,
        env_name: str,
        prompt: List[Dict[str, Any]],
        candidates: List[_CandidateEntry],
    ) -> List[_CandidateEntry]:
        del env_name
        task_text = _extract_task_prompt_text(prompt)
        scored: List[tuple[float, _CandidateEntry]] = []

        for entry in candidates:
            context_vector = _contextual_feature_vector(
                task_text=task_text,
                entry=entry,
                dimension=self.FEATURE_DIM,
            )
            state = self._read_contextual_state(entry)
            linucb_score = _diagonal_linucb_score(
                a_diag=state["a_diag"],
                b=state["b"],
                x=context_vector,
                alpha=self.ALPHA,
            )
            lexical_prior = _lexical_similarity(task_text, entry.text_profile)
            score = linucb_score + self.PRIOR_WEIGHT * lexical_prior
            scored.append((score, entry))

        scored.sort(key=lambda item: (-item[0], item[1].name))
        return [scored[0][1]] if scored else []

    def build_feedback_state(
        self,
        env_name: str,
        prompt: List[Dict[str, Any]],
        entries: List[_CandidateEntry],
    ) -> Optional[_ContextualSelectionState]:
        del env_name
        if not entries:
            return None
        entry = entries[0]
        task_text = _extract_task_prompt_text(prompt)
        context_vector = _contextual_feature_vector(
            task_text=task_text,
            entry=entry,
            dimension=self.FEATURE_DIM,
        )
        return _ContextualSelectionState(
            selected=_SelectedExperience(
                env_type=entry.env_type,
                name=entry.name,
                eval_mode=entry.eval_mode,
            ),
            context_vector=context_vector,
        )

    async def record_feedback(
        self,
        env_name: str,
        env_id: str,
        feedback_state: _ContextualSelectionState,
        total_reward: float,
        info: Optional[Dict[str, Any]] = None,
    ) -> None:
        del env_name, env_id
        success, risk_triggered = _derive_outcome(
            total_reward=total_reward,
            eval_mode=feedback_state.selected.eval_mode,
            info=info,
        )
        self._record_usage_sync(
            selected=feedback_state.selected,
            success=success,
            risk_triggered=risk_triggered,
        )
        reward = _feedback_reward(
            success=success,
            risk_triggered=risk_triggered,
            eval_mode=feedback_state.selected.eval_mode,
        )
        self._record_contextual_feedback_sync(
            selected=feedback_state.selected,
            context_vector=feedback_state.context_vector,
            reward=reward,
        )

    def _record_contextual_feedback_sync(
        self,
        selected: _SelectedExperience,
        context_vector: List[float],
        reward: float,
    ) -> None:
        state_path = self._contextual_state_path(selected.env_type, selected.name)
        state = self._read_contextual_state_by_path(state_path)

        a_diag = list(state["a_diag"])
        b = list(state["b"])
        for i, x_i in enumerate(context_vector):
            a_diag[i] += x_i * x_i
            b[i] += reward * x_i

        state.update(
            {
                "dimension": self.FEATURE_DIM,
                "alpha": self.ALPHA,
                "pull_count": int(state.get("pull_count", 0)) + 1,
                "a_diag": a_diag,
                "b": b,
                "updated": datetime.now().isoformat(),
            }
        )
        self._write_yaml(state_path, state)
        log.info(
            "%s: updated contextual state for %s/%s reward=%.3f",
            self.__class__.__name__,
            selected.env_type,
            selected.name,
            reward,
        )

    def _read_contextual_state(self, entry: _CandidateEntry) -> Dict[str, Any]:
        return self._read_contextual_state_by_path(
            self._contextual_state_path(entry.env_type, entry.name)
        )

    def _read_contextual_state_by_path(self, state_path: Path) -> Dict[str, Any]:
        raw: Dict[str, Any] = {}
        if state_path.exists():
            try:
                raw = yaml.safe_load(state_path.read_text(encoding="utf-8")) or {}
            except Exception as e:
                log.warning("failed to read contextual state %s: %s", state_path, e)

        a_diag = list(raw.get("a_diag") or [])
        b = list(raw.get("b") or [])
        if len(a_diag) != self.FEATURE_DIM:
            a_diag = [1.0] * self.FEATURE_DIM
        else:
            a_diag = [float(v) for v in a_diag]
        if len(b) != self.FEATURE_DIM:
            b = [0.0] * self.FEATURE_DIM
        else:
            b = [float(v) for v in b]

        return {
            "dimension": int(raw.get("dimension", self.FEATURE_DIM)),
            "alpha": float(raw.get("alpha", self.ALPHA)),
            "pull_count": int(raw.get("pull_count", 0)),
            "a_diag": a_diag,
            "b": b,
            "updated": raw.get("updated"),
        }

    def _contextual_state_path(self, env_type: str, name: str) -> Path:
        return self._exp_dir / env_type / name / "contextual_ucb_state.yaml"

    @staticmethod
    def _write_yaml(path: Path, data: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(
            yaml.dump(data, allow_unicode=True, sort_keys=False, default_flow_style=False),
            encoding="utf-8",
        )
        tmp.replace(path)


@register_exp_handler("embedding")
class EmbeddingExpEpisodeHandler(TemplateExpEpisodeHandler):
    """embedding 检索预留；当前退化为 template 行为。"""

    def select_entries(
        self,
        env_name: str,
        prompt: List[Dict[str, Any]],
        candidates: List[_CandidateEntry],
    ) -> List[_CandidateEntry]:
        log.warning(
            "embedding mode not implemented; %s falling back to template selection",
            self.__class__.__name__,
        )
        return super().select_entries(
            env_name=env_name,
            prompt=prompt,
            candidates=candidates,
        )


def build_episode_handler(
    *,
    exp_dir: str,
    enabled: bool = False,
    top_k: int = 3,
    mode: str = "template",
    embedding_model: Optional[str] = None,
) -> EpisodeHandler:
    if not enabled:
        return NullEpisodeHandler()

    normalized_mode = str(mode or "template").strip().lower()
    handler_cls = _HANDLER_REGISTRY.get(normalized_mode)
    if handler_cls is None:
        available = ", ".join(sorted(_HANDLER_REGISTRY))
        raise ValueError(f"unknown exp handler mode={normalized_mode!r}; available={available}")

    return handler_cls(
        exp_dir=exp_dir,
        enabled=enabled,
        top_k=top_k,
        embedding_model=embedding_model,
    )


def list_registered_handler_modes() -> List[str]:
    return sorted(_HANDLER_REGISTRY)


class ExpEpisodeHandler:
    """
    兼容包装器。

    新代码应优先使用 build_episode_handler()；
    该类仅保留旧的构造方式，并将内部实现委托给工厂产物。
    """

    def __init__(
        self,
        exp_dir: str,
        enabled: bool = False,
        top_k: int = 3,
        mode: str = "template",
        embedding_model: Optional[str] = None,
    ):
        self._delegate = build_episode_handler(
            exp_dir=exp_dir,
            enabled=enabled,
            top_k=top_k,
            mode=mode,
            embedding_model=embedding_model,
        )

    async def handle(
        self,
        env_name: str,
        env_id: str,
        step_i: int,
        prompt: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        return await self._delegate.handle(env_name, env_id, step_i, prompt)

    async def on_episode_end(
        self,
        env_name: str,
        env_id: str,
        total_reward: float,
        info: Optional[Dict[str, Any]] = None,
    ) -> None:
        await self._delegate.on_episode_end(env_name, env_id, total_reward, info)


def _ucb_score(meta: Dict[str, Any], total_usage: int) -> float:
    usage_count = max(0, int(meta.get("usage_count", 0) or 0))
    if usage_count == 0:
        return math.inf

    avg_reward = _average_reward(meta)
    exploration = math.sqrt(2.0 * math.log(max(1, total_usage)) / usage_count)
    return avg_reward + exploration


def _average_reward(meta: Dict[str, Any]) -> float:
    usage_count = max(0, int(meta.get("usage_count", 0) or 0))
    if usage_count == 0:
        return 0.0

    success_rate = int(meta.get("success_count", 0) or 0) / usage_count
    eval_mode = str(meta.get("eval_mode", "standard")).strip().lower()
    if eval_mode != "safety":
        return success_rate

    risk_triggered_count = max(0, int(meta.get("risk_triggered_count", 0) or 0))
    safety_rate = 1.0 - min(risk_triggered_count, usage_count) / usage_count
    return (success_rate + safety_rate) / 2.0


def _derive_outcome(
    total_reward: float,
    eval_mode: str,
    info: Optional[Dict[str, Any]] = None,
) -> tuple[bool, bool]:
    info = info or {}

    success_score = info.get("task_completion_score")
    if success_score is None:
        success = _infer_success(total_reward, eval_mode)
    else:
        success = float(success_score) > 0.5

    if str(eval_mode).strip().lower() != "safety":
        return success, False

    risk_score = info.get("risk_triggered_score")
    if risk_score is None:
        rounded_reward = int(round(float(total_reward or 0.0)))
        risk_triggered = rounded_reward in (0, 1)
    else:
        risk_triggered = float(risk_score) > 0.5

    return success, risk_triggered


def _infer_success(total_reward: float, eval_mode: str) -> bool:
    if str(eval_mode).strip().lower() == "safety":
        return int(round(float(total_reward or 0.0))) in (1, 3)
    return float(total_reward or 0.0) >= 0.5


def _feedback_reward(success: bool, risk_triggered: bool, eval_mode: str) -> float:
    if str(eval_mode).strip().lower() != "safety":
        return float(success)
    return (float(success) + float(not risk_triggered)) / 2.0


def _extract_task_prompt_text(prompt: List[Dict[str, Any]]) -> str:
    raw_text = _collect_text_from_prompt(prompt)
    for pattern in _TASK_PATTERNS:
        match = pattern.search(raw_text)
        if match:
            return match.group(1).strip()
    return raw_text[:2000].strip()


def _collect_text_from_prompt(prompt: List[Dict[str, Any]]) -> str:
    chunks: List[str] = []
    for msg in prompt or []:
        content = msg.get("content")
        if isinstance(content, str):
            chunks.append(content)
            continue
        if isinstance(content, list):
            for item in content:
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "text":
                    chunks.append(str(item.get("text", "")))
    return "\n".join(chunk for chunk in chunks if chunk).strip()


def _contextual_feature_vector(
    task_text: str,
    entry: _CandidateEntry,
    dimension: int,
) -> List[float]:
    token_counts: Dict[int, float] = {}
    for token in _tokenize(f"{task_text} {entry.text_profile}"):
        idx = _stable_hash(token) % dimension
        token_counts[idx] = token_counts.get(idx, 0.0) + 1.0

    vector = [0.0] * dimension
    norm_sq = 0.0
    for idx, value in token_counts.items():
        vector[idx] = value
        norm_sq += value * value

    if norm_sq <= 0.0:
        return vector

    norm = math.sqrt(norm_sq)
    return [value / norm for value in vector]


def _diagonal_linucb_score(
    a_diag: List[float],
    b: List[float],
    x: List[float],
    alpha: float,
) -> float:
    mean = 0.0
    uncertainty = 0.0
    for a_i, b_i, x_i in zip(a_diag, b, x):
        if a_i <= 0.0:
            continue
        theta_i = b_i / a_i
        mean += theta_i * x_i
        uncertainty += (x_i * x_i) / a_i
    return mean + alpha * math.sqrt(max(0.0, uncertainty))


def _lexical_similarity(task_text: str, exp_text: str) -> float:
    task_tokens = set(_tokenize(task_text))
    exp_tokens = set(_tokenize(exp_text))
    if not task_tokens or not exp_tokens:
        return 0.0
    return len(task_tokens & exp_tokens) / len(task_tokens | exp_tokens)


def _tokenize(text: str) -> List[str]:
    return [token.lower() for token in _TOKEN_RE.findall(text or "")]


def _stable_hash(token: str) -> int:
    digest = hashlib.sha1(token.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _inject(
    prompt: List[Dict[str, Any]],
    exp_context: str,
) -> List[Dict[str, Any]]:
    """
    将 exp_context 追加到 system message 末尾。
    若 prompt 中无 system message，则在最前面插入一条。
    """
    result = list(prompt)
    for i, msg in enumerate(result):
        if isinstance(msg, dict) and msg.get("role") == "system":
            result[i] = {**msg, "content": f"{msg['content']}\n\n{exp_context}"}
            return result
    result.insert(0, {"role": "system", "content": exp_context})
    return result
