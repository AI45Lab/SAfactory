"""
ExpUpdater: 管理 Experience 的 merge / evolve / prune 三种更新策略。

merge  — 将新 ExpDraft 合并进 ExpBank（同名更新，不同名创建）
evolve — 根据失败轨迹用 LLM 改进已有 experience
prune  — 删除低质量 experience（低 success_rate 且低 usage_count）
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from .config import UpdaterConfig
from .generator import LLMGenerateFn, ExpDraft, _extract_json_block, _load_prompt
from .exp_bank import Experience, ExpMeta, ExpBank, exp_similarity
from .trajectory import TrajectoryRecord

log = logging.getLogger("exp_service.updater")


# ------------------------------------------------------------------ #
# 结果数据类
# ------------------------------------------------------------------ #

@dataclass
class UpdateResult:
    created: int = 0
    updated: int = 0
    skipped: int = 0
    pruned: int = 0
    evolved: int = 0
    errors: List[str] = field(default_factory=list)

    def __add__(self, other: "UpdateResult") -> "UpdateResult":
        return UpdateResult(
            created=self.created + other.created,
            updated=self.updated + other.updated,
            skipped=self.skipped + other.skipped,
            pruned=self.pruned + other.pruned,
            evolved=self.evolved + other.evolved,
            errors=self.errors + other.errors,
        )

    def summary(self) -> str:
        return (
            f"created={self.created} updated={self.updated} "
            f"skipped={self.skipped} pruned={self.pruned} "
            f"evolved={self.evolved} errors={len(self.errors)}"
        )


# ------------------------------------------------------------------ #
# ExpUpdater
# ------------------------------------------------------------------ #

class ExpUpdater:
    """
    Experience 更新器。

    Args:
        bank:           ExpBank 实例
        llm_generate:   异步 LLM 调用函数（evolve 时使用）
        cfg:            UpdaterConfig
    """

    def __init__(
        self,
        bank: ExpBank,
        llm_generate: Optional[LLMGenerateFn] = None,
        cfg: Optional[UpdaterConfig] = None,
    ):
        self._bank = bank
        self._llm = llm_generate
        self._cfg = cfg or UpdaterConfig()
        self._evolve_tmpl = _load_prompt("evolve_exp.md")

    # ---------------------------------------------------------------- #
    # merge
    # ---------------------------------------------------------------- #

    def merge_into_bank(self, drafts: List[ExpDraft]) -> UpdateResult:
        """
        将 ExpDraft 列表写入 ExpBank。
        - 同名 experience：更新 source_sessions 和 content（取更丰富的），保留 meta 统计
        - 相似 experience（name Jaccard >= threshold）：同上
        - 全新 experience：直接写入
        """
        result = UpdateResult()
        for draft in drafts:
            try:
                r = self._merge_one(draft)
                if r == "created":
                    result.created += 1
                elif r == "updated":
                    result.updated += 1
                else:
                    result.skipped += 1
            except Exception as e:
                msg = f"merge error for {draft.env_type}/{draft.name}: {e}"
                log.error(msg)
                result.errors.append(msg)
        log.info("merge_into_bank: %s", result.summary())
        return result

    def _merge_one(self, draft: ExpDraft) -> str:
        existing = self._bank.read_exp(draft.env_type, draft.name)
        if existing is None:
            existing = self._find_similar(draft)

        if existing is None:
            exp = _draft_to_exp(draft)
            self._bank.write_exp(exp)
            return "created"

        merged = _merge_exp(existing, draft)
        self._bank.write_exp(merged)
        return "updated"

    def _find_similar(self, draft: ExpDraft) -> Optional[Experience]:
        threshold = self._cfg.merge_similarity_threshold
        draft_exp = _draft_to_exp(draft)
        for name in self._bank.list_exps(draft.env_type):
            s = self._bank.read_exp(draft.env_type, name)
            if s and exp_similarity(draft_exp, s) >= threshold:
                log.debug(
                    "found similar experience: %s/%s (threshold=%.2f)",
                    draft.env_type, name, threshold,
                )
                return s
        return None

    # ---------------------------------------------------------------- #
    # evolve
    # ---------------------------------------------------------------- #

    async def evolve_with_failures(
        self,
        failure_trajectories: List[TrajectoryRecord],
        update_threshold: Optional[float] = None,
        max_new_exps: Optional[int] = None,
    ) -> UpdateResult:
        """
        根据失败轨迹进化 ExpBank 中的低质量 experience。
        对每个 env_type，找出 success_rate < threshold 的 experience，
        用 LLM 分析失败轨迹并改进。
        """
        if not self._llm:
            log.warning("evolve_with_failures: no LLM configured, skipping")
            return UpdateResult()
        if not failure_trajectories:
            return UpdateResult()

        threshold = update_threshold if update_threshold is not None else self._cfg.evolve_threshold
        max_new = max_new_exps if max_new_exps is not None else self._cfg.max_new_skills_per_evolve

        by_env: Dict[str, List[TrajectoryRecord]] = {}
        for t in failure_trajectories:
            by_env.setdefault(t.env_type, []).append(t)

        result = UpdateResult()
        evolved_count = 0

        for env_type, trajs in by_env.items():
            exps_to_evolve = [
                e for e in self._bank.list_all_exps()
                if e.env_type == env_type and e.success_rate < threshold
            ]
            for exp in exps_to_evolve:
                if evolved_count >= max_new:
                    break
                try:
                    improved = await self._evolve_one(exp, trajs[:5])
                    if improved:
                        self._bank.write_exp(improved)
                        result.evolved += 1
                        evolved_count += 1
                        log.info("evolved experience: %s/%s", env_type, exp.name)
                except Exception as e:
                    msg = f"evolve error for {env_type}/{exp.name}: {e}"
                    log.error(msg)
                    result.errors.append(msg)

        log.info("evolve_with_failures: %s", result.summary())
        return result

    async def _evolve_one(
        self, exp: Experience, trajs: List[TrajectoryRecord]
    ) -> Optional[Experience]:
        if not self._evolve_tmpl:
            log.warning("evolve_exp.md prompt not found, skipping evolve")
            return None
        traj_text = "\n\n---\n\n".join(t.to_text(max_turns=10) for t in trajs)
        meta_info = f"name: {exp.name}\ndescription: {exp.description}\ntrigger: {exp.trigger}"
        prompt_text = (
            self._evolve_tmpl
            .replace("{existing_experience}", f"## Meta\n{meta_info}\n\n## Content\n{exp.content}")
            .replace("{failure_trajectories}", traj_text)
            .replace("{success_rate}", f"{exp.success_rate:.2f}")
        )
        messages = [{"role": "user", "content": prompt_text}]
        try:
            response = await self._llm(messages)
        except Exception as e:
            log.error("LLM evolve call failed: %s", e)
            return None

        raw = _extract_json_block(response)
        if not raw:
            return None
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return None

        if not data.get("updated"):
            log.debug("LLM decided not to update experience %s: %s", exp.name, data.get("reason", ""))
            return None

        # 保留 meta（尤其是 usage_count / success_count），仅更新内容
        new_exp = Experience(
            name=exp.name,
            env_type=exp.env_type,
            description=str(data.get("description") or exp.description).strip(),
            trigger=str(data.get("trigger") or exp.trigger).strip(),
            content=str(data.get("content") or exp.content).strip(),
            eval_mode=exp.eval_mode,
            meta=exp.meta,
        )
        if new_exp.meta:
            new_exp.meta.version = _increment_version(new_exp.meta.version)
            # 合并 source_sessions
            new_sources = list({*exp.meta.source_sessions, *(data.get("source_sessions") or [])})
            new_exp.meta.source_sessions = new_sources[-20:]
        return new_exp

    # ---------------------------------------------------------------- #
    # prune
    # ---------------------------------------------------------------- #

    def prune(
        self,
        min_success_rate: Optional[float] = None,
        min_usage_count: Optional[int] = None,
    ) -> UpdateResult:
        """
        删除 success_rate < min_success_rate 且 usage_count < min_usage_count
        的 experience（两条件同时满足才删除）。
        注意：usage_count=0 的 experience（从未被调用）不受此规则影响。
        """
        sr_threshold = min_success_rate if min_success_rate is not None else self._cfg.prune_min_success_rate
        uc_threshold = min_usage_count if min_usage_count is not None else self._cfg.prune_min_usage_count

        result = UpdateResult()
        for exp in self._bank.list_all_exps():
            uc = exp.meta.usage_count if exp.meta else 0
            sr = exp.success_rate
            # 只裁剪有实际使用记录的低质量经验
            if uc >= uc_threshold and sr < sr_threshold:
                try:
                    self._bank.delete_exp(exp.env_type, exp.name)
                    result.pruned += 1
                    log.info(
                        "pruned experience: %s/%s (sr=%.2f uc=%d)",
                        exp.env_type, exp.name, sr, uc,
                    )
                except Exception as e:
                    msg = f"prune error for {exp.env_type}/{exp.name}: {e}"
                    log.error(msg)
                    result.errors.append(msg)

        log.info("prune: removed %d experiences", result.pruned)
        return result


# ------------------------------------------------------------------ #
# 工具函数
# ------------------------------------------------------------------ #

def _draft_to_exp(draft: ExpDraft) -> Experience:
    meta = ExpMeta(
        name=draft.name,
        env_type=draft.env_type,
        eval_mode=draft.eval_mode,
        source_sessions=draft.source_sessions[-20:],
    )
    return Experience(
        name=draft.name,
        env_type=draft.env_type,
        description=draft.description,
        trigger=draft.trigger,
        content=draft.content,
        eval_mode=draft.eval_mode,
        meta=meta,
    )


def _merge_exp(existing: Experience, draft: ExpDraft) -> Experience:
    """
    合并已有 experience 与新 draft。
    - content: 取更丰富的（长度 > 1.2 倍时才替换）
    - description / trigger: 取更长的
    - source_sessions: 合并去重，保留最近 20 条
    - meta 统计（usage_count / success_count）: 保留原值，由 record_usage() 更新
    """
    new_content = (
        draft.content
        if len(draft.content) > len(existing.content) * 1.2
        else existing.content
    )
    new_desc = (
        draft.description
        if len(draft.description) > len(existing.description)
        else existing.description
    )
    new_trigger = draft.trigger if draft.trigger else existing.trigger

    meta = existing.meta or ExpMeta(name=existing.name, env_type=existing.env_type)
    new_sources = list(dict.fromkeys(meta.source_sessions + draft.source_sessions))[-20:]
    meta.source_sessions = new_sources
    meta.version = _increment_version(meta.version)

    return Experience(
        name=existing.name,
        env_type=existing.env_type,
        description=new_desc,
        trigger=new_trigger,
        content=new_content,
        eval_mode=draft.eval_mode or existing.eval_mode,
        meta=meta,
    )


def _increment_version(version: str) -> str:
    try:
        parts = str(version).split(".")
        if len(parts) >= 2:
            return f"{parts[0]}.{int(parts[1]) + 1}"
        return f"{parts[0]}.1"
    except (ValueError, IndexError):
        return "1.1"
