"""
ExpService: 主编排。

run_once()  — 完整执行一次 DB 读取 -> 轨迹选取 -> Experience 生成 -> 写入
run_loop()  — 持续按 poll_interval_s 调用 run_once
"""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Set

from .config import ServiceConfig
from .db_reader import DBReader
from .generator import ExpGenerator, ExpDraft, make_openai_llm
from .exp_bank import ExpBank
from .trajectory import TrajectoryBuilder, TrajectoryRecord, TrajectorySelector
from .trajectory import make_success_selector, make_failure_selector
from .updater import ExpUpdater, UpdateResult

log = logging.getLogger("exp_service.service")


# ------------------------------------------------------------------ #
# 状态持久化
# ------------------------------------------------------------------ #

@dataclass
class ServiceState:
    processed_session_ids: Set[str] = field(default_factory=set)
    last_run: Optional[datetime] = None
    total_runs: int = 0
    total_exps_created: int = 0
    total_exps_updated: int = 0

    def save(self, path: str) -> None:
        data = {
            "processed_session_ids": list(self.processed_session_ids),
            "last_run": self.last_run.isoformat() if self.last_run else None,
            "total_runs": self.total_runs,
            "total_exps_created": self.total_exps_created,
            "total_exps_updated": self.total_exps_updated,
        }
        p = Path(path)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(p)
        log.debug("state saved: %s", path)

    @classmethod
    def load(cls, path: str) -> "ServiceState":
        p = Path(path)
        if not p.exists():
            return cls()
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            inst = cls()
            inst.processed_session_ids = set(data.get("processed_session_ids", []))
            inst.total_runs = int(data.get("total_runs", 0))
            inst.total_exps_created = int(data.get("total_exps_created", 0))
            inst.total_exps_updated = int(data.get("total_exps_updated", 0))
            if data.get("last_run"):
                try:
                    inst.last_run = datetime.fromisoformat(data["last_run"])
                except (ValueError, TypeError):
                    pass
            log.info("state loaded: %d processed sessions", len(inst.processed_session_ids))
            return inst
        except Exception as e:
            log.warning("failed to load state from %s: %s, starting fresh", path, e)
            return cls()


# ------------------------------------------------------------------ #
# 单次运行结果
# ------------------------------------------------------------------ #

@dataclass
class RunResult:
    success_sessions: int = 0
    failure_sessions: int = 0
    success_drafts: int = 0
    failure_drafts: int = 0
    update: UpdateResult = field(default_factory=UpdateResult)
    elapsed_s: float = 0.0
    error: Optional[str] = None

    def summary(self) -> str:
        return (
            f"success_traj={self.success_sessions} failure_traj={self.failure_sessions} "
            f"drafts={self.success_drafts + self.failure_drafts} "
            f"{self.update.summary()} elapsed={self.elapsed_s:.1f}s"
        )


# ------------------------------------------------------------------ #
# ExpService
# ------------------------------------------------------------------ #

class ExpService:
    """
    ExpService 编排整个流水线。

    可通过 from_config() 从 ServiceConfig 构建，也可手动注入各组件。
    """

    def __init__(
        self,
        db_reader: DBReader,
        exp_bank: ExpBank,
        generator: ExpGenerator,
        updater: ExpUpdater,
        success_selector: TrajectorySelector,
        failure_selector: TrajectorySelector,
        state_path: str = "./exp_service_state.json",
        min_steps: int = 2,
    ):
        self._db = db_reader
        self._bank = exp_bank
        self._generator = generator
        self._updater = updater
        self._success_sel = success_selector
        self._failure_sel = failure_selector
        self._state_path = state_path
        self._min_steps = min_steps
        self._builder = TrajectoryBuilder(db_reader)
        self._state = ServiceState.load(state_path)

    # ---------------------------------------------------------------- #
    # 公开接口
    # ---------------------------------------------------------------- #

    async def run_once(self) -> RunResult:
        """执行一次完整流水线"""
        t0 = _now()
        result = RunResult()
        try:
            result = await self._run_once_inner()
        except Exception as e:
            log.exception("run_once failed: %s", e)
            result.error = str(e)
        result.elapsed_s = _now() - t0
        log.info("run_once done: %s", result.summary())
        return result

    async def run_loop(self, interval_s: Optional[float] = None) -> None:
        """持续轮询，直到被取消"""
        poll = interval_s or 300.0
        log.info("run_loop started: interval=%.0fs", poll)
        while True:
            await self.run_once()
            log.info("sleeping %.0fs before next run", poll)
            await asyncio.sleep(poll)

    def get_state(self) -> ServiceState:
        return self._state

    # ---------------------------------------------------------------- #
    # 内部流程
    # ---------------------------------------------------------------- #

    async def _run_once_inner(self) -> RunResult:
        result = RunResult()
        exclude = set(self._state.processed_session_ids)

        # 1. 选取成功轨迹
        success_sessions = self._success_sel.select(self._db, exclude)
        success_trajs: List[TrajectoryRecord] = self._builder.build_many(
            success_sessions, min_steps=self._min_steps
        )
        result.success_sessions = len(success_trajs)

        # 2. 选取失败轨迹
        failure_sessions = self._failure_sel.select(self._db, exclude)
        failure_trajs: List[TrajectoryRecord] = self._builder.build_many(
            failure_sessions, min_steps=self._min_steps
        )
        result.failure_sessions = len(failure_trajs)

        if not success_trajs and not failure_trajs:
            log.info("no new trajectories to process")
            return result

        # 3. 从成功轨迹生成正向 experience
        success_drafts: List[ExpDraft] = await self._generator.generate_from_success(success_trajs)
        result.success_drafts = len(success_drafts)

        # 4. 从失败轨迹生成失败教训 experience
        failure_drafts: List[ExpDraft] = await self._generator.generate_from_failure(failure_trajs)
        result.failure_drafts = len(failure_drafts)

        # 5. 合并写入 ExpBank
        all_drafts = success_drafts + failure_drafts
        merge_result = self._updater.merge_into_bank(all_drafts)
        result.update = merge_result

        # 6. 用失败轨迹进化已有 experience
        if failure_trajs:
            evolve_result = await self._updater.evolve_with_failures(failure_trajs)
            result.update = result.update + evolve_result

        # 7. 更新已处理 session_id
        processed = (
            {s.session_id for s in success_sessions}
            | {s.session_id for s in failure_sessions}
        )
        self._state.processed_session_ids |= processed
        self._state.last_run = datetime.now()
        self._state.total_runs += 1
        self._state.total_exps_created += merge_result.created
        self._state.total_exps_updated += merge_result.updated
        self._state.save(self._state_path)

        return result

    # ---------------------------------------------------------------- #
    # 工厂
    # ---------------------------------------------------------------- #

    @classmethod
    def from_config(cls, cfg: ServiceConfig) -> "ExpService":
        """从 ServiceConfig 一次性构建所有组件"""
        db_reader = DBReader(cfg.db.url)
        exp_bank = ExpBank(cfg.exp_dir)
        llm_fn = make_openai_llm(cfg.llm)
        generator = ExpGenerator(llm_fn, cfg.generator)
        updater = ExpUpdater(exp_bank, llm_fn, cfg.updater)
        success_selector = make_success_selector(cfg.selector)
        failure_selector = make_failure_selector(cfg.selector)
        return cls(
            db_reader=db_reader,
            exp_bank=exp_bank,
            generator=generator,
            updater=updater,
            success_selector=success_selector,
            failure_selector=failure_selector,
            state_path=cfg.state_path,
            min_steps=cfg.selector.min_steps,
        )


def _now() -> float:
    import time
    return time.perf_counter()


# ------------------------------------------------------------------ #
# Dry-run 测试入口
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    """
    用法：
      python -m exp_service.service --db-url /mnt/shared-storage-gpfs2/evobox-share-gpfs2/liubangwei/codes/AIEvoBox/os_envs_base_2.db \\
                                    --exp-dir ./experiences \\
                                    --limit 3

    从真实 DB 读取轨迹，打印发给 LLM 的完整 prompt，但不实际调用 LLM。
    """
    import argparse
    import sys

    from .config import SelectorConfig, GeneratorConfig
    from .generator import _select_prompt, _infer_group_env_type, _batch, _PROMPTS_DIR

    # ── 参数解析 ──────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(description="exp_service dry-run: print LLM prompts without calling LLM")
    parser.add_argument("--db-url", required=True, help="SQLite DB URL, e.g. sqlite:///evobox.db")
    parser.add_argument("--exp-dir", default="./experiences", help="Experience 文件系统根目录")
    parser.add_argument("--limit", type=int, default=3, help="最多抓取的 session 数量")
    parser.add_argument("--mode", choices=["success", "failure", "both"], default="both")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    # ── Mock LLM：打印 prompt，返回占位 JSON ──────────────────────────
    _call_index = 0

    async def mock_llm(messages):
        global _call_index
        _call_index += 1
        prompt_text = messages[-1]["content"]

        sep = "=" * 70
        print(f"\n{sep}")
        print(f"  LLM CALL #{_call_index}")
        print(sep)
        # 截断轨迹正文，只保留 prompt 模板部分可读
        MAX_PRINT = 4000
        if len(prompt_text) > MAX_PRINT:
            print(prompt_text[:MAX_PRINT])
            print(f"\n... [truncated {len(prompt_text) - MAX_PRINT} chars] ...")
        else:
            print(prompt_text)
        print(sep)

        # 返回合法 JSON，让 parser 不报错
        fake = {
            "experiences": [
                {
                    "name": f"dry_run_exp_{_call_index:02d}",
                    "description": f"[DRY RUN] placeholder from call #{_call_index}",
                    "trigger": "This is a dry-run placeholder, no real LLM was called.",
                    "content": (
                        "# Dry Run Experience\n\n"
                        "## Note\nThis experience was generated by the dry-run test.\n"
                        "Replace with real LLM output.\n"
                    ),
                }
            ]
        }
        return f"```json\n{json.dumps(fake, indent=2, ensure_ascii=False)}\n```"

    # ── 真实 DB 读取 + 轨迹构建 ───────────────────────────────────────
    async def main():
        from .db_reader import DBReader
        from .trajectory import TrajectoryBuilder, RewardThresholdSelector, FailureSelector

        db = DBReader(args.db_url)
        builder = TrajectoryBuilder(db)

        sel_cfg = SelectorConfig()
        gen_cfg = GeneratorConfig(batch_size=2, max_exps_per_batch=2)
        generator = ExpGenerator(mock_llm, gen_cfg)

        success_trajs, failure_trajs = [], []

        if args.mode in ("success", "both"):
            success_sessions = RewardThresholdSelector(
                min_reward=0.5, top_k=args.limit, min_steps=2
            ).select(db, set())
            success_trajs = builder.build_many(success_sessions, min_steps=2)
            print(f"\n>> Found {len(success_trajs)} success trajectories")
            for t in success_trajs:
                print(f"   • {t.session_id[:16]}…  env={t.env_type}  "
                      f"eval_mode={t.eval_mode}  score={t.score_label()}  steps={t.total_steps}")

        if args.mode in ("failure", "both"):
            failure_sessions = FailureSelector(
                max_reward=0.35, top_k=args.limit, min_steps=2
            ).select(db, set())
            failure_trajs = builder.build_many(failure_sessions, min_steps=2)
            print(f"\n>> Found {len(failure_trajs)} failure trajectories")
            for t in failure_trajs:
                print(f"   • {t.session_id[:16]}…  env={t.env_type}  "
                      f"eval_mode={t.eval_mode}  score={t.score_label()}  steps={t.total_steps}")

        if not success_trajs and not failure_trajs:
            print("\n[No trajectories found. Check --db-url and filter thresholds.]")
            return

        # 预览第一条轨迹的 to_text() 输出，确认 observation 解析正确
        sample = (success_trajs + failure_trajs)[0]
        print(f"\n{'─'*70}")
        print(f"  TRAJECTORY PREVIEW: {sample.session_id[:16]}… ({sample.total_steps} steps)")
        print(f"{'─'*70}")
        print(sample.to_text(max_turns=3))   # 只打印前 3 步，避免输出过长

        print("\n" + "─" * 70)
        print("  GENERATING PROMPTS (mock LLM, no real API call)")
        print("─" * 70)

        success_drafts = await generator.generate_from_success(success_trajs)
        failure_drafts = await generator.generate_from_failure(failure_trajs)
        all_drafts = success_drafts + failure_drafts

        print(f"\n>> Drafts produced: {len(all_drafts)}")
        for d in all_drafts:
            print(f"   • [{d.eval_mode}] {d.env_type}/{d.name}")
            print(f"     trigger: {d.trigger[:80]}")

    asyncio.run(main())
