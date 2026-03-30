"""
ExpService 配置数据类。

所有字段均带默认值，可通过 from_dict() 从 YAML / dict 加载。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DBConfig:
    """AIEvoBox DB 连接配置（只读）"""
    url: str = "sqlite:///evobox.db"
    # sqlite:///path/to/file.db  或  postgresql://user:pass@host/db
    # 对于 SQLite 仅支持同步读取（sqlite3 标准库）


@dataclass
class LLMConfig:
    """LLM 调用配置"""
    base_url: str = "http://localhost:8000/v1"
    api_key: str = "EMPTY"
    model: str = "Qwen2.5-72B-Instruct"
    temperature: float = 0.3
    max_tokens: int = 4096
    timeout_s: float = 120.0
    max_concurrency: int = 3        # 并发 LLM 调用数上限


@dataclass
class SelectorConfig:
    """轨迹选取配置"""
    # 成功轨迹选取
    success_min_reward: float = 0.5         # 成功轨迹 reward 下限
    success_top_k: Optional[int] = 20       # 每轮最多处理多少条成功轨迹

    # 失败轨迹选取
    failure_max_reward: float = 0.2         # 失败轨迹 reward 上限
    failure_top_k: Optional[int] = 10       # 每轮最多处理多少条失败轨迹

    # 通用过滤
    min_steps: int = 2                      # 有效轨迹的最少步骤数
    env_names: List[str] = field(default_factory=list)   # 为空表示处理所有 env


@dataclass
class GeneratorConfig:
    """Skill 生成配置"""
    max_exps_per_batch: int = 3           # 每批轨迹最多生成几条 experience
    min_reward_for_extraction: float = 0.0  # 生成前再次确认的 reward 下限
    batch_size: int = 5                     # 一次提交给 LLM 几条轨迹做批量提炼


@dataclass
class UpdaterConfig:
    """Experience 更新/进化配置"""
    # evolve 触发条件
    evolve_threshold: float = 0.4           # success_rate 低于此值才触发 evolve
    max_new_skills_per_evolve: int = 3      # 每次 evolve 最多新增几条 experience

    # prune 条件（两条件同时满足才删除）
    prune_min_success_rate: float = 0.15    # success_rate 低于此值的 experience 候选删除
    prune_min_usage_count: int = 3          # usage_count >= 此值时才参与 prune 评估

    # merge 相似度阈值
    merge_similarity_threshold: float = 0.8


@dataclass
class ServiceConfig:
    """ExpService 顶层配置"""
    db: DBConfig = field(default_factory=DBConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    selector: SelectorConfig = field(default_factory=SelectorConfig)
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    updater: UpdaterConfig = field(default_factory=UpdaterConfig)

    exp_dir: str = "./experiences"             # Experience 文件系统根目录
    state_path: str = "./exp_service_state.json"   # 已处理 session_id 记录

    poll_interval_s: float = 300.0          # run_loop 轮询间隔（秒）
    log_level: str = "INFO"

    # ------------------------------------------------------------------ #
    @classmethod
    def from_dict(cls, d: dict) -> "ServiceConfig":
        """从嵌套 dict（如 YAML 解析结果）构建配置"""
        cfg = cls()

        if "db" in d:
            db = d["db"]
            cfg.db = DBConfig(url=str(db.get("url", cfg.db.url)))

        if "llm" in d:
            lm = d["llm"]
            cfg.llm = LLMConfig(
                base_url=str(lm.get("base_url", cfg.llm.base_url)),
                api_key=str(lm.get("api_key", cfg.llm.api_key)),
                model=str(lm.get("model", cfg.llm.model)),
                temperature=float(lm.get("temperature", cfg.llm.temperature)),
                max_tokens=int(lm.get("max_tokens", cfg.llm.max_tokens)),
                timeout_s=float(lm.get("timeout_s", cfg.llm.timeout_s)),
                max_concurrency=int(lm.get("max_concurrency", cfg.llm.max_concurrency)),
            )

        if "selector" in d:
            sel = d["selector"]
            cfg.selector = SelectorConfig(
                success_min_reward=float(sel.get("success_min_reward", cfg.selector.success_min_reward)),
                success_top_k=sel.get("success_top_k", cfg.selector.success_top_k),
                failure_max_reward=float(sel.get("failure_max_reward", cfg.selector.failure_max_reward)),
                failure_top_k=sel.get("failure_top_k", cfg.selector.failure_top_k),
                min_steps=int(sel.get("min_steps", cfg.selector.min_steps)),
                env_names=list(sel.get("env_names") or []),
            )

        if "generator" in d:
            gen = d["generator"]
            cfg.generator = GeneratorConfig(
                max_exps_per_batch=int(gen.get("max_exps_per_batch", cfg.generator.max_exps_per_batch)),
                min_reward_for_extraction=float(gen.get("min_reward_for_extraction", cfg.generator.min_reward_for_extraction)),
                batch_size=int(gen.get("batch_size", cfg.generator.batch_size)),
            )

        if "updater" in d:
            upd = d["updater"]
            cfg.updater = UpdaterConfig(
                evolve_threshold=float(upd.get("evolve_threshold", cfg.updater.evolve_threshold)),
                max_new_skills_per_evolve=int(upd.get("max_new_skills_per_evolve", cfg.updater.max_new_skills_per_evolve)),
                prune_min_success_rate=float(upd.get("prune_min_success_rate", cfg.updater.prune_min_success_rate)),
                prune_min_usage_count=int(upd.get("prune_min_usage_count", cfg.updater.prune_min_usage_count)),
                merge_similarity_threshold=float(upd.get("merge_similarity_threshold", cfg.updater.merge_similarity_threshold)),
            )

        cfg.exp_dir = str(d.get("exp_dir", cfg.exp_dir))
        cfg.state_path = str(d.get("state_path", cfg.state_path))
        cfg.poll_interval_s = float(d.get("poll_interval_s", cfg.poll_interval_s))
        cfg.log_level = str(d.get("log_level", cfg.log_level)).upper()

        return cfg
