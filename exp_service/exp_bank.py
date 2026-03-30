"""
Experience 文件系统管理（ExpBank）。

目录结构：
  experiences/
    index.yaml                     # 全局索引（精简版：名称 + 触发条件 + 成功率）
    {env_type}/
      {exp_name}/
        meta.yaml                  # 元数据（版本、时间、调用/成功计数、来源 session）
        SKILL.md                   # 经验正文（纯 Markdown，不含元数据）
        details.md                 # 可选：更详细的步骤说明
        examples.md                # 可选：案例片段

meta.yaml 字段：
  name, version, env_type, eval_mode, created, updated,
  usage_count, success_count, risk_triggered_count, source_sessions

index.yaml 精简字段（只保留检索必要信息）：
  environments:
    {env_type}:
      - name, description, trigger, success_rate, eval_mode, path
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

log = logging.getLogger("exp_service.exp_bank")


# ------------------------------------------------------------------ #
# 数据模型
# ------------------------------------------------------------------ #

@dataclass
class ExpMeta:
    """
    经验元数据，对应 meta.yaml。
    与经验正文分离存储，避免 SKILL.md 含冗余信息。
    """
    name: str
    version: str = "1.0"
    env_type: str = ""
    eval_mode: str = "standard"   # "standard" | "safety"
    created: Optional[datetime] = None
    updated: Optional[datetime] = None
    usage_count: int = 0          # agent 调用该经验的次数
    success_count: int = 0        # 调用后任务成功的次数
    risk_triggered_count: int = 0 # 调用后触发风险的次数（safety 模式）
    source_sessions: List[str] = field(default_factory=list)

    def __post_init__(self):
        if self.created is None:
            self.created = datetime.now()
        if self.updated is None:
            self.updated = self.created

    @property
    def success_rate(self) -> float:
        """基于实际调用统计的成功率；未使用时返回 None（避免假性 0）"""
        if self.usage_count > 0:
            return round(self.success_count / self.usage_count, 4)
        return 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "version": str(self.version),
            "env_type": self.env_type,
            "eval_mode": self.eval_mode,
            "created": self.created.isoformat() if self.created else None,
            "updated": self.updated.isoformat() if self.updated else None,
            "usage_count": self.usage_count,
            "success_count": self.success_count,
            "risk_triggered_count": self.risk_triggered_count,
            "source_sessions": self.source_sessions[-20:],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any], name: str = "", env_type: str = "") -> "ExpMeta":
        def _dt(v):
            if not v:
                return None
            try:
                return datetime.fromisoformat(str(v))
            except (ValueError, TypeError):
                return None

        return cls(
            name=str(d.get("name", name)),
            version=str(d.get("version", "1.0")),
            env_type=str(d.get("env_type", env_type)),
            eval_mode=str(d.get("eval_mode", "standard")),
            created=_dt(d.get("created")),
            updated=_dt(d.get("updated")),
            usage_count=int(d.get("usage_count", 0)),
            success_count=int(d.get("success_count", 0)),
            risk_triggered_count=int(d.get("risk_triggered_count", 0)),
            source_sessions=list(d.get("source_sessions") or []),
        )


@dataclass
class Experience:
    """
    完整经验对象，包含正文和元数据。
    - content  → 写入 SKILL.md（纯 Markdown，无 frontmatter）
    - meta     → 写入 meta.yaml
    """
    name: str
    env_type: str
    description: str   # 一句话描述，用于 index
    trigger: str       # 触发条件，用于 index
    content: str       # 经验正文（Markdown）
    eval_mode: str = "standard"
    meta: Optional[ExpMeta] = None

    def __post_init__(self):
        if self.meta is None:
            self.meta = ExpMeta(
                name=self.name,
                env_type=self.env_type,
                eval_mode=self.eval_mode,
            )

    @property
    def success_rate(self) -> float:
        return self.meta.success_rate if self.meta else 0.0

    @property
    def exp_md_path(self) -> str:
        return f"{self.env_type}/{self.name}/SKILL.md"

    @property
    def meta_path(self) -> str:
        return f"{self.env_type}/{self.name}/meta.yaml"

    def to_index_entry(self) -> Dict[str, Any]:
        """index.yaml 中的精简条目：只含检索所需字段，不含 tags。"""
        return {
            "name": self.name,
            "description": self.description,
            "trigger": self.trigger,
            "success_rate": self.success_rate,
            "eval_mode": self.eval_mode,
            "path": self.exp_md_path,
        }


@dataclass
class ExpIndex:
    last_updated: datetime = field(default_factory=datetime.now)
    total_exps: int = 0
    environments: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)

    def upsert(self, exp: Experience) -> None:
        env = exp.env_type
        if env not in self.environments:
            self.environments[env] = []
        entries = self.environments[env]
        for i, e in enumerate(entries):
            if e.get("name") == exp.name:
                entries[i] = exp.to_index_entry()
                self.last_updated = datetime.now()
                return
        entries.append(exp.to_index_entry())
        self.total_exps += 1
        self.last_updated = datetime.now()

    def remove(self, env_type: str, name: str) -> bool:
        entries = self.environments.get(env_type, [])
        for i, e in enumerate(entries):
            if e.get("name") == name:
                entries.pop(i)
                self.total_exps = max(0, self.total_exps - 1)
                self.last_updated = datetime.now()
                return True
        return False

    def list_env(self, env_type: str) -> List[Dict[str, Any]]:
        return list(self.environments.get(env_type, []))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "last_updated": self.last_updated.isoformat(),
            "stats": {
                "total_exps": self.total_exps,
                "environments": list(self.environments.keys()),
            },
            "environments": self.environments,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExpIndex":
        inst = cls()
        if d.get("last_updated"):
            try:
                inst.last_updated = datetime.fromisoformat(str(d["last_updated"]))
            except (ValueError, TypeError):
                pass
        stats = d.get("stats") or {}
        inst.total_exps = int(stats.get("total_exps", 0))
        inst.environments = {k: list(v) for k, v in (d.get("environments") or {}).items()}
        return inst


# ------------------------------------------------------------------ #
# ExpBank
# ------------------------------------------------------------------ #

class ExpBank:
    """
    Experience 文件系统的 CRUD 操作。
    每个经验目录包含：meta.yaml（元数据） + SKILL.md（正文）。
    写操作均原子化（先写临时文件再 rename）。
    """

    def __init__(self, exp_dir: str):
        self._root = Path(exp_dir)
        self._root.mkdir(parents=True, exist_ok=True)
        self._index_path = self._root / "index.yaml"
        if not self._index_path.exists():
            self._write_yaml(self._index_path, ExpIndex().to_dict())
        log.info("ExpBank initialized: root=%s", self._root)

    # -- 索引 --

    def read_index(self) -> ExpIndex:
        try:
            d = yaml.safe_load(self._index_path.read_text(encoding="utf-8")) or {}
            return ExpIndex.from_dict(d)
        except Exception as e:
            log.error("failed to read global index: %s", e)
            return ExpIndex()

    def _save_index(self, index: ExpIndex) -> None:
        self._write_yaml(self._index_path, index.to_dict())

    # -- Experience CRUD --

    def read_exp(self, env_type: str, name: str) -> Optional[Experience]:
        """读取 meta.yaml + SKILL.md，组装为 Experience 对象。"""
        exp_dir = self._root / env_type / name
        skill_md = exp_dir / "SKILL.md"
        meta_yaml = exp_dir / "meta.yaml"

        if not skill_md.exists():
            return None
        try:
            content = skill_md.read_text(encoding="utf-8").strip()
            meta = None
            if meta_yaml.exists():
                raw = yaml.safe_load(meta_yaml.read_text(encoding="utf-8")) or {}
                meta = ExpMeta.from_dict(raw, name=name, env_type=env_type)
            else:
                meta = ExpMeta(name=name, env_type=env_type)

            # description / trigger 从 index 获取（或降级为空字符串）
            index = self.read_index()
            entry = next(
                (e for e in index.environments.get(env_type, []) if e.get("name") == name),
                {},
            )
            return Experience(
                name=name,
                env_type=env_type,
                description=str(entry.get("description", "")),
                trigger=str(entry.get("trigger", "")),
                content=content,
                eval_mode=meta.eval_mode,
                meta=meta,
            )
        except Exception as e:
            log.error("failed to read experience %s/%s: %s", env_type, name, e)
            return None

    def write_exp(self, exp: Experience) -> None:
        """
        写入 meta.yaml + SKILL.md，并更新全局 index.yaml。
        若 meta 已存在则保留 usage_count / success_count / risk_triggered_count。
        """
        exp_dir = self._root / exp.env_type / exp.name
        exp_dir.mkdir(parents=True, exist_ok=True)

        # 若已有 meta，保留调用统计，不覆盖
        meta_path = exp_dir / "meta.yaml"
        if meta_path.exists() and exp.meta is not None:
            existing_raw = yaml.safe_load(meta_path.read_text(encoding="utf-8")) or {}
            exp.meta.usage_count = int(existing_raw.get("usage_count", exp.meta.usage_count))
            exp.meta.success_count = int(existing_raw.get("success_count", exp.meta.success_count))
            exp.meta.risk_triggered_count = int(
                existing_raw.get("risk_triggered_count", exp.meta.risk_triggered_count)
            )
        if exp.meta is None:
            exp.meta = ExpMeta(name=exp.name, env_type=exp.env_type, eval_mode=exp.eval_mode)
        exp.meta.updated = datetime.now()

        self._write_yaml(meta_path, exp.meta.to_dict())
        self._write_text(exp_dir / "SKILL.md", exp.content)

        index = self.read_index()
        index.upsert(exp)
        self._save_index(index)
        log.info(
            "wrote experience: %s/%s  eval_mode=%s  sr=%.2f",
            exp.env_type, exp.name, exp.eval_mode, exp.success_rate,
        )

    def record_usage(
        self,
        env_type: str,
        name: str,
        success: bool,
        risk_triggered: Optional[bool] = None,
    ) -> None:
        """
        记录一次 agent 调用结果（由 ExpEpisodeHandler 在 episode 结束后回调）。
        更新 meta.yaml 中的 usage_count / success_count / risk_triggered_count。
        """
        meta_path = self._root / env_type / name / "meta.yaml"
        if not meta_path.exists():
            log.warning("record_usage: meta not found for %s/%s", env_type, name)
            return
        try:
            raw = yaml.safe_load(meta_path.read_text(encoding="utf-8")) or {}
            raw["usage_count"] = int(raw.get("usage_count", 0)) + 1
            if success:
                raw["success_count"] = int(raw.get("success_count", 0)) + 1
            if risk_triggered is True:
                raw["risk_triggered_count"] = int(raw.get("risk_triggered_count", 0)) + 1
            else:
                raw.setdefault("risk_triggered_count", int(raw.get("risk_triggered_count", 0)))
            raw["updated"] = datetime.now().isoformat()
            self._write_yaml(meta_path, raw)

            # 同步 index.yaml 中的 success_rate
            exp = self.read_exp(env_type, name)
            if exp:
                index = self.read_index()
                index.upsert(exp)
                self._save_index(index)
        except Exception as e:
            log.error("record_usage failed for %s/%s: %s", env_type, name, e)

    def delete_exp(self, env_type: str, name: str) -> bool:
        exp_dir = self._root / env_type / name
        if not exp_dir.exists():
            return False
        import shutil
        shutil.rmtree(str(exp_dir))
        index = self.read_index()
        removed = index.remove(env_type, name)
        if removed:
            self._save_index(index)
        log.info("deleted experience: %s/%s", env_type, name)
        return removed

    def list_env_types(self) -> List[str]:
        return [p.name for p in self._root.iterdir() if p.is_dir()]

    def list_exps(self, env_type: str) -> List[str]:
        env_dir = self._root / env_type
        if not env_dir.exists():
            return []
        return [p.name for p in env_dir.iterdir() if p.is_dir() and (p / "SKILL.md").exists()]

    def list_all_exps(self) -> List[Experience]:
        exps = []
        for env_type in self.list_env_types():
            for name in self.list_exps(env_type):
                e = self.read_exp(env_type, name)
                if e:
                    exps.append(e)
        return exps

    def write_extra(self, env_type: str, name: str, filename: str, content: str) -> None:
        """写入 details.md / examples.md 等附加文件"""
        p = self._root / env_type / name / filename
        p.parent.mkdir(parents=True, exist_ok=True)
        self._write_text(p, content)

    # -- 内部工具 --

    @staticmethod
    def _write_yaml(path: Path, data: Dict[str, Any]) -> None:
        tmp = path.with_suffix(".tmp")
        tmp.write_text(
            yaml.dump(data, allow_unicode=True, sort_keys=False, default_flow_style=False),
            encoding="utf-8",
        )
        tmp.replace(path)

    @staticmethod
    def _write_text(path: Path, text: str) -> None:
        tmp = path.with_suffix(".tmp")
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(path)


# ------------------------------------------------------------------ #
# 相似度工具（供 ExpUpdater 使用）
# ------------------------------------------------------------------ #

def name_similarity(a: str, b: str) -> float:
    if a == b:
        return 1.0
    w1 = set(a.lower().split("_"))
    w2 = set(b.lower().split("_"))
    if not w1 or not w2:
        return 0.0
    return len(w1 & w2) / len(w1 | w2)


def description_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    w1 = set(re.findall(r"\w+", a.lower()))
    w2 = set(re.findall(r"\w+", b.lower()))
    if not w1 or not w2:
        return 0.0
    return len(w1 & w2) / len(w1 | w2)


def exp_similarity(s1: Experience, s2: Experience) -> float:
    return 0.7 * name_similarity(s1.name, s2.name) + 0.3 * description_similarity(s1.description, s2.description)
