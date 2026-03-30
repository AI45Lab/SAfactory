"""
ExpService CLI 入口。

用法示例：
  python -m exp_service run-once --config exp_service_config.yaml
  python -m exp_service run-loop --interval 300 --config ...
  python -m exp_service status --exp-dir ./experiences
  python -m exp_service evolve --config ... --hours 24
  python -m exp_service prune --config ...
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import yaml


def _build_cfg_from_args(args: argparse.Namespace):
    from .config import ServiceConfig

    # 优先从 --config 文件读取，再用 CLI 参数覆盖
    cfg_dict: dict = {}
    if args.config and Path(args.config).exists():
        with open(args.config, encoding="utf-8") as f:
            cfg_dict = yaml.safe_load(f) or {}

    # CLI 参数直接覆盖对应字段
    if getattr(args, "db_url", None):
        cfg_dict.setdefault("db", {})["url"] = args.db_url
    if getattr(args, "exp_dir", None):
        cfg_dict["exp_dir"] = args.exp_dir
    if getattr(args, "llm_base_url", None):
        cfg_dict.setdefault("llm", {})["base_url"] = args.llm_base_url
    if getattr(args, "llm_model", None):
        cfg_dict.setdefault("llm", {})["model"] = args.llm_model
    if getattr(args, "llm_api_key", None):
        cfg_dict.setdefault("llm", {})["api_key"] = args.llm_api_key
    if getattr(args, "min_reward", None) is not None:
        cfg_dict.setdefault("selector", {})["success_min_reward"] = args.min_reward
    if getattr(args, "state_path", None):
        cfg_dict["state_path"] = args.state_path
    if getattr(args, "interval", None):
        cfg_dict["poll_interval_s"] = args.interval

    return ServiceConfig.from_dict(cfg_dict)


def _setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# ------------------------------------------------------------------ #
# Sub-commands
# ------------------------------------------------------------------ #

def _cmd_run_once(args: argparse.Namespace) -> None:
    from .service import ExpService
    cfg = _build_cfg_from_args(args)
    _setup_logging(cfg.log_level)
    service = ExpService.from_config(cfg)
    result = asyncio.run(service.run_once())
    print(f"[run-once] {result.summary()}")
    if result.error:
        print(f"[run-once] ERROR: {result.error}", file=sys.stderr)
        sys.exit(1)


def _cmd_run_loop(args: argparse.Namespace) -> None:
    from .service import ExpService
    cfg = _build_cfg_from_args(args)
    _setup_logging(cfg.log_level)
    interval = getattr(args, "interval", None) or cfg.poll_interval_s
    service = ExpService.from_config(cfg)
    try:
        asyncio.run(service.run_loop(interval_s=interval))
    except KeyboardInterrupt:
        print("\n[run-loop] stopped by user")


def _cmd_status(args: argparse.Namespace) -> None:
    from .exp_bank import ExpBank
    from .service import ServiceState
    _setup_logging("INFO")

    exp_dir = args.exp_dir
    state_path = getattr(args, "state_path", "./exp_service_state.json")

    bank = ExpBank(exp_dir)
    index = bank.read_index()
    state = ServiceState.load(state_path)

    print(f"Exp dir      : {exp_dir}")
    print(f"Total skills : {index.total_exps}")
    print(f"Env types    : {list(index.environments.keys())}")
    print(f"Last updated : {index.last_updated.isoformat()}")
    print()
    print(f"State file   : {state_path}")
    print(f"Processed    : {len(state.processed_session_ids)} sessions")
    print(f"Total runs   : {state.total_runs}")
    print(f"Last run     : {state.last_run.isoformat() if state.last_run else 'never'}")
    print()
    for env_type in bank.list_env_types():
        names = bank.list_exps(env_type)
        print(f"  [{env_type}]  {len(names)} skills")
        for name in names:
            skill = bank.read_exp(env_type, name)
            if exp:
                print(f"    - {name:<40} sr={skill.success_rate:.2f}  uc={skill.usage_count}")


def _cmd_evolve(args: argparse.Namespace) -> None:
    from .config import ServiceConfig
    from .db_reader import DBReader
    from .exp_bank import ExpBank
    from .trajectory import TrajectoryBuilder, FailureSelector
    from .generator import make_openai_llm
    from .updater import ExpUpdater

    cfg = _build_cfg_from_args(args)
    _setup_logging(cfg.log_level)
    hours = getattr(args, "hours", 24.0)
    threshold = getattr(args, "update_threshold", cfg.updater.evolve_threshold)

    db = DBReader(cfg.db.url)
    bank = ExpBank(cfg.exp_dir)
    llm_fn = make_openai_llm(cfg.llm)
    updater = ExpUpdater(bank, llm_fn, cfg.updater)
    builder = TrajectoryBuilder(db)

    from datetime import datetime, timedelta
    cutoff = datetime.now() - timedelta(hours=hours)

    failure_sel = FailureSelector(
        max_reward=cfg.selector.failure_max_reward,
        top_k=cfg.selector.failure_top_k,
        min_steps=cfg.selector.min_steps,
    )
    sessions = failure_sel.select(db, set())
    sessions = [s for s in sessions if s.created_at is None or s.created_at >= cutoff]
    trajs = builder.build_many(sessions, min_steps=cfg.selector.min_steps)

    print(f"[evolve] found {len(trajs)} failure trajectories (last {hours}h)")
    result = asyncio.run(updater.evolve_with_failures(trajs, update_threshold=threshold))
    print(f"[evolve] {result.summary()}")


def _cmd_prune(args: argparse.Namespace) -> None:
    from .exp_bank import ExpBank
    from .updater import ExpUpdater

    cfg = _build_cfg_from_args(args)
    _setup_logging(cfg.log_level)

    bank = ExpBank(cfg.exp_dir)
    updater = ExpUpdater(bank, cfg=cfg.updater)
    result = updater.prune()
    print(f"[prune] {result.summary()}")


# ------------------------------------------------------------------ #
# Argument parser
# ------------------------------------------------------------------ #

def _common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", type=str, default=None,
                   help="Path to exp_service YAML config file")
    p.add_argument("--db-url", type=str, default=None,
                   help="DB url, e.g. sqlite:///evobox.db")
    p.add_argument("--exp-dir", type=str, default=None,
                   help="Skill filesystem root directory")
    p.add_argument("--llm-base-url", type=str, default=None)
    p.add_argument("--llm-model", type=str, default=None)
    p.add_argument("--llm-api-key", type=str, default=None)
    p.add_argument("--state-path", type=str, default=None,
                   help="Path to state JSON file")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="exp_service",
        description="ExpService: synthesize and manage agent experiences from trajectory DB",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # run-once
    p_once = sub.add_parser("run-once", help="Run pipeline once and exit")
    _common_args(p_once)
    p_once.add_argument("--min-reward", type=float, default=None)
    p_once.set_defaults(func=_cmd_run_once)

    # run-loop
    p_loop = sub.add_parser("run-loop", help="Run pipeline continuously")
    _common_args(p_loop)
    p_loop.add_argument("--interval", type=float, default=None,
                        help="Poll interval in seconds")
    p_loop.set_defaults(func=_cmd_run_loop)

    # status
    p_status = sub.add_parser("status", help="Show experience bank status")
    p_status.add_argument("--exp-dir", type=str, default="./experiences")
    p_status.add_argument("--state-path", type=str, default="./exp_service_state.json")
    p_status.set_defaults(func=_cmd_status)

    # evolve
    p_evolve = sub.add_parser("evolve", help="Evolve skills from recent failures")
    _common_args(p_evolve)
    p_evolve.add_argument("--hours", type=float, default=24.0,
                          help="Look back N hours for failure trajectories")
    p_evolve.add_argument("--update-threshold", type=float, default=None,
                          help="Skill success_rate threshold for evolve")
    p_evolve.set_defaults(func=_cmd_evolve)

    # prune
    p_prune = sub.add_parser("prune", help="Prune low-quality experiences")
    _common_args(p_prune)
    p_prune.set_defaults(func=_cmd_prune)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
