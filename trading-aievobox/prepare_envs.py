#!/usr/bin/env python3
"""
Prepare TradingGym environments and clear trajectories.

- Connects to AIEvoBox DB (AIEVOBOX_DB_URL or default trading-aievobox/db/trading_rollout.db)
- Optionally clears all trajectories (interaction_steps + interaction_sessions)
- Ensures there is at least one base trading_gym env; if not, creates from TRADING_* envs
- Replicates the base env N times (with unique env_id)

Usage examples:
  python3 trading-aievobox/prepare_envs.py --n 1000 --clear-trajectories
  AIEVOBOX_DB_URL=sqlite:////abs/path/to/db.sqlite3 python3 trading-aievobox/prepare_envs.py --n 500
"""

import argparse
import asyncio
import os
import sys
from typing import Any, Dict


def _ensure_aievobox_importable() -> None:
    root = os.environ.get("AIEVOBOX_ROOT", "/root/AIEvoBox")
    if os.path.isdir(root) and root not in sys.path:
        sys.path.insert(0, root)


async def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare trading envs and clear trajectories")
    parser.add_argument("--n", type=int, default=1000, help="Number of new trading_gym envs to add")
    parser.add_argument("--env-name", default="trading_gym", help="Environment name to create")
    parser.add_argument("--clear-trajectories", action="store_true", help="Delete all interaction steps and sessions")
    args = parser.parse_args()

    _ensure_aievobox_importable()
    from core.data_manager.manager import DataManager
    from core.data_manager.models import EnvironmentConfig, InteractionSession, InteractionStep

    db_url = os.environ.get("AIEVOBOX_DB_URL")
    if not db_url:
        # default to repo db path
        script_dir = os.path.dirname(os.path.abspath(__file__))
        db_path = os.path.join(script_dir, "db", "trading_rollout.db")
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        db_url = f"sqlite:////{db_path}"
        os.environ["AIEVOBOX_DB_URL"] = db_url

    print(f"DB: {db_url}")

    dm = DataManager(db_url=db_url)
    await dm.init()

    # Optionally clear trajectories
    if args.clear_trajectories:
        deleted_steps = await InteractionStep.all().delete()
        deleted_sess = await InteractionSession.all().delete()
        print(f"Cleared trajectories: steps={deleted_steps} sessions={deleted_sess}")

    # Ensure base env exists or create from TRADING_*
    existing = await EnvironmentConfig.filter(env_name=args.env_name).first()
    if existing is None:
        data_dir = os.environ.get("TRADING_DATA_DIR")
        price_file = os.environ.get("TRADING_PRICE_FILE")
        tweet_file = os.environ.get("TRADING_TWEET_FILE") or None
        window = int(os.environ.get("TRADING_WINDOW", "7"))
        assert data_dir and price_file, "TRADING_DATA_DIR / TRADING_PRICE_FILE must be set to create base env"
        base = await dm.add_environment_config(
            env_name=args.env_name,
            data_dir=data_dir,
            price_filename=price_file,
            tweet_filename=tweet_file,
            window_size=window,
        )
        print(f"Created base env: {base.env_name} {base.env_id}")
        env_params: Dict[str, Any] = {
            "data_dir": data_dir,
            "price_filename": price_file,
            "tweet_filename": tweet_file,
            "window_size": window,
        }
    else:
        env_params = existing.env_params or {}
        print(
            f"Found base env: {existing.env_name} {existing.env_id} | params: "
            f"data_dir={env_params.get('data_dir')} price={env_params.get('price_filename')} window={env_params.get('window_size')}"
        )

    # Replicate N times
    created = 0
    for _ in range(args.n):
        await dm.add_environment_config(env_name=args.env_name, **{k: v for k, v in env_params.items() if v is not None})
        created += 1
    total = await EnvironmentConfig.filter(env_name=args.env_name).count()
    print(f"Replicated env '{args.env_name}' {created} times. Total envs with this name: {total}")

    await dm.close()


if __name__ == "__main__":
    asyncio.run(main())

