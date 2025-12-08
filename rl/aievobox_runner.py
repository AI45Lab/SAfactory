#!/usr/bin/env python3
"""
AIEvoBox Rollout Runner

This script is launched by the Buffer Server via subprocess.
It reads configuration from environment variables, runs the Interactor,
and the data is written to the shared database by DataManager.
Buffer Server queries the database via /get_rollout_data.
"""

import asyncio
import json
import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from typing import Any, Dict

import yaml

# Add AIEvoBox to path
AIEVOBOX_ROOT = os.environ.get("AIEVOBOX_ROOT", "/root/AIEvoBox")
if AIEVOBOX_ROOT not in sys.path:
    sys.path.insert(0, AIEVOBOX_ROOT)

# Setup logging
LOG_DIR = os.path.join(AIEVOBOX_ROOT, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "aievobox_runner.log")

logger = logging.getLogger("aievobox_runner")
logger.setLevel(logging.DEBUG)

# File handler with rotation
file_handler = RotatingFileHandler(
    LOG_FILE, maxBytes=50*1024*1024, backupCount=5, encoding="utf-8"
)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
))
logger.addHandler(file_handler)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter(
    "%(asctime)s [%(levelname)s] %(message)s"
))
logger.addHandler(console_handler)

logger.info(f"AIEvoBox Runner logging to: {LOG_FILE}")

from core.data_manager.manager import DataManager
from core.interactor import Interactor


async def run_rollout(config: Dict[str, Any]):
    """Main rollout execution logic."""
    # Use LLM Proxy URL instead of remote engine URL directly
    llm_proxy_url = os.environ.get("LLM_PROXY_URL", "http://127.0.0.1:8890")
    num_repeat_per_sample = int(config.get("num_repeat_per_sample", os.environ.get("NUM_REPEAT_PER_SAMPLE", 1)))
    max_steps = int(config.get("max_steps", os.environ.get("ROLLOUT_MAX_STEPS", 10)))

    logger.info(f"Starting rollout")
    logger.info(f"LLM Proxy URL: {llm_proxy_url}")
    logger.info(f"Num repeat per sample: {num_repeat_per_sample}")
    logger.info(f"Max steps: {max_steps}")

    # Initialize DataManager with shared database
    db_url = os.environ.get("AIEVOBOX_DB_URL", "sqlite:////root/AIEvoBox/rollout.db")
    dm = DataManager(db_url=db_url)
    await dm.init()
    logger.info(f"DataManager initialized with DB: {db_url}")

    # Load environment configs
    env_config_path = os.path.join(AIEVOBOX_ROOT, "env/search/search_env_configs.yaml")
    if os.path.exists(env_config_path):
        with open(env_config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        for env in cfg.get("environments", []):
            try:
                await dm.add_environment_config(
                    env_name=env["env_name"],
                    **env.get("env_params", {})
                )
            except Exception as e:
                logger.error(f"Error adding environment config: {e}")

    envs = await dm.get_all_environments()
    logger.info(f"Loaded {len(envs)} environments")

    # Setup Interactor with LLM Proxy
    # The LLM Proxy expects URLs like: /v1/{session_id}/chat/completions
    # So we use SessionSuffixBaseURLProvider to append session_id to the URL
    from core.interactor import SessionSuffixBaseURLProvider

    api_key = os.environ.get("OPENAI_API_KEY", "openai_api_key")
    model = os.environ.get("AIEVOBOX_MODEL", "model")
    max_workers = int(os.environ.get("ROLLOUT_BATCH_SIZE", "128"))

    base_url_provider = SessionSuffixBaseURLProvider(base_url_root=f"{llm_proxy_url}/v1")
    logger.info(f"Using SessionSuffixBaseURLProvider with root: {llm_proxy_url}/v1")

    # Get temperature from sampling_params, default to 1.0
    sampling_params = config.get("sampling_params", {})
    temperature = sampling_params.get("temperature", 1.0)
    logger.info(f"Temperature: {temperature}")

    interactor = Interactor(
        base_url_provider=base_url_provider,
        data_manager=dm,
        max_workers=max_workers,
        max_steps=max_steps,
        api_key=api_key,
        model=model,
        temperature=temperature,
        enable_render=False,
    )

    try:
        await interactor.run_all_environments(episodes_per_env=num_repeat_per_sample)
        logger.info("Rollout completed successfully")
    except Exception as e:
        logger.error(f"Rollout error: {e}")
        raise


def main():
    # Read config from environment variable
    config_json = os.environ.get("AIEVOBOX_ROLLOUT_CONFIG", "{}")
    try:
        config = json.loads(config_json)
    except json.JSONDecodeError:
        config = {}

    logger.info(f"Config: {config}")

    # Run the async rollout
    asyncio.run(run_rollout(config))


if __name__ == "__main__":
    main()
