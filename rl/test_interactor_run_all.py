#!/usr/bin/env python3
"""
测试 AIEvoBox Interactor 独立运行（脱离 slime、LLM Proxy、Buffer Server）。

使用方式：
    直接指向 LLM 引擎运行测试:
    LLM_BASE_URL=http://your-llm-server:8000/v1 python3 test_interactor_run_all.py
"""

import asyncio
import importlib
import logging
import os
import resource
import sys

import yaml

# Increase file descriptor limit for high concurrency
_soft, _hard = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (min(65536, _hard), _hard))
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logging.getLogger("env.search.search_env").setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)


# ===== 配置区域：可通过环境变量覆盖 =====
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_AIEVOBOX_ROOT = os.environ.get("AIEVOBOX_ROOT", os.path.dirname(_SCRIPT_DIR))

# 测试使用 rl/test.db
DB_URL = os.environ.get("AIEVOBOX_DB_URL", f"sqlite:///{_AIEVOBOX_ROOT}/rl/test.db")

# LLM 引擎地址（直接连接，不经过 proxy）
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", os.environ.get("LLM_PROXY_URL", "http://100.99.102.150:30000/v1"))

# LLM / Agent 配置
API_KEY = os.environ.get("OPENAI_API_KEY", "test")
MODEL = os.environ.get("AIEVOBOX_MODEL", "custom")
TEMPERATURE = float(os.environ.get("TEMPERATURE", "1.0"))

# 交互与可视化配置
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", "128"))
MAX_STEPS = int(os.environ.get("MAX_STEPS", "10"))
EPISODES_PER_ENV = int(os.environ.get("EPISODES_PER_ENV", "1"))
VISUAL_DIR = os.environ.get("VISUAL_DIR", "/tmp/aievobox_vis")
ENABLE_RENDER = os.environ.get("ENABLE_RENDER", "false").lower() == "true"

# 环境配置文件路径
ENV_CONFIG_PATH = os.environ.get(
    "ENV_CONFIG_PATH",
    os.path.join(_AIEVOBOX_ROOT, "env/search/search_env_configs.yaml")
)
# ===========================================


def _ensure_project_root_in_path() -> None:
    current_file = os.path.abspath(__file__)
    rl_dir = os.path.dirname(current_file)
    project_root = os.path.dirname(rl_dir)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)


_ensure_project_root_in_path()

from core.data_manager.manager import DataManager
from core.interactor import Interactor
from core.llm import StaticBaseURLProvider


async def main() -> None:
    # 确保所有环境完成注册
    importlib.import_module("env")

    logger.info("=" * 60)
    logger.info("AIEvoBox Interactor 独立测试")
    logger.info("=" * 60)
    logger.info(f"DB URL: {DB_URL}")
    logger.info(f"LLM Base URL: {LLM_BASE_URL}")
    logger.info(f"Model: {MODEL}")
    logger.info(f"Temperature: {TEMPERATURE}")
    logger.info(f"Max Workers: {MAX_WORKERS}")
    logger.info(f"Max Steps: {MAX_STEPS}")
    logger.info(f"Episodes per Env: {EPISODES_PER_ENV}")
    logger.info(f"Enable Render: {ENABLE_RENDER}")
    logger.info("=" * 60)

    # 初始化 DataManager
    data_manager = DataManager(db_url=DB_URL)
    await data_manager.init()
    logger.info("DataManager 初始化完成")

    # 加载环境配置
    # if os.path.exists(ENV_CONFIG_PATH):
    #     with open(ENV_CONFIG_PATH, "r", encoding="utf-8") as f:
    #         cfg = yaml.safe_load(f)

    #     for env in tqdm(cfg.get("environments", []), desc="Adding environment configs"):
    #         try:
    #             await data_manager.add_environment_config(
    #                 env_name=env["env_name"],
    #                 **env.get("env_params", {})
    #             )
    #         except Exception as e:
    #             logger.error(f"Error adding environment config: {e}")
    # else:
    #     logger.warning(f"环境配置文件不存在: {ENV_CONFIG_PATH}")

    env_configs = await data_manager.get_all_environments()
    logger.info(f"已加载 {len(env_configs)} 个环境")
    for cfg in env_configs[:5]:  # 只显示前 5 个
        logger.info(f"  - {cfg.env_name}_{cfg.env_id}")
    if len(env_configs) > 5:
        logger.info(f"  ... 共 {len(env_configs)} 个环境")

    if not env_configs:
        logger.error("没有可用的环境，退出")
        await data_manager.close()
        return

    # 创建 BaseURLProvider
    base_url_provider = StaticBaseURLProvider(base_url=LLM_BASE_URL)

    # 创建 Interactor
    interactor = Interactor(
        base_url_provider=base_url_provider,
        data_manager=data_manager,
        max_workers=MAX_WORKERS,
        max_steps=MAX_STEPS,
        n_episodes=EPISODES_PER_ENV,
        visual_save_path=VISUAL_DIR,
        enable_render=ENABLE_RENDER,
        api_key=API_KEY,
        model=MODEL,
        temperature=TEMPERATURE,
    )

    logger.info(f"开始运行 Interactor.run_all_environments, episodes_per_env={EPISODES_PER_ENV}")

    try:
        await interactor.run_all_environments()
        logger.info("Interactor 运行完成")
    except Exception as e:
        logger.error(f"Interactor 运行出错: {e}")
        raise
    finally:
        await data_manager.close()
        logger.info("DataManager 已关闭")


if __name__ == "__main__":
    asyncio.run(main())
