"""
ExpService — 从 AIEvoBox trajectory DB 合成和管理 Agent Experiences。

快速使用：
    from exp_service import ExpService
    from exp_service.config import ServiceConfig

    cfg = ServiceConfig.from_dict({
        "db": {"url": "sqlite:///evobox.db"},
        "llm": {"base_url": "http://...", "model": "Qwen2.5-72B-Instruct"},
        "exp_dir": "./experiences",
    })
    service = ExpService.from_config(cfg)
    await service.run_once()

CLI：
    python -m exp_service run-once --config exp_service_config.yaml
    python -m exp_service status --exp-dir ./experiences
"""

from .config import ServiceConfig, DBConfig, LLMConfig, SelectorConfig, GeneratorConfig, UpdaterConfig
from .db_reader import DBReader
from .exp_bank import Experience, ExpMeta, ExpBank, ExpIndex
from .trajectory import (
    TrajectoryRecord,
    TrajectoryBuilder,
    TrajectorySelector,
    RewardThresholdSelector,
    FailureSelector,
    TimeWindowSelector,
    EnvTypeSelector,
    CompositeSelector,
)
from .generator import ExpGenerator, ExpDraft, make_openai_llm
from .updater import ExpUpdater, UpdateResult
from .service import ExpService, ServiceState, RunResult

__all__ = [
    "ServiceConfig", "DBConfig", "LLMConfig", "SelectorConfig", "GeneratorConfig", "UpdaterConfig",
    "DBReader",
    "Experience", "ExpMeta", "ExpBank", "ExpIndex",
    "TrajectoryRecord", "TrajectoryBuilder", "TrajectorySelector",
    "RewardThresholdSelector", "FailureSelector", "TimeWindowSelector",
    "EnvTypeSelector", "CompositeSelector",
    "ExpGenerator", "ExpDraft", "make_openai_llm",
    "ExpUpdater", "UpdateResult",
    "ExpService", "ServiceState", "RunResult",
]
