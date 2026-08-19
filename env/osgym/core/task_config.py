"""Task configuration helpers for OSGym."""

import copy
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from ..evaluation.risk_adapter import (
    is_locally_supported_evaluator_config,
    requires_local_evaluator_adapter,
)


def load_credentials(base_dir: Path, logger) -> Dict[str, Any]:
    """Load optional credentials.yaml used by authenticated tasks."""
    credentials_path = os.path.join(base_dir, "credentials.yaml")
    if not os.path.exists(credentials_path):
        logger.debug(f"No credentials file found at {credentials_path}")
        return {}

    try:
        import yaml

        with open(credentials_path, "r") as f:
            credentials = yaml.safe_load(f) or {}
        logger.info(f"Loaded credentials from {credentials_path}")
        return credentials
    except Exception as e:
        logger.warning(f"Failed to load credentials: {e}")
        return {}


def prepare_task_config(
    dataset: Dict[str, Any],
    eval_mode: str,
    credentials: Dict[str, Any],
    logger,
) -> Tuple[str, str, Dict[str, Any], str]:
    """Validate dataset, inject credentials, and return common task fields."""
    for field in ("id", "instruction"):
        if field not in dataset:
            raise ValueError(f"Dataset missing required field: {field}")

    task_id = dataset["id"]
    task_domain = dataset.get("domain", "default")
    task_config = dataset
    instruction = dataset["instruction"]

    if "evaluator" not in dataset:
        logger.warning(f"Task {task_id} missing evaluator")
    if eval_mode == "safety" and "risk_evaluator" not in dataset:
        logger.warning(f"Safety mode task {task_id} missing risk_evaluator")

    inject_credentials(task_config, credentials, logger)
    logger.info(f"Loaded task: {task_id} (domain: {task_domain}, eval_mode: {eval_mode})")
    return task_id, task_domain, task_config, instruction


def inject_credentials(task_config: Dict[str, Any], credentials: Dict[str, Any], logger) -> None:
    """Replace settings_file or credential_key references with credential payloads."""
    for section in ("config", "evaluator"):
        if section in task_config:
            _inject_credentials(task_config[section], credentials, logger)


def build_desktop_env_task_config(task_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build the DesktopEnv reset config.

    DesktopEnv resolves evaluator functions eagerly. For evaluators handled by
    OSGym itself, pass a dummy evaluator to DesktopEnv and keep the original
    task config for final scoring.
    """
    desktop_task_config = copy.deepcopy(task_config)
    evaluator = desktop_task_config.get("evaluator")

    if isinstance(evaluator, dict) and _should_bypass_desktop_evaluator(evaluator):
        desktop_task_config["evaluator"] = {"func": "infeasible"}

    return desktop_task_config


def _inject_credentials(obj: Any, credentials: Dict[str, Any], logger) -> None:
    if isinstance(obj, dict):
        _inject_from_settings_file(obj, credentials, logger)
        _inject_from_credential_key(obj, credentials, logger)
        for value in obj.values():
            _inject_credentials(value, credentials, logger)
    elif isinstance(obj, list):
        for item in obj:
            _inject_credentials(item, credentials, logger)


def _inject_from_settings_file(obj: Dict[str, Any], credentials: Dict[str, Any], logger) -> None:
    settings_path = obj.get("settings_file")
    if not settings_path:
        return

    parts = settings_path.split("/")
    if "settings" not in parts:
        return
    idx = parts.index("settings")
    if idx + 1 >= len(parts):
        return

    _set_credentials(obj, parts[idx + 1], credentials, logger)


def _inject_from_credential_key(obj: Dict[str, Any], credentials: Dict[str, Any], logger) -> None:
    credential_key = obj.get("credential_key")
    if credential_key:
        _set_credentials(obj, credential_key, credentials, logger)


def _set_credentials(obj: Dict[str, Any], credential_key: str, credentials: Dict[str, Any], logger) -> None:
    creds = credentials.get(credential_key)
    if creds:
        obj["_credentials"] = creds
        logger.debug(f"Injected credentials for '{credential_key}'")
    else:
        logger.warning(f"No credentials found for '{credential_key}'")


def _should_bypass_desktop_evaluator(evaluator: Dict[str, Any]) -> bool:
    return (
        requires_local_evaluator_adapter(evaluator)
        or not is_locally_supported_evaluator_config(evaluator)
    )
