"""DesktopEnv construction helpers for OSGym."""

import os
from pathlib import Path
from typing import Optional, Tuple

from ..desktop_env.desktop_env import DesktopEnv


SUPPORTED_OBSERVATION_TYPES = {"screenshot", "a11y_tree", "screenshot_a11y_tree"}


def validate_observation_type(
    observation_type: Optional[str],
    field_name: str,
) -> Optional[str]:
    if observation_type is None:
        return None

    normalized = observation_type.strip()
    if normalized not in SUPPORTED_OBSERVATION_TYPES:
        raise ValueError(
            f"Invalid {field_name}: {observation_type}. "
            f"Expected one of {sorted(SUPPORTED_OBSERVATION_TYPES)}."
        )
    return normalized


def resolve_prompt_observation_type(prompt_observation_type: Optional[str]) -> str:
    return validate_observation_type(
        prompt_observation_type,
        "prompt_observation_type",
    ) or "screenshot"


def requires_a11y_tree(
    prompt_observation_type: str,
    eval_mode: str,
    task_id: Optional[str],
) -> bool:
    if prompt_observation_type in {"a11y_tree", "screenshot_a11y_tree"}:
        return True

    if eval_mode == "safety":
        task_id = task_id or ""
        return "popup" in task_id or "induced_text" in task_id

    return False


def resolve_vm_path(
    configured_vm_path: Optional[str],
    current_dir: Path,
    logger,
) -> str:
    if configured_vm_path:
        expanded_vm_path = os.path.expandvars(os.path.expanduser(configured_vm_path))
        if not os.path.isabs(expanded_vm_path):
            expanded_vm_path = os.path.join(current_dir, expanded_vm_path)
        vm_path = os.path.abspath(expanded_vm_path)
        if not os.path.exists(vm_path):
            logger.warning(f"Configured vm_path does not exist yet: {vm_path}")
        return vm_path

    try:
        from ..desktop_env.providers.docker.manager import DockerVMManager

        vm_manager = DockerVMManager()
        return vm_manager.get_vm_path(os_type="Ubuntu", region=None)
    except Exception as exc:
        vm_path = os.path.join(current_dir, "docker_vm_data", "Ubuntu.qcow2")
        logger.warning(f"Auto-download VM failed, falling back to: {vm_path}")
        logger.debug(f"Reason: {exc}")
        return vm_path


def create_desktop_env(
    *,
    provider_name: str,
    vm_path: Optional[str],
    action_space: str,
    screen_size: Tuple[int, int],
    headless: bool,
    require_a11y_tree: bool,
    cache_dir: str,
    host_ip: Optional[str],
) -> DesktopEnv:
    kwargs = {
        "provider_name": provider_name,
        "path_to_vm": vm_path,
        "action_space": action_space,
        "screen_size": screen_size,
        "headless": headless,
        "require_a11y_tree": require_a11y_tree,
        "require_terminal": False,
        "os_type": "Ubuntu",
        "cache_dir": cache_dir,
    }
    if provider_name == "containerd":
        kwargs["host_ip"] = host_ip
    return DesktopEnv(**kwargs)


def run_halfway_setup(env, task_config, logger) -> None:
    """Execute RiOSWorld-style halfway setup directly from task_config."""
    halfway_config = task_config.get("halfway_config") or []
    if not halfway_config:
        return

    logger.info("Running halfway setup...")
    use_proxy = bool(task_config.get("proxy", False) and getattr(env, "enable_proxy", False))

    if hasattr(env.setup_controller, "halfway_setup"):
        env.setup_controller.halfway_setup(halfway_config)
    else:
        env.setup_controller.setup(halfway_config, use_proxy=use_proxy)
