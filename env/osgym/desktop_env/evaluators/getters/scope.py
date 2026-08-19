import json
import logging
import os
import posixpath
from typing import Any, Dict

import requests


logger = logging.getLogger("desktopenv.getters.scope")


def _vm_quote(path: str) -> str:
    return "'" + path.replace("'", "'\"'\"'") + "'"


def get_scope_task(env, config: Dict[str, Any]) -> Dict[str, Any]:
    """Upload and execute a task-local SCOPE reward.py inside the VM."""
    task_dir = config["task_dir"]
    reward_path = config.get("reward_path") or os.path.join(task_dir, "reward.py")
    task_id = config.get("task_id") or os.path.basename(os.path.normpath(task_dir))
    python_bin = config.get("python", "python3")
    timeout = int(config.get("timeout", 300))

    if not os.path.exists(reward_path):
        raise FileNotFoundError(f"SCOPE reward file not found: {reward_path}")

    remote_dir = posixpath.join(config.get("remote_root", "/tmp/osworld_scope"), task_id)
    remote_reward = posixpath.join(remote_dir, "reward.py")

    env.setup_controller._upload_file_setup([
        {"local_path": reward_path, "path": remote_reward}
    ])

    for helper in config.get("helper_files", []):
        local_path = helper["local_path"]
        remote_path = helper["path"]
        if os.path.exists(local_path):
            env.setup_controller._upload_file_setup([
                {"local_path": local_path, "path": remote_path}
            ])
        else:
            logger.warning("Skipping missing SCOPE helper file: %s", local_path)

    command = [
        "bash",
        "-lc",
        f"cd {_vm_quote(remote_dir)} && {python_bin} reward.py",
    ]
    response = requests.post(
        f"http://{env.vm_ip}:{env.server_port}/execute",
        headers={"Content-Type": "application/json"},
        data=json.dumps({"command": command, "shell": False}),
        timeout=timeout,
    )

    if response.status_code != 200:
        logger.error(
            "SCOPE reward execution failed with status %s: %s",
            response.status_code,
            response.text,
        )
        return {
            "output": "",
            "error": response.text,
            "returncode": None,
            "status_code": response.status_code,
        }

    result = response.json()
    result["status_code"] = response.status_code
    result["task_id"] = task_id
    return result
