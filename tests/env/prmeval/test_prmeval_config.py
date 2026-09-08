from __future__ import annotations

import json
from pathlib import Path

import yaml

from manager.simulation_config import load_agent_start_config

ROOT = Path(__file__).parents[3]
ENV_ROOT = ROOT / "env/prmeval"


def test_prmeval_config_schedules_one_self_contained_progress_trajectory() -> None:
    config = yaml.safe_load((ENV_ROOT / "prmeval_config.yaml").read_text(encoding="utf-8"))
    environments = config["environments"]

    assert len(environments) == 1
    environment = environments[0]
    assert environment["env_name"] == "prmeval"
    assert environment["env_image"]
    assert environment["dataset_load_mode"] == "eager"
    assert environment["env_params"]["prmeval"]["infer"]["options"]["keep_base_url"] is True

    rows = [
        json.loads(line)
        for line in (ENV_ROOT / "datasets/prmeval_smoke.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == 1
    trajectory = rows[0]
    assert trajectory["id"] == "prmeval-progress-smoke-001"
    assert trajectory["quality_label"] == "successful"
    assert trajectory["partial_success"] == 1.0
    assert trajectory["is_simulation"] is True
    assert len(trajectory["frames"]) == len(trajectory["target_progress"]) == 3


def test_prmeval_start_config_mounts_only_the_adapter_and_results() -> None:
    docker = load_agent_start_config(str(ENV_ROOT / "prmeval_start.yaml"))["prmeval"]["docker"]

    assert docker["run_command"] == "python /tmp/safactory-prmeval-runner.py"
    assert docker["workdir"] == "/workspace"
    assert any(volume["target"] == "/tmp/safactory-prmeval-runner.py" for volume in docker["volumes"])
    assert any(volume["target"] == "/workspace/Safactory/results" for volume in docker["volumes"])
