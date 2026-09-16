from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).parents[3]
ENV_ROOT = ROOT / "env/prmeval"


def _read_yaml_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_standard_file_set_and_matching_names() -> None:
    expected = {
        "runner.py",
        "adapter.py",
        "rule_evaluator.py",
        "prmeval_config.yaml",
        "prmeval_start.yaml",
        "prmeval_config.rjob.yaml",
        "prmeval_start.rjob.yaml",
        "request.smoke.json",
    }
    assert expected.issubset({path.name for path in ENV_ROOT.iterdir()})
    for path in expected - {"request.smoke.json", "runner.py", "adapter.py", "rule_evaluator.py"}:
        text = _read_yaml_text(ENV_ROOT / path)
        assert "env_name: prmeval" in text or "agent_name: prmeval" in text


def test_smoke_request_has_one_dataset_row_and_native_config() -> None:
    request = json.loads((ENV_ROOT / "request.smoke.json").read_text(encoding="utf-8"))
    assert request["env_params"]["dataset"]["id"] == "prmeval-contract-001"
    assert request["env_params"]["prmeval"]["infer"]["name"] == "openai_compatible"
    assert len(request["env_params"]["dataset"]["frames"]) == 3
