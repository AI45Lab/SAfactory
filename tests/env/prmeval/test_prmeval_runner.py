from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).parents[3]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


runner = _load("prmeval_runner_under_test", ROOT / "env/prmeval/runner.py")
adapter = _load("prmeval_adapter_under_test", ROOT / "env/prmeval/adapter.py")
evaluator = _load("prmeval_rule_evaluator_under_test", ROOT / "env/prmeval/rule_evaluator.py")


def _request() -> dict:
    return {
        "job_id": "prmeval-smoke",
        "session_id": "session-1",
        "model": "route",
        "gateway_base_url": "http://127.0.0.1:8000/v1/sessions",
        "env_params": {
            "dataset": {
                "id": "trajectory-1",
                "task": "Move the object.",
                "frames": "episode.npz",
            },
            "media_root": "/tmp/media",
            "results_root": "/tmp/results",
            "prmeval": {"sampling": {"base_frames": 3}, "infer": {"name": "openai_compatible"}},
        },
    }


def test_runner_module_import_does_not_require_prmeval() -> None:
    assert callable(runner.read_request)
    assert callable(runner.run_episode)


def test_relative_frame_path_requires_an_explicit_media_root() -> None:
    request = _request()
    request["env_params"].pop("media_root")
    try:
        adapter._prepare_trajectory(request["env_params"]["dataset"], request["env_params"])
    except ValueError as exc:
        assert "media_root" in str(exc)
    else:
        raise AssertionError("relative frames path should require media_root")


def test_relative_frame_path_is_resolved_from_media_root() -> None:
    task = adapter._prepare_trajectory(
        {"id": "one", "task": "Move", "frames": "episode.npz"},
        {"media_root": "/tmp/media"},
    )
    assert task["frames"] == str((Path("/tmp/media") / "episode.npz").resolve())


def test_summary_is_flattened_for_rule_evaluator() -> None:
    summary = {
        "coverage": {"successful": 1},
        "metrics": {"progress": {"mse": 0.125, "pearson": 0.75, "num_samples": 1}},
    }
    metrics = adapter._flatten_summary(summary)
    assert metrics["mse"] == 0.125
    assert metrics["pearson"] == 0.75
    assert evaluator.mse_to_reward(0.0) == 10.0
    assert 0.0 < evaluator.mse_to_reward(0.125) < 10.0


def test_malformed_request_emits_one_controlled_failure() -> None:
    completed = subprocess.run(
        [sys.executable, str(ROOT / "env/prmeval/runner.py")],
        input="[1]",
        capture_output=True,
        text=True,
        env={"SAFACTORY_SESSION_ID": "fallback"},
        check=False,
    )
    assert completed.returncode == 0
    result = json.loads(completed.stdout)
    assert result["session_id"] == "fallback"
    assert result["status"] == "failed"


def test_rule_evaluator_score_mapping_is_available_without_runtime_dependencies() -> None:
    assert evaluator.mse_to_reward(0.125) > 0
    try:
        evaluator.mse_to_reward(float("nan"))
    except ValueError as exc:
        assert "finite" in str(exc)
    else:
        raise AssertionError("invalid MSE must fail closed")
