from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

RUNNER_PATH = Path(__file__).parents[3] / "env/prmeval/runner.py"
SPEC = importlib.util.spec_from_file_location("safactory_prmeval_runner", RUNNER_PATH)
assert SPEC and SPEC.loader
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


def _request() -> dict:
    return {
        "job_id": "prmeval-smoke",
        "session_id": "session-1",
        "agent_start_timeout_s": 60,
        "env_params": {
            "dataset": {
                "id": "trajectory-1",
                "task": "Move the object.",
                "frames": [
                    [[[0, 0, 0]]],
                    [[[127, 127, 127]]],
                    [[[255, 255, 255]]],
                ],
                "quality_label": "successful",
                "partial_success": 1.0,
                "target_progress": [0.0, 0.5, 1.0],
            },
            "prmeval": {
                "command": "prmeval",
                "sampling": {"base_frames": 3},
                "infer": {"options": {"custom_option": "preserved"}},
            },
        },
    }


def test_runner_evaluates_only_the_current_trajectory(monkeypatch, tmp_path) -> None:
    result_path = tmp_path / "safactory_result.json"
    monkeypatch.setenv("SAFACTORY_RESULT_PATH", str(result_path))
    observed: dict[str, object] = {}

    def fake_run(command, **kwargs):
        observed["command"] = command
        config_path = Path(command[command.index("--config") + 1])
        config = json.loads(config_path.read_text(encoding="utf-8"))
        observed["config"] = config
        rows = [json.loads(line) for line in Path(config["sampling"]["paths"][0]).read_text().splitlines()]
        assert rows == [_request()["env_params"]["dataset"]]

        output = Path(config["output_dir"]) / config["run_name"]
        output.mkdir(parents=True)
        (output / "metrics.json").write_text(
            json.dumps(
                {
                    "coverage": {"successful": 1, "failed": 0},
                    "metrics": {"progress": {"mse": 0.125, "pearson": 0.75, "num_samples": 1}},
                }
            ),
            encoding="utf-8",
        )
        (output / "metrics_detail.jsonl").write_text(
            json.dumps(
                {
                    "detail_type": "record",
                    "sample_id": "sample-1",
                    "target": {"values": [0.0, 0.5, 1.0]},
                    "prediction": {"values": [0.0, 0.25, 1.0]},
                    "input": {"items": [{"frame_indices": [0, 1, 2]}]},
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0, stdout="{}", stderr="")

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    result = runner.run_episode(_request())

    assert observed["command"][0] == "prmeval"
    config = observed["config"]
    assert config["sampling"]["max_trajectories"] == 1
    assert config["sampling"]["eval_types"] == ["progress"]
    assert config["infer"]["base_url"] == "SAFACTORY_GATEWAY_SESSION_URL_CONTAINER"
    assert config["infer"]["model_id"] == "SAFACTORY_ROUTE_MODEL"
    assert config["infer"]["options"] == {"custom_option": "preserved", "keep_base_url": True}
    assert result["status"] == "succeeded"
    assert result["metrics"]["mse"] == 0.125
    assert result["metrics"]["pearson"] == 0.75
    assert result["metrics"]["predicted_progress"] == [0.0, 0.25, 1.0]


def test_relative_frame_path_requires_an_explicit_media_root() -> None:
    request = _request()
    request["env_params"]["dataset"]["frames"] = "episode.npz"
    try:
        runner._prepare_trajectory(request["env_params"]["dataset"], request["env_params"])
    except ValueError as exc:
        assert "media_root" in str(exc)
    else:
        raise AssertionError("relative frames path should require media_root")
