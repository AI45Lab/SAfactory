"""Local contract fixture for the production deepsafe-cyber adapter.

The real benchmark image and nested dockerd are not available on a contract
workstation.  This fixture imports the production adapter, replaces only
``_wait_for_nested_docker`` and ``_run_native``, and makes the fake native run
read the generated providers.yaml and issue one request to the mock Gateway
session.  Provider generation, native argv, result parsing, and metric mapping
therefore remain covered by the production code.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Any
from urllib.request import Request, urlopen


def run_case(request: dict[str, Any], task: dict[str, Any], session_url: str):
    production_path = Path.cwd() / "env" / "deepsafe_cyber" / "adapter.py"
    spec = importlib.util.spec_from_file_location("deepsafe_cyber_production_adapter", production_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import production adapter: {production_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    module._wait_for_nested_docker = lambda timeout_s: None
    module._run_native = lambda command, **kwargs: _fake_native(command, **kwargs)
    return module.run_case(request, task, session_url)


def _fake_native(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    timeout_s: float,
) -> subprocess.CompletedProcess[str]:
    provider_path = Path(env["DEEPSAFE_CYBER_PROVIDER_CONFIG"])
    assert provider_path.stat().st_mode & 0o777 == 0o600
    provider = provider_path.read_text(encoding="utf-8")
    base_url = _yaml_scalar(provider, "base_url")
    model = _yaml_scalar(provider, "model")
    assert base_url and model

    payload = json.dumps(
        {"model": model, "messages": [{"role": "user", "content": "contract fixture"}], "stream": False}
    ).encode()
    call = Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urlopen(call, timeout=10) as response:
        assert response.status == 200
        json.load(response)

    task = command[command.index("--task") + 1]
    runs_root = Path(command[command.index("--runs-root") + 1])
    run_id = "contract-run-001"
    result_directory = runs_root / run_id
    public = result_directory / "public"
    evidence = result_directory / "private" / "evidence" / "patcheval"
    public.mkdir(parents=True, exist_ok=True)
    evidence.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "schema": "deepsafe-cyber.published-result.v2",
        "run_id": run_id,
        "terminal": {"code": "MODEL_INCORRECT", "reason": "contract fixture"},
        "evaluation": {"model_verdict": "INCORRECT"},
        "cleanup": {"status": "COMPLETED"},
        "artifacts": {"eval_log": {"ref": f"{run_id}/private/eval/contract.eval"}},
    }
    (public / "result.json").write_text(json.dumps(result) + "\n", encoding="utf-8")
    (evidence / "patcheval-score-evidence.json").write_text("{}\n", encoding="utf-8")
    cli = {
        "schema": "deepsafe-cyber.cli-result.v2",
        "run_id": run_id,
        "result_ref": run_id,
        "task": task,
    }
    return subprocess.CompletedProcess(
        command,
        0,
        stdout=json.dumps(cli, separators=(",", ":")) + "\n",
        stderr="deepsafe-cyber contract fixture diagnostics\n",
    )


def _yaml_scalar(document: str, key: str) -> str:
    match = re.search(rf"^\s*{re.escape(key)}:\s*(\".*?\"|'.*?'|[^#\n]+)\s*$", document, re.MULTILINE)
    if not match:
        return ""
    value = match.group(1).strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        value = value[1:-1]
    return value
