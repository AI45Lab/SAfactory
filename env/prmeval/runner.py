"""PRMEval runner 模板：第一步，只读取 SAfactory request。

本文件可独立运行，只依赖 Python 标准库。当前 succeeded 仅表示请求读取
成功；metrics.evaluation_executed=False 表示尚未执行 PRMEval 评测。
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from prmeval.core import EvalConfig, Evaluator  # type: ignore

RESULT_JSON_PREFIX = "SAFACTORY_RESULT_JSON "
RESULT_PATH_ENV = "SAFACTORY_RESULT_PATH"


def read_request() -> dict[str, Any]:
    raw = sys.stdin.read().strip() or os.environ.get("SAFACTORY_START_REQUEST_JSON", "")
    if not raw:
        raise RuntimeError("missing SimulationStartRequest JSON")
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise TypeError("SimulationStartRequest must be a JSON object")
    return data


def test_read_request():
    import json

    with open(
        "/mnt/shared-storage-user/liuyicong/SAfactory/env/prmeval/config.json",
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    return data


def post_process_result(result: dict[str, Any], session_id, job_id) -> dict[str, Any]:
    """在写入 stdout 之前，可在此处对 result 做最后处理，例如：
    - 补充 metrics 中的评测结果；
    - 对 result 中的敏感信息做脱敏处理；
    - 对 result 中的浮点数做精度截断。
    """

    # 下一步：使用 dataset、prmeval_settings 和上述模型参数构建评测输入。
    # 完整的其他字段仍可从 request 获取，例如 request.get("metadata", {})。
    # 此处只返回输入摘要，避免把完整图像数组写进日志。
    return {
        "session_id": session_id,
        "status": "succeeded",
        "total_reward": 0.0,
        "step_count": 0,
        "terminated": True,
        "truncated": False,
        "error_text": None,
        "metrics": result,
    }


def post_process_config(config: dict[str, Any], request) -> EvalConfig:
    #  在此处对 prmeval_config 做后处理，例如：
    # - 补充模型ID，推理信息

    # 3. 模型调用参数。容器内调用优先使用 SAfactory 注入的 session URL，
    # 它已经处理 localhost 到宿主机地址的转换，并包含 session_id。
    gateway_base_url = request["gateway_base_url"]
    model = os.environ.get("SAFACTORY_ROUTE_MODEL") or request["model"]
    # api_key = request["api_key"]

    # temperature = request.get("temperature", 0.0)
    # timeout_s = request.get("agent_start_timeout_s", 600.0)
    # gateway_session_url = os.environ.get("SAFACTORY_GATEWAY_SESSION_URL_CONTAINER")

    config["infer"]["model_id"] = model
    config["infer"]["base_url"] = gateway_base_url
    # config["infer"]["api_key"]= api_key

    return config


def run_episode(request: dict[str, Any]) -> dict[str, Any]:
    """从完整 request 取出业务输入；后续在这里接入 PRMEval。"""
    # 1. 调度信息：由 SAfactory 为本次任务生成。
    job_id = request["job_id"]
    session_id = request["session_id"]
    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError("session_id must be a non-empty string")

    # 2. 任务输入：dataset 是当前 JSONL 行，不是整个数据集。
    env_params = request.get("env_params")
    if not isinstance(env_params, dict):
        raise TypeError("env_params must be a JSON object")
    dataset = env_params["dataset"]
    output_path = Path("/tmp/safactory_prmeval_datasets/temp_sample.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with output_path.open("w", encoding="utf-8") as f:
            f.write(json.dumps(dataset, ensure_ascii=False) + "\n")
        # 3. PRMEval 执行
        prmeval_config = env_params.get("prmeval", {})
        if not isinstance(dataset, dict) or not isinstance(prmeval_config, dict):
            raise TypeError(
                "env_params.dataset and env_params.prmeval must be JSON objects"
            )
        prmeval_config = post_process_config(prmeval_config, request)
        prmeval_config = EvalConfig.model_validate(prmeval_config)
        summary = Evaluator(prmeval_config).run()
        result = post_process_result(summary, session_id, job_id)

        return result

    finally:
        output_path.unlink(missing_ok=True)


def main() -> int:
    session_id = os.environ.get("SAFACTORY_SESSION_ID", "")
    try:
        request = read_request()
        # request = test_read_request()
        session_id = request.get("session_id", session_id)
        result = run_episode(request)
    except Exception as exc:
        result = {
            "session_id": session_id,
            "status": "failed",
            "total_reward": 0.0,
            "step_count": 0,
            "terminated": True,
            "truncated": False,
            "error_text": str(exc),
            "metrics": {"stage": "request_received", "evaluation_executed": False},
        }
    # stdout 专用于 SAfactory 结果协议；调试信息应写到 stderr。
    _write_result(result)
    return 0


def _write_result(result: dict[str, Any]) -> None:
    _persist_result_artifact(result)
    print(RESULT_JSON_PREFIX + json.dumps(result, ensure_ascii=False), flush=True)


def _persist_result_artifact(result: dict[str, Any]) -> None:
    raw_path = str(os.environ.get(RESULT_PATH_ENV) or "").strip()
    if not raw_path:
        return
    try:
        path = Path(raw_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(path.name + ".tmp")
        tmp_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        tmp_path.replace(path)
    except Exception as exc:
        print(
            f"SAFACTORY_RUNNER_DIAGNOSTIC result_artifact_write_failed: {exc}",
            file=sys.stderr,
            flush=True,
        )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(
            json.dumps(
                {
                    "session_id": os.environ.get("SAFACTORY_SESSION_ID", ""),
                    "status": "failed",
                    "total_reward": 0.0,
                    "step_count": 0,
                    "terminated": True,
                    "truncated": False,
                    "error_text": str(exc),
                    "metrics": {},
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        raise SystemExit(0)
