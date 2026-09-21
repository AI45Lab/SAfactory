#!/usr/bin/env python3
"""Check a runner locally with an owned mock model endpoint, without Launcher/cluster.

Only non-streaming OpenAI chat completions are emulated. Native harness dependencies
must be locally available or replaced by an environment-specific test fixture.
This does not validate Gateway persistence, images, mounts, or cluster submission.
"""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading

from live_smoke import stop_process


def validate_result(result, session_id, expected_status):
    if not isinstance(result, dict) or result.get("session_id") != session_id:
        raise ValueError("result must be an object with the request session_id")
    if result.get("status") != expected_status:
        raise ValueError(f"unexpected result status: {result.get('status')}: {result.get('error_text')}")
    reward = result.get("total_reward")
    if type(reward) not in (int, float) or not math.isfinite(reward):
        raise ValueError("total_reward must be a finite number (0.0 is valid without evaluation)")
    if type(result.get("step_count")) is not int or result["step_count"] < 0:
        raise ValueError("step_count must be a nonnegative integer")
    if any(type(result.get(key)) is not bool for key in ("terminated", "truncated")):
        raise ValueError("terminated/truncated must be booleans")
    if not isinstance(result.get("metrics", {}), dict):
        raise ValueError("metrics must be an object")
    if result.get("error_text") is not None and not isinstance(result["error_text"], str):
        raise ValueError("error_text must be null or a string")
    if expected_status == "failed" and not result.get("error_text"):
        raise ValueError("controlled failure must include error_text")


def run_smoke(runner, request, *, response=None, timeout=30, input_mode="stdin",
              expected_status="succeeded", require_model_call=False, adapter=None):
    request = dict(request)
    session_id = request["session_id"]
    calls, errors = [], []
    response_body = response if response is not None else {
        "id": "contract-response", "object": "chat.completion", "created": 0,
        "model": request["model"],
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "fixture answer"},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            try:
                if self.path != f"/v1/sessions/{session_id}/chat/completions":
                    raise ValueError(f"unexpected model path: {self.path}")
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                if body.get("model") != request["model"]:
                    raise ValueError("model route differs from request.model")
                if body.get("stream"):
                    raise ValueError("streaming needs an environment-specific fixture")
                calls.append(body)
                payload, status = response_body, 200
            except Exception as exc:
                errors.append(str(exc))
                payload, status = {"error": str(exc)}, 400
            raw = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with tempfile.TemporaryDirectory(prefix="safactory-contract-") as temp:
            # An optional adapter copy lets a protocol test replace native
            # dependencies with an explicit fixture without changing the
            # production environment directory.
            runner_path = Path(runner).resolve()
            if adapter is not None:
                isolated = Path(temp) / "runner"
                isolated.mkdir()
                shutil.copy2(runner_path, isolated / "runner.py")
                shutil.copy2(Path(adapter).resolve(), isolated / "adapter.py")
                runner_path = isolated / "runner.py"
            root_url = f"http://127.0.0.1:{server.server_port}/v1/sessions"
            request["gateway_base_url"] = root_url
            request["storage_config"] = {"db_url": f"sqlite://{temp}/contract.db"}
            session_url = f"{root_url}/{session_id}"
            env = dict(os.environ)
            env.update({
                "SAFACTORY_SESSION_ID": session_id,
                "SAFACTORY_START_REQUEST_JSON": json.dumps(request) if input_mode == "env" else "",
                "SAFACTORY_GATEWAY_BASE_URL": root_url,
                "SAFACTORY_GATEWAY_SESSION_URL": session_url,
                "SAFACTORY_GATEWAY_SESSION_URL_CONTAINER": session_url,
                "SAFACTORY_ROUTE_MODEL": request["model"],
                "SAFACTORY_MODEL_REF": f"safactory/{request['model']}",
                "OPENROUTER_BASE_URL": session_url,
                "SAFACTORY_RESULT_PATH": f"{temp}/result.json",
                "NO_PROXY": "127.0.0.1,localhost", "no_proxy": "127.0.0.1,localhost",
                "PYTHONDONTWRITEBYTECODE": "1",
            })
            process = subprocess.Popen(
                [sys.executable, str(runner_path)],
                env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, start_new_session=True,
            )
            try:
                stdout, stderr = process.communicate(
                    json.dumps(request) if input_mode == "stdin" else "", timeout=timeout
                )
            finally:
                stop_process(process)
                for pipe in (process.stdin, process.stdout, process.stderr):
                    pipe.close()
            if stderr:
                print(stderr, file=sys.stderr, end="")
            if process.returncode:
                raise ValueError(f"runner process failed: exit {process.returncode}")
            # json.loads deliberately rejects extra stdout, even extra JSON objects.
            result = json.loads(stdout)
            validate_result(result, session_id, expected_status)
            artifact = Path(env["SAFACTORY_RESULT_PATH"])
            if artifact.exists() and json.loads(artifact.read_text()) != result:
                raise ValueError("stdout and result artifact differ")
            if errors:
                raise ValueError("; ".join(errors))
            if require_model_call and not calls:
                raise ValueError("runner did not use the mock session endpoint")
            return {"validation": "local-contract-only", "model_calls": len(calls), "result": result}
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, help="Optional fixture adapter copied beside the runner")
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--response", type=Path, help="Mock non-streaming chat response JSON")
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--input-mode", choices=["stdin", "env"], default="stdin")
    parser.add_argument("--expect-status", choices=["succeeded", "failed"], default="succeeded")
    parser.add_argument("--require-model-call", action="store_true")
    args = parser.parse_args()
    summary = run_smoke(
        args.runner, json.loads(args.request.read_text()),
        response=json.loads(args.response.read_text()) if args.response else None,
        timeout=args.timeout, input_mode=args.input_mode, expected_status=args.expect_status,
        require_model_call=args.require_model_call, adapter=args.adapter,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
