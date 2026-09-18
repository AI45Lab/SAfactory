#!/usr/bin/env python3
"""Own a Gateway for one bounded Launcher run; no second terminal required."""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, build_opener

REPO_ROOT = Path(__file__).resolve().parents[3]


def stop_process(process):
    if process is None:
        return
    # Each child owns a new process group, including any descendants it starts.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass
    finally:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def run_live(gateway_command, launcher_command, ready_url, log_path, *,
             startup_timeout=60, run_timeout=600):
    address = urlsplit(ready_url)
    try:
        with socket.create_connection((address.hostname, address.port or 80), timeout=1):
            raise ValueError("Gateway port already occupied; verify that Gateway and use launcher.py directly")
    except (ConnectionRefusedError, TimeoutError):
        pass
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    gateway = launcher = None
    opener = build_opener(ProxyHandler({}))
    env = dict(os.environ, SAFACTORY_GATEWAY_LOG_PATH=str(log_path.resolve()) + ".events")
    try:
        with log_path.open("w", encoding="utf-8") as log:
            gateway = subprocess.Popen(gateway_command, stdout=log, stderr=subprocess.STDOUT,
                                       env=env, start_new_session=True)
            deadline = time.monotonic() + startup_timeout
            while True:
                if gateway.poll() is not None:
                    raise RuntimeError(f"Gateway exited before readiness; see {log_path}")
                try:
                    with opener.open(ready_url, timeout=1) as response:
                        if response.status == 200 and json.load(response).get("status") == "ready":
                            break
                except (OSError, ValueError):
                    pass
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Gateway readiness timed out; see {log_path}")
                time.sleep(0.1)
            launcher = subprocess.Popen(launcher_command, start_new_session=True)
            deadline = time.monotonic() + run_timeout
            while launcher.poll() is None:
                if gateway.poll() is not None:
                    raise RuntimeError(f"Gateway exited during Launcher run; see {log_path}")
                if time.monotonic() >= deadline:
                    raise TimeoutError("Launcher smoke run timed out")
                time.sleep(0.1)
            return launcher.returncode
    finally:
        stop_process(launcher)
        stop_process(gateway)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gateway-config", required=True)
    parser.add_argument("--gateway-log", default="logs/smoke/gateway.log")
    parser.add_argument("--startup-timeout", type=float, default=60)
    parser.add_argument("--run-timeout", type=float, default=600)
    parser.add_argument("launcher_args", nargs=argparse.REMAINDER, help="-- followed by launcher.py flags")
    options = parser.parse_args()
    argv = options.launcher_args
    if argv[:1] == ["--"]:
        argv = argv[1:]
    if not argv or options.startup_timeout <= 0 or options.run_timeout <= 0:
        parser.error("provide Launcher flags after -- and positive timeouts")
    if Path.cwd().resolve() != REPO_ROOT:
        parser.error("run this helper from the SAfactory repository root")
    sys.path.insert(0, str(REPO_ROOT))
    from args import parse_simulation_args
    from gateway.config import load_gateway_config

    cfg = load_gateway_config(options.gateway_config)
    args = parse_simulation_args(argv)
    if args.llm_model not in cfg.llm_routes:
        parser.error("--llm-model must match a Gateway llm_routes key")
    if args.storage_type != cfg.storage_type:
        parser.error("Launcher and Gateway storage_type must match")
    if cfg.storage_type == "sqlite":
        db_url = cfg.storage_config["db_url"]
        if args.db_path and args.db_path != db_url:
            parser.error("Launcher --db-path must match Gateway storage_config.db_url")
        if not args.db_path:
            argv += ["--db-path", db_url]
    host = cfg.listen_host
    if host in {"0.0.0.0", "::"}:
        host = "127.0.0.1" if host == "0.0.0.0" else "::1"
    local_host = f"[{host}]" if ":" in host else host
    local_root = f"http://{local_host}:{cfg.listen_port}"
    explicit_gateway_url = any(
        value == "--gateway-base-url" or value.startswith("--gateway-base-url=") for value in argv
    )
    if not explicit_gateway_url:
        if args.mode == "rjob":
            parser.error("RJob requires explicit --gateway-base-url reachable from the cluster")
        argv += ["--gateway-base-url", local_root + cfg.base_session_path]
    else:
        address = urlsplit(args.gateway_base_url)
        if (address.port or 80) != cfg.listen_port or address.path.rstrip('/') != cfg.base_session_path.rstrip('/'):
            parser.error("Gateway URL port and session path must match the owned Gateway")
        if args.mode == "rjob" and address.hostname in {"localhost", "127.0.0.1", "::1"}:
            parser.error("RJob Gateway URL must be reachable from the cluster")
    # SIGINT already raises KeyboardInterrupt; make SIGTERM run the same cleanup path.
    def interrupted(signum, frame):
        raise KeyboardInterrupt

    previous_handler = signal.signal(signal.SIGTERM, interrupted)
    try:
        return run_live(
            [sys.executable, "-m", "gateway", "--config", options.gateway_config],
            [sys.executable, "launcher.py", *argv], local_root + "/readyz", options.gateway_log,
            startup_timeout=options.startup_timeout, run_timeout=options.run_timeout,
        )
    finally:
        signal.signal(signal.SIGTERM, previous_handler)


if __name__ == "__main__":
    raise SystemExit(main())
