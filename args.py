from __future__ import annotations

import argparse
from collections.abc import Sequence


def parse_simulation_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Safactory task-level simulation launcher",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--job-id", type=str, default="", help="Simulation workflow id")
    parser.add_argument("--exp-config", type=str, default="./core/exp/config.yaml")
    parser.add_argument("--mode", choices=["docker"], default="docker")

    parser.add_argument("--agent-config", type=str, default=None, help="Single agent YAML config")
    parser.add_argument("--agent-start-config", type=str, default=None, help="Agent container startup YAML config")
    parser.add_argument("--agent-root", type=str, default="env", help="Directory scanned for agent YAML configs")
    parser.add_argument("--storage-type", type=str, default="sqlite", choices=["sqlite", "cloud"])
    parser.add_argument("--db-path", type=str, default="sqlite://env_trajs.db")
    parser.add_argument("--rebuild-table", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--disable-buffer", dest="enable_buffer", action="store_false", default=True)
    parser.add_argument("--buffer-size", type=int, default=100)
    parser.add_argument("--flush-interval", type=float, default=5.0)

    parser.add_argument("--pool-size", type=int, default=1, help="Base Docker lease pool size")
    parser.add_argument("--multiplier", type=float, default=1.2, help="Warm-pool multiplier")
    parser.add_argument("--max-workers", type=int, default=0, help="0 means use warm-pool size")
    parser.add_argument("--docker-bin", type=str, default="docker", help="Docker executable")
    parser.add_argument("--docker-pull-policy", type=str, default="never", choices=["never", "always"])
    parser.add_argument("--docker-startup-concurrency", type=int, default=8)

    parser.add_argument("--gateway-base-url", type=str, default="http://127.0.0.1:8080/v1/sessions")
    parser.add_argument("--agent-start-timeout-s", type=float, default=3600.0)
    parser.add_argument("--agent-runtime", choices=["agent_start"], default="agent_start")
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--llm-model", type=str, default="default")
    parser.add_argument("--llm-temperature", type=float, default=0.3)
    parser.add_argument("--rl-group-size", type=int, default=0)
    parser.add_argument("--rl-epoch", type=int, default=1)
    parser.add_argument(
        "--enable-evaluation",
        dest="evaluation_enabled",
        action="store_true",
        default=False,
        help="Enable evaluator flow after rollout. Disabled means rollout containers are always removed after use.",
    )

    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--console-log-level", type=str, default="INFO")
    parser.add_argument("--file-log-level", type=str, default="DEBUG")
    parser.add_argument("--log-max-bytes", type=int, default=50 * 1024 * 1024)
    parser.add_argument("--log-backup-count", type=int, default=20)
    parser.add_argument("--debug-log", action="store_true", default=False)
    return parser.parse_args(argv)
