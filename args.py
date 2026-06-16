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
    parser.add_argument(
        "--evaluation-config",
        type=str,
        default="",
        help="Optional evaluator runtime YAML for judge endpoints, evaluator pools, and default specs.",
    )
    parser.add_argument("--mode", choices=["docker", "rjob"], default="docker")

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
    parser.add_argument(
        "--cleanup-docker-container",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Whether to remove rollout Docker containers after evaluation or rollout finishes. "
            "Use --no-cleanup-docker-container to keep containers for debugging."
        ),
    )
    parser.add_argument("--rjob-cluster-entry", type=str, default="", help="RJob cluster entry URL")
    parser.add_argument("--rjob-namespace", type=str, default="", help="RJob namespace")
    parser.add_argument("--rjob-access-key", type=str, default="", help="RJob access key id")
    parser.add_argument("--rjob-secret-key", type=str, default="", help="RJob secret key")
    parser.add_argument(
        "--rjob-verifyssl",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether RJobClient verifies SSL certificates.",
    )
    parser.add_argument("--rjob-retries", type=int, default=3, help="RJobClient retry count")
    parser.add_argument("--rjob-poll-interval-s", type=float, default=5.0, help="RJob status poll interval")
    parser.add_argument(
        "--rjob-cleanup",
        dest="rjob_cleanup_on_finish",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to delete RJob jobs after rollout finishes.",
    )
    parser.add_argument(
        "--rjob-gateway-base-url",
        type=str,
        default="",
        help="Gateway base URL reachable from RJob containers. Defaults to --gateway-base-url.",
    )
    parser.add_argument("--rjob-name-prefix", type=str, default="safactory", help="RJob name prefix")
    parser.add_argument(
        "--rjob-no-packaging",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pass no_packaging to RJobClient.submit.",
    )
    parser.add_argument("--rjob-charged-group", type=str, default="", help="Default RJob charged group")
    parser.add_argument(
        "--rjob-auto-delete-duration",
        type=str,
        default="",
        help="Default RJob auto_delete_duration, for example 12h. Empty means RJob default.",
    )
    parser.add_argument(
        "--rjob-keep-failed-jobs",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Keep failed RJob jobs for debugging.",
    )
    parser.add_argument(
        "--rjob-submit-concurrency",
        type=int,
        default=0,
        help="Optional RJob submit concurrency limit. 0 means use worker concurrency.",
    )

    parser.add_argument("--gateway-base-url", type=str, default="http://127.0.0.1:8080/v1/sessions")
    parser.add_argument("--agent-start-timeout-s", type=float, default=3600.0)
    parser.add_argument("--agent-runtime", choices=["agent_start"], default="agent_start")
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--llm-model", type=str, default="default")
    parser.add_argument("--llm-temperature", type=float, default=0.3)
    parser.add_argument(
        "--evaluation-model",
        "--evaluation_model",
        dest="evaluation_model",
        type=str,
        default="",
        help=(
            "Optional gateway llm_routes key used for evaluation. When set, "
            "LLM judge requests and agent-eval model configuration use this model."
        ),
    )
    parser.add_argument("--rl-group-size", type=int, default=0)
    parser.add_argument("--rl-epoch", type=int, default=1)
    parser.add_argument(
        "--enable-evaluation",
        dest="evaluation_enabled",
        action="store_true",
        default=False,
        help=(
            "Enable evaluator flow after rollout. When disabled, rollout containers are released immediately "
            "after rollout according to --cleanup-docker-container."
        ),
    )
    parser.add_argument(
        "--eval-task-dir-name",
        type=str,
        default="eval_tasks",
        help="Directory name under each env config folder used for markdown evaluation tasks.",
    )
    parser.add_argument(
        "--strict-eval-tasks",
        action="store_true",
        default=False,
        help="Fail evaluation when the expected markdown eval task file is missing.",
    )

    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--console-log-level", type=str, default="INFO")
    parser.add_argument("--file-log-level", type=str, default="DEBUG")
    parser.add_argument("--log-max-bytes", type=int, default=50 * 1024 * 1024)
    parser.add_argument("--log-backup-count", type=int, default=20)
    parser.add_argument("--debug-log", action="store_true", default=False)
    return parser.parse_args(argv)
