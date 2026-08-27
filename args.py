from __future__ import annotations

import argparse
from collections.abc import Sequence

DEFAULT_RJOB_CONFIG_PATH = "config.yaml"
DEFAULT_SANDBOX_CONFIG_PATH = "config.yaml"


def parse_simulation_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Safactory task-level simulation launcher",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--job-id", type=str, default="", help="Simulation workflow id")
    parser.add_argument("--mode", choices=["docker", "rjob", "sandbox"], default="docker")
    parser.add_argument(
        "--rjob-config",
        type=str,
        default=DEFAULT_RJOB_CONFIG_PATH,
        help="YAML file for global RJob connection/auth settings",
    )
    parser.add_argument(
        "--sandbox-config",
        type=str,
        default=DEFAULT_SANDBOX_CONFIG_PATH,
        help="YAML file for global OpenSandbox/Brainbox connection settings",
    )

    parser.add_argument("--agent-config", type=str, default=None, help="Single agent YAML config")
    parser.add_argument("--agent-start-config", type=str, default=None, help="Agent container startup YAML config")
    parser.add_argument("--agent-root", type=str, default="env", help="Directory scanned for agent YAML configs")
    parser.add_argument("--storage-type", type=str, default="sqlite", choices=["sqlite", "cloud"])
    parser.add_argument(
        "--db-path",
        type=str,
        default=None,
        help="SQLite storage DB URI. Cloud storage ignores this and uses wt-data-gateway defaults.",
    )
    parser.add_argument(
        "--rebuild-table",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Delete this job's environment and landing rows before rebuilding. "
            "Cloud mode requires exact deletion confirmation and explicit "
            "production acknowledgement."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Continue a job and skip finished environments. Before continuing, "
            "all landing rows for unfinished sessions are deleted; Cloud mode "
            "requires the same destructive-operation safeguards as rebuild."
        ),
    )
    parser.add_argument(
        "--confirm-cloud-delete-job-id",
        default="",
        help=(
            "Exact job_id confirmation required before Cloud resume/rebuild may "
            "delete landing rows"
        ),
    )
    parser.add_argument(
        "--confirm-production",
        action="store_true",
        help=(
            "Acknowledge that the resolved Cloud landing target is production."
        ),
    )
    parser.add_argument("--disable-buffer", dest="enable_buffer", action="store_false", default=True)
    parser.add_argument("--buffer-size", type=int, default=100)
    parser.add_argument("--flush-interval", type=float, default=5.0)

    parser.add_argument("--pool-size", type=int, default=1, help="Base Docker lease pool size")
    parser.add_argument("--multiplier", type=float, default=1.2, help="Warm-pool multiplier")
    parser.add_argument("--max-workers", type=int, default=0, help="0 means use warm-pool size")
    parser.add_argument("--docker-bin", type=str, default="docker", help="Docker executable")
    parser.add_argument("--docker-pull-policy", type=str, default="never", choices=["never", "always"])
    parser.add_argument(
        "--docker-image-archive-dir",
        type=str,
        default="",
        help="Directory containing Docker image archives named <repository>-<tag>.tar[.gz]",
    )
    parser.add_argument(
        "--cleanup-docker-image",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Remove images loaded from --docker-image-archive-dir after their containers finish",
    )
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
    parser.add_argument(
        "--cleanup-stale-docker-containers",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Remove stale Safactory Docker containers from the same job_id before startup.",
    )
    parser.add_argument("--gateway-base-url", type=str, default="http://127.0.0.1:8080/v1/sessions")
    parser.add_argument("--agent-start-timeout-s", type=float, default=600.0)
    parser.add_argument(
        "--agent-start-timeout-grace-s",
        type=float,
        default=120.0,
        help="Extra outer runner timeout budget after the in-container agent timeout.",
    )
    parser.add_argument(
        "--container-refill-timeout-s",
        type=float,
        default=300.0,
        help="Maximum time to release one finished runtime resource and prepare its replacement.",
    )
    parser.add_argument(
        "--row-wait-timeout-s",
        type=float,
        default=60.0,
        help="Maximum time to wait for newly produced DB rows while refilling the runtime pool.",
    )
    parser.add_argument(
        "--row-fetch-timeout-s",
        type=float,
        default=30.0,
        help="Maximum time for one scheduler DB fetch before stopping new row scheduling.",
    )
    parser.add_argument(
        "--gateway-close-timeout-s",
        type=float,
        default=15.0,
        help="HTTP timeout for gateway session close requests.",
    )
    parser.add_argument(
        "--gateway-close-retries",
        type=int,
        default=1,
        help="Retry count for gateway session close requests.",
    )
    parser.add_argument(
        "--gateway-close-retry-backoff-s",
        type=float,
        default=1.0,
        help="Backoff between gateway session close retries.",
    )
    parser.add_argument(
        "--shutdown-timeout-s",
        type=float,
        default=120.0,
        help="Maximum time allowed for launcher shutdown and container cleanup.",
    )
    parser.add_argument(
        "--docker-command-timeout-s",
        type=float,
        default=300.0,
        help="Default timeout for Docker lifecycle commands issued by the manager.",
    )
    parser.add_argument(
        "--docker-start-timeout-s",
        type=float,
        default=300.0,
        help="Timeout for Docker run/copy commands while starting rollout containers.",
    )
    parser.add_argument(
        "--docker-remove-timeout-s",
        type=float,
        default=120.0,
        help="Timeout for Docker rm commands while releasing rollout containers.",
    )
    parser.add_argument(
        "--docker-stop-timeout-s",
        type=float,
        default=10.0,
        help="Grace period passed to docker stop before forced removal.",
    )
    parser.add_argument(
        "--docker-inspect-timeout-s",
        type=float,
        default=10.0,
        help="Timeout for Docker inspect checks during container cleanup.",
    )
    parser.add_argument(
        "--docker-remove-retries",
        type=int,
        default=3,
        help="Number of stop/inspect/rm attempts when releasing rollout containers.",
    )
    parser.add_argument(
        "--docker-remove-retry-delay-s",
        type=float,
        default=2.0,
        help="Delay between Docker container removal retry attempts.",
    )
    parser.add_argument(
        "--docker-lifecycle-timeout-s",
        type=float,
        default=60.0,
        help="Timeout for optional per-container cleanup and healthcheck commands.",
    )
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
        help=(
            "Enable evaluator flow after rollout. When disabled, rollout containers are released immediately "
            "after rollout according to --cleanup-docker-container."
        ),
    )
    parser.add_argument(
        "--circuit-breaker",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stop scheduling new episodes when recent rollout failures/timeouts exceed configured thresholds.",
    )
    parser.add_argument("--circuit-breaker-window", type=int, default=50)
    parser.add_argument("--circuit-breaker-min-samples", type=int, default=20)
    parser.add_argument("--circuit-breaker-failure-rate", type=float, default=0.8)
    parser.add_argument("--circuit-breaker-timeout-rate", type=float, default=0.5)
    parser.add_argument("--circuit-breaker-consecutive-timeouts", type=int, default=5)

    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--console-log-level", type=str, default="INFO")
    parser.add_argument("--file-log-level", type=str, default="DEBUG")
    parser.add_argument("--log-backup-count", type=int, default=20)
    parser.add_argument("--debug-log", action="store_true", default=False)
    return parser.parse_args(argv)
