from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence

from args import parse_simulation_args
from log_setup import setup_launcher_logging
from manager.simulation_config import load_simulation_run_config
from manager.simulation_flow import SimulationFlow

log = logging.getLogger("launcher")


async def main(argv: Sequence[str] | None = None) -> int:
    args = parse_simulation_args(argv)
    log_session = setup_launcher_logging(
        log_dir=args.log_dir,
        run_name=args.run_name,
        console_level=args.console_log_level,
        file_level=args.file_log_level,
        max_bytes=args.log_max_bytes,
        backup_count=args.log_backup_count,
        debug_loggers=["sqlite_strategy", "yaml_aggregator"] if args.debug_log else None,
    )

    log.info("main log file: %s", log_session.main_log_path)
    log.info("log run directory: %s", log_session.run_dir)

    flow = SimulationFlow(load_simulation_run_config(args))
    try:
        summary = await flow.run()
        log.info(
            "RUN SUMMARY: status=%s episodes=%d succeeded=%d failed=%d",
            summary.status,
            summary.total_episodes,
            summary.succeeded_episodes,
            summary.failed_episodes,
        )
        return 0 if summary.status == "succeeded" and summary.failed_episodes == 0 else 2
    finally:
        await flow.shutdown()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
