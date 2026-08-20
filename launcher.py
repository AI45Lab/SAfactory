from __future__ import annotations

import asyncio
import logging
import signal
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
        backup_count=args.log_backup_count,
        debug_loggers=["sqlite_strategy", "yaml_aggregator"] if args.debug_log else None,
    )

    log.debug("main log file: %s", log_session.main_log_path)
    log.debug("log run directory: %s", log_session.run_dir)

    cfg = load_simulation_run_config(args)
    log.info("JOB INITIALIZED | job_id=\033[1;96m%s\033[0m", cfg.job_id)
    flow = SimulationFlow(cfg)
    stop_event = asyncio.Event()
    _install_signal_handlers(stop_event)
    run_task: asyncio.Task | None = None
    stop_task: asyncio.Task | None = None
    try:
        run_task = asyncio.create_task(flow.run(), name="simulation-flow-run")
        stop_task = asyncio.create_task(stop_event.wait(), name="launcher-stop-signal")
        done, _pending = await asyncio.wait({run_task, stop_task}, return_when=asyncio.FIRST_COMPLETED)
        if stop_task in done and not run_task.done():
            log.warning("launcher received shutdown signal; cancelling simulation flow")
            run_task.cancel()
            try:
                await run_task
            except asyncio.CancelledError:
                pass
            return 130

        summary = await run_task
        log.info(
            "RUN SUMMARY: status=%s episodes=%d succeeded=%d failed=%d",
            summary.status,
            summary.total_episodes,
            summary.succeeded_episodes,
            summary.failed_episodes,
        )
        return 0 if summary.status == "succeeded" and summary.failed_episodes == 0 else 2
    finally:
        if stop_task is not None:
            stop_task.cancel()
            await asyncio.gather(stop_task, return_exceptions=True)
        await _shutdown_with_timeout(flow, timeout_s=cfg.shutdown_timeout_s)


def _install_signal_handlers(stop_event: asyncio.Event) -> None:
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop_event.set)
        except (NotImplementedError, RuntimeError):
            log.debug("signal handler unavailable for %s", sig, exc_info=True)


async def _shutdown_with_timeout(flow: SimulationFlow, *, timeout_s: float) -> None:
    timeout = max(1.0, float(timeout_s or 120.0))
    try:
        await asyncio.wait_for(flow.shutdown(), timeout=timeout)
    except asyncio.TimeoutError:
        log.error("simulation shutdown timed out after %.1fs; some runtime resources may remain", timeout)


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
