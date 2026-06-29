from __future__ import annotations

import argparse
import logging
import os
import sys
from collections.abc import Sequence
from pathlib import Path

import uvicorn

from gateway.app import create_app
from gateway.config import load_gateway_config


DEPENDENCY_WARNING_LOGGERS = (
    "aiohttp",
    "aiosqlite",
    "asyncio",
    "httpcore",
    "httpx",
    "tortoise",
    "urllib3",
)


def _build_formatter() -> logging.Formatter:
    return logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the AIEvo API Gateway.")
    parser.add_argument(
        "--config",
        required=True,
        help=(
            "Path to the gateway YAML/JSON config file. All runtime settings "
            "such as listen address, storage, telemetry, and LLM routes are "
            "loaded from this file."
        ),
    )
    return parser.parse_args(argv)


def _setup_gateway_logging() -> str:
    log_path = os.environ.get("SAFACTORY_GATEWAY_LOG_PATH", "logs/gateway.log")
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    for handler in list(root.handlers):
        root.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass

    formatter = _build_formatter()
    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    root.addHandler(console_handler)
    root.addHandler(file_handler)

    for logger_name in DEPENDENCY_WARNING_LOGGERS:
        logging.getLogger(logger_name).setLevel(logging.WARNING)

    for logger_name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logger = logging.getLogger(logger_name)
        logger.handlers.clear()
        logger.propagate = True
        logger.setLevel(logging.INFO)

    logging.captureWarnings(True)
    logging.getLogger("gateway.__main__").info("gateway logging initialized: log_file=%s", path)
    return str(path)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    _setup_gateway_logging()
    cfg = load_gateway_config(args.config)
    uvicorn.run(
        create_app(cfg),
        host=cfg.listen_host,
        port=cfg.listen_port,
        reload=False,
        log_config=None,
    )


if __name__ == "__main__":
    main()
