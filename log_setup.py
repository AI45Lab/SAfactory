from __future__ import annotations

import logging
import os
import re
import shutil
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime


DEFAULT_INFO_LOGGERS = ("launcher", "manager", "evaluator")
DEFAULT_SUPPRESS_PREFIXES = (
    "core.llm",
    "httpx",
    "httpcore",
    "urllib3",
    "aiosqlite",
    "tortoise",
    "kubernetes",
    "brainpp",
)
DEPENDENCY_WARNING_LOGGERS = (
    "httpx",
    "httpcore",
    "urllib3",
    "core.llm",
    "core.llm.base",
    "aiosqlite",
    "tortoise",
    "tortoise.db_client",
    "kubernetes",
    "kubernetes.client",
    "kubernetes.client.rest",
    "brainpp",
    "brainpp.rjob",
)


@dataclass(frozen=True, slots=True)
class LauncherLogSession:
    run_id: str
    run_dir: str
    main_log_path: str


def _normalize_patterns(patterns: Sequence[str] | None) -> set[str]:
    normalized: set[str] = set()
    for pattern in patterns or []:
        value = str(pattern).strip()
        if value:
            normalized.add(value)
    return normalized


def _matches_logger(name: str, patterns: set[str]) -> bool:
    for pattern in patterns:
        if name == pattern or name.startswith(pattern + "."):
            return True
    return False


def _parse_level(level_name: str, default: int) -> int:
    level = getattr(logging, str(level_name).upper(), None)
    return level if isinstance(level, int) else default


def _build_formatter() -> logging.Formatter:
    return logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _sanitize_run_name(run_name: str | None) -> str:
    cleaned = (run_name or "").strip()
    if not cleaned:
        return ""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", cleaned)
    return cleaned.strip("._-")


class ConsoleFilter(logging.Filter):
    """
    Keep console output focused on the run's main control flow.

    Rules:
    1. Respect the configured console level.
    2. Allow WARNING and above when they meet the configured console level.
    3. Suppress noisy dependency prefixes below WARNING.
    4. Allow DEBUG only for explicit debug loggers.
    5. Allow INFO only for explicit info loggers.
    """

    def __init__(
        self,
        *,
        console_level: str = "INFO",
        info_loggers: Sequence[str] | None = None,
        debug_loggers: Sequence[str] | None = None,
        suppress_prefixes: Sequence[str] | None = None,
    ):
        super().__init__()
        self._console_level = _parse_level(console_level, logging.INFO)
        self._info_loggers = _normalize_patterns(DEFAULT_INFO_LOGGERS if info_loggers is None else info_loggers)
        self._debug_loggers = _normalize_patterns(debug_loggers)
        self._suppress_prefixes = _normalize_patterns(
            DEFAULT_SUPPRESS_PREFIXES if suppress_prefixes is None else suppress_prefixes
        )

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno >= logging.WARNING:
            return record.levelno >= self._console_level

        if _matches_logger(record.name, self._suppress_prefixes):
            return False

        if record.levelno < logging.INFO:
            return _matches_logger(record.name, self._debug_loggers)

        if record.levelno == logging.INFO:
            if record.levelno < self._console_level:
                return False
            return _matches_logger(record.name, self._info_loggers)

        return False


def build_run_id(run_name: str | None = None) -> str:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    prefix = _sanitize_run_name(run_name)
    return f"{prefix}-{stamp}" if prefix else stamp


def build_log_session(log_dir: str, run_name: str | None) -> LauncherLogSession:
    os.makedirs(log_dir, exist_ok=True)
    current_run_path = os.path.join(log_dir, ".current_run")
    run_dir = ""
    if os.path.isfile(current_run_path):
        with open(current_run_path, "r", encoding="utf-8") as f:
            run_dir = f.read().strip()

    if run_dir:
        run_dir = run_dir if os.path.isabs(run_dir) else os.path.join(log_dir, run_dir)
        run_id = os.path.basename(os.path.normpath(run_dir))
    else:
        run_id = build_run_id(run_name)
        run_dir = os.path.join(log_dir, run_id)
        with open(current_run_path, "w", encoding="utf-8") as f:
            f.write(run_dir + "\n")

    os.makedirs(run_dir, exist_ok=True)
    return LauncherLogSession(
        run_id=run_id,
        run_dir=run_dir,
        main_log_path=os.path.join(run_dir, "main.log"),
    )


def build_console_handler(
    *,
    console_level: str,
    info_loggers: Sequence[str] | None = None,
    debug_loggers: Sequence[str] | None = None,
    suppress_prefixes: Sequence[str] | None = None,
) -> logging.Handler:
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(_build_formatter())
    handler.addFilter(
        ConsoleFilter(
            console_level=console_level,
            info_loggers=info_loggers,
            debug_loggers=debug_loggers,
            suppress_prefixes=suppress_prefixes,
        )
    )
    return handler


def build_main_file_handler(*, session: LauncherLogSession, file_level: str) -> logging.Handler:
    handler = logging.FileHandler(session.main_log_path, encoding="utf-8")
    handler.setLevel(_parse_level(file_level, logging.DEBUG))
    handler.setFormatter(_build_formatter())
    return handler


def _reset_root_handlers(root: logging.Logger) -> None:
    for handler in list(root.handlers):
        root.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass


def _set_logger_levels(logger_names: Sequence[str], level: int) -> None:
    for logger_name in logger_names:
        logging.getLogger(logger_name).setLevel(level)


def cleanup_old_log_runs(log_dir: str, keep_runs: int = 20) -> None:
    if keep_runs <= 0 or not os.path.isdir(log_dir):
        return

    candidates = []
    for name in os.listdir(log_dir):
        path = os.path.join(log_dir, name)
        if not os.path.isdir(path):
            continue
        main_log = os.path.join(path, "main.log")
        if not os.path.isfile(main_log):
            continue
        candidates.append((os.path.getmtime(path), path))

    candidates.sort(reverse=True)
    for _, old_path in candidates[keep_runs:]:
        shutil.rmtree(old_path, ignore_errors=True)


def setup_launcher_logging(
    *,
    log_dir: str,
    run_name: str | None,
    console_level: str = "INFO",
    file_level: str = "DEBUG",
    backup_count: int = 0,
    info_loggers: Sequence[str] | None = None,
    debug_loggers: Sequence[str] | None = None,
    suppress_prefixes: Sequence[str] | None = None,
) -> LauncherLogSession:
    """
    Configure root logging for a single launcher run.

    Notes:
    - Logs are organized by run directory rather than file rotation.
    - `backup_count` is repurposed as "how many recent run directories to keep".
    """
    os.makedirs(log_dir, exist_ok=True)

    session = build_log_session(log_dir, run_name)
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    _reset_root_handlers(root)

    root.addHandler(
        build_console_handler(
            console_level=console_level,
            info_loggers=info_loggers,
            debug_loggers=debug_loggers,
            suppress_prefixes=suppress_prefixes,
        )
    )
    root.addHandler(build_main_file_handler(session=session, file_level=file_level))

    # Keep especially chatty dependency loggers at WARNING in both console and file logs.
    _set_logger_levels(DEPENDENCY_WARNING_LOGGERS, logging.WARNING)

    logging.captureWarnings(True)

    cleanup_old_log_runs(log_dir, keep_runs=int(backup_count or 0))

    root.debug(
        "logging initialized: console_level=%s file_level=%s run_dir=%s main_log=%s",
        console_level.upper(),
        file_level.upper(),
        session.run_dir,
        session.main_log_path,
    )

    return session
