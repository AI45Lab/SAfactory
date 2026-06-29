from __future__ import annotations

import json
import logging
import os
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterator


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _rounded_ms(value: float) -> float:
    return round(value, 3)


def _json_default(value: Any) -> str:
    return str(value)


@dataclass(slots=True)
class PerfMark:
    name: str
    timestamp_utc: str
    elapsed_ms: float
    delta_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        item: dict[str, Any] = {
            "name": self.name,
            "timestamp_utc": self.timestamp_utc,
            "elapsed_ms": self.elapsed_ms,
            "delta_ms": self.delta_ms,
        }
        if self.metadata:
            item["metadata"] = self.metadata
        return item


class PerfTrace:
    """
    Lightweight end-to-end timing timeline for logs.

    A trace stores wall-clock timestamps plus monotonic elapsed/delta values for
    key nodes. Emit one summary line and grep for "perf_trace summary".
    """

    def __init__(
        self,
        name: str,
        *,
        logger: logging.Logger | None = None,
        trace_id: str | None = None,
        context: dict[str, Any] | None = None,
        log_marks: bool | None = None,
    ) -> None:
        self.name = name
        self.trace_id = trace_id or uuid.uuid4().hex[:12]
        self.logger = logger or logging.getLogger("perf_trace")
        self.context: dict[str, Any] = dict(context or {})
        self.started_at_utc = _utc_timestamp()
        self._started_perf = time.perf_counter()
        self._last_perf = self._started_perf
        self._marks: list[PerfMark] = []
        self._summary_emitted = False
        self.log_marks = (
            os.environ.get("SAFACTORY_PERF_TRACE_MARKS", "").strip().lower()
            in {"1", "true", "yes", "on"}
            if log_marks is None
            else bool(log_marks)
        )
        self.mark("start")

    def update_context(self, **context: Any) -> None:
        self.context.update({key: value for key, value in context.items() if value is not None})

    def mark(self, name: str, **metadata: Any) -> None:
        now = time.perf_counter()
        mark = PerfMark(
            name=name,
            timestamp_utc=_utc_timestamp(),
            elapsed_ms=_rounded_ms((now - self._started_perf) * 1000),
            delta_ms=_rounded_ms((now - self._last_perf) * 1000),
            metadata={key: value for key, value in metadata.items() if value is not None},
        )
        self._last_perf = now
        self._marks.append(mark)
        if self.log_marks:
            payload = self._payload(status="mark", extra={"mark": mark.as_dict()})
            self.logger.info(
                "perf_trace mark: %s",
                json.dumps(payload, ensure_ascii=False, default=_json_default),
            )

    @contextmanager
    def span(self, name: str, **metadata: Any) -> Iterator[None]:
        self.mark(f"{name}.begin", **metadata)
        started = time.perf_counter()
        try:
            yield
        except Exception as exc:
            self.mark(
                f"{name}.error",
                duration_ms=_rounded_ms((time.perf_counter() - started) * 1000),
                error_type=type(exc).__name__,
                error=str(exc),
                **metadata,
            )
            raise
        else:
            self.mark(
                f"{name}.end",
                duration_ms=_rounded_ms((time.perf_counter() - started) * 1000),
                **metadata,
            )

    def emit_summary(self, *, status: str, force: bool = False, **extra: Any) -> None:
        if self._summary_emitted and not force:
            return
        self._summary_emitted = True
        self.logger.info(
            "perf_trace summary: %s",
            json.dumps(self._payload(status=status, extra=extra), ensure_ascii=False, default=_json_default),
        )

    def _payload(self, *, status: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "trace_id": self.trace_id,
            "name": self.name,
            "status": status,
            "started_at_utc": self.started_at_utc,
            "completed_at_utc": _utc_timestamp(),
            "total_ms": _rounded_ms((time.perf_counter() - self._started_perf) * 1000),
            "context": self.context,
            "marks": [mark.as_dict() for mark in self._marks],
        }
        if extra:
            payload.update({key: value for key, value in extra.items() if value is not None})
        return payload
