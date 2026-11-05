#!/usr/bin/env python3
"""Export delivered Mining manifests into one-case-per-row SAfactory JSONL.

This command reads local files only. It neither loads Docker images nor calls
models. It checks each whole partition because run_mining_400.sh does the same
even for a single selected case. Existing output files are never overwritten.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
import re
import sys
from typing import Any


def read_partition(batch_dir: Path, runtime_batch_dir: str) -> list[dict[str, Any]]:
    """Validate a delivered partition and preserve its selected_tasks order."""
    batch = batch_dir.expanduser().resolve()
    runtime_path = PurePosixPath(runtime_batch_dir)
    if not runtime_path.is_absolute() or ".." in runtime_path.parts:
        raise ValueError(f"runtime batch directory must be an absolute container path: {runtime_batch_dir}")
    raw_names = (batch / "selected_tasks.txt").read_text(encoding="utf-8").splitlines()
    names = [name.strip() for name in raw_names if name.strip()]
    if not names or len(set(names)) != len(names):
        raise ValueError(f"{batch}: selected_tasks.txt must be nonempty and unique")
    if raw_names != names:
        raise ValueError(f"{batch}: remove empty lines and surrounding whitespace from selected_tasks.txt")
    for name in names:
        if not re.fullmatch(r"arvo-[A-Za-z0-9][A-Za-z0-9_.-]*-lv0", name):
            raise ValueError(f"{batch}: invalid ARVO lv0 task name: {name!r}")
    index = json.loads((batch / "tasks.json").read_text(encoding="utf-8"))
    if not isinstance(index, dict) or type(index.get("task_count")) is not int:
        raise ValueError(f"{batch}: tasks.json must contain integer task_count")
    if index["task_count"] != len(names):
        raise ValueError(f"{batch}: task_count differs from selected_tasks.txt")
    actual = {p.name for p in batch.iterdir() if (p / "harbor-runtime-profile/task.toml").is_file()}
    if actual != set(names):
        raise ValueError(f"{batch}: profile set differs; missing={sorted(set(names)-actual)}, extra={sorted(actual-set(names))}")
    if not (batch / "bin/load_image.py").is_file():
        raise ValueError(f"{batch}: missing bin/load_image.py")
    if not (batch / "shared-images").is_dir():
        raise ValueError(f"{batch}: missing shared-images directory")
    rows = []
    for name in names:
        case = batch / name
        if not any(p.is_file() for p in (case / "images").glob("*.tar*")):
            raise ValueError(f"{case}: missing images/*.tar* archive")
        lock = json.loads((case / "image-lock.json").read_text(encoding="utf-8"))
        if not isinstance(lock, dict) or not isinstance(lock.get("images"), list):
            raise ValueError(f"{case}: image-lock.json must contain an images array")
        rows.append({"task_id": name, "task_name": name, "batch_dir": str(runtime_path)})
    return rows


def collect_rows(
    batch_dirs: list[Path], runtime_batch_dirs: list[str] | None = None,
    tasks: list[str] | None = None, limit: int | None = None,
) -> list[dict[str, Any]]:
    if runtime_batch_dirs is not None and len(runtime_batch_dirs) != len(batch_dirs):
        raise ValueError("--runtime-batch-dir must be supplied once per --batch-dir, in the same order")
    if limit is not None and limit < 1:
        raise ValueError("--limit must be positive")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for i, batch in enumerate(batch_dirs):
        runtime = runtime_batch_dirs[i] if runtime_batch_dirs is not None else str(batch.expanduser().resolve())
        for row in read_partition(batch, runtime):
            if row["task_id"] in seen:
                raise ValueError(f"duplicate task across partitions: {row['task_id']}")
            seen.add(row["task_id"])
            rows.append(row)
    if tasks:
        wanted = set(tasks)
        missing = wanted - seen
        if missing:
            raise ValueError(f"requested task not present in supplied manifests: {sorted(missing)}")
        rows = [row for row in rows if row["task_name"] in wanted]
    if limit is not None:
        rows = rows[:limit]
    if not rows:
        raise ValueError("no tasks selected")
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-dir", type=Path, action="append", required=True,
                        help="local delivered partition; repeat to merge partitions")
    parser.add_argument("--runtime-batch-dir", action="append",
                        help="container path, once per --batch-dir; defaults to the same absolute path")
    parser.add_argument("--task", action="append", help="select only this manifest task; repeat for multiple cases")
    parser.add_argument("--limit", type=int, help="emit the first N selected rows after validating all partitions")
    parser.add_argument("--output", type=Path, required=True, help="new JSONL path; existing files are refused")
    args = parser.parse_args(argv)
    try:
        if args.output.exists():
            raise ValueError(f"output already exists: {args.output}; choose a new path")
        rows = collect_rows(args.batch_dir, args.runtime_batch_dir, args.task, args.limit)
        content = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(content)
    except (OSError, ValueError) as exc:
        print(f"build_dataset: {exc}", file=sys.stderr)
        return 2
    print(f"Wrote {len(rows)} case rows to {args.output}; no Docker/model calls performed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
