#!/usr/bin/env python3
"""Materialize benchmark bundles into one runnable Harbor task."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


ARVO_AGENT_TIMEOUT_SEC = 7200.0


@dataclass(frozen=True)
class BundleSpec:
    kind: str
    package_dir: Path
    task: str
    variant: str | None = None
    level: str | None = None
    validation: str = "full"


def _param(
    dataset: Mapping[str, Any], params: Mapping[str, Any], key: str, default: Any = None
) -> Any:
    value = dataset.get(key)
    return value if value is not None else params.get(key, default)


def parse_bundle_spec(
    dataset: Mapping[str, Any], params: Mapping[str, Any]
) -> BundleSpec | None:
    kind = str(_param(dataset, params, "bundle_type", "") or "").strip()
    task = str(_param(dataset, params, "bundle_task", "") or "").strip()
    variant = str(_param(dataset, params, "bundle_variant", "") or "").strip()
    level = str(_param(dataset, params, "bundle_level", "") or "").strip()
    package_dir_text = str(
        _param(dataset, params, "bundle_package_dir", "") or ""
    ).strip()
    validation = str(params.get("bundle_validation") or "full").strip()
    if not package_dir_text:
        if any((kind, task, variant, level)):
            raise RuntimeError(
                "bundle_package_dir is required when bundle parameters are set"
            )
        return None

    if not kind:
        raise RuntimeError("bundle_type is required when bundle_package_dir is set")
    if not task:
        raise RuntimeError("bundle_task is required when bundle_package_dir is set")

    package_dir = Path(package_dir_text).resolve()
    if not package_dir.is_dir():
        raise RuntimeError(f"Bundle package directory does not exist: {package_dir}")
    if kind == "vulhub":
        if variant not in {"zero-day", "one-day"}:
            raise RuntimeError("Vulhub bundle_variant must be zero-day or one-day")
        if validation != "full":
            raise RuntimeError("bundle_validation is only configurable for ARVO")
    elif kind == "arvo":
        if level not in {"level0", "level1", "level2", "level3"}:
            raise RuntimeError(
                "ARVO bundle_level must be level0, level1, level2, or level3"
            )
        if validation not in {"full", "selected"}:
            raise RuntimeError("ARVO bundle_validation must be full or selected")
    else:
        raise RuntimeError(f"Unsupported bundle_type: {kind!r}")

    return BundleSpec(
        kind=kind,
        package_dir=package_dir,
        task=task,
        variant=variant or None,
        level=level or None,
        validation=validation,
    )


def materialize_bundle(
    spec: BundleSpec,
    *,
    output_dir: Path,
    env: Mapping[str, str],
) -> Path:
    if spec.kind == "vulhub":
        tool = spec.package_dir / "bin" / "vulhub_task.py"
        command = [
            sys.executable,
            str(tool),
            "--package-dir",
            str(spec.package_dir),
            "materialize",
            "--output-dir",
            str(output_dir),
            "--task",
            spec.task,
            "--variant",
            str(spec.variant),
            "--load-images",
        ]
    elif spec.kind == "arvo" and spec.validation == "selected":
        _materialize_selected_arvo_task(spec, output_dir=output_dir)
        command = None
    elif spec.kind == "arvo":
        tool = spec.package_dir / "bin" / "arvo_task.py"
        command = [
            sys.executable,
            str(tool),
            "materialize",
            "--package-dir",
            str(spec.package_dir),
            "--output-dir",
            str(output_dir),
            "--task",
            spec.task,
            "--level",
            str(spec.level),
            "--no-load-images",
        ]
    else:  # BundleSpec construction guards this; keep the execution boundary safe.
        raise RuntimeError(f"Unsupported bundle_type: {spec.kind!r}")

    if command is not None:
        if not tool.is_file():
            raise RuntimeError(f"Bundle materializer does not exist: {tool}")
        subprocess.run(command, env=dict(env), check=True)

    task_dirs = (
        sorted(path for path in output_dir.iterdir() if (path / "task.toml").is_file())
        if output_dir.is_dir()
        else []
    )
    if len(task_dirs) != 1:
        raise RuntimeError(
            f"Bundle materializer must produce exactly one Harbor task, got "
            f"{len(task_dirs)} in {output_dir}"
    )
    if spec.kind == "arvo":
        _set_arvo_agent_timeout(task_dirs[0] / "task.toml")
        _load_arvo_images(spec, env=env)
    return task_dirs[0]


def _bundle_path(package_dir: Path, relative: str) -> Path:
    root = package_dir.resolve()
    path = (root / relative).resolve()
    if path != root and root not in path.parents:
        raise RuntimeError(f"Bundle path escapes package root: {relative!r}")
    return path


def _link_or_copy(source: str, destination: str) -> str:
    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)
    return destination


def _replace_materialized_file(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.materialize.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.chmod(path.stat().st_mode)
    temporary.replace(path)


def _set_arvo_agent_timeout(task_toml: Path) -> None:
    """Set the copied ARVO task timeout without mutating a hard-linked bundle file."""
    content = task_toml.read_text(encoding="utf-8")
    section_pattern = re.compile(
        r"^(?P<header>[ \t]*\[agent\][ \t]*(?:#.*)?\r?\n)"
        r"(?P<body>.*?)"
        r"(?=^[ \t]*\[\[?[^\r\n]+\][ \t]*(?:#.*)?\r?$|\Z)",
        re.MULTILINE | re.DOTALL,
    )
    timeout_pattern = re.compile(
        r"^(?P<prefix>[ \t]*timeout_sec[ \t]*=[ \t]*)"
        r"(?P<value>[^#\r\n]+?)"
        r"(?P<suffix>[ \t]*(?:#.*)?)(?P<newline>\r?\n|\Z)",
        re.MULTILINE,
    )

    def replace_section(match: re.Match[str]) -> str:
        body, count = timeout_pattern.subn(
            rf"\g<prefix>{ARVO_AGENT_TIMEOUT_SEC:.1f}\g<suffix>\g<newline>",
            match.group("body"),
        )
        if count != 1:
            raise RuntimeError(
                f"expected exactly one [agent].timeout_sec in {task_toml}, got {count}"
            )
        return match.group("header") + body

    updated, section_count = section_pattern.subn(replace_section, content)
    if section_count != 1:
        raise RuntimeError(
            f"expected exactly one [agent] section in {task_toml}, got {section_count}"
        )
    if updated != content:
        _replace_materialized_file(task_toml, updated)


def _materialize_selected_arvo_task(spec: BundleSpec, *, output_dir: Path) -> None:
    """Copy one manifest-selected prebuilt task without full-bundle hash validation."""
    manifest = json.loads((spec.package_dir / "manifest.json").read_text())
    if manifest.get("delivery_format") != "arvo-task-bundle":
        raise RuntimeError(f"Not an ARVO Task Bundle: {spec.package_dir}")
    task_row = next(
        (
            row
            for row in manifest.get("tasks") or []
            if str(row.get("task_id") or "") == spec.task
        ),
        None,
    )
    if task_row is None:
        raise RuntimeError(f"ARVO task is not in bundle manifest: {spec.task}")
    if spec.level not in (task_row.get("levels") or []):
        raise RuntimeError(f"ARVO task {spec.task} does not provide {spec.level}")

    profile_id = str(manifest.get("default_runtime_profile") or "")
    profile_row = next(
        (
            row
            for row in manifest.get("runtime_profiles") or []
            if str(row.get("profile_id") or "") == profile_id
        ),
        None,
    )
    if profile_row is None:
        raise RuntimeError(f"ARVO default runtime profile is missing: {profile_id!r}")
    profile_root = _bundle_path(
        spec.package_dir, str(profile_row.get("relative_path") or "")
    )
    profile = json.loads((profile_root / "profile.json").read_text())
    adapter = profile.get("adapter") or {}
    if adapter.get("kind") != "prebuilt-harbor-tasks":
        raise RuntimeError("selected ARVO validation requires prebuilt-harbor-tasks")

    selected = [
        row
        for row in manifest.get("profile_tasks") or []
        if str(row.get("task_id") or "") == spec.task
        and str(row.get("level") or "") == spec.level
    ]
    if len(selected) != 1:
        raise RuntimeError(
            f"expected exactly one ARVO profile task for {spec.task}/{spec.level}, "
            f"got {len(selected)}"
        )
    row = selected[0]
    source = _bundle_path(spec.package_dir, str(row.get("relative_path") or ""))
    if not (source / "task.toml").is_file():
        raise RuntimeError(f"ARVO profile task is missing task.toml: {source}")
    name = str(row.get("name") or source.name)
    if not name or Path(name).name != name:
        raise RuntimeError(f"Invalid ARVO materialized task name: {name!r}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"ARVO materialization output is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / name
    shutil.copytree(
        source,
        destination,
        symlinks=False,
        copy_function=_link_or_copy,
    )
    for relative in ("task.toml", "environment/docker-compose.yaml"):
        path = destination / relative
        if not path.is_file():
            continue
        content = path.read_text(encoding="utf-8")
        if "__ARVO_TASK_ROOT__" in content:
            _replace_materialized_file(
                path,
                content.replace("__ARVO_TASK_ROOT__", str(destination.resolve())),
            )


def _load_arvo_images(spec: BundleSpec, *, env: Mapping[str, str]) -> None:
    manifest = json.loads((spec.package_dir / "manifest.json").read_text())
    profile_id = str(manifest["default_runtime_profile"])
    profile_row = next(
        row
        for row in manifest["runtime_profiles"]
        if str(row["profile_id"]) == profile_id
    )
    profile_root = spec.package_dir / str(profile_row["relative_path"])
    profile = json.loads((profile_root / "profile.json").read_text())
    image_lock = json.loads(
        (profile_root / str(profile.get("image_lock") or "image-lock.json")).read_text()
    )
    rows = [
        row
        for row in image_lock["images"]
        if row.get("scope") != "task" or str(row.get("task_id")) == spec.task
    ]
    if not rows or any(not str(row.get("ref") or "") for row in rows):
        raise RuntimeError(f"ARVO image lock is incomplete for task: {spec.task}")
    for row in rows:
        archive = _bundle_path(spec.package_dir, str(row["archive"]))
        if not archive.is_file():
            raise RuntimeError(f"ARVO image archive does not exist: {archive}")
        expected_size = row.get("archive_size")
        if expected_size is not None and archive.stat().st_size != int(expected_size):
            raise RuntimeError(
                f"ARVO image archive size mismatch: {archive}; "
                f"expected={expected_size} actual={archive.stat().st_size}"
            )
    archives = sorted(
        {_bundle_path(spec.package_dir, str(row["archive"])) for row in rows}
    )
    for archive in archives:
        subprocess.run(
            ["docker", "load", "--input", str(archive)],
            env=dict(env),
            check=True,
        )
    subprocess.run(
        [
            "docker",
            "image",
            "inspect",
            *sorted({str(row["ref"]) for row in rows}),
        ],
        env=dict(env),
        stdout=subprocess.DEVNULL,
        check=True,
    )
