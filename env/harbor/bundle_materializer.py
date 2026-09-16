#!/usr/bin/env python3
"""Materialize benchmark bundles into one runnable Harbor task."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


ARVO_AGENT_TIMEOUT_SEC = 7200.0
CVEFACTORY_BATCH_BUNDLE_TYPE = "cvefactory-harbor-batch"


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
    elif kind in {"cvedockergen", "vulhub-local", CVEFACTORY_BATCH_BUNDLE_TYPE}:
        _task_name(task)
        if variant or level:
            raise RuntimeError(f"{kind} bundles do not accept variant or level overrides")
        if validation != "full":
            raise RuntimeError("bundle_validation is only configurable for ARVO")
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
    if spec.kind == CVEFACTORY_BATCH_BUNDLE_TYPE:
        return _load_cvefactory_batch(spec, env=env)
    if spec.kind == "vulhub":
        if _has_vulhub_delivery_loaders(spec.package_dir):
            return _load_vulhub_runtime_profile_task(spec, env=env)
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
    elif spec.kind == "cvedockergen":
        _prepare_bundle_output(spec, output_dir)
        package, tool = _cvedockergen_package(spec)
        command = [
            sys.executable,
            str(tool),
            "--package-dir",
            str(package),
            "materialize",
            "--output-dir",
            str(output_dir),
            "--load-images",
        ]
    elif spec.kind == "vulhub-local":
        if _uses_preexpanded_harbor_layout(spec.package_dir):
            return _load_preexpanded_harbor_task(spec, env=env)
        _prepare_bundle_output(spec, output_dir)
        _task_name(spec.task)
        tool = _bundle_path(spec.package_dir, "vulhub_task_local.py")
        for relative in (
            f"tasks/{spec.task}/instruction.md",
            f"images/{spec.task}/{spec.task}.tar.gz",
            "user_base_docker.tar",
        ):
            path = _bundle_path(spec.package_dir, relative)
            if not path.is_file():
                raise RuntimeError(f"Local Vulhub bundle input does not exist: {path}")
        source = _bundle_path(spec.package_dir, f"tasks/{spec.task}")
        _validate_copy_source(spec.package_dir, source)
        # The delivery CLI loads user_base_docker.tar itself. The pre/post archive
        # is copied into the Docker build context and loaded later by the verifier.
        command = [
            sys.executable,
            str(tool),
            "--package-dir",
            str(spec.package_dir),
            "harbor",
            "--task",
            spec.task,
            "--output-dir",
            str(output_dir),
            "--nested",
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
    # Some single-task delivery CLIs write directly to OUT, others to OUT/task-id
    # (or OUT/task-id-variant). Never infer the name from the bundle directory.
    if (output_dir / "task.toml").is_file():
        task_dirs.insert(0, output_dir)
    if len(task_dirs) != 1:
        raise RuntimeError(
            f"Bundle materializer must produce exactly one Harbor task, got "
            f"{len(task_dirs)} in {output_dir}"
    )
    if spec.kind == "arvo":
        _set_arvo_agent_timeout(task_dirs[0] / "task.toml")
        _load_arvo_images(spec, env=env)
    elif spec.kind in {"cvedockergen", "vulhub-local"}:
        task_dir = task_dirs[0]
        _bundle_path(output_dir, str(task_dir.relative_to(output_dir)))
        if spec.kind == "vulhub-local":
            _configure_local_verifier_docker(task_dir, env=env)
    return task_dirs[0]


def _has_vulhub_delivery_loaders(package_dir: Path) -> bool:
    """Detect the reusable pre-expanded Vulhub delivery-script contract."""
    return all(
        (package_dir / relative).is_file()
        for relative in ("bin/load_shared_images.py", "bin/load_image.py")
    )


def _load_vulhub_runtime_profile_task(
    spec: BundleSpec, *, env: Mapping[str, str]
) -> Path:
    """Load delivery images and return one pre-expanded Harbor runtime profile."""
    task = _task_name(spec.task)
    task_dir = _bundle_path(spec.package_dir, task)
    runtime = None
    for relative in ("harbor-runtime-profile", "harbor-runtime-profiles"):
        candidate = _bundle_path(task_dir, relative)
        if not candidate.is_dir():
            continue
        if runtime is not None:
            raise RuntimeError(
                f"Vulhub task has multiple runtime profiles: {task_dir}"
            )
        runtime = candidate
    if runtime is None:
        raise RuntimeError(
            f"Vulhub runtime profile does not exist: "
            f"{task_dir / 'harbor-runtime-profile'}"
        )

    docker_env = dict(env)
    subprocess.run(
        [sys.executable, str(spec.package_dir / "bin/load_shared_images.py")],
        cwd=spec.package_dir,
        env=docker_env,
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(spec.package_dir / "bin/load_image.py"),
            "--root",
            str(spec.package_dir),
            "--task",
            task,
        ],
        cwd=spec.package_dir,
        env=docker_env,
        check=True,
    )
    return runtime


def _uses_preexpanded_harbor_layout(package_dir: Path) -> bool:
    """Detect the delivery contract used by ready-to-run Harbor task batches."""
    task_list = package_dir / "task_list.csv"
    tasks_dir = package_dir / "harbor_tasks"
    present = (task_list.exists(), tasks_dir.exists())
    if any(present) and not all(present):
        raise RuntimeError(
            "Pre-expanded Harbor bundle layout is incomplete; expected both "
            f"{task_list} and {tasks_dir}"
        )
    if all(present) and (not task_list.is_file() or not tasks_dir.is_dir()):
        raise RuntimeError(
            "Pre-expanded Harbor bundle layout requires a task_list.csv file "
            "and a harbor_tasks directory"
        )
    return all(present)


def _load_preexpanded_harbor_task(
    spec: BundleSpec, *, env: Mapping[str, str]
) -> Path:
    """Resolve one pre-expanded task and load its root-level base image archive."""
    task = _task_name(spec.task)
    task_list = _bundle_path(spec.package_dir, "task_list.csv")
    try:
        with task_list.open(encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source)
            if not {"task_id", "task_type"}.issubset(reader.fieldnames or []):
                raise RuntimeError(
                    "Pre-expanded Harbor task_list.csv must contain task_id and "
                    "task_type columns"
                )
            matches = [
                row for row in reader if str(row.get("task_id") or "") == task
            ]
    except (OSError, UnicodeError, csv.Error) as error:
        raise RuntimeError(
            f"Cannot read pre-expanded Harbor task list: {error}"
        ) from error
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one pre-expanded Harbor task named {task!r}, got {len(matches)}"
        )
    if not str(matches[0].get("task_type") or "").strip():
        raise RuntimeError(f"Pre-expanded Harbor task has no task_type: {task}")

    task_path = _bundle_path(spec.package_dir, f"harbor_tasks/{task}")
    required_paths = (
        task_path / "task.toml",
        task_path / "instruction.md",
        task_path / "environment/Dockerfile",
        task_path / "tests/test.sh",
    )
    missing = [str(path) for path in required_paths if not path.is_file()]
    if missing:
        raise RuntimeError(
            "Pre-expanded Harbor task is incomplete; missing: " + ", ".join(missing)
        )

    base_images = _dockerfile_base_images(required_paths[2])
    _load_root_image_archives(spec.package_dir, base_images, env=env)
    return task_path


def _dockerfile_base_images(dockerfile: Path) -> tuple[str, ...]:
    """Return concrete external FROM images, excluding named build stages."""
    images: list[str] = []
    stages: set[str] = set()
    for number, raw_line in enumerate(
        dockerfile.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not re.match(r"^\s*FROM(?:\s|$)", raw_line, re.IGNORECASE):
            continue
        try:
            tokens = shlex.split(raw_line, comments=True)
        except ValueError as error:
            raise RuntimeError(
                f"Cannot parse Dockerfile line {number} in {dockerfile}: {error}"
            ) from error
        if not tokens or tokens[0].upper() != "FROM":
            raise RuntimeError(f"Cannot parse Dockerfile FROM at {dockerfile}:{number}")
        index = 1
        while index < len(tokens) and tokens[index].startswith("--"):
            index += 1
        if index >= len(tokens):
            raise RuntimeError(f"Dockerfile FROM has no image at {dockerfile}:{number}")
        image = tokens[index]
        if "$" in image:
            raise RuntimeError(
                f"Pre-expanded Harbor Dockerfile uses a variable FROM image: {image}"
            )
        if (
            image.lower() != "scratch"
            and image.lower() not in stages
            and image not in images
        ):
            images.append(image)
        if index + 2 < len(tokens) and tokens[index + 1].upper() == "AS":
            stages.add(tokens[index + 2].lower())
    if not images:
        raise RuntimeError(f"Dockerfile contains no concrete base image: {dockerfile}")
    return tuple(images)


def _load_root_image_archives(
    package_dir: Path,
    images: tuple[str, ...],
    *,
    env: Mapping[str, str],
) -> None:
    pending = [image for image in images if not _docker_image_exists(image, env=env)]
    if not pending:
        return

    archives = sorted(
        path
        for path in package_dir.iterdir()
        if path.is_file()
        and (
            path.name.endswith(".tar")
            or path.name.endswith(".tar.gz")
            or path.name.endswith(".tgz")
        )
    )
    archive_tags = {archive: _docker_archive_tags(archive) for archive in archives}
    selected: dict[Path, dict[str, str]] = {}
    for image in pending:
        exact = [
            (archive, tag)
            for archive, tags in archive_tags.items()
            for tag in tags
            if tag == image
        ]
        aliases = [
            (archive, tag)
            for archive, tags in archive_tags.items()
            for tag in tags
            if _short_image_ref(tag) == image
        ]
        matches = exact or aliases
        if len(matches) != 1:
            raise RuntimeError(
                f"Expected one root image archive for {image!r}, got {len(matches)} "
                f"under {package_dir}"
            )
        archive, source_tag = matches[0]
        selected.setdefault(archive, {})[image] = source_tag

    docker_env = dict(env)
    for archive, targets in selected.items():
        _verify_optional_archive_sha256(archive)
        subprocess.run(
            ["docker", "load", "--input", str(archive)],
            env=docker_env,
            check=True,
        )
        for target, source_tag in targets.items():
            if not _docker_image_exists(target, env=docker_env):
                if not _docker_image_exists(source_tag, env=docker_env):
                    raise RuntimeError(
                        f"Loaded archive {archive} provides neither {target!r} nor "
                        f"its source tag {source_tag!r}"
                    )
                subprocess.run(
                    ["docker", "tag", source_tag, target],
                    env=docker_env,
                    check=True,
                )
            if not _docker_image_exists(target, env=docker_env):
                raise RuntimeError(
                    f"Docker image is unavailable after loading: {target}"
                )


def _docker_archive_tags(archive: Path) -> tuple[str, ...]:
    try:
        with tarfile.open(archive, mode="r:*") as bundle:
            manifest_file = bundle.extractfile("manifest.json")
            if manifest_file is None:
                return ()
            manifest = json.load(manifest_file)
    except (KeyError, OSError, tarfile.TarError, UnicodeError, json.JSONDecodeError):
        return ()
    tags: list[str] = []
    for row in manifest if isinstance(manifest, list) else []:
        if not isinstance(row, dict):
            continue
        for tag in row.get("RepoTags") or []:
            if isinstance(tag, str) and tag and tag not in tags:
                tags.append(tag)
    return tuple(tags)


def _short_image_ref(image: str) -> str:
    return image.rsplit("/", 1)[-1]


def _verify_optional_archive_sha256(archive: Path) -> None:
    digest_path = Path(f"{archive}.sha256")
    if not digest_path.is_file():
        return
    fields = digest_path.read_text(encoding="utf-8").strip().split()
    expected = fields[0].lower() if fields else ""
    if len(expected) != 64 or any(
        character not in "0123456789abcdef" for character in expected
    ):
        raise RuntimeError(f"Invalid image archive SHA-256 file: {digest_path}")
    digest = hashlib.sha256()
    with archive.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != expected:
        raise RuntimeError(
            f"Image archive SHA-256 mismatch: {archive}; expected={expected} actual={actual}"
        )


def _docker_image_exists(image: str, *, env: Mapping[str, str]) -> bool:
    return subprocess.run(
        ["docker", "image", "inspect", image],
        env=dict(env),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0


def _load_cvefactory_batch(
    spec: BundleSpec, *, env: Mapping[str, str]
) -> Path:
    task = _task_name(spec.task)
    if not re.fullmatch(r"cve-\d{4}-\d{4,}", task):
        raise RuntimeError(f"CVE-Factory batch task is not a CVE: {task}")

    manifest_path = _bundle_path(spec.package_dir, "package-manifest.json")
    load_images = _bundle_path(spec.package_dir, "load_images.py")
    if not manifest_path.is_file():
        raise RuntimeError(f"CVE-Factory package manifest does not exist: {manifest_path}")
    if not load_images.is_file():
        raise RuntimeError(f"CVE-Factory image loader does not exist: {load_images}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = manifest.get("tasks")
    if not isinstance(rows, list) or not rows:
        raise RuntimeError("CVE-Factory package manifest has no tasks")
    if manifest.get("task_count") != len(rows):
        raise RuntimeError(
            "CVE-Factory package task count mismatch: "
            f"declared={manifest.get('task_count')!r} actual={len(rows)}"
        )
    matches = [row for row in rows if str(row.get("task_id") or "") == task]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one CVE-Factory manifest task named {task!r}, got {len(matches)}"
        )

    task_path = _bundle_path(spec.package_dir, task)
    required_paths = (
        task_path / "task.toml",
        task_path / "instruction.md",
        task_path / "environment/docker-compose.yaml",
        task_path / "tests/test.sh",
    )
    missing = [str(path) for path in required_paths if not path.is_file()]
    if missing:
        raise RuntimeError(
            "CVE-Factory batch task is incomplete; missing: " + ", ".join(missing)
        )

    subprocess.run(
        [sys.executable, str(load_images), "--skip-sha256"],
        cwd=spec.package_dir,
        env=dict(env),
        check=True,
    )
    return task_path


def _task_name(value: str) -> str:
    if (
        not value
        or value in {".", ".."}
        or Path(value).name != value
        or "\\" in value
        or any(ord(character) < 32 for character in value)
    ):
        raise RuntimeError(f"Invalid bundle task name: {value!r}")
    return value


def _prepare_bundle_output(spec: BundleSpec, output_dir: Path) -> None:
    """Delivery scripts may remove existing tasks; allow only a fresh output."""
    root = spec.package_dir.resolve()
    destination = output_dir.resolve()
    if destination == root or root in destination.parents:
        raise RuntimeError(f"Bundle output must be outside the source package: {output_dir}")
    if output_dir.is_symlink() or (
        output_dir.exists()
        and (not output_dir.is_dir() or any(output_dir.iterdir()))
    ):
        raise RuntimeError(f"Bundle materialization output is not empty: {output_dir}")


def _cvedockergen_package(spec: BundleSpec) -> tuple[Path, Path]:
    """Select one CTF/ARVO/CVE delivery in a directory of single-task bundles."""
    package = _bundle_path(spec.package_dir, _task_name(spec.task))
    manifest = json.loads(_bundle_path(package, "manifest.json").read_text())
    tasks = manifest.get("tasks") or []
    if len(tasks) != 1:
        raise RuntimeError(f"CVEDockerGen child bundle must contain one task: {package}")
    task = tasks[0]
    _task_name(str(task.get("task_id") or ""))
    profile = task["runtime_profile"]
    source = _bundle_path(package, str(profile["runnable_task"]))
    if not (source / "task.toml").is_file():
        raise RuntimeError(f"CVEDockerGen runnable task is missing task.toml: {source}")
    # Copytree follows symlinks. Allow shared files within the collection, but
    # reject inputs that would copy arbitrary files outside the mounted root.
    _validate_copy_source(spec.package_dir, source)
    for row in (task.get("images") or {}).values():
        archive = _bundle_path(spec.package_dir, str(package / row["archive"]))
        if not archive.is_file():
            raise RuntimeError(f"CVEDockerGen task image archive is missing: {archive}")
    profile_root = _bundle_path(package, str(profile["root"]))
    lock_path = _bundle_path(package, str(profile_root / "image-lock.json"))
    if lock_path.is_file():
        lock = json.loads(lock_path.read_text())
        for row in lock.get("images") or []:
            # The delivered tools can fall back to collection/_runtime for a
            # missing profile-local archive. Preserve that shared mount layout.
            _bundle_path(spec.package_dir, str(profile_root / row["archive"]))
    tools = [
        _bundle_path(package, f"bin/{name}")
        for name in ("ctf_task.py", "arvo_task.py")
        if (package / "bin" / name).is_file()
    ]
    if len(tools) != 1:
        raise RuntimeError(f"Expected one CVEDockerGen materializer in {package}/bin")
    return package, tools[0]


def _validate_copy_source(package_dir: Path, source: Path) -> None:
    # DirEntry caches file-type metadata, avoiding a GPFS stat for every asset.
    # Follow in-root directory links too: copytree will dereference their files.
    pending = [source]
    visited: set[Path] = set()
    while pending:
        directory = pending.pop()
        resolved = _bundle_path(package_dir, str(directory))
        if resolved in visited:
            continue
        visited.add(resolved)
        with os.scandir(directory) as entries:
            for entry in entries:
                if entry.is_symlink():
                    _bundle_path(package_dir, entry.path)
                if entry.is_dir():
                    pending.append(Path(entry.path))


def _configure_local_verifier_docker(
    task_dir: Path, *, env: Mapping[str, str]
) -> None:
    """Bind the runner's nested daemon into the delivery verifier's container."""
    docker_host = env.get("DOCKER_HOST", "unix:///var/run/docker.sock")
    if not docker_host.startswith("unix:///"):
        raise RuntimeError("vulhub-local verifier requires a local UNIX Docker socket")
    socket = docker_host[len("unix://"):]
    docker = shutil.which("docker", path=env.get("PATH", os.defpath))
    if docker is None:
        raise RuntimeError("Docker CLI is required for the vulhub-local verifier")
    compose = _bundle_path(task_dir, "environment/docker-compose.yaml")
    content = compose.read_text(encoding="utf-8")
    for source, destination in (
        ("/var/run/docker.sock:/var/run/docker.sock", f"{socket}:/var/run/docker.sock"),
        ("/usr/bin/docker:/usr/bin/docker:ro", f"{Path(docker).resolve()}:/usr/bin/docker:ro"),
    ):
        if content.count(source) != 1:
            raise RuntimeError(f"Unexpected local Vulhub Docker bind in {compose}: {source}")
        content = content.replace(source, json.dumps(destination))
    if content.count("    environment:\n") != 1:
        raise RuntimeError(f"Expected one main environment block in {compose}")
    content = content.replace(
        "    environment:\n",
        "    environment:\n      DOCKER_HOST: unix:///var/run/docker.sock\n",
        1,
    )
    _replace_materialized_file(compose, content)


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
