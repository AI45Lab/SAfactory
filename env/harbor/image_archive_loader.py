#!/usr/bin/env python3
"""Validate and load offline image archives into nested Docker."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


LOAD_TIMEOUT_S = 1800
INSPECT_TIMEOUT_S = 20
ARCHIVE_BUNDLE_TYPE = "cvedockergen-harbor-archive"
PROPENTBENCH_ARCHIVE_BUNDLE_TYPE = "cvedockergen-propentbench-archive"
TASK_CATEGORY_FIELD = "task_type"
PROPENTBENCH_AGENT_IMAGE = "propentbench-agent:2.1.228"
PROPENTBENCH_ALPINE_IMAGE = "alpine:3.23.4"
PROPENTBENCH_EGRESS_IMAGE = (
    "harbor-prebuilt:harbor-docker-egress-control-sidecar--9af3dc172792cee8"
)
PROPENTBENCH_AGENT_ARCHIVE = (
    "agent-image/agent-image.tar",
    2514088960,
    "bd04872296bc0ebb6ec458ab01cb7954798af8012e896615e93e0dfd7450937c",
    (PROPENTBENCH_AGENT_IMAGE,),
)
PROPENTBENCH_SIDECAR_ARCHIVES = (
    (
        "sidecar-images/alpine-probe-oci.tar",
        3883520,
        "6b6ed995465ff2ebd500fccfadea2dc96f13bc01bd651fe90a84571624122611",
        (PROPENTBENCH_ALPINE_IMAGE,),
    ),
    (
        "sidecar-images/egress-sidecar-prebuilt.tar",
        19385344,
        "af8d89cc80815a8a6343bb8cee887e0a19364b17998e447ebe80d833c74a058a",
        (PROPENTBENCH_EGRESS_IMAGE,),
    ),
)


@dataclass(frozen=True)
class ImageArchive:
    source: Path
    sha256: str | None
    size_bytes: int | None
    expected_images: tuple[str, ...]


@dataclass(frozen=True)
class ImageArchiveBundle:
    package_dir: Path
    task: str
    task_path: Path
    image_archives: tuple[ImageArchive, ...]
    harbor_env: tuple[tuple[str, str], ...] = field(default=(), repr=False)


def parse_image_archive_bundle(
    dataset: Mapping[str, Any], params: Mapping[str, Any]
) -> ImageArchiveBundle | None:
    """Resolve one prebuilt Harbor task and the Docker archives it requires."""
    kind = str(_param(dataset, params, "bundle_type", "") or "").strip()
    if kind not in {
        ARCHIVE_BUNDLE_TYPE,
        PROPENTBENCH_ARCHIVE_BUNDLE_TYPE,
    }:
        return None

    package_dir_text = _required_text(
        _param(dataset, params, "bundle_package_dir"), "bundle_package_dir"
    )
    task = _task_name(
        _required_text(_param(dataset, params, "bundle_task"), "bundle_task")
    )
    for key in ("bundle_variant", "bundle_level"):
        if str(_param(dataset, params, key, "") or "").strip():
            raise RuntimeError(f"{kind} does not accept {key}")
    if str(params.get("bundle_validation") or "full").strip() != "full":
        raise RuntimeError("bundle_validation is only configurable for ARVO")

    package_dir = Path(package_dir_text).resolve()
    if not package_dir.is_dir():
        raise RuntimeError(f"Bundle package directory does not exist: {package_dir}")
    if kind == PROPENTBENCH_ARCHIVE_BUNDLE_TYPE:
        common_package_dir_text = str(
            _param(dataset, params, "bundle_common_package_dir", "") or ""
        ).strip()
        common_package_dir = (
            Path(common_package_dir_text).resolve()
            if common_package_dir_text
            else package_dir
        )
        return _parse_propentbench_archive_bundle(
            package_dir, task, common_package_dir
        )
    manifest_path = _bundle_path(package_dir, "manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = []
    for row in manifest.get("tasks") or []:
        harbor = row.get("harbor") or {}
        task_path_text = str(harbor.get("task_path") or "")
        if task_path_text and Path(task_path_text).name == task:
            rows.append(row)
    if len(rows) != 1:
        raise RuntimeError(
            f"Expected one manifest task named {task!r}, got {len(rows)}"
        )
    row = rows[0]
    category = str(_param(dataset, params, TASK_CATEGORY_FIELD, "") or "").strip().lower()
    if category and category not in {"cve", "ctf"}:
        raise RuntimeError(
            f"{TASK_CATEGORY_FIELD} must be 'cve' or 'ctf' for archive bundles"
        )
    manifest_category = "cve" if str(row.get("cve_id") or "").strip() else "ctf"
    if category and category != manifest_category:
        raise RuntimeError(
            f"Task category mismatch for {task}: dataset={category} manifest={manifest_category}"
        )
    if str(row.get("status") or "") != "runnable":
        raise RuntimeError(f"Archive bundle task is not runnable: {task}")

    # Resolve the runtime profile from this task's manifest entry.  CVE and CTF
    # tasks intentionally use different profiles; do not hard-code a batch path
    # or assume all CTF tasks share one profile.
    task_path_text = str((row.get("harbor") or {}).get("task_path") or "")
    profile = Path(task_path_text).parts[1] if len(Path(task_path_text).parts) > 1 else ""
    if category == "cve" and profile.startswith("docker-"):
        raise RuntimeError(f"CVE task {task} unexpectedly uses profile {profile!r}")
    if category == "ctf" and profile not in {"docker-prebuilt-amd64", "docker-linux-amd64", "docker-compose-amd64"}:
        raise RuntimeError(f"CTF task {task} unexpectedly uses profile {profile!r}")
    task_path = _bundle_path(package_dir, task_path_text)
    task_toml_path = task_path / "task.toml"
    if not task_path.is_dir() or not task_toml_path.is_file():
        raise RuntimeError(f"Harbor task directory is incomplete: {task_path}")
    task_toml = tomllib.loads(task_toml_path.read_text(encoding="utf-8"))
    image = _required_text(
        (task_toml.get("environment") or {}).get("docker_image"),
        f"{task_toml_path} [environment].docker_image",
    )

    archive_path = _bundle_path(package_dir, str(row.get("image_archive") or ""))
    if not archive_path.is_file():
        raise RuntimeError(f"Image archive does not exist: {archive_path}")
    sha256 = _required_text(row.get("image_tar_sha256"), "image_tar_sha256").lower()
    if len(sha256) != 64 or any(
        character not in "0123456789abcdef" for character in sha256
    ):
        raise RuntimeError("image_tar_sha256 must be 64 hex digits")

    return ImageArchiveBundle(
        package_dir=package_dir,
        task=task,
        task_path=task_path,
        image_archives=(
            ImageArchive(
                source=archive_path,
                sha256=sha256,
                size_bytes=archive_path.stat().st_size,
                expected_images=(image,),
            ),
        ),
    )


def _parse_propentbench_archive_bundle(
    package_dir: Path, task: str, common_package_dir: Path
) -> ImageArchiveBundle:
    if not re.fullmatch(r"CVE-\d{4}-\d{4,}", task):
        raise RuntimeError(f"PropEntBench archive bundle task is not a CVE: {task}")

    task_path = _bundle_path(package_dir, task)
    task_toml_path = task_path / "task.toml"
    compose_path = task_path / "environment/docker-compose.yaml"
    archive_path = task_path / "images.tar.gz"
    digest_path = task_path / "PREBUILT-IMAGE-SHA256"
    flag_path = _bundle_path(package_dir, f"run/{task}.json")
    required_paths = (task_toml_path, compose_path, archive_path, digest_path, flag_path)
    missing = [str(path) for path in required_paths if not path.is_file()]
    if missing:
        raise RuntimeError(
            "PropEntBench archive bundle is incomplete; missing: " + ", ".join(missing)
        )

    task_toml = tomllib.loads(task_toml_path.read_text(encoding="utf-8"))
    metadata = task_toml.get("metadata") or {}
    if str(metadata.get("cve") or "").upper() != task:
        raise RuntimeError(f"PropEntBench task metadata CVE does not match {task}")
    agent_image = _required_text(
        (task_toml.get("environment") or {}).get("docker_image"),
        f"{task_toml_path} [environment].docker_image",
    )
    if agent_image != PROPENTBENCH_AGENT_IMAGE:
        raise RuntimeError(
            f"Unexpected PropEntBench agent image for {task}: {agent_image}"
        )
    network_mode = (task_toml.get("environment") or {}).get(
        "network_mode", "public"
    )
    if not isinstance(network_mode, str) or network_mode not in {
        "public",
        "no-network",
        "allowlist",
    }:
        raise RuntimeError(
            f"Unexpected PropEntBench network_mode for {task}: {network_mode!r}"
        )

    expected_task_images = _compose_images(compose_path)
    sha256 = _sha256_text(digest_path.read_text(encoding="utf-8"), str(digest_path))
    receipt_path = task_path / "materialization-receipt.json"
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt_sha256 = _sha256_text(receipt.get("image_sha256"), "image_sha256")
        if receipt_sha256 != sha256:
            raise RuntimeError(f"PropEntBench image digests disagree for {task}")

    flag_data = json.loads(flag_path.read_text(encoding="utf-8"))
    if str(flag_data.get("cve") or "").upper() != task:
        raise RuntimeError(f"PropEntBench flag metadata CVE does not match {task}")
    expected_flag = _required_text(flag_data.get("expected_flag"), "expected_flag")
    if str(flag_data.get("flag_mode") or "") not in {
        "runtime-placeholder",
        "baked-only",
    }:
        raise RuntimeError(f"Unexpected PropEntBench flag mode for {task}")

    agent_relative, agent_size, agent_sha256, agent_images = (
        PROPENTBENCH_AGENT_ARCHIVE
    )
    agent_root = (
        package_dir
        if (package_dir / agent_relative).is_file()
        else common_package_dir
    )
    agent_archive = ImageArchive(
        source=_required_sized_bundle_file(
            agent_root, agent_relative, agent_size
        ),
        sha256=agent_sha256,
        size_bytes=agent_size,
        expected_images=agent_images,
    )
    sidecar_archives = tuple(
        ImageArchive(
            source=_required_sized_bundle_file(
                common_package_dir,
                relative,
                size_bytes,
            ),
            sha256=sha256,
            size_bytes=size_bytes,
            expected_images=expected_images,
        )
        for relative, size_bytes, sha256, expected_images in (
            PROPENTBENCH_SIDECAR_ARCHIVES if network_mode != "public" else ()
        )
    )
    task_archive = ImageArchive(
        source=archive_path,
        sha256=sha256,
        size_bytes=archive_path.stat().st_size,
        expected_images=expected_task_images,
    )
    return ImageArchiveBundle(
        package_dir=package_dir,
        task=task,
        task_path=task_path,
        image_archives=(agent_archive, *sidecar_archives, task_archive),
        harbor_env=(("FLAG", expected_flag), ("HARBOR_EXPECTED_SECRET", expected_flag)),
    )


def parse_image_archives(value: Any) -> tuple[ImageArchive, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise TypeError("image_archives must be a list")

    archives: list[ImageArchive] = []
    for index, item in enumerate(value):
        name = f"image_archives[{index}]"
        if not isinstance(item, dict):
            raise TypeError(f"{name} must be an object")
        source = Path(_required_text(item.get("source"), f"{name}.source"))
        if not source.is_absolute():
            raise RuntimeError(f"{name}.source must be absolute")
        if not source.is_file():
            raise RuntimeError(f"{name}.source is not a regular file: {source}")

        raw_size = item.get("size_bytes")
        size_bytes: int | None = None
        if raw_size is not None:
            if isinstance(raw_size, bool) or not isinstance(raw_size, int):
                raise TypeError(f"{name}.size_bytes must be an integer")
            size_bytes = raw_size
            if size_bytes <= 0:
                raise RuntimeError(f"{name}.size_bytes must be positive")
            actual_size = source.stat().st_size
            if actual_size != size_bytes:
                raise RuntimeError(
                    f"image archive size mismatch: {source}; "
                    f"expected={size_bytes} actual={actual_size}"
                )

        raw_sha256 = item.get("sha256")
        sha256: str | None = None
        if raw_sha256 is not None:
            sha256 = _required_text(raw_sha256, f"{name}.sha256").lower()
            if len(sha256) != 64 or any(
                character not in "0123456789abcdef" for character in sha256
            ):
                raise RuntimeError(f"{name}.sha256 must be 64 hex digits")

        raw_images = item.get("expected_images")
        if not isinstance(raw_images, list) or not raw_images:
            raise RuntimeError(f"{name}.expected_images must be a non-empty list")
        expected_images = tuple(
            _required_text(image, f"{name}.expected_images") for image in raw_images
        )
        if len(expected_images) != len(set(expected_images)):
            raise RuntimeError(f"{name}.expected_images contains duplicates")
        archives.append(
            ImageArchive(
                source=source,
                sha256=sha256,
                size_bytes=size_bytes,
                expected_images=expected_images,
            )
        )
    return tuple(archives)


def load_image_archives(
    archives: tuple[ImageArchive, ...], *, env: Mapping[str, str]
) -> None:
    docker_env = dict(env)
    for archive in archives:
        if archive.sha256 is not None:
            actual_sha256 = sha256_file(archive.source)
            if actual_sha256 != archive.sha256:
                raise RuntimeError(
                    f"image archive SHA-256 mismatch: {archive.source}; "
                    f"expected={archive.sha256} actual={actual_sha256}"
                )
        try:
            completed = subprocess.run(
                ["docker", "load", "--input", str(archive.source)],
                env=docker_env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
                timeout=LOAD_TIMEOUT_S,
            )
        except subprocess.TimeoutExpired as error:
            raise RuntimeError(
                f"timed out loading image archive after {LOAD_TIMEOUT_S}s: "
                f"{archive.source}"
            ) from error
        if completed.returncode != 0:
            output = str(completed.stdout or "").strip()
            raise RuntimeError(
                f"failed to load image archive {archive.source}: {output[-2000:]}"
            )
        for image in archive.expected_images:
            _inspect_image(image, env=docker_env)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _compose_images(path: Path) -> tuple[str, ...]:
    images: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^\s*image:\s*['\"]?([^'\"\s#]+)", line)
        if match and match.group(1) not in images:
            images.append(match.group(1))
    if not images:
        raise RuntimeError(f"Docker Compose file contains no images: {path}")
    return tuple(images)


def _sha256_text(value: Any, name: str) -> str:
    sha256 = _required_text(value, name).lower()
    if len(sha256) != 64 or any(
        character not in "0123456789abcdef" for character in sha256
    ):
        raise RuntimeError(f"{name} must be 64 hex digits")
    return sha256


def _required_bundle_file(package_dir: Path, relative: str) -> Path:
    path = _bundle_path(package_dir, relative)
    if not path.is_file():
        raise RuntimeError(f"Bundle file does not exist: {path}")
    return path


def _required_sized_bundle_file(
    package_dir: Path, relative: str, size_bytes: int
) -> Path:
    path = _required_bundle_file(package_dir, relative)
    if path.stat().st_size != size_bytes:
        raise RuntimeError(
            f"Bundle file size mismatch: {path}; "
            f"expected={size_bytes} actual={path.stat().st_size}"
        )
    return path


def _inspect_image(image: str, *, env: Mapping[str, str]) -> None:
    completed = subprocess.run(
        ["docker", "image", "inspect", image],
        env=dict(env),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=INSPECT_TIMEOUT_S,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stdout.strip() or "docker command failed")


def _required_text(value: Any, name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise RuntimeError(f"SimulationStartRequest missing {name}")
    return text


def _param(
    dataset: Mapping[str, Any], params: Mapping[str, Any], key: str, default: Any = None
) -> Any:
    value = dataset.get(key)
    return value if value is not None else params.get(key, default)


def _task_name(value: str) -> str:
    if (
        value in {".", ".."}
        or Path(value).name != value
        or "\\" in value
        or any(ord(character) < 32 for character in value)
    ):
        raise RuntimeError(f"Invalid bundle task name: {value!r}")
    return value


def _bundle_path(package_dir: Path, relative: str) -> Path:
    if not relative or Path(relative).is_absolute():
        raise RuntimeError(f"Invalid bundle-relative path: {relative!r}")
    root = package_dir.resolve()
    path = (root / relative).resolve()
    if path != root and root not in path.parents:
        raise RuntimeError(f"Bundle path escapes package root: {relative!r}")
    return path
