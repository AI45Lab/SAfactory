#!/usr/bin/env python3
"""Validate and load offline image archives into nested Docker."""

from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


LOAD_TIMEOUT_S = 1800
INSPECT_TIMEOUT_S = 20


@dataclass(frozen=True)
class ImageArchive:
    source: Path
    sha256: str | None
    size_bytes: int | None
    expected_images: tuple[str, ...]


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
