"""Make Docker SDK runs lazily load PatchEval image archives.

Python imports ``sitecustomize`` automatically when this directory is placed on
PYTHONPATH. This keeps the upstream PatchEval implementation unchanged.
"""
from __future__ import annotations

import os
import subprocess
import threading
from collections import defaultdict
from pathlib import Path
from typing import Any


ARCHIVE_DIR = Path(os.environ["PATCHEVAL_IMAGE_ARCHIVE_DIR"]).expanduser()
CLEANUP = os.environ.get("PATCHEVAL_CLEANUP_IMAGE", "1").lower() not in {"0", "false", "no", "off"}
DOCKER_BIN = os.environ.get("DOCKER_BIN", "docker")
LOAD_TIMEOUT_S = float(os.environ.get("PATCHEVAL_IMAGE_LOAD_TIMEOUT_S", "1800"))


def _with_proxy_environment(existing: Any) -> Any:
    http_proxy = os.environ.get("PATCHEVAL_HTTP_PROXY", "").strip()
    https_proxy = os.environ.get("PATCHEVAL_HTTPS_PROXY", "").strip() or http_proxy
    no_proxy = os.environ.get("PATCHEVAL_NO_PROXY", "").strip()
    configured = {
        key: value
        for key, value in {
            "HTTP_PROXY": http_proxy,
            "HTTPS_PROXY": https_proxy,
            "NO_PROXY": no_proxy,
            "http_proxy": http_proxy,
            "https_proxy": https_proxy,
            "no_proxy": no_proxy,
        }.items()
        if value
    }
    if not configured:
        return existing
    if existing is None:
        return configured
    if isinstance(existing, dict):
        return {**configured, **existing}
    if isinstance(existing, list):
        names = {str(item).split("=", 1)[0] for item in existing}
        return [*existing, *(f"{key}={value}" for key, value in configured.items() if key not in names)]
    return existing


def _install() -> None:
    if not ARCHIVE_DIR.is_dir():
        raise RuntimeError(f"PatchEval image archive directory does not exist: {ARCHIVE_DIR}")

    from docker.errors import ImageNotFound
    from docker.models.containers import Container, ContainerCollection

    original_run = ContainerCollection.run
    original_remove = Container.remove
    locks: defaultdict[str, threading.Lock] = defaultdict(threading.Lock)
    loaded_images: set[str] = set()
    container_images: dict[str, str] = {}

    def archive_for(image: str) -> Path:
        stem = image.rsplit("/", 1)[-1].replace(":", "-")
        candidates = (
            ARCHIVE_DIR / f"{stem}.tar",
            ARCHIVE_DIR / f"{stem}.tar.gz",
            ARCHIVE_DIR / f"{stem}.tgz",
        )
        archive = next((path for path in candidates if path.is_file() and path.stat().st_size > 0), None)
        if archive is None:
            raise FileNotFoundError(f"No archive found for Docker image {image!r} in {ARCHIVE_DIR}")
        return archive

    def ensure_image(collection: Any, image: str) -> bool:
        with locks[image]:
            try:
                collection.client.images.get(image)
                return image in loaded_images
            except ImageNotFound:
                archive = archive_for(image)
                print(f"[PatchEval image adapter] loading {image} from {archive}", flush=True)
                subprocess.run(
                    [DOCKER_BIN, "load", "-i", str(archive)],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=LOAD_TIMEOUT_S,
                )
                collection.client.images.get(image)
                loaded_images.add(image)
                return True

    def run_with_archive(self: Any, image: str, command: Any = None, *args: Any, **kwargs: Any) -> Any:
        loaded_by_adapter = ensure_image(self, image)
        kwargs["environment"] = _with_proxy_environment(kwargs.get("environment"))
        try:
            container = original_run(self, image, command, *args, **kwargs)
        except Exception:
            if loaded_by_adapter:
                with locks[image]:
                    try:
                        self.client.images.remove(image)
                    except Exception:
                        pass
                    loaded_images.discard(image)
            raise
        if loaded_by_adapter:
            container_images[container.id] = image
        return container

    def remove_with_cleanup(self: Any, *args: Any, **kwargs: Any) -> Any:
        container_id = self.id
        image = container_images.pop(container_id, None)
        result = original_remove(self, *args, **kwargs)
        if image and CLEANUP:
            with locks[image]:
                try:
                    remaining = self.client.containers.list(all=True, filters={"ancestor": image})
                    if not remaining:
                        self.client.images.remove(image)
                        loaded_images.discard(image)
                        print(f"[PatchEval image adapter] removed {image}", flush=True)
                except Exception as exc:
                    print(f"[PatchEval image adapter] failed to remove {image}: {exc}", flush=True)
        return result

    ContainerCollection.run = run_with_archive
    Container.remove = remove_with_cleanup


_install()
