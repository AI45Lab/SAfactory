"""
Lightweight resource manager for the multiprocess backend.

The implementation mirrors ``myray.resource_manager`` but does not
depend on Ray, keeping the interface compatible while running fully
inside the current process.
"""

import os
import shutil
import threading
from pathlib import Path
from typing import Dict, Optional, Set

import logging


logger = logging.getLogger("MPResourceManager")


class PortPool:
    """Thread-safe port allocator."""

    def __init__(self, start: int, end: int, name: str = "PortPool"):
        if start >= end:
            raise ValueError(f"start ({start}) must be less than end ({end})")
        if start < 0 or end > 65536:
            raise ValueError("Port range must be within 0-65535")

        self.name = name
        self.start = start
        self.end = end
        self.available: Set[int] = set(range(start, end))
        self.allocated: Set[int] = set()
        self.lock = threading.Lock()

        logger.info("[%s] Initialized with %d ports (%d-%d)", self.name, len(self.available), start, end - 1)

    def allocate(self) -> int:
        with self.lock:
            if not self.available:
                raise RuntimeError(f"[{self.name}] No available ports! All {len(self.allocated)} ports are allocated.")

            port = self.available.pop()
            self.allocated.add(port)
            logger.info("[%s] Allocated port %d (%d remaining)", self.name, port, len(self.available))
            return port

    def release(self, port: int):
        with self.lock:
            if port in self.allocated:
                self.allocated.remove(port)
                self.available.add(port)
                logger.info("[%s] Released port %d (%d available)", self.name, port, len(self.available))
            else:
                logger.warning("[%s] Port %d was not allocated", self.name, port)

    def get_stats(self) -> Dict:
        with self.lock:
            return {
                "name": self.name,
                "total": len(self.available) + len(self.allocated),
                "available": len(self.available),
                "allocated": len(self.allocated),
                "range": f"{self.start}-{self.end-1}",
            }


class WorkingDirManager:
    """Manage per-env working directories to avoid collisions."""

    def __init__(self, base_dir: str = "/tmp/raycraft_mc"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        self.directories: Dict[str, Path] = {}
        self.lock = threading.Lock()

        logger.info("[WorkingDirManager] Initialized with base %s", self.base_dir)

    def create(self, uuid: str) -> Path:
        with self.lock:
            if uuid in self.directories:
                return self.directories[uuid]

            working_dir = self.base_dir / f"mc-{uuid}"
            working_dir.mkdir(parents=True, exist_ok=True)
            self.directories[uuid] = working_dir
            logger.info("[WorkingDirManager] Created directory for %s: %s", uuid[:8], working_dir)
            return working_dir

    def cleanup(self, uuid: str):
        with self.lock:
            if uuid not in self.directories:
                return

            working_dir = self.directories[uuid]
            if working_dir.exists():
                try:
                    shutil.rmtree(working_dir)
                    logger.info("[WorkingDirManager] Cleaned directory for %s: %s", uuid[:8], working_dir)
                except Exception as e:
                    logger.warning("[WorkingDirManager] Failed to clean %s: %s", uuid[:8], e)

            del self.directories[uuid]

    def get_stats(self) -> Dict:
        with self.lock:
            total_size = 0
            for working_dir in self.directories.values():
                if working_dir.exists():
                    try:
                        total_size += sum(
                            f.stat().st_size
                            for f in working_dir.rglob('*')
                            if f.is_file()
                        )
                    except Exception:
                        pass

            return {
                "count": len(self.directories),
                "total_size_mb": total_size / 1024 / 1024,
                "base_dir": str(self.base_dir),
            }


class GlobalResourceManager:
    """Centralized allocator for ports and working directories."""

    def __init__(
        self,
        display_port_range=(1, 100),
        working_dir_base="/tmp/raycraft_mc",
    ):
        self.display_pool = PortPool(*display_port_range, name="DISPLAY")
        self.working_dir_manager = WorkingDirManager(working_dir_base)

        self.allocations: Dict[str, Dict] = {}
        logger.info(
            "[GlobalResourceManager] Initialized display=%s workdir=%s",
            display_port_range,
            working_dir_base,
        )

    def allocate_resources(self, uuid: str) -> Dict:
        if uuid in self.allocations:
            raise RuntimeError(f"UUID {uuid} already has resources allocated")

        display_port = self.display_pool.allocate()
        working_dir = self.working_dir_manager.create(uuid)

        self.allocations[uuid] = {
            "display_port": display_port,
            "working_dir": str(working_dir),
        }

        return self.allocations[uuid]

    def release_resources(self, uuid: str):
        if uuid not in self.allocations:
            return

        resources = self.allocations[uuid]
        self.display_pool.release(resources["display_port"])

        # Keep working dirs for debugging; callers can clean manually if needed.
        del self.allocations[uuid]

    def get_stats(self) -> Dict:
        return {
            "display_ports": self.display_pool.get_stats(),
            "working_dirs": self.working_dir_manager.get_stats(),
            "active_instances": len(self.allocations),
            "allocated_uuids": [uuid[:8] for uuid in self.allocations.keys()],
        }

