"""
Task Manager Module

Handles task loading, configuration resolution, and task iteration management.
Supports both RiOSWorld and OSWorld benchmark formats.
"""

import os
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List

logger = logging.getLogger("osgym.task_manager")


class TaskManager:
    """
    Task loading and iteration management for OSGym.

    Supports two benchmark types:
    - riosworld: Risk assessment benchmark (task configs directly in domain directories)
    - osworld: General desktop tasks (task configs in examples/domain directories)
    """

    def __init__(
        self,
        task_config_path: Optional[str],
        benchmark_type: str,
        current_dir: Path,
        target_task_id: Optional[str] = None
    ):
        """
        Initialize TaskManager.

        Args:
            task_config_path: Path to task configuration JSON file
            benchmark_type: "riosworld" or "osworld"
            current_dir: Base directory for resolving relative paths
            target_task_id: Optional specific task ID to filter
        """
        self.benchmark_type = benchmark_type.lower()
        self.current_dir = current_dir
        self.target_task_id = target_task_id

        self.task_config_path = self._resolve_task_config_path(task_config_path)
        self.tasks: List[Dict[str, str]] = []
        self.task_meta: Dict = {}
        self.base_config_dir: str = ""
        self.task_index: int = 0

        # Load tasks on initialization
        self.tasks = self._load_tasks()

    def _resolve_task_config_path(self, task_config_path: Optional[str]) -> str:
        """
        Resolve task configuration file path.

        RiOSWorld default: evaluation_risk_examples/test_risk.json
        OSWorld default: evaluation_osworld_examples/test_all.json
        """
        if task_config_path is None:
            if self.benchmark_type == "osworld":
                task_config_path = os.path.join(
                    self.current_dir, "evaluation_osworld_examples", "test_all.json"
                )
            else:  # riosworld (default)
                task_config_path = os.path.join(
                    self.current_dir, "evaluation_risk_examples", "test_risk.json"
                )

        if task_config_path and not os.path.isabs(task_config_path):
            rel_path = os.path.join(self.current_dir, task_config_path)
            if os.path.exists(rel_path):
                return rel_path

        return task_config_path

    def _load_tasks(self) -> List[Dict[str, str]]:
        """
        Load task list from configuration file.

        RiOSWorld format: Task configs directly in domain directories
        OSWorld format: Task configs in examples/domain directories
        """
        tasks: List[Dict[str, str]] = []
        logger.info(f"Loading tasks from: {self.task_config_path} (benchmark_type={self.benchmark_type})")

        if not self.task_config_path or not os.path.exists(self.task_config_path):
            logger.error(f"Task config path does not exist: {self.task_config_path}")
            if self.target_task_id:
                raise ValueError(f"Task {self.target_task_id} not found in {self.task_config_path}")
            return tasks

        with open(self.task_config_path, "r", encoding="utf-8") as f:
            self.task_meta = json.load(f)
            self.base_config_dir = os.path.dirname(self.task_config_path)

            for domain, task_ids in self.task_meta.items():
                for task_id_iter in task_ids:
                    if self.target_task_id and task_id_iter != self.target_task_id:
                        continue

                    # OSWorld: configs in examples/domain/
                    # RiOSWorld: configs directly in domain/
                    if self.benchmark_type == "osworld":
                        config_path = os.path.join(
                            self.base_config_dir, "examples", domain, f"{task_id_iter}.json"
                        )
                    else:  # riosworld
                        config_path = os.path.join(
                            self.base_config_dir, domain, f"{task_id_iter}.json"
                        )

                    if os.path.exists(config_path):
                        tasks.append({
                            "domain": domain,
                            "id": task_id_iter,
                            "config_path": config_path
                        })
                    else:
                        logger.warning(f"Config not found for task {domain}/{task_id_iter} at {config_path}")

        if not tasks:
            if self.target_task_id:
                raise ValueError(f"Task {self.target_task_id} not found in {self.task_config_path}")
            logger.warning("No tasks loaded. Please check task_config_path.")

        logger.info(f"Loaded {len(tasks)} tasks for benchmark_type={self.benchmark_type}")
        return tasks

    def get_current_task_info(self) -> Optional[Dict[str, str]]:
        """
        Get current task info without advancing the index.

        Returns:
            Task info dict with 'domain', 'id', 'config_path', or None if no tasks
        """
        if not self.tasks or self.task_index >= len(self.tasks):
            return None
        return self.tasks[self.task_index]

    def get_next_task_info(self) -> Optional[Dict[str, str]]:
        """
        Get next task info and advance the index.

        Returns:
            Task info dict with 'domain', 'id', 'config_path', or None if no more tasks
        """
        if not self.tasks or self.task_index >= len(self.tasks):
            return None

        task_info = self.tasks[self.task_index]
        self.task_index += 1
        return task_info

    def load_task_config(self, task_info: Dict[str, str]) -> Dict[str, Any]:
        """
        Load full task configuration from file.

        Args:
            task_info: Task info dict containing 'config_path'

        Returns:
            Full task configuration dict
        """
        with open(task_info["config_path"], "r", encoding="utf-8") as f:
            return json.load(f)

    def has_more_tasks(self) -> bool:
        """Check if there are more tasks to process."""
        return self.task_index < len(self.tasks)

    def reset_index(self, index: int = 0):
        """Reset task iteration index."""
        self.task_index = index

    def set_index(self, index: int):
        """Set task iteration index, clamping to valid range."""
        if index >= len(self.tasks):
            logger.info("Task index out of range. Restarting from the first task.")
            self.task_index = 0
        else:
            self.task_index = index

    @property
    def total_tasks(self) -> int:
        """Get total number of tasks."""
        return len(self.tasks)

    @property
    def remaining_tasks(self) -> int:
        """Get number of remaining tasks."""
        return max(0, len(self.tasks) - self.task_index)
