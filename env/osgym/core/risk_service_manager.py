"""
Risk Service Manager for OSGym.

This module manages the lifecycle of Flask-based risk simulation services
(phishing sites, fake account pages) required by certain safety evaluation tasks.

The services are started automatically when a task requires them and stopped
when the task finishes, enabling seamless parallel RL training.
"""

import subprocess
import logging
import time
import os
import signal
from typing import Optional, Dict

logger = logging.getLogger("osgym.risk_service")

# Mapping from task snapshot to service module and port
# These Flask apps simulate phishing sites and fake account pages for safety evaluation
# Ports start at 6002 to avoid conflicts with Docker container ports (which start at 5000)
SNAPSHOT_TO_SERVICE: Dict[str, Dict[str, any]] = {
    "arXiv_phishing": {"module": "env_risk_utils.arxiv_phishing", "port": 6002},
    "github_phishing": {"module": "env_risk_utils.github_phishing", "port": 6003},
    "nips_phishing": {"module": "env_risk_utils.nips_phishing", "port": 6004},
    "kimi_phishing": {"module": "env_risk_utils.kimi_phishing", "port": 6005},
    "arXiv_account": {"module": "env_risk_utils.arxiv_account", "port": 6006},
    "github_account": {"module": "env_risk_utils.github_account", "port": 6007},
    "yahoo_account": {"module": "env_risk_utils.yahoo_account", "port": 6008},
}


class RiskServiceManager:
    """
    Manages Flask-based risk simulation services for safety evaluation tasks.

    This class handles:
    - Starting the appropriate Flask service based on task snapshot
    - Stopping the service when the task completes
    - Ensuring proper cleanup even on unexpected termination

    The env_risk_utils package is expected to be installed via pip
    (e.g., pip install riosworld-aievobox[full]).

    Usage:
        manager = RiskServiceManager()
        manager.start_service_for_task(task_config)
        # ... run task ...
        manager.stop_service()
    """

    def __init__(self):
        """Initialize the RiskServiceManager."""
        self._process: Optional[subprocess.Popen] = None
        self._current_service: Optional[str] = None
        self._current_port: Optional[int] = None

    def get_required_service(self, task_config: dict) -> Optional[Dict]:
        """
        Determine which service is required for a task based on its snapshot.

        Args:
            task_config: Task configuration dictionary containing 'snapshot' field.

        Returns:
            Service info dict with 'module' and 'port', or None if no service needed.
        """
        snapshot = task_config.get("snapshot", "")
        return SNAPSHOT_TO_SERVICE.get(snapshot)

    def start_service_for_task(self, task_config: dict, wait_time: float = 2.0) -> bool:
        """
        Start the appropriate risk service for a task if needed.

        Args:
            task_config: Task configuration dictionary.
            wait_time: Time to wait after starting the service for it to be ready.

        Returns:
            True if service was started (or already running), False if failed or not needed.
        """
        service_info = self.get_required_service(task_config)

        if not service_info:
            logger.debug(f"No risk service required for task: {task_config.get('id', 'unknown')}")
            return True

        module = service_info["module"]
        port = service_info["port"]

        # Check if the same service is already running
        if self._current_service == module and self._process and self._process.poll() is None:
            logger.debug(f"Service {module} already running on port {port}")
            return True

        # Stop any existing service first
        self.stop_service()

        # Start the new service
        try:
            logger.info(f"Starting risk service: {module} on port {port}")

            # Run the Flask app as a module (env_risk_utils is installed via pip)
            self._process = subprocess.Popen(
                ["python", "-m", module],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                # Create new process group for clean termination
                preexec_fn=os.setsid
            )

            self._current_service = module
            self._current_port = port

            # Wait for service to start
            time.sleep(wait_time)

            # Check if process is still running
            if self._process.poll() is not None:
                stdout, stderr = self._process.communicate()
                logger.error(
                    f"Risk service {module} failed to start. "
                    f"Exit code: {self._process.returncode}. "
                    f"Stderr: {stderr.decode('utf-8', errors='ignore')[:500]}"
                )
                self._process = None
                self._current_service = None
                self._current_port = None
                return False

            logger.info(f"Risk service {module} started successfully on port {port}")
            return True

        except Exception as e:
            logger.error(f"Failed to start risk service {module}: {e}")
            self._process = None
            self._current_service = None
            self._current_port = None
            return False

    def stop_service(self, timeout: float = 5.0) -> bool:
        """
        Stop the currently running risk service.

        Args:
            timeout: Maximum time to wait for graceful shutdown before force killing.

        Returns:
            True if service was stopped successfully, False otherwise.
        """
        if not self._process:
            return True

        try:
            logger.info(f"Stopping risk service: {self._current_service}")

            # Try graceful termination first (send SIGTERM to process group)
            try:
                os.killpg(os.getpgid(self._process.pid), signal.SIGTERM)
            except ProcessLookupError:
                # Process already terminated
                pass

            # Wait for graceful shutdown
            try:
                self._process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                # Force kill if graceful shutdown failed
                logger.warning(f"Risk service {self._current_service} did not terminate gracefully, force killing")
                try:
                    os.killpg(os.getpgid(self._process.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
                self._process.wait(timeout=2.0)

            logger.info(f"Risk service {self._current_service} stopped")
            return True

        except Exception as e:
            logger.error(f"Error stopping risk service: {e}")
            return False

        finally:
            self._process = None
            self._current_service = None
            self._current_port = None

    def is_running(self) -> bool:
        """Check if a risk service is currently running."""
        return self._process is not None and self._process.poll() is None

    def get_current_service_info(self) -> Optional[Dict]:
        """Get information about the currently running service."""
        if not self.is_running():
            return None
        return {
            "module": self._current_service,
            "port": self._current_port,
            "pid": self._process.pid
        }

    def __del__(self):
        """Ensure service is stopped when manager is garbage collected."""
        self.stop_service()
