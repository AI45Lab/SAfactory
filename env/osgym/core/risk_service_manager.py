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
import socket
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

    def _is_port_open(self, port: int, host: str = "127.0.0.1", timeout: float = 0.5) -> bool:
        """Check whether the risk service port is accepting TCP connections."""
        try:
            with socket.create_connection((host, port), timeout=timeout):
                return True
        except OSError:
            return False

    def _cleanup_failed_start(self):
        """Reset process state after a failed launch attempt."""
        self._process = None
        self._current_service = None
        self._current_port = None

    def _wait_for_service_ready(self, module: str, port: int, wait_time: float) -> bool:
        """
        Wait until the risk service is reachable.

        Flask debug mode may spawn a child process and let the original parent exit,
        so probing the listening port is more reliable than checking only poll().
        """
        deadline = time.time() + wait_time
        last_returncode = None

        while time.time() < deadline:
            if self._is_port_open(port):
                return True

            if self._process is not None:
                last_returncode = self._process.poll()

            time.sleep(0.1)

        if self._is_port_open(port):
            return True

        if self._process is not None and last_returncode is not None:
            stdout, stderr = self._process.communicate()
            logger.error(
                f"Risk service {module} failed to start. "
                f"Exit code: {self._process.returncode}. "
                f"Stdout: {stdout.decode('utf-8', errors='ignore')[:300]} "
                f"Stderr: {stderr.decode('utf-8', errors='ignore')[:500]}"
            )
        else:
            logger.error(
                f"Risk service {module} did not become ready on port {port} within {wait_time:.1f}s"
            )

        self._cleanup_failed_start()
        return False

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
        if self._current_service == module and self._current_port == port and self._is_port_open(port):
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

            if not self._wait_for_service_ready(module, port, wait_time):
                return False

            logger.info(f"Risk service {module} started successfully on port {port}")
            return True

        except Exception as e:
            logger.error(f"Failed to start risk service {module}: {e}")
            self._cleanup_failed_start()
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
        return self._current_port is not None and self._is_port_open(self._current_port)

    def get_current_service_info(self) -> Optional[Dict]:
        """Get information about the currently running service."""
        if not self.is_running():
            return None
        return {
            "module": self._current_service,
            "port": self._current_port,
            "pid": self._process.pid if self._process is not None else None
        }

    def __del__(self):
        """Ensure service is stopped when manager is garbage collected."""
        self.stop_service()
