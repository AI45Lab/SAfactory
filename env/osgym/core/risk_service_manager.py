"""Coordinate the local web services used by OSGym safety tasks.

Ray runs every OSGym environment in a separate worker process, while all of
those workers share the pod network namespace.  The safety task URLs use fixed
ports, so one service per task would make workers race for the same port.  This
module instead treats each risk service as a pod-local shared resource and
coordinates its lifetime with a file lock and a small state file in ``/tmp``.
"""

from __future__ import annotations

import fcntl
from http.client import HTTPConnection, HTTPException
import json
import logging
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional
import uuid


logger = logging.getLogger("osgym.risk_service")

HEALTH_PATH = "/__osgym_health__"
PROJECT_ROOT = Path(__file__).resolve().parents[3]
RUNTIME_DIR = Path(
    os.environ.get("OSGYM_RISK_SERVICE_RUNTIME_DIR", "/tmp/osgym-risk-services")
)

# The browser tasks address these ports through Docker's 172.17.0.1 bridge.
# The Flask modules in this repository are imported by risk_service_server,
# which overrides their standalone development-server ports.
SNAPSHOT_TO_SERVICE: Dict[str, Dict[str, Any]] = {
    "arXiv_phishing": {
        "module": "env.osgym.env_risk_utils.arxiv_phishing",
        "port": 6002,
    },
    "github_phishing": {
        "module": "env.osgym.env_risk_utils.github_phishing",
        "port": 6003,
    },
    "nips_phishing": {
        "module": "env.osgym.env_risk_utils.nips_phishing",
        "port": 6004,
    },
    "kimi_phishing": {
        "module": "env.osgym.env_risk_utils.kimi_phishing",
        "port": 6005,
    },
    "arXiv_account": {
        "module": "env.osgym.env_risk_utils.arxiv_account",
        "port": 6006,
    },
    "github_account": {
        "module": "env.osgym.env_risk_utils.github_account",
        "port": 6007,
    },
    "yahoo_account": {
        "module": "env.osgym.env_risk_utils.yahoo_account",
        "port": 6008,
    },
}


class RiskServiceManager:
    """Acquire and release pod-local risk services for one OSGym instance."""

    def __init__(self) -> None:
        self._client_id = f"{os.getpid()}-{uuid.uuid4().hex}"
        self._current_service: Optional[str] = None
        self._current_module: Optional[str] = None
        self._current_port: Optional[int] = None
        self._process: Optional[subprocess.Popen] = None

    @staticmethod
    def get_required_service(task_config: dict) -> Optional[Dict[str, Any]]:
        """Return the service required by a task snapshot, if any."""
        return SNAPSHOT_TO_SERVICE.get(str(task_config.get("snapshot", "")))

    @staticmethod
    def _paths(port: int) -> tuple[Path, Path, Path]:
        RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
        return (
            RUNTIME_DIR / f"{port}.lock",
            RUNTIME_DIR / f"{port}.json",
            RUNTIME_DIR / f"{port}.log",
        )

    @contextmanager
    def _service_lock(self, port: int) -> Iterator[tuple[Path, Path]]:
        lock_path, state_path, log_path = self._paths(port)
        with lock_path.open("a+") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield state_path, log_path
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def _read_state(state_path: Path) -> Dict[str, Any]:
        try:
            value = json.loads(state_path.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else {}
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return {}

    @staticmethod
    def _write_state(state_path: Path, state: Dict[str, Any]) -> None:
        state_path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            dir=str(state_path.parent), prefix=f".{state_path.name}.", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as tmp_file:
                json.dump(state, tmp_file, sort_keys=True)
                tmp_file.flush()
                os.fsync(tmp_file.fileno())
            os.replace(tmp_name, state_path)
        finally:
            try:
                os.unlink(tmp_name)
            except FileNotFoundError:
                pass

    @staticmethod
    def _pid_is_alive(pid: Any) -> bool:
        try:
            pid = int(pid)
        except (TypeError, ValueError):
            return False
        if pid <= 0:
            return False
        try:
            # ``kill(pid, 0)`` also succeeds for an unreaped zombie.  Treat a
            # zombie as stopped so shutdown does not wait for the full grace
            # period before trying an ineffective SIGKILL.
            process_state = Path(f"/proc/{pid}/stat").read_text(
                encoding="utf-8"
            ).split()[2]
            if process_state == "Z":
                return False
        except (FileNotFoundError, IndexError, OSError):
            pass
        try:
            os.kill(pid, 0)
            return True
        except PermissionError:
            return True
        except ProcessLookupError:
            return False

    @classmethod
    def _prune_clients(cls, state: Dict[str, Any]) -> Dict[str, int]:
        clients = state.get("clients", {})
        if not isinstance(clients, dict):
            return {}
        return {
            str(client_id): int(pid)
            for client_id, pid in clients.items()
            if cls._pid_is_alive(pid)
        }

    @staticmethod
    def _is_port_open(
        port: int, host: str = "127.0.0.1", timeout: float = 0.5
    ) -> bool:
        try:
            with socket.create_connection((host, port), timeout=timeout):
                return True
        except OSError:
            return False

    @staticmethod
    def _probe_service(
        service: str, port: int, timeout: float = 0.5
    ) -> Optional[Dict[str, Any]]:
        connection = HTTPConnection("127.0.0.1", port, timeout=timeout)
        try:
            connection.request("GET", HEALTH_PATH)
            response = connection.getresponse()
            if response.status != 200:
                return None
            payload = json.loads(response.read().decode("utf-8"))
        except (OSError, HTTPException, ValueError, json.JSONDecodeError):
            return None
        finally:
            connection.close()
        if not isinstance(payload, dict) or payload.get("service") != service:
            return None
        return payload

    @classmethod
    def _wait_for_service_ready(
        cls,
        service: str,
        port: int,
        process: subprocess.Popen,
        wait_time: float,
    ) -> Optional[Dict[str, Any]]:
        deadline = time.monotonic() + wait_time
        while time.monotonic() < deadline:
            health = cls._probe_service(service, port)
            if health is not None:
                return health
            if process.poll() is not None:
                return None
            time.sleep(0.1)
        return cls._probe_service(service, port)

    @staticmethod
    def _terminate_process(pid: Any, timeout: float = 5.0) -> bool:
        try:
            pid = int(pid)
        except (TypeError, ValueError):
            return True
        if pid <= 0:
            return True

        try:
            cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().replace(
                b"\0", b" "
            )
        except OSError:
            return True
        if b"env.osgym.core.risk_service_server" not in cmdline:
            logger.error(
                "Refusing to terminate unmanaged process pid=%s cmdline=%r",
                pid,
                cmdline[:300],
            )
            return False

        try:
            process_group = os.getpgid(pid)
            os.killpg(process_group, signal.SIGTERM)
        except ProcessLookupError:
            return True
        except OSError:
            logger.warning("Failed to terminate risk service pid=%s", pid, exc_info=True)
            return False

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not RiskServiceManager._pid_is_alive(pid):
                return True
            time.sleep(0.1)

        try:
            os.killpg(process_group, signal.SIGKILL)
        except ProcessLookupError:
            return True
        except OSError:
            logger.warning("Failed to kill risk service pid=%s", pid, exc_info=True)
            return False
        return True

    @staticmethod
    def _tail_log(log_path: Path, max_bytes: int = 2000) -> str:
        try:
            with log_path.open("rb") as log_file:
                log_file.seek(0, os.SEEK_END)
                size = log_file.tell()
                log_file.seek(max(0, size - max_bytes))
                return log_file.read().decode("utf-8", errors="replace").strip()
        except OSError:
            return ""

    def _set_current(
        self,
        service: str,
        module: str,
        port: int,
        process: Optional[subprocess.Popen] = None,
    ) -> None:
        self._current_service = service
        self._current_module = module
        self._current_port = port
        self._process = process

    def _clear_current(self) -> None:
        self._current_service = None
        self._current_module = None
        self._current_port = None
        self._process = None

    def start_service_for_task(
        self, task_config: dict, wait_time: float = 10.0
    ) -> bool:
        """Acquire the shared service required by ``task_config``.

        The call is serialized across Ray workers.  A healthy existing service
        is reused; only the first caller starts a process.
        """
        service_info = self.get_required_service(task_config)
        if not service_info:
            if self._current_port is not None:
                self.stop_service()
            logger.debug(
                "No risk service required for task: %s",
                task_config.get("id", "unknown"),
            )
            return True

        service = str(task_config.get("snapshot", ""))
        module = str(service_info["module"])
        port = int(service_info["port"])

        if (
            self._current_service == service
            and self._current_port == port
            and self._probe_service(service, port) is not None
        ):
            return True

        if self._current_port is not None:
            self.stop_service()

        with self._service_lock(port) as (state_path, log_path):
            state = self._read_state(state_path)
            clients = self._prune_clients(state)
            health = self._probe_service(service, port)

            if health is not None:
                server_pid = health.get("pid")
                clients[self._client_id] = os.getpid()
                self._write_state(
                    state_path,
                    {
                        "service": service,
                        "module": module,
                        "port": port,
                        "server_pid": server_pid,
                        "instance_id": health.get("instance_id"),
                        "clients": clients,
                    },
                )
                self._set_current(service, module, port)
                logger.info(
                    "Reusing shared risk service %s on port %s (clients=%s)",
                    service,
                    port,
                    len(clients),
                )
                return True

            if self._is_port_open(port):
                logger.error(
                    "Port %s is occupied by a process that is not the expected "
                    "OSGym risk service %s",
                    port,
                    service,
                )
                return False

            stale_pid = state.get("server_pid")
            if stale_pid and self._pid_is_alive(stale_pid):
                logger.warning(
                    "Restarting unhealthy risk service %s with %s live clients",
                    service,
                    len(clients),
                )
                if not self._terminate_process(stale_pid):
                    return False

            logger.info("Starting shared risk service %s on port %s", service, port)
            log_file = log_path.open("ab", buffering=0)
            try:
                process = subprocess.Popen(
                    [
                        sys.executable,
                        "-B",
                        "-m",
                        "env.osgym.core.risk_service_server",
                        "--service",
                        service,
                        "--module",
                        module,
                        "--port",
                        str(port),
                    ],
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    close_fds=True,
                    cwd=str(PROJECT_ROOT),
                )
            except Exception:
                logger.error(
                    "Failed to launch risk service %s", service, exc_info=True
                )
                return False
            finally:
                log_file.close()

            health = self._wait_for_service_ready(
                service, port, process, wait_time
            )
            if health is None:
                return_code = process.poll()
                self._terminate_process(process.pid)
                try:
                    state_path.unlink()
                except FileNotFoundError:
                    pass
                logger.error(
                    "Risk service %s failed to become ready on port %s "
                    "within %.1fs (returncode=%s). Log tail: %s",
                    service,
                    port,
                    wait_time,
                    return_code,
                    self._tail_log(log_path),
                )
                return False

            clients[self._client_id] = os.getpid()
            self._write_state(
                state_path,
                {
                    "service": service,
                    "module": module,
                    "port": port,
                    "server_pid": health.get("pid", process.pid),
                    "instance_id": health.get("instance_id"),
                    "clients": clients,
                },
            )
            self._set_current(service, module, port, process)
            logger.info("Shared risk service %s is ready on port %s", service, port)
            return True

    def stop_service(self, timeout: float = 5.0) -> bool:
        """Release this manager's reference and stop the last-user service."""
        service = self._current_service
        port = self._current_port
        if service is None or port is None:
            return True

        stopped = True
        with self._service_lock(port) as (state_path, _):
            state = self._read_state(state_path)
            clients = self._prune_clients(state)
            clients.pop(self._client_id, None)

            if clients:
                state["clients"] = clients
                self._write_state(state_path, state)
                logger.info(
                    "Released shared risk service %s (remaining_clients=%s)",
                    service,
                    len(clients),
                )
            else:
                server_pid = state.get("server_pid")
                health = self._probe_service(service, port)
                if health is not None and health.get("pid") is not None:
                    server_pid = health["pid"]
                if server_pid is not None:
                    stopped = self._terminate_process(server_pid, timeout=timeout)
                try:
                    state_path.unlink()
                except FileNotFoundError:
                    pass
                if stopped:
                    logger.info("Stopped last-user risk service %s", service)
                else:
                    logger.error(
                        "Failed to stop last-user risk service %s", service
                    )

        self._clear_current()
        return stopped

    def is_running(self) -> bool:
        """Return whether this manager's acquired service is healthy."""
        return (
            self._current_service is not None
            and self._current_port is not None
            and self._probe_service(
                self._current_service, self._current_port
            )
            is not None
        )

    def get_current_service_info(self) -> Optional[Dict[str, Any]]:
        """Return health and ownership information for the acquired service."""
        if self._current_service is None or self._current_port is None:
            return None
        health = self._probe_service(self._current_service, self._current_port)
        if health is None:
            return None
        return {
            "service": self._current_service,
            "module": self._current_module,
            "port": self._current_port,
            "pid": health.get("pid"),
            "instance_id": health.get("instance_id"),
        }
