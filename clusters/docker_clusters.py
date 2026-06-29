from __future__ import annotations

import asyncio
import json
import logging
import re
import secrets
import shlex
import string
import subprocess
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

from manager.binding_plan import BindingPlan
from .base import ClusterBackend

log = logging.getLogger("manager.docker_clusters")

_INVALID_NAME_CHARS = re.compile(r"[^a-zA-Z0-9_.-]+")
_DEFAULT_RUNNER_CONTAINER_PATH = "/tmp/safactory-openclaw-runner.mjs"


@dataclass(slots=True)
class DockerContainerRecord:
    key: str
    env_name: str
    image: str
    container_id: str
    container_name: str
    reuse_container: bool
    workdir: str
    run_command: str
    result_mode: str
    cleanup_command: str
    healthcheck_command: str
    max_runs: int
    run_count: int = 0


def _random_suffix(length: int = 8) -> str:
    alphabet = string.ascii_lowercase + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(max(1, length)))


def _safe_name(value: str, *, max_len: int = 48) -> str:
    normalized = _INVALID_NAME_CHARS.sub("-", str(value or "").strip()).strip("-._")
    if not normalized:
        normalized = "agent"
    return normalized[:max_len].strip("-._") or "agent"


def _as_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return shlex.split(value)
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if str(item).strip()]
    return [str(value)]


def _positive_float(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return max(1.0, number)


def _merge_dicts(*values: Any) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    for value in values:
        if isinstance(value, dict):
            merged.update(value)
    return merged


class DockerContainerBackend(ClusterBackend):
    """
    OpenClaw Docker backend.

    It owns Docker container lifecycle only. Agent runs are executed later with
    docker exec, so containers no longer need to expose an HTTP service.
    """

    def __init__(self, *, cluster_cfg: Dict[str, Any]) -> None:
        self._cluster_cfg = dict(cluster_cfg or {})
        self._docker_cfg = dict(self._cluster_cfg.get("docker", {}) or {})
        self._env_types = dict(self._cluster_cfg.get("env_types", {}) or {})

        self.docker_bin = str(self._docker_cfg.get("bin", "docker") or "docker")
        self._name_prefix = _safe_name(str(self._docker_cfg.get("name_prefix", "safactory-openclaw")))
        self._startup_concurrency = max(1, int(self._docker_cfg.get("startup_concurrency", 8) or 8))
        self._cleanup_container_on_finish = bool(self._docker_cfg.get("cleanup_container_on_finish", True))
        self._remove_on_close = bool(
            self._docker_cfg.get("remove_on_close", self._cleanup_container_on_finish)
        )
        self._pull_policy = str(self._docker_cfg.get("pull_policy", "never") or "never").strip().lower()
        self._reuse_policy = str(self._docker_cfg.get("reuse_policy", "explicit") or "explicit").strip().lower()
        self._default_reuse = bool(self._docker_cfg.get("default_reuse_container", False))
        self._default_workdir = str(self._docker_cfg.get("workdir", "") or "")
        self._idle_command = str(self._docker_cfg.get("idle_command", "tail -f /dev/null") or "tail -f /dev/null")
        self._default_run_command = str(
            self._docker_cfg.get("run_command", f"node {_DEFAULT_RUNNER_CONTAINER_PATH}")
            or f"node {_DEFAULT_RUNNER_CONTAINER_PATH}"
        )
        self._runner_container_path = str(
            self._docker_cfg.get("runner_container_path", _DEFAULT_RUNNER_CONTAINER_PATH)
            or _DEFAULT_RUNNER_CONTAINER_PATH
        )
        self._runner_host_path = self._resolve_runner_host_path(self._docker_cfg.get("runner_script_host_path"))
        self._default_cleanup_command = str(self._docker_cfg.get("cleanup_command", "") or "")
        self._default_healthcheck_command = str(self._docker_cfg.get("healthcheck_command", "") or "")
        self._default_max_runs = int(self._docker_cfg.get("max_runs_per_container", 0) or 0)
        self._command_timeout_s = _positive_float(self._docker_cfg.get("command_timeout_s"), 300.0)
        self._start_timeout_s = _positive_float(self._docker_cfg.get("start_timeout_s"), self._command_timeout_s)
        self._remove_timeout_s = _positive_float(self._docker_cfg.get("remove_timeout_s"), 120.0)
        self._lifecycle_timeout_s = _positive_float(self._docker_cfg.get("lifecycle_timeout_s"), 60.0)

        self._containers: Dict[str, DockerContainerRecord] = {}
        self._idle: Dict[str, Deque[str]] = {}
        self._lock = asyncio.Lock()
        self._startup_sem = asyncio.Semaphore(self._startup_concurrency)

    async def start(self, plan: BindingPlan) -> None:
        if not plan.env_to_image:
            log.warning("empty Docker binding plan")
            return
        await self.validate_images(plan)
        log.info("OpenClaw Docker backend ready for %d agent image(s)", len(plan.env_to_image))

    async def validate_images(self, plan: BindingPlan) -> None:
        missing_local: List[str] = []
        pull_failures: List[str] = []

        for image in sorted(plan.images_needed):
            image = str(image or "").strip()
            if not image:
                continue

            env_name = str(plan.image_to_env.get(image) or "")
            docker_cfg = self._docker_cfg_for_env(env_name)
            pull_policy = self._resolve_pull_policy(docker_cfg)
            if pull_policy == "always":
                result = await self._run_command([self.docker_bin, "pull", image], check=False)
                if result.returncode != 0:
                    pull_failures.append(
                        f"{image} (env={env_name or 'unknown'}, stderr={self._tail(result.stderr)})"
                    )
                continue

            result = await self._run_command([self.docker_bin, "image", "inspect", image], check=False)
            if result.returncode != 0:
                missing_local.append(
                    f"{image} (env={env_name or 'unknown'}, stderr={self._tail(result.stderr)})"
                )

        errors: List[str] = []
        if missing_local:
            errors.append(
                "Required Docker image(s) are not available locally: "
                + "; ".join(missing_local)
                + ". Build or pull the image first, or update env_image in the agent YAML. "
                + "If the image is private, run docker login. To let the manager pull before startup, "
                + "set cluster.docker.pull_policy: always."
            )
        if pull_failures:
            errors.append(
                "Required Docker image pull failed: "
                + "; ".join(pull_failures)
                + ". Verify the image name exists and that Docker is logged in if the registry is private."
            )
        if errors:
            raise RuntimeError(" ".join(errors))

    async def acquire(self, *, env_name: str, image: str) -> DockerContainerRecord:
        key = self._container_key(env_name, image)
        async with self._lock:
            idle = self._idle.get(key)
            while idle:
                container_id = idle.popleft()
                record = self._containers.get(container_id)
                if record is not None:
                    return record

        async with self._startup_sem:
            return await self._start_container(env_name=env_name, image=image)

    async def release(self, container_id: str, *, succeeded: bool) -> None:
        record = self._containers.get(str(container_id or ""))
        if record is None:
            return

        record.run_count += 1
        keep = bool(succeeded and record.reuse_container)
        if keep and record.cleanup_command:
            keep = await self._exec_lifecycle_command(record, record.cleanup_command, "cleanup")
        if keep and record.healthcheck_command:
            keep = await self._exec_lifecycle_command(record, record.healthcheck_command, "healthcheck")
        if keep and record.max_runs > 0 and record.run_count >= record.max_runs:
            keep = False

        if keep:
            async with self._lock:
                self._idle.setdefault(record.key, deque()).append(record.container_id)
            return

        await self.remove(container_id)

    async def remove(self, container_id: str) -> None:
        record = self._containers.pop(str(container_id or ""), None)
        if record is None:
            return
        async with self._lock:
            idle = self._idle.get(record.key)
            if idle is not None:
                self._idle[record.key] = deque(item for item in idle if item != record.container_id)

        ident = record.container_id or record.container_name
        if not ident:
            return
        if not self._cleanup_container_on_finish:
            log.info(
                "preserving Docker container after run: agent=%s container=%s id=%s",
                record.env_name,
                record.container_name,
                record.container_id,
            )
            return
        result = await self._run_command(
            [self.docker_bin, "rm", "-f", ident],
            check=False,
            timeout_s=self._remove_timeout_s,
        )
        if result.returncode != 0:
            log.warning("docker rm failed for %s: %s", ident, (result.stderr or "").strip())

    async def close(self) -> None:
        if not self._cleanup_container_on_finish:
            if self._containers:
                log.info("preserving %d Docker container(s) on backend close", len(self._containers))
            self._containers.clear()
            self._idle.clear()
            return
        for container_id in list(self._containers):
            await self.remove(container_id)

    async def _install_runner_script_if_needed(
        self,
        record: DockerContainerRecord,
        docker_cfg: Dict[str, Any],
    ) -> None:
        install_runner = docker_cfg.get("install_runner_script")
        if install_runner is None:
            install_runner = self._runner_container_path in record.run_command
        if not install_runner:
            return

        runner_host_path = self._resolve_runner_host_path(docker_cfg.get("runner_script_host_path"))
        if runner_host_path is None:
            runner_host_path = self._runner_host_path
        if runner_host_path is None or not runner_host_path.is_file():
            raise RuntimeError(f"OpenClaw runner script not found: {runner_host_path}")

        await self._run_required(
            [self.docker_bin, "cp", str(runner_host_path), f"{record.container_id}:{self._runner_container_path}"],
            action=f"install Safactory OpenClaw runner in {record.container_name}",
            timeout_s=self._start_timeout_s,
        )
        log.debug(
            "installed Safactory OpenClaw runner into %s:%s",
            record.container_name,
            self._runner_container_path,
        )

    async def _start_container(self, *, env_name: str, image: str) -> DockerContainerRecord:
        docker_cfg = self._docker_cfg_for_env(env_name)
        pull_policy = self._resolve_pull_policy(docker_cfg)
        if pull_policy == "always":
            await self._run_required([self.docker_bin, "pull", image], action=f"pull image {image}")

        container_name = f"{self._name_prefix}-{_safe_name(env_name)}-{_random_suffix(6)}"
        workdir = str(docker_cfg.get("workdir", self._default_workdir) or "").strip()
        idle_command = str(docker_cfg.get("idle_command", self._idle_command) or self._idle_command)
        resolved_run_command = str(
            docker_cfg.get("run_command", self._default_run_command) or self._default_run_command
        )
        resolved_result_mode = str(
            docker_cfg.get("result_mode", docker_cfg.get("run_result_mode", "json")) or "json"
        ).strip().lower()
        resolved_cleanup_command = str(docker_cfg.get("cleanup_command", self._default_cleanup_command) or "")
        resolved_healthcheck_command = str(
            docker_cfg.get("healthcheck_command", self._default_healthcheck_command) or ""
        )
        resolved_max_runs = int(docker_cfg.get("max_runs_per_container", self._default_max_runs) or 0)
        reuse_container = self._resolve_reuse(docker_cfg)
        install_runner = docker_cfg.get("install_runner_script")
        if install_runner is None:
            install_runner = self._runner_container_path in resolved_run_command
        run_cmd = self._build_run_command(
            image=image,
            name=container_name,
            docker_cfg=docker_cfg,
            idle_command=idle_command,
            workdir=workdir,
        )
        log.info(
            "Docker container create command: agent=%s image=%s command=%s",
            env_name,
            image,
            self._cmd_for_log(run_cmd),
        )
        log.info(
            "Docker container create params: agent=%s params=%s",
            env_name,
            self._json_for_log(
                {
                    "agent": env_name,
                    "image": image,
                    "container_name": container_name,
                    "docker_bin": self.docker_bin,
                    "pull_policy": pull_policy,
                    "cleanup_container_on_finish": self._cleanup_container_on_finish,
                    "remove_on_close": self._remove_on_close,
                    "network": str(docker_cfg.get("network", "") or "").strip(),
                    "platform": str(docker_cfg.get("platform", "") or "").strip(),
                    "env": _merge_dicts(docker_cfg.get("env")),
                    "volumes": docker_cfg.get("volumes", []) or [],
                    "extra_args": _as_list(docker_cfg.get("extra_args")),
                    "workdir": workdir,
                    "idle_command": idle_command,
                    "run_command": resolved_run_command,
                    "result_mode": resolved_result_mode,
                    "cleanup_command": resolved_cleanup_command,
                    "healthcheck_command": resolved_healthcheck_command,
                    "reuse_container": reuse_container,
                    "max_runs_per_container": resolved_max_runs,
                    "runner_container_path": self._runner_container_path,
                    "install_runner_script": bool(install_runner),
                }
            ),
        )
        result = await self._run_required(
            run_cmd,
            action=f"start OpenClaw container for {env_name}",
            timeout_s=self._start_timeout_s,
        )
        container_id = result.stdout.strip().splitlines()[-1].strip()
        if not container_id:
            raise RuntimeError(f"docker run returned empty container id for agent={env_name} image={image}")

        record = DockerContainerRecord(
            key=self._container_key(env_name, image),
            env_name=env_name,
            image=image,
            container_id=container_id,
            container_name=container_name,
            reuse_container=reuse_container,
            workdir=workdir,
            run_command=resolved_run_command,
            result_mode=resolved_result_mode,
            cleanup_command=resolved_cleanup_command,
            healthcheck_command=resolved_healthcheck_command,
            max_runs=resolved_max_runs,
        )
        try:
            await self._install_runner_script_if_needed(record, docker_cfg)
        except Exception:
            await self._run_command(
                [self.docker_bin, "rm", "-f", container_id],
                check=False,
                timeout_s=self._remove_timeout_s,
            )
            raise
        self._containers[container_id] = record
        log.info(
            "OpenClaw container ready: agent=%s image=%s container=%s reuse=%s",
            env_name,
            image,
            container_name,
            record.reuse_container,
        )
        return record

    def _resolve_reuse(self, docker_cfg: Dict[str, Any]) -> bool:
        if self._reuse_policy == "always":
            return True
        if self._reuse_policy == "never":
            return False
        return bool(docker_cfg.get("reuse_container", self._default_reuse))

    def _docker_cfg_for_env(self, env_name: str) -> Dict[str, Any]:
        env_cfg = dict(self._env_types.get(env_name, {}) or {})
        return _merge_dicts(self._docker_cfg, env_cfg.get("docker"))

    def _resolve_pull_policy(self, docker_cfg: Dict[str, Any]) -> str:
        return str(docker_cfg.get("pull_policy", self._pull_policy) or self._pull_policy).strip().lower()

    def _build_run_command(
        self,
        *,
        image: str,
        name: str,
        docker_cfg: Dict[str, Any],
        idle_command: str,
        workdir: str,
    ) -> List[str]:
        cmd = [self.docker_bin, "run", "-d"]
        platform = str(docker_cfg.get("platform", "") or "").strip()
        if platform:
            cmd.extend(["--platform", platform])
        if self._remove_on_close:
            cmd.append("--rm")
        cmd.extend(["--name", name])

        if workdir:
            cmd.extend(["-w", workdir])

        network = str(docker_cfg.get("network", "") or "").strip()
        if network:
            cmd.extend(["--network", network])

        for key, value in _merge_dicts(docker_cfg.get("env")).items():
            cmd.extend(["-e", f"{key}={value}"])

        volumes = docker_cfg.get("volumes", docker_cfg.get("mounts", [])) or []
        for volume in volumes:
            spec = self._volume_spec(volume)
            if spec:
                cmd.extend(["-v", spec])

        for arg in _as_list(docker_cfg.get("extra_args")):
            cmd.append(arg)

        cmd.extend([image, "sh", "-lc", idle_command])
        return cmd

    async def _exec_lifecycle_command(self, record: DockerContainerRecord, command: str, action: str) -> bool:
        cmd = [self.docker_bin, "exec"]
        if record.workdir:
            cmd.extend(["-w", record.workdir])
        cmd.extend([record.container_id, "sh", "-lc", command])
        result = await self._run_command(cmd, check=False, timeout_s=self._lifecycle_timeout_s)
        if result.returncode == 0:
            return True
        log.warning(
            "OpenClaw container %s failed %s command: stdout=%r stderr=%r",
            record.container_name,
            action,
            (result.stdout or "").strip()[-1000:],
            (result.stderr or "").strip()[-1000:],
        )
        return False

    async def _run_required(
        self,
        cmd: List[str],
        *,
        action: str,
        timeout_s: float | None = None,
    ) -> subprocess.CompletedProcess[str]:
        try:
            return await self._run_command(cmd, check=True, timeout_s=timeout_s)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"{action} timed out after {float(exc.timeout or 0):.1f}s: cmd={cmd!r}") from exc
        except subprocess.CalledProcessError as exc:
            stdout = (exc.stdout or "").strip()
            stderr = (exc.stderr or "").strip()
            raise RuntimeError(f"{action} failed: cmd={cmd!r} stdout={stdout!r} stderr={stderr!r}") from exc

    async def _run_command(
        self,
        cmd: List[str],
        *,
        check: bool,
        timeout_s: float | None = None,
    ) -> subprocess.CompletedProcess[str]:
        timeout = _positive_float(timeout_s, self._command_timeout_s)

        def _run() -> subprocess.CompletedProcess[str]:
            return subprocess.run(cmd, check=check, capture_output=True, text=True, timeout=timeout)

        return await asyncio.to_thread(_run)

    @staticmethod
    def _cmd_for_log(cmd: List[str]) -> str:
        return shlex.join([str(part) for part in cmd])

    @staticmethod
    def _json_for_log(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, default=str)

    @staticmethod
    def _tail(value: str, limit: int = 500) -> str:
        return (value or "").strip()[-int(limit):]

    @staticmethod
    def _container_key(env_name: str, image: str) -> str:
        return f"{str(env_name)}\0{str(image)}"

    @staticmethod
    def _resolve_runner_host_path(value: Any) -> Optional[Path]:
        if value is None or str(value).strip() == "":
            return Path(__file__).resolve().parents[1] / "env" / "openclaw" / "runner.mjs"
        path = Path(str(value)).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        return path

    @staticmethod
    def _volume_spec(volume: Any) -> str:
        if isinstance(volume, str):
            return volume.strip()
        if not isinstance(volume, dict):
            return ""
        source = volume.get("source") or volume.get("hostPath") or volume.get("host_path")
        target = volume.get("target") or volume.get("containerPath") or volume.get("container_path")
        if not source or not target:
            return ""
        mode = str(volume.get("mode") or "").strip()
        if not mode and bool(volume.get("read_only", volume.get("readonly", False))):
            mode = "ro"
        spec = f"{source}:{target}"
        if mode:
            spec = f"{spec}:{mode}"
        return spec
