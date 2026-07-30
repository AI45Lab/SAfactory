from __future__ import annotations

import asyncio
import base64
import inspect
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import urlsplit

from core.perf_trace import PerfTrace
from manager.binding_plan import BindingPlan
from manager.types import PoolEntry, SimulationAgentLease, SimulationStartRequest

from .base import ClusterBackend

log = logging.getLogger("clusters.rjob_cluster")

RJOB_RUNNING_STATUSES = {"Created", "Pending", "Starting", "Running", "Inqueue", "Restarting", "Killing", "Deleting"}
RJOB_SUCCEEDED_STATUSES = {"Succeeded"}
RJOB_FAILED_STATUSES = {"Failed", "Stopped", "Killed"}

_DEFAULT_RUNNER_CONTAINER_PATH = "/tmp/safactory-openclaw-runner.mjs"
_DEFAULT_RUN_COMMAND = f"node {_DEFAULT_RUNNER_CONTAINER_PATH}"
_INVALID_NAME_CHARS = re.compile(r"[^a-z0-9.-]+")
_MAX_RJOB_NAME_LEN = 49


class RJobClusterBackend(ClusterBackend):
    """RJob cluster adapter used by leases and per-episode job execution."""

    runtime = "rjob"

    def __init__(self, *, cluster_cfg: Dict[str, Any] | None = None) -> None:
        self._cluster_cfg = dict(cluster_cfg or {})
        self._rjob_cfg = dict(self._cluster_cfg.get("rjob", {}) or {})
        self._env_types = dict(self._cluster_cfg.get("env_types", {}) or {})
        self.startup_concurrency = max(1, int(self._rjob_cfg.get("submit_concurrency", 0) or 1))
        self._clients: Dict[tuple[Any, ...], Any] = {}

    async def start(self, plan: BindingPlan) -> None:
        if not plan.env_to_image:
            log.warning("empty RJob binding plan")
            return
        for env_name in sorted(plan.env_to_image):
            docker_cfg = self._docker_cfg_for_env(env_name)
            rjob_cfg = self._rjob_cfg_for_env(env_name)
            self._validate_mounts(env_name, docker_cfg, rjob_cfg)
        log.debug("RJob cluster backend ready for %d agent image(s)", len(plan.env_to_image))

    async def allocate(
        self,
        *,
        row_id: Any,
        env_name: str,
        env_id: str,
        image: str,
        env_params: Dict[str, Any],
        group_id: str,
    ) -> PoolEntry:
        docker_cfg = self._docker_cfg_for_env(env_name)
        rjob_cfg = self._rjob_cfg_for_env(env_name)
        self._validate_mounts(env_name, docker_cfg, rjob_cfg)

        run_command = str(
            rjob_cfg.get("run_command") or docker_cfg.get("run_command") or _DEFAULT_RUN_COMMAND
        ).strip()
        if not run_command:
            raise RuntimeError(f"RJob lease for {env_name}/{env_id} is missing run_command")

        workdir = str(rjob_cfg.get("workdir") or docker_cfg.get("workdir") or "").strip()
        result_mode = str(
            rjob_cfg.get("result_mode") or docker_cfg.get("result_mode") or docker_cfg.get("run_result_mode") or "json"
        ).strip().lower()

        runtime_config = dict(rjob_cfg)
        runtime_config["env"] = merge_env_dicts(docker_cfg.get("env"), rjob_cfg.get("env"))
        runtime_config["docker_volumes"] = list(docker_cfg.get("volumes", []) or [])
        runtime_config["embedded_files"] = _merge_embedded_files(docker_cfg, rjob_cfg)

        pending_name = f"rjob-pending-{str(env_id).replace('-', '')[:12]}"
        return PoolEntry(
            env_name=str(env_name),
            env_id=str(env_id),
            row_id=row_id,
            image=str(image),
            job_name=pending_name,
            env_params=dict(env_params or {}),
            group_id=str(group_id or ""),
            status="ready",
            runtime=self.runtime,
            runtime_config=runtime_config,
            resource_id="",
            resource_name=pending_name,
            workdir=workdir,
            run_command=run_command,
            result_mode=result_mode,
            reuse_container=False,
        )

    async def release(self, entry: PoolEntry, *, succeeded: bool) -> None:
        return

    async def remove(self, entry: PoolEntry) -> None:
        return

    async def close(self) -> None:
        return

    def build_job(
        self,
        *,
        cfg: Dict[str, Any],
        lease: SimulationAgentLease,
        request: SimulationStartRequest,
        rjob_name: str,
        env: Dict[str, str],
    ) -> Any:
        symbols = self.symbols()
        resources_cfg = dict(cfg.get("resources") or cfg.get("default_resources") or {})
        requests_cfg = dict(cfg.get("requests") or {})

        resources = self._build_resources(symbols, resources_cfg)
        requests = self._build_resources(symbols, requests_cfg) if requests_cfg else None
        image_pull_policy = self._coerce_enum(symbols.get("ImagePullPolicy"), cfg.get("image_pull_policy"))
        run_command = self._command_with_embedded_files(cfg, lease.run_command)

        container_kwargs: Dict[str, Any] = {
            "name": str(cfg.get("container_name") or "main"),
            "image": lease.image,
            "command": ["sh", "-lc", run_command],
            "environments": env,
        }
        if lease.workdir:
            container_kwargs["working_dir"] = lease.workdir
        if resources is not None:
            container_kwargs["resources"] = resources
        if requests is not None:
            container_kwargs["requests"] = requests
        if image_pull_policy is not None:
            container_kwargs["image_pull_policy"] = image_pull_policy
        if "privileged" in cfg:
            container_kwargs["privileged"] = bool(cfg.get("privileged"))

        container = _make(symbols["Container"], **container_kwargs)
        template_kwargs: Dict[str, Any] = {
            "containers": [container],
            "environments": env,
        }
        if lease.workdir:
            template_kwargs["working_dir"] = lease.workdir
        if cfg.get("before_script"):
            template_kwargs["before_script"] = list(cfg.get("before_script") or [])
        if cfg.get("user"):
            template_kwargs["user"] = str(cfg.get("user"))
        if cfg.get("mount"):
            template_kwargs["mount"] = list(cfg.get("mount") or [])
        template = _make(symbols["Template"], **template_kwargs)

        task_kwargs: Dict[str, Any] = {
            "replicas": int(cfg.get("replicas", 1) or 1),
            "template": template,
        }
        restart_policy = self._coerce_enum(symbols.get("RestartPolicy"), cfg.get("restart_policy") or "Never")
        if restart_policy is not None:
            task_kwargs["restart_policy"] = restart_policy
        private_machine = self._coerce_enum(symbols.get("PrivateMachine"), cfg.get("private_machine"))
        if private_machine is not None:
            task_kwargs["private_machine"] = private_machine
        for key in (
            "gang_start",
            "host_network",
            "enable_sshd",
            "share_host_shm",
            "daemon",
            "termination_grace_period_seconds",
            "local_storage_in_mb",
            "max_wait_duration",
            "max_running_duration",
        ):
            if key in cfg:
                task_kwargs[key] = cfg[key]
        if cfg.get("mount_config"):
            task_kwargs["mount_config"] = list(cfg.get("mount_config") or [])
        if cfg.get("depends_on"):
            task_kwargs["depends_on"] = list(cfg.get("depends_on") or [])
        if cfg.get("labels"):
            task_kwargs["labels"] = dict(cfg.get("labels") or {})
        task = _make(symbols["Task"], **task_kwargs)

        spec_kwargs: Dict[str, Any] = {"tasks": {str(cfg.get("task_name") or "main"): task}}
        for key in ("preemptible", "gang_start", "host_network", "enable_sshd", "topo_group"):
            if key in cfg:
                spec_kwargs[key] = cfg[key]
        if cfg.get("auto_delete_duration"):
            spec_kwargs["auto_delete_duration"] = str(cfg.get("auto_delete_duration"))
        spec = _make(symbols["Spec"], **spec_kwargs)

        labels = _safe_labels(
            {
                "safactory.brainpp.cn/job-id": request.job_id,
                "safactory.brainpp.cn/session-id": request.session_id,
                "safactory.brainpp.cn/agent-name": lease.agent_name,
                "safactory.brainpp.cn/agent-id": lease.agent_id,
                **dict(cfg.get("labels") or {}),
            }
        )
        annotations = {
            "safactory.brainpp.cn/job-id": request.job_id,
            "safactory.brainpp.cn/session-id": request.session_id,
            "safactory.brainpp.cn/agent-name": lease.agent_name,
            "safactory.brainpp.cn/agent-id": lease.agent_id,
            **{str(k): str(v) for k, v in dict(cfg.get("annotations") or {}).items()},
        }
        metadata_kwargs: Dict[str, Any] = {
            "name": rjob_name,
            "labels": labels,
            "annotations": annotations,
        }
        if cfg.get("charged_group"):
            metadata_kwargs["charged_group"] = str(cfg.get("charged_group"))
        metadata = _make(symbols["Metadata"], **metadata_kwargs)
        return _make(symbols["Job"], metadata=metadata, spec=spec)

    async def submit_job(self, client: Any, job: Any, submit_kwargs: Dict[str, Any]) -> str:
        submitted = await asyncio.to_thread(client.submit, job, **submit_kwargs)
        return str(submitted or "").strip()

    async def wait_terminal(
        self,
        client: Any,
        job_name: str,
        *,
        timeout_s: float,
        poll_interval_s: float,
        trace: PerfTrace | None = None,
    ) -> tuple[str, int]:
        deadline = time.monotonic() + max(1.0, float(timeout_s))
        unknown_count = 0
        poll_count = 0
        last_status = ""
        wait_started = time.perf_counter()
        first_status_ms: float | None = None
        submit_to_starting_ms: float | None = None
        submit_to_running_ms: float | None = None
        while True:
            status = await self._get_status(client, job_name)
            poll_count += 1
            elapsed_ms = round((time.perf_counter() - wait_started) * 1000, 3)
            if status != "Unknown" and first_status_ms is None:
                first_status_ms = elapsed_ms
                if trace is not None:
                    trace.mark(
                        "rjob_first_status",
                        status=status,
                        poll_count=poll_count,
                        job_name=job_name,
                        submit_to_first_status_ms=first_status_ms,
                    )
                    trace.update_context(rjob_submit_to_first_status_ms=first_status_ms)
            if status == "Starting" and submit_to_starting_ms is None:
                submit_to_starting_ms = elapsed_ms
                if trace is not None:
                    trace.mark(
                        "rjob_starting",
                        status=status,
                        poll_count=poll_count,
                        job_name=job_name,
                        submit_to_starting_ms=submit_to_starting_ms,
                    )
                    trace.update_context(rjob_submit_to_starting_ms=submit_to_starting_ms)
            if status == "Running" and submit_to_running_ms is None:
                submit_to_running_ms = elapsed_ms
                if trace is not None:
                    trace.mark(
                        "rjob_running",
                        status=status,
                        poll_count=poll_count,
                        job_name=job_name,
                        submit_to_running_ms=submit_to_running_ms,
                    )
                    trace.update_context(rjob_submit_to_running_ms=submit_to_running_ms)
            if status != last_status:
                if trace is not None:
                    trace.mark(
                        "status_poll",
                        status=status,
                        poll_count=poll_count,
                        job_name=job_name,
                        submit_to_status_ms=elapsed_ms,
                    )
                log.info("RJob status changed: name=%s status=%s poll_count=%d", job_name, status, poll_count)
                last_status = status
            if status in RJOB_SUCCEEDED_STATUSES or status in RJOB_FAILED_STATUSES:
                return status, poll_count
            if status == "Unknown":
                unknown_count += 1
                if unknown_count >= 5:
                    return status, poll_count
            if time.monotonic() >= deadline:
                if trace is not None:
                    trace.mark(
                        "wait_terminal.timeout",
                        last_status=status,
                        poll_count=poll_count,
                        timeout_s=timeout_s,
                    )
                exc = TimeoutError(f"RJob {job_name} timed out after {timeout_s:.1f}s; last_status={status}")
                exc.last_status = status  # type: ignore[attr-defined]
                exc.poll_count = poll_count  # type: ignore[attr-defined]
                raise exc
            if status not in RJOB_RUNNING_STATUSES and status != "Unknown":
                log.warning("RJob %s returned unrecognized status=%s; keep polling", job_name, status)
            await asyncio.sleep(max(0.1, float(poll_interval_s)))

    async def logs_text(self, client: Any, job_name: str, *, suppress_errors: bool = False) -> str:
        try:
            raw = await asyncio.to_thread(client.logs_rjob, job_name)
        except Exception:
            if suppress_errors:
                log.warning("RJob logs_rjob failed for %s", job_name, exc_info=True)
                return ""
            raise
        return extract_logs_text(raw)

    async def stop_job(self, client: Any, job_name: str) -> None:
        try:
            await asyncio.to_thread(client.stop, job_name)
        except Exception:
            log.warning("RJob stop failed for %s", job_name, exc_info=True)

    async def cleanup_job(
        self,
        client: Any,
        job_name: str,
        *,
        cfg: Dict[str, Any],
        terminal_status: str,
    ) -> None:
        if not bool(cfg.get("cleanup_on_finish", True)):
            return
        if terminal_status in RJOB_FAILED_STATUSES and bool(cfg.get("keep_failed_jobs", False)):
            log.info("keeping failed RJob for debugging: %s status=%s", job_name, terminal_status)
            return
        try:
            await asyncio.to_thread(client.delete, [job_name], async_=True)
        except TypeError:
            await asyncio.to_thread(client.delete, [job_name])
        except Exception:
            log.warning("RJob delete failed for %s", job_name, exc_info=True)

    def client(self, cfg: Dict[str, Any]) -> Any:
        symbols = self.symbols()
        kwargs = {
            "cluster_entry": str(cfg.get("cluster_entry") or "").strip() or None,
            "namespace": str(cfg.get("namespace") or "").strip() or None,
            "access_key": str(cfg.get("access_key") or "").strip() or None,
            "secret_key": str(cfg.get("secret_key") or "").strip() or None,
            "verifyssl": bool(cfg.get("verifyssl", True)),
            "retries": int(cfg.get("retries", 3) or 0),
        }
        key = tuple((name, kwargs[name]) for name in sorted(kwargs))
        if key not in self._clients:
            client_kwargs = {name: value for name, value in kwargs.items() if value is not None}
            self._clients[key] = symbols["RJobClient"](**client_kwargs)
        return self._clients[key]

    @staticmethod
    def symbols() -> Dict[str, Any]:
        try:
            import brainpp.rjob as rjob_mod
            from brainpp.rjob import RJobClient
        except ImportError as exc:
            raise RuntimeError(
                "RJob mode requires brainpp.rjob / RJobClient to be installed in the launcher environment."
            ) from exc

        try:
            import brainpp.rjob.struct as struct_mod
        except ImportError:
            struct_mod = rjob_mod

        names = [
            "Job",
            "Metadata",
            "Spec",
            "Task",
            "Template",
            "Container",
            "Resources",
            "RestartPolicy",
            "PrivateMachine",
            "ImagePullPolicy",
        ]
        symbols: Dict[str, Any] = {"RJobClient": RJobClient}
        for name in names:
            value = getattr(rjob_mod, name, None) or getattr(struct_mod, name, None)
            if value is not None:
                symbols[name] = value
        missing = [
            name
            for name in ("Job", "Metadata", "Spec", "Task", "Template", "Container", "Resources")
            if name not in symbols
        ]
        if missing:
            raise RuntimeError(f"RJobClient package is missing required struct(s): {missing}")
        return symbols

    @staticmethod
    def submit_kwargs(cfg: Dict[str, Any]) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "no_packaging": bool(cfg.get("no_packaging", True)),
            "dry_run": bool(cfg.get("dry_run", False)),
            "predict_only": bool(cfg.get("predict_only", False)),
            "top": bool(cfg.get("top", False)),
        }
        for key in ("packaging_dir", "yaml_file_dump_path"):
            if cfg.get(key):
                kwargs[key] = str(cfg.get(key))
        if "name_normalized" in cfg:
            kwargs["name_normalized"] = bool(cfg.get("name_normalized"))
        else:
            kwargs["name_normalized"] = True
        return kwargs

    @staticmethod
    def resolve_gateway_base_url(cfg: Dict[str, Any], request: SimulationStartRequest) -> str:
        base_url = str(cfg.get("gateway_base_url") or request.gateway_base_url).rstrip("/")
        try:
            host = urlsplit(base_url).hostname
        except Exception:
            host = ""
        if host in {"127.0.0.1", "localhost", "::1"}:
            raise RuntimeError(
                "RJob mode requires a gateway base URL reachable from the RJob cluster. "
                "Set gateway_base_url in the YAML passed by --rjob-config."
            )
        return base_url

    @staticmethod
    def build_job_name(
        cfg: Dict[str, Any],
        lease: SimulationAgentLease,
        request: SimulationStartRequest,
    ) -> str:
        prefix = _safe_name(str(cfg.get("name_prefix") or "safactory"), max_len=12)
        agent = _safe_name(lease.agent_name, max_len=8)
        job = _safe_name(request.job_id, max_len=10)
        session = _safe_name(request.session_id, max_len=10)
        return f"{prefix}-{job}-{agent}-{session}".strip("-")[:_MAX_RJOB_NAME_LEN].strip("-") or "safactory-rjob"

    def _docker_cfg_for_env(self, env_name: str) -> Dict[str, Any]:
        env_cfg = dict(self._env_types.get(env_name, {}) or {})
        docker_cfg = dict(self._cluster_cfg.get("docker", {}) or {})
        return merge_dicts(docker_cfg, env_cfg.get("docker"))

    def _rjob_cfg_for_env(self, env_name: str) -> Dict[str, Any]:
        env_cfg = dict(self._env_types.get(env_name, {}) or {})
        return merge_dicts(self._rjob_cfg, env_cfg.get("rjob"))

    @staticmethod
    def _validate_mounts(env_name: str, docker_cfg: Dict[str, Any], rjob_cfg: Dict[str, Any]) -> None:
        docker_volumes = docker_cfg.get("volumes", docker_cfg.get("mounts", [])) or []
        docker_volumes = [
            volume for volume in docker_volumes
            if not _is_embeddable_file_mount(volume)
        ]
        rjob_mounts = rjob_cfg.get("mount_config") or rjob_cfg.get("mount") or []
        if docker_volumes and not rjob_mounts:
            raise RuntimeError(
                f"RJob mode cannot map local Docker mounts for agent {env_name!r}. "
                "Configure rjob.mount_config or rjob.mount with cluster-accessible storage."
            )

    @staticmethod
    def _build_resources(symbols: Dict[str, Any], cfg: Dict[str, Any]) -> Optional[Any]:
        if not cfg:
            return None
        kwargs: Dict[str, Any] = {}
        for key in ("cpu", "gpu", "memory_in_mb", "ephemeral_storage_in_mb"):
            if key in cfg and cfg.get(key) is not None:
                kwargs[key] = cfg.get(key)
        if "custom_resources" in cfg and cfg.get("custom_resources") is not None:
            custom_resources = _normalize_custom_resources(cfg.get("custom_resources"))
            if custom_resources:
                kwargs["custom_resources"] = custom_resources
        return _make(symbols["Resources"], **kwargs) if kwargs else None

    @staticmethod
    def _command_with_embedded_files(cfg: Dict[str, Any], run_command: str) -> str:
        files = [
            item for item in (cfg.get("embedded_files") or [])
            if isinstance(item, dict) and str(item.get("source") or "").strip() and str(item.get("target") or "").strip()
        ]
        if not files:
            return run_command

        writes = []
        for item in files:
            source = Path(str(item.get("source") or "")).expanduser()
            target = str(item.get("target") or "").strip()
            if not source.is_file():
                raise RuntimeError(f"RJob embedded file source does not exist: {source}")
            writes.append((target, base64.b64encode(source.read_bytes()).decode("ascii")))

        python_bin = str(cfg.get("embed_python_bin") or cfg.get("python_bin") or "python").strip() or "python"
        lines = [
            "set -eu",
            f"{python_bin} - <<'PY'",
            "import base64",
            "from pathlib import Path",
        ]
        for target, content in writes:
            lines.append(f"_p = Path({target!r})")
            lines.append("_p.parent.mkdir(parents=True, exist_ok=True)")
            lines.append(f"_p.write_bytes(base64.b64decode({content!r}))")
        lines.extend(["PY", f"exec {run_command}"])
        return "\n".join(lines)

    @staticmethod
    def _coerce_enum(enum_cls: Any, value: Any) -> Any:
        text = str(value or "").strip()
        if not text:
            return None
        if enum_cls is None:
            return text
        for candidate in (text, text.lower(), text.upper(), text.capitalize()):
            if hasattr(enum_cls, candidate):
                return getattr(enum_cls, candidate)
        try:
            for item in enum_cls:
                if (
                    str(getattr(item, "name", "")).lower() == text.lower()
                    or str(getattr(item, "value", "")).lower() == text.lower()
                ):
                    return item
        except TypeError:
            pass
        return text

    @staticmethod
    async def _get_status(client: Any, job_name: str) -> str:
        jobs = await asyncio.to_thread(client.list, [job_name])
        if not jobs:
            return "Unknown"
        job = jobs[0]
        if isinstance(job, dict):
            return str(job.get("status") or _nested_name(job.get("status")) or "Unknown")
        status = getattr(job, "status", None)
        if isinstance(status, str):
            return status
        current = getattr(status, "current", None)
        if current is not None:
            return str(getattr(current, "name", current))
        return str(getattr(status, "name", status) or "Unknown")


def merge_dicts(*values: Any) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    for value in values:
        if isinstance(value, dict):
            merged.update(value)
    return merged


def merge_env_dicts(*values: Any) -> Dict[str, str]:
    merged: Dict[str, str] = {}
    for value in values:
        if isinstance(value, dict):
            merged.update({str(key): str(val) for key, val in value.items()})
    return merged


def _normalize_custom_resources(value: Any) -> List[str]:
    """Convert YAML-friendly custom resources to the RJob SDK list[str] form."""
    if isinstance(value, dict):
        items = [f"{str(name).strip()}={str(quantity).strip()}" for name, quantity in value.items()]
    elif isinstance(value, str):
        items = [value]
    elif isinstance(value, (list, tuple)):
        if not all(isinstance(item, str) for item in value):
            raise ValueError(
                "RJob resources.custom_resources list entries must be strings "
                "in resource-name=value format"
            )
        items = list(value)
    else:
        raise ValueError(
            "RJob resources.custom_resources must be a mapping, string, or list of strings"
        )

    normalized: List[str] = []
    for item in items:
        text = item.strip()
        name, separator, quantity = text.partition("=")
        if not separator or not name.strip() or not quantity.strip():
            raise ValueError(
                "RJob resources.custom_resources entries must use resource-name=value format"
            )
        normalized.append(f"{name.strip()}={quantity.strip()}")
    return normalized


def extract_logs_text(raw: Any) -> str:
    if raw is None:
        return ""
    if not isinstance(raw, str):
        return str(raw)
    text = raw
    try:
        import json

        payload = json.loads(raw)
    except json.JSONDecodeError:
        return text
    chunks = list(_walk_log_payload(payload))
    return "\n".join(chunk for chunk in chunks if chunk) or text


def _merge_embedded_files(docker_cfg: Dict[str, Any], rjob_cfg: Dict[str, Any]) -> List[Dict[str, str]]:
    files: List[Dict[str, str]] = []
    for item in rjob_cfg.get("embedded_files") or []:
        normalized = _normalize_embedded_file(item)
        if normalized:
            files.append(normalized)
    for volume in docker_cfg.get("volumes", docker_cfg.get("mounts", [])) or []:
        normalized = _embedded_file_from_mount(volume)
        if normalized and normalized not in files:
            files.append(normalized)
    return files


def _normalize_embedded_file(value: Any) -> Dict[str, str] | None:
    if isinstance(value, str) or not isinstance(value, dict):
        return None
    source = str(value.get("source") or value.get("hostPath") or value.get("host_path") or "").strip()
    target = str(value.get("target") or value.get("containerPath") or value.get("container_path") or "").strip()
    if not source or not target:
        return None
    source_path = Path(source).expanduser()
    if not source_path.is_absolute():
        source_path = (Path.cwd() / source_path).resolve(strict=False)
    return {"source": str(source_path), "target": target}


def _is_embeddable_file_mount(value: Any) -> bool:
    return _embedded_file_from_mount(value) is not None


def _embedded_file_from_mount(value: Any) -> Dict[str, str] | None:
    if not isinstance(value, dict):
        return None
    source = str(value.get("source") or value.get("hostPath") or value.get("host_path") or "").strip()
    target = str(value.get("target") or value.get("containerPath") or value.get("container_path") or "").strip()
    if not source or not target:
        return None
    source_path = Path(source).expanduser()
    if not source_path.is_file():
        return None
    return {"source": str(source_path), "target": target}


def _make(cls: Any, **kwargs: Any) -> Any:
    try:
        sig = inspect.signature(cls)
    except (TypeError, ValueError):
        return cls(**kwargs)
    accepts_var_kw = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in sig.parameters.values())
    if accepts_var_kw:
        return cls(**kwargs)
    filtered = {key: value for key, value in kwargs.items() if key in sig.parameters}
    return cls(**filtered)


def _safe_name(value: str, *, max_len: int) -> str:
    normalized = _INVALID_NAME_CHARS.sub("-", str(value or "").lower()).strip("-")
    return (normalized or "x")[:max_len].strip("-") or "x"


def _safe_labels(values: Dict[str, Any]) -> Dict[str, str]:
    labels: Dict[str, str] = {}
    for key, value in values.items():
        text = _safe_label_value(str(value or ""))
        if text:
            labels[str(key)] = text
    return labels


def _safe_label_value(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value or "")).strip("-._")
    return text[:63].strip("-._")


def _nested_name(value: Any) -> str:
    if isinstance(value, dict):
        current = value.get("current")
        if isinstance(current, dict):
            return str(current.get("name") or current.get("value") or "")
        return str(value.get("name") or value.get("value") or "")
    return str(getattr(value, "name", "") or getattr(value, "value", "") or "")


def _walk_log_payload(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from _walk_log_payload(item)
    elif isinstance(value, dict):
        for key in ("log", "logs", "content", "stdout", "stderr", "message", "data"):
            if key in value:
                yield from _walk_log_payload(value[key])
    else:
        yield str(value)
