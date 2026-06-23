from __future__ import annotations

import asyncio
import base64
import inspect
import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional
from urllib.parse import urlsplit

from .episode_common import (
    RESULT_JSON_PREFIX,
    json_for_log,
    normalize_result,
    parse_result_output,
    request_env,
    request_payload,
    tail,
)
from .types import SimulationAgentLease, SimulationStartRequest, SimulationStartResult

log = logging.getLogger("manager.rjob_episode_runner")

_INVALID_NAME_CHARS = re.compile(r"[^a-z0-9.-]+")
_RUNNING_STATUSES = {"Created", "Pending", "Starting", "Running", "Inqueue", "Restarting", "Killing", "Deleting"}
_SUCCEEDED_STATUSES = {"Succeeded"}
_FAILED_STATUSES = {"Failed", "Stopped", "Killed"}
_MAX_RJOB_NAME_LEN = 49


class RJobEpisodeRunner:
    """Runs one episode as one RJob job."""

    def __init__(self, *, timeout_s: float) -> None:
        self.timeout_s = float(timeout_s)
        self._clients: Dict[tuple[Any, ...], Any] = {}

    async def start(
        self,
        lease: SimulationAgentLease,
        request: SimulationStartRequest,
    ) -> SimulationStartResult:
        if not lease.image:
            raise RuntimeError(f"RJob lease missing image: {lease.agent_name}/{lease.agent_id}")
        if not lease.run_command:
            raise RuntimeError(f"RJob lease missing run_command: {lease.agent_name}/{lease.agent_id}")

        cfg = dict(lease.runtime_config or {})
        gateway_base_url = self._resolve_gateway_base_url(cfg, request)
        client = self._client(cfg)
        request_params, payload = request_payload(request)
        rjob_name = self._build_job_name(cfg, lease, request)
        merged_env = _merge_dicts(
            cfg.get("env"),
            request_env(
                request,
                payload,
                gateway_base_url=gateway_base_url,
                containerize_local_gateway=False,
            ),
        )
        job = self._build_job(
            cfg=cfg,
            lease=lease,
            request=request,
            rjob_name=rjob_name,
            env=merged_env,
        )
        submit_kwargs = self._submit_kwargs(cfg)

        log.info(
            "OpenClaw RJob submit params: agent=%s/%s name=%s params=%s",
            lease.agent_name,
            lease.agent_id,
            rjob_name,
            self._safe_json_for_log(
                {
                    "request": request_params,
                    "image": lease.image,
                    "workdir": lease.workdir,
                    "run_command": lease.run_command,
                    "submit": submit_kwargs,
                    "resources": cfg.get("resources") or cfg.get("default_resources") or {},
                    "mount_config": cfg.get("mount_config") or [],
                }
            ),
        )

        submitted_name = ""
        terminal_status = "Unknown"
        logs_text = ""
        try:
            submitted = await asyncio.to_thread(client.submit, job, **submit_kwargs)
            submitted_name = str(submitted or rjob_name).strip()
            if not submitted_name and (submit_kwargs.get("dry_run") or submit_kwargs.get("predict_only")):
                return SimulationStartResult(
                    session_id=request.session_id,
                    status="succeeded",
                    total_reward=0.0,
                    step_count=0,
                    terminated=True,
                    truncated=False,
                    metrics={"runtime": "rjob", "dry_run": bool(submit_kwargs.get("dry_run"))},
                )
            if not submitted_name:
                raise RuntimeError("RJobClient.submit returned empty job name")

            terminal_status = await self._wait_terminal(
                client,
                submitted_name,
                timeout_s=float(request.agent_start_timeout_s or self.timeout_s),
                poll_interval_s=float(cfg.get("poll_interval_s", 5.0) or 5.0),
            )
            try:
                logs_text = await self._logs_text(client, submitted_name)
            except Exception as exc:
                return SimulationStartResult(
                    session_id=request.session_id,
                    status="failed",
                    total_reward=0.0,
                    step_count=0,
                    terminated=True,
                    truncated=False,
                    error_text=f"RJob {submitted_name} finished with status={terminal_status}, but logs_rjob failed: {exc}",
                    metrics={
                        "runtime": "rjob",
                        "rjob_name": submitted_name,
                        "rjob_status": terminal_status,
                        "logs_error": str(exc),
                    },
                )
            result = self._result_from_terminal_status(
                terminal_status=terminal_status,
                logs_text=logs_text,
                lease=lease,
                request=request,
                job_name=submitted_name,
            )
            return result
        except TimeoutError as exc:
            if submitted_name:
                await self._stop_job(client, submitted_name)
                logs_text = await self._logs_text(client, submitted_name, suppress_errors=True)
            return SimulationStartResult(
                session_id=request.session_id,
                status="failed",
                total_reward=0.0,
                step_count=0,
                terminated=True,
                truncated=False,
                error_text=str(exc),
                metrics={
                    "runtime": "rjob",
                    "rjob_name": submitted_name or rjob_name,
                    "rjob_status": terminal_status,
                    "logs_tail": tail(logs_text),
                },
            )
        finally:
            if submitted_name:
                await self._cleanup_job(
                    client,
                    submitted_name,
                    cfg=cfg,
                    terminal_status=terminal_status,
                )

    async def close(self) -> None:
        return

    def _build_job(
        self,
        *,
        cfg: Dict[str, Any],
        lease: SimulationAgentLease,
        request: SimulationStartRequest,
        rjob_name: str,
        env: Dict[str, str],
    ) -> Any:
        symbols = self._symbols()
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

    def _build_resources(self, symbols: Dict[str, Any], cfg: Dict[str, Any]) -> Optional[Any]:
        if not cfg:
            return None
        kwargs: Dict[str, Any] = {}
        for key in ("cpu", "gpu", "memory_in_mb", "ephemeral_storage_in_mb", "custom_resources"):
            if key in cfg and cfg.get(key) is not None:
                kwargs[key] = cfg.get(key)
        return _make(symbols["Resources"], **kwargs) if kwargs else None

    def _command_with_embedded_files(self, cfg: Dict[str, Any], run_command: str) -> str:
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

    async def _wait_terminal(
        self,
        client: Any,
        job_name: str,
        *,
        timeout_s: float,
        poll_interval_s: float,
    ) -> str:
        deadline = time.monotonic() + max(1.0, float(timeout_s))
        unknown_count = 0
        while True:
            status = await self._get_status(client, job_name)
            if status in _SUCCEEDED_STATUSES or status in _FAILED_STATUSES:
                return status
            if status == "Unknown":
                unknown_count += 1
                if unknown_count >= 5:
                    return status
            if time.monotonic() >= deadline:
                raise TimeoutError(f"RJob {job_name} timed out after {timeout_s:.1f}s; last_status={status}")
            if status not in _RUNNING_STATUSES and status != "Unknown":
                log.warning("RJob %s returned unrecognized status=%s; keep polling", job_name, status)
            await asyncio.sleep(max(0.1, float(poll_interval_s)))

    async def _get_status(self, client: Any, job_name: str) -> str:
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

    async def _logs_text(self, client: Any, job_name: str, *, suppress_errors: bool = False) -> str:
        try:
            raw = await asyncio.to_thread(client.logs_rjob, job_name)
        except Exception:
            if suppress_errors:
                log.warning("RJob logs_rjob failed for %s", job_name, exc_info=True)
                return ""
            raise
        return _extract_logs_text(raw)

    def _result_from_terminal_status(
        self,
        *,
        terminal_status: str,
        logs_text: str,
        lease: SimulationAgentLease,
        request: SimulationStartRequest,
        job_name: str,
    ) -> SimulationStartResult:
        result_mode = str(lease.result_mode or "json").strip().lower()
        if terminal_status in _SUCCEEDED_STATUSES and result_mode == "exit_code":
            return SimulationStartResult(
                session_id=request.session_id,
                status="succeeded",
                total_reward=0.0,
                step_count=0,
                terminated=True,
                truncated=False,
                metrics={
                    "runtime": "rjob",
                    "rjob_name": job_name,
                    "rjob_status": terminal_status,
                    "result_mode": result_mode,
                    "logs_tail": tail(logs_text),
                },
            )

        try:
            body = parse_result_output(logs_text)
            result = normalize_result(body, session_id=request.session_id)
        except Exception as exc:
            return SimulationStartResult(
                session_id=request.session_id,
                status="failed",
                total_reward=0.0,
                step_count=0,
                terminated=True,
                truncated=False,
                error_text=(
                    f"RJob {job_name} finished with status={terminal_status}, "
                    f"but no SimulationStartResult JSON could be parsed: {exc}"
                ),
                metrics={
                    "runtime": "rjob",
                    "rjob_name": job_name,
                    "rjob_status": terminal_status,
                    "logs_tail": tail(logs_text),
                },
            )

        result.metrics = dict(result.metrics or {})
        result.metrics.update(
            {
                "runtime": "rjob",
                "rjob_name": job_name,
                "rjob_status": terminal_status,
                "logs_tail": tail(logs_text),
            }
        )
        if terminal_status in _FAILED_STATUSES and result.status == "succeeded":
            result.status = "failed"
            result.error_text = f"RJob {job_name} failed with status={terminal_status}"
        return result

    async def _stop_job(self, client: Any, job_name: str) -> None:
        try:
            await asyncio.to_thread(client.stop, job_name)
        except Exception:
            log.warning("RJob stop failed for %s", job_name, exc_info=True)

    async def _cleanup_job(
        self,
        client: Any,
        job_name: str,
        *,
        cfg: Dict[str, Any],
        terminal_status: str,
    ) -> None:
        cleanup = bool(cfg.get("cleanup_on_finish", True))
        if not cleanup:
            return
        if terminal_status in _FAILED_STATUSES and bool(cfg.get("keep_failed_jobs", False)):
            log.info("keeping failed RJob for debugging: %s status=%s", job_name, terminal_status)
            return
        try:
            await asyncio.to_thread(client.delete, [job_name], async_=True)
        except TypeError:
            await asyncio.to_thread(client.delete, [job_name])
        except Exception:
            log.warning("RJob delete failed for %s", job_name, exc_info=True)

    def _client(self, cfg: Dict[str, Any]) -> Any:
        symbols = self._symbols()
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
    def _symbols() -> Dict[str, Any]:
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
        missing = [name for name in ("Job", "Metadata", "Spec", "Task", "Template", "Container", "Resources") if name not in symbols]
        if missing:
            raise RuntimeError(f"RJobClient package is missing required struct(s): {missing}")
        return symbols

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
                if str(getattr(item, "name", "")).lower() == text.lower() or str(getattr(item, "value", "")).lower() == text.lower():
                    return item
        except TypeError:
            pass
        return text

    def _submit_kwargs(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
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

    def _resolve_gateway_base_url(self, cfg: Dict[str, Any], request: SimulationStartRequest) -> str:
        base_url = str(cfg.get("gateway_base_url") or request.gateway_base_url).rstrip("/")
        try:
            host = urlsplit(base_url).hostname
        except Exception:
            host = ""
        if host in {"127.0.0.1", "localhost", "::1"}:
            raise RuntimeError(
                "RJob mode requires a gateway base URL reachable from the RJob cluster. "
                "Set --rjob-gateway-base-url or cluster.rjob.gateway_base_url."
            )
        return base_url

    @staticmethod
    def _build_job_name(
        cfg: Dict[str, Any],
        lease: SimulationAgentLease,
        request: SimulationStartRequest,
    ) -> str:
        prefix = _safe_name(str(cfg.get("name_prefix") or "safactory"), max_len=12)
        agent = _safe_name(lease.agent_name, max_len=8)
        job = _safe_name(request.job_id, max_len=10)
        session = _safe_name(request.session_id, max_len=10)
        return f"{prefix}-{job}-{agent}-{session}".strip("-")[:_MAX_RJOB_NAME_LEN].strip("-") or "safactory-rjob"

    @staticmethod
    def _safe_json_for_log(value: Any) -> str:
        scrubbed = json.loads(json_for_log(value))
        if isinstance(scrubbed, dict):
            submit = scrubbed.get("submit")
            if isinstance(submit, dict):
                for key in ("access_key", "secret_key"):
                    if key in submit:
                        submit[key] = "***"
        return json_for_log(scrubbed)


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


def _merge_dicts(*values: Any) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    for value in values:
        if isinstance(value, dict):
            merged.update({str(key): str(val) for key, val in value.items()})
    return merged


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


def _extract_logs_text(raw: Any) -> str:
    if raw is None:
        return ""
    if not isinstance(raw, str):
        return str(raw)
    text = raw
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return text
    chunks = list(_walk_log_payload(payload))
    return "\n".join(chunk for chunk in chunks if chunk) or text


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
