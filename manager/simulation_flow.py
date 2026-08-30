from __future__ import annotations

import asyncio
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import quote, urlsplit

import requests

from core.data_manager.manager import DataManager
from core.data_manager.yaml_aggregator import (
    all_env_yaml_load,
    is_job_db_processing_done,
    sync_configs_to_db,
    wait_for_pending_inserts,
)
from core.perf_trace import PerfTrace
from evaluator.factory import build_evaluation_service
from evaluator.gateway_client import GatewayClient
from evaluator.reward_committer import RewardCommitter
from evaluator.service import EvaluationService
from evaluator.trajectory_reader import TrajectoryReader
from .agent_start_client import AgentStartClient
from .manager import AgentPoolManager
from .resume_cleanup import cleanup_resume_artifacts
from .simulation_config import (
    build_manager_runtime_config,
    expand_rl_epoch,
    expand_rl_group_size,
)
from .simulation_lease_pool import SimulationLeasePool
from .simulation_worker import SimulationWorkerGroup
from .types import SimulationRunConfig, SimulationRunSummary

log = logging.getLogger("manager.simulation_flow")

_GATEWAY_ENV_ROW_POLL_INTERVAL_S = 30.0
_GATEWAY_ENV_ROW_MAX_ATTEMPTS = 10


class SimulationFlow:
    def __init__(self, cfg: SimulationRunConfig) -> None:
        self.cfg = cfg
        self.data_manager: Optional[DataManager] = None
        self.manager_cfg: Optional[Dict[str, Any]] = None
        self.agent_pool_manager: Optional[AgentPoolManager] = None
        self.lease_pool: Optional[SimulationLeasePool] = None
        self.agent_start_client: Optional[AgentStartClient] = None
        self.worker_group: Optional[SimulationWorkerGroup] = None
        self.gateway_client: Optional[GatewayClient] = None
        self.evaluation_service: Optional[EvaluationService] = None
        self.reward_committer: Optional[RewardCommitter] = None
        self._shutdown_started = False

    async def run(self) -> SimulationRunSummary:
        trace = PerfTrace(
            "simulation_flow.run",
            logger=log,
            context={
                "job_id": self.cfg.job_id,
                "mode": self.cfg.mode,
                "storage_type": self.cfg.storage_type,
                "pool_size": self.cfg.pool_size,
                "warm_pool_size": self.cfg.warm_pool_size,
                "evaluation_enabled": self.cfg.evaluation_enabled,
            },
        )
        try:
            with trace.span("prepare_storage"):
                await self.prepare_storage()
            with trace.span("check_gateway_ready"):
                await self.check_gateway_ready()
            if self.cfg.resume:
                with trace.span("clear_gateway_session_cache"):
                    await self.clear_resume_gateway_session_cache()
            with trace.span("wait_gateway_environment_rows"):
                await self.wait_gateway_environment_rows()
            with trace.span("start_agent_scheduler"):
                await self.start_agent_scheduler()
            with trace.span("run_workers"):
                summary = await self.run_workers()
            trace.update_context(
                summary_status=summary.status,
                total_episodes=summary.total_episodes,
                succeeded_episodes=summary.succeeded_episodes,
                truncated_episodes=summary.truncated_episodes,
                failed_episodes=summary.failed_episodes,
            )
            trace.emit_summary(status=summary.status)
            return summary
        except asyncio.CancelledError:
            trace.emit_summary(status="cancelled", error_type="CancelledError")
            raise
        except Exception as exc:
            trace.emit_summary(status="failed", error_type=type(exc).__name__, error=str(exc))
            raise

    async def prepare_storage(self) -> None:
        storage_config: Dict[str, Any] = {
            "enable_buffer": self.cfg.enable_buffer,
            "buffer_size": self.cfg.buffer_size,
            "flush_interval": self.cfg.flush_interval,
        }
        if self.cfg.storage_type == "sqlite":
            storage_config["db_url"] = self.cfg.db_url
        else:
            storage_config.update({
                "confirm_cloud_delete_job_id": self.cfg.confirm_cloud_delete_job_id,
                "confirm_production": self.cfg.confirm_production,
            })

        self.data_manager = DataManager(
            job_id=self.cfg.job_id,
            storage_type=self.cfg.storage_type,
            **storage_config,
        )

        yaml_config_list = all_env_yaml_load(env_root=self.cfg.agent_root, env_config=self.cfg.agent_config)
        yaml_config_list = expand_rl_group_size(yaml_config_list, self.cfg.rl_group_size)
        yaml_config_list = expand_rl_epoch(yaml_config_list, self.cfg.rl_epoch)

        await sync_configs_to_db(
            self.data_manager,
            yaml_config_list,
            self.cfg.storage_type,
            self.cfg.startup_submit_count,
            self.cfg.followup_submit_batch,
            rebuild_table=self.cfg.rebuild_table,
            resume=self.cfg.resume,
        )
        self.manager_cfg = build_manager_runtime_config(self.cfg)
        if self.cfg.resume and self.cfg.mode == "rjob":
            await cleanup_resume_artifacts(
                job_id=self.cfg.job_id,
                model=self.cfg.llm_model,
                data_manager=self.data_manager,
                manager_cfg=self.manager_cfg,
            )
        log.info(
            "storage prepared: job_id=%s base_pool_size=%d warm_pool_size=%d startup_submit_count=%d followup_submit_batch=%d",
            self.cfg.job_id,
            self.cfg.pool_size,
            self.cfg.warm_pool_size,
            self.cfg.startup_submit_count,
            self.cfg.followup_submit_batch,
        )

    async def check_gateway_ready(self) -> None:
        ready_url = self._gateway_origin() + "/readyz"

        def _probe() -> tuple[int, str]:
            response = requests.get(ready_url, timeout=5.0)
            return response.status_code, response.text

        try:
            status_code, body = await asyncio.to_thread(_probe)
        except Exception as exc:
            raise RuntimeError(f"gateway is not reachable at {ready_url}: {exc}") from exc

        if status_code != 200:
            raise RuntimeError(f"gateway is not ready at {ready_url}: status={status_code} body={body[:500]}")
        self._validate_gateway_storage(body, ready_url)
        await self.check_gateway_model_route()
        log.info("gateway ready: %s", ready_url)

    async def wait_gateway_environment_rows(self) -> None:
        target = int(self.cfg.warm_pool_size)
        job_id = quote(self.cfg.job_id, safe="")
        count_url = self._gateway_origin() + f"/internal/jobs/{job_id}/environment-row-count"
        last_error = "no response"

        def _fetch_count() -> tuple[int, str]:
            response = requests.get(count_url, timeout=5.0)
            return response.status_code, response.text

        for attempt in range(1, _GATEWAY_ENV_ROW_MAX_ATTEMPTS + 1):
            try:
                status_code, body = await asyncio.to_thread(_fetch_count)
                if status_code != 200:
                    raise RuntimeError(f"status={status_code} body={body[:500]}")
                payload = json.loads(body)
                row_count = int(payload["row_count"])
                if row_count < 0:
                    raise ValueError("row_count must be non-negative")
                if row_count >= target:
                    log.info(
                        "gateway environment rows visible: job_id=%s row_count=%d target=%d",
                        self.cfg.job_id,
                        row_count,
                        target,
                    )
                    return
                last_error = f"row_count={row_count} target={target}"
            except Exception as exc:
                last_error = str(exc)

            if attempt < _GATEWAY_ENV_ROW_MAX_ATTEMPTS:
                log.info(
                    "gateway environment rows not ready: job_id=%s attempt=%d/%d "
                    "last_result=%s retry_in_s=%.0f",
                    self.cfg.job_id,
                    attempt,
                    _GATEWAY_ENV_ROW_MAX_ATTEMPTS,
                    last_error,
                    _GATEWAY_ENV_ROW_POLL_INTERVAL_S,
                )
                await asyncio.sleep(_GATEWAY_ENV_ROW_POLL_INTERVAL_S)

        raise RuntimeError(
            "gateway environment rows are not visible after "
            f"{_GATEWAY_ENV_ROW_MAX_ATTEMPTS} attempts: "
            f"job_id={self.cfg.job_id} target={target} last_result={last_error}"
        )

    async def clear_resume_gateway_session_cache(self) -> None:
        if self.data_manager is None:
            raise RuntimeError("data manager is not prepared")
        rows = await self.data_manager.get_all_environments(self.cfg.job_id)
        session_ids = [
            str(row.get("env_id"))
            for row in rows
            if row.get("env_id")
            and not bool(row.get("finished"))
            and not bool(row.get("is_deleted"))
        ]
        if not session_ids:
            return
        client = GatewayClient(gateway_base_url=self.cfg.gateway_base_url)
        try:
            result = await client.clear_session_cache(session_ids)
        finally:
            await client.aclose()
        log.info(
            "gateway resume session cache cleared: job_id=%s sessions=%d removed=%s",
            self.cfg.job_id,
            len(session_ids),
            result.get("removed", 0),
        )

    async def check_gateway_model_route(self) -> None:
        metrics_url = self._gateway_origin() + "/metrics"

        def _fetch_metrics() -> tuple[int, str]:
            response = requests.get(metrics_url, timeout=5.0)
            return response.status_code, response.text

        try:
            status_code, body = await asyncio.to_thread(_fetch_metrics)
        except Exception:
            log.debug("gateway metrics are not reachable at %s; skip route-key preflight", metrics_url, exc_info=True)
            return

        if status_code != 200:
            log.debug("gateway metrics returned status=%s; skip route-key preflight", status_code)
            return

        available = sorted(set(re.findall(r'gateway_llm_route_inflight\{model="([^"]+)"\}', body)))
        if not available:
            log.debug("gateway metrics exposed no llm route labels; skip route-key preflight")
            return
        required_models = [self.cfg.llm_model]
        missing = [model for model in required_models if model not in available]
        if missing:
            raise RuntimeError(
                f"gateway model route(s) are not configured: {missing}; "
                f"available={available}"
            )
        log.debug("gateway model routes ready: models=%s available=%s", required_models, available)

    async def start_agent_scheduler(self) -> None:
        if self.manager_cfg is None:
            self.manager_cfg = build_manager_runtime_config(self.cfg)
        if self.data_manager is None:
            raise RuntimeError("data manager is not prepared")
        self.agent_pool_manager = AgentPoolManager(
            self.manager_cfg,
            self.data_manager,
            job_id=self.cfg.job_id,
            db_processing_done_checker=lambda: is_job_db_processing_done(self.cfg.job_id),
        )
        self.lease_pool = SimulationLeasePool(
            self.agent_pool_manager,
            pool_size=self.cfg.warm_pool_size,
            refill_timeout_s=self.cfg.container_refill_timeout_s,
        )
        await self.lease_pool.start()

    async def run_workers(self) -> SimulationRunSummary:
        if self.lease_pool is None:
            raise RuntimeError("lease pool is not started")
        if self.data_manager is None:
            raise RuntimeError("data manager is not prepared")

        self.agent_start_client = AgentStartClient(
            timeout_s=self.cfg.agent_start_timeout_s + self.cfg.agent_start_timeout_grace_s,
        )
        self.gateway_client = GatewayClient(
            gateway_base_url=self.cfg.gateway_base_url,
            close_timeout_s=self.cfg.gateway_close_timeout_s,
            close_retries=self.cfg.gateway_close_retries,
            retry_backoff_s=self.cfg.gateway_close_retry_backoff_s,
        )
        self.reward_committer = RewardCommitter(
            db_url=self.cfg.db_url,
            storage_type=self.cfg.storage_type,
            data_manager=self.data_manager,
        )
        evaluation_service = None
        if self.cfg.evaluation_enabled:
            log.info("EVAL FLOW enabled: rule evaluator only")
            self.evaluation_service = build_evaluation_service(
                trajectory_reader=TrajectoryReader(
                    db_url=self.cfg.db_url,
                    storage_type=self.cfg.storage_type,
                    data_manager=self.data_manager,
                ),
                max_concurrency=self.cfg.max_workers or self.cfg.warm_pool_size,
            )
            evaluation_service = self.evaluation_service
        else:
            log.debug("EVAL FLOW disabled: launcher was not started with --enable-evaluation")
        self.worker_group = SimulationWorkerGroup(
            lease_pool=self.lease_pool,
            data_manager=self.data_manager,
            agent_start_client=self.agent_start_client,
            cfg=self.cfg,
            gateway_client=self.gateway_client,
            evaluation_service=evaluation_service,
            reward_committer=self.reward_committer,
        )
        return await self.worker_group.run_all()

    async def shutdown(self) -> None:
        if self._shutdown_started:
            return
        self._shutdown_started = True
        trace = PerfTrace(
            "simulation_flow.shutdown",
            logger=log,
            context={"job_id": self.cfg.job_id},
        )

        if self.lease_pool is not None:
            try:
                with trace.span("lease_pool_close"):
                    await self.lease_pool.aclose()
            except Exception:
                log.exception("lease pool close failed (ignored)")

        if self.agent_start_client is not None:
            try:
                with trace.span("agent_start_client_close"):
                    await self.agent_start_client.close()
            except Exception:
                log.exception("agent start client close failed (ignored)")

        if self.gateway_client is not None:
            try:
                with trace.span("gateway_client_close"):
                    await self.gateway_client.aclose()
            except Exception:
                log.exception("gateway client close failed (ignored)")

        try:
            with trace.span("wait_for_pending_inserts"):
                await wait_for_pending_inserts()
        except Exception:
            log.exception("wait_for_pending_inserts failed (ignored)")

        if self.data_manager is not None:
            try:
                with trace.span("data_manager_close"):
                    await self.data_manager.close()
            except Exception:
                log.exception("data manager close failed (ignored)")

        trace.emit_summary(status="complete")

    def _gateway_origin(self) -> str:
        parsed = urlsplit(self.cfg.gateway_base_url)
        if not parsed.scheme or not parsed.netloc:
            raise ValueError(f"gateway_base_url must be absolute: {self.cfg.gateway_base_url!r}")
        return f"{parsed.scheme}://{parsed.netloc}"

    def _validate_gateway_storage(self, ready_body: str, ready_url: str) -> None:
        try:
            payload = json.loads(ready_body)
        except json.JSONDecodeError:
            log.warning("gateway ready response is not JSON; cannot validate shared trajectory DB: %s", ready_url)
            return
        if not isinstance(payload, dict):
            log.warning("gateway ready response is not an object; cannot validate shared trajectory DB: %s", ready_url)
            return

        gateway_storage_type = str(payload.get("storage_type") or "").strip()
        if not gateway_storage_type:
            log.warning("gateway ready response has no storage_type; cannot validate shared trajectory DB")
            return
        if gateway_storage_type != self.cfg.storage_type:
            raise RuntimeError(
                "gateway storage_type does not match launcher storage_type: "
                f"gateway={gateway_storage_type!r} launcher={self.cfg.storage_type!r}. "
                "Start gateway with the same storage backend used by launcher."
            )

        if self.cfg.storage_type != "sqlite":
            return

        storage_config = payload.get("storage_config")
        storage_config = storage_config if isinstance(storage_config, dict) else {}
        gateway_db_url = str(storage_config.get("db_url") or "").strip()
        if not gateway_db_url:
            raise RuntimeError(
                "gateway ready response does not expose storage_config.db_url. "
                "Use a gateway build that reports its SQLite DB so evaluator can verify trajectory storage."
            )

        gateway_db = _normalize_sqlite_db_path(gateway_db_url)
        launcher_db = _normalize_sqlite_db_path(self.cfg.db_url)
        if gateway_db != launcher_db:
            raise RuntimeError(
                "gateway SQLite DB does not match launcher --db-path; evaluator would read an empty or partial "
                f"trajectory. gateway={gateway_db} launcher={launcher_db}. "
                "Set gateway storage_config.db_url to the same value as launcher --db-path."
            )


def _normalize_sqlite_db_path(db_url: str) -> str:
    value = str(db_url or "").strip()
    if value.startswith("sqlite://"):
        value = value[len("sqlite://") :]
    value = value.split("?", 1)[0]
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return str(path.resolve(strict=False))
