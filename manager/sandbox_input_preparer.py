"""S3 input preparation, used only by the Sandbox episode workflow."""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import shlex
import threading
import time
import uuid
from pathlib import PurePosixPath
from typing import Any

from clusters.protected_execd import delete_private_plan, write_private_plan
from clusters.sandbox_cluster import run_sandbox_command

from .sandbox_s3_inputs import S3InputPlanner
from .types import SimulationAgentLease, SimulationStartRequest

log = logging.getLogger(__name__)


class InputPreparationError(RuntimeError):
    """Only controlled codes, never an SDK error containing a signed URL."""

    def __init__(self, phase: str, code: str) -> None:
        self.phase, self.code = phase, code
        super().__init__(f"Sandbox input {phase} failed ({code})")


class SandboxInputPreparer:
    def __init__(self, config: dict[str, Any]) -> None:
        self.planner = S3InputPlanner(config)

    async def prepare(
        self, *, lease: SimulationAgentLease, request: SimulationStartRequest, deadline: float,
    ) -> dict[str, Any]:
        # Shield the operation so repeated cancellation cannot race remote
        # interruption with deletion of the private plan.
        task = asyncio.create_task(self._prepare(lease, request, deadline))
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            task.cancel()
            while not task.done():
                try:
                    await asyncio.shield(task)
                except asyncio.CancelledError:
                    continue
                except Exception:
                    break
            if not task.cancelled():
                task.exception()
            raise

    async def _prepare(
        self, lease: SimulationAgentLease, request: SimulationStartRequest, deadline: float,
    ) -> dict[str, Any]:
        sandbox = lease.runtime_handle
        params = request.env_params
        source = params.get("bundle_source")
        if (
            request.agent_name != "harbor" or not isinstance(source, dict)
            or source.get("type") != "s3" or params.get("image_load_mode") != "task"
            or set(source) - {"type", "profile", "download_timeout_s", "download_concurrency"}
        ):
            raise InputPreparationError("config", "invalid_s3_source")
        if not getattr(sandbox, "protected_execd", False):
            raise InputPreparationError("config", "protected_execd_required")
        root = PurePosixPath(str(params.get("bundle_root") or ""))
        if not root.is_absolute() or ".." in root.parts or PurePosixPath("/work") not in root.parents:
            raise InputPreparationError("config", "bundle_root_must_be_under_work")
        try:
            seconds = float(source.get("download_timeout_s", 1800))
            concurrency = int(source.get("download_concurrency", 4))
            if not math.isfinite(seconds) or seconds <= 0 or not 1 <= concurrency <= 16:
                raise ValueError
        except (TypeError, ValueError, OverflowError):
            raise InputPreparationError("config", "invalid_download_limits") from None
        task_id = (params.get("dataset") or {}).get("task_id")
        profile_subdir = params.get("profile_subdir", "harbor-runtime-profile")
        attempt_id = uuid.uuid4().hex
        loop = asyncio.get_running_loop()
        end = min(deadline, loop.time() + seconds)
        phase, uploaded = "plan", False
        cancelled = threading.Event()
        failed = False
        try:
            async with asyncio.timeout_at(end):
                plan = await asyncio.to_thread(
                    self.planner.plan, source.get("profile"), task_id, profile_subdir,
                    session_id=request.session_id, attempt_id=attempt_id,
                    cancel_event=cancelled, deadline=time.monotonic() + max(0, end - loop.time()),
                )
                body = json.dumps(plan, ensure_ascii=False).encode("utf-8")
                phase, uploaded = "transfer", True
                path = await write_private_plan(sandbox, attempt_id, body)
                phase = "download"
                argv = [
                    "/opt/harbor-env/bin/python", "/work/safactory/input_bundle.py",
                    "--plan", path, "--plan-sha256", hashlib.sha256(body).hexdigest(),
                    "--session-id", request.session_id, "--attempt-id", attempt_id,
                    "--task-id", task_id, "--profile-subdir", profile_subdir,
                    "--bundle-root", str(root), "--timeout-s", str(max(0.001, end - loop.time())),
                    "--concurrency", str(concurrency),
                ]
                output = await run_sandbox_command(
                    sandbox, shlex.join(argv), timeout_s=max(0.001, end - loop.time()),
                )
                if output.exit_code:
                    raise InputPreparationError(phase, "download_failed")
                phase = "receipt"
                receipt = json.loads(output.stdout)
                expected = {
                    "session_id": request.session_id, "attempt_id": attempt_id,
                    "task_id": task_id, "bundle_root": str(root),
                    "lock_sha256": plan["lock_sha256"], "file_count": len(plan["files"]),
                    "bytes": sum(item["size"] for item in plan["files"]),
                }
                if not isinstance(receipt, dict) or any(receipt.get(k) != v for k, v in expected.items()):
                    raise InputPreparationError(phase, "receipt_mismatch")
                elapsed = float(receipt["elapsed_s"])
                if not math.isfinite(elapsed) or elapsed < 0:
                    raise InputPreparationError(phase, "invalid_elapsed_time")
                # Only a fixed allowlist crosses back into persistent metadata.
                return {**expected, "elapsed_s": elapsed}
        except (InputPreparationError, TimeoutError, asyncio.CancelledError):
            failed = True
            raise
        except Exception as exc:
            failed = True
            code = getattr(exc, "code", "input_prepare_failed")
            if not isinstance(code, str) or not code.replace("_", "").isalnum():
                code = "input_prepare_failed"
            raise InputPreparationError(phase, code) from None
        finally:
            cancelled.set()
            if uploaded:
                # A failed cleanup must prevent execution of untrusted task code.
                cleanup = asyncio.create_task(delete_private_plan(sandbox, attempt_id))
                cancellation = None
                while True:
                    try:
                        await asyncio.shield(cleanup)
                        break
                    except asyncio.CancelledError as exc:
                        cancellation = cancellation or exc
                        if cleanup.done():
                            break
                    except Exception:
                        if not failed and cancellation is None:
                            raise InputPreparationError("cleanup", "private_plan_cleanup_failed") from None
                        log.warning("Private input cleanup failed: sandbox=%s", lease.resource_id)
                        break
                if cancellation is not None:
                    raise cancellation

    async def close(self) -> None:
        self.planner.close()
