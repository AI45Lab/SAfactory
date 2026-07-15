from __future__ import annotations

from typing import Any

from evaluator.eval_types import EvalRequest, EvalSpec, TargetAgentRef


class TargetContainerAccessService:
    def build_target_ref(
        self,
        *,
        request: EvalRequest,
        spec: EvalSpec,
    ) -> TargetAgentRef:
        lease = request.lease
        runtime = str(getattr(lease, "runtime", "docker") or "docker")
        resource_id = str(getattr(lease, "resource_id", "") or "")
        container_id = str(getattr(lease, "container_id", "") or (resource_id if runtime == "sandbox" else ""))
        resource_name = str(getattr(lease, "resource_name", "") or getattr(lease, "resource_id", "") or "")
        container_name = str(getattr(lease, "container_name", "") or container_id or resource_name)
        image = str(getattr(lease, "image", "") or "")
        alias = spec.target_container_alias or container_name or container_id
        workspace_path = _first_present(
            request.env_params,
            ["workspace_path", "workspace_snapshot_path", "output_dir"],
        )
        artifact_paths = dict(request.env_params.get("artifact_paths") or {})
        alias_map = {alias: container_name or container_id} if alias else {}
        runtime_config = dict(getattr(lease, "runtime_config", {}) or {})
        sandbox_endpoint = None
        sandbox_headers: dict[str, str] = {}
        if spec.target_access_mode == "sandbox_proxy":
            endpoint = str(runtime_config.get("endpoint") or "").rstrip("/")
            sandbox_endpoint = f"{endpoint}/command" if endpoint else None
            sandbox_headers = {
                str(key): str(value)
                for key, value in dict(runtime_config.get("endpoint_headers", {}) or {}).items()
            }
        return TargetAgentRef(
            session_id=request.session_id,
            container_id=container_id,
            container_name=container_name,
            container_alias=alias,
            image=image,
            runtime=runtime,
            resource_id=resource_id,
            sandbox_endpoint=sandbox_endpoint,
            sandbox_headers=sandbox_headers,
            workspace_path=workspace_path,
            artifact_paths=artifact_paths,
            access_mode=spec.target_access_mode,
            docker_socket_path="/var/run/docker.sock" if spec.target_access_mode == "direct_docker" else None,
            alias_map=alias_map,
            exec_hint=_build_exec_hint(alias, container_name or container_id, spec.target_access_mode),
        )

    async def prepare_target_access(
        self,
        *,
        request: EvalRequest,
        spec: EvalSpec,
    ) -> TargetAgentRef:
        target = self.build_target_ref(request=request, spec=spec)
        if target.access_mode == "direct_docker" and target.runtime != "docker":
            raise ValueError(f"direct_docker target access is unavailable for runtime={target.runtime!r}")
        if target.access_mode == "sandbox_proxy":
            if target.runtime != "sandbox" or not target.sandbox_endpoint or not target.sandbox_headers:
                raise ValueError("sandbox_proxy requires an active Sandbox endpoint and access headers")
        if target.access_mode == "snapshot":
            target.artifact_paths.update(
                await self.create_snapshot(
                    request=request,
                    paths=list(target.artifact_paths.values()),
                )
            )
        elif target.access_mode == "brokered_container":
            target.broker_base_url, target.broker_token = await self.create_broker_session(
                request=request,
                allowed_commands=list(spec.evaluator_task_input.get("allowed_commands") or []),
                ttl_s=float(spec.evaluator_task_input.get("broker_ttl_s") or spec.timeout_s),
            )
        return target

    async def create_snapshot(
        self,
        *,
        request: EvalRequest,
        paths: list[str],
    ) -> dict[str, str]:
        # The scheduler may already have copied artifacts out. This method keeps
        # the contract explicit without coupling evaluator/ to manager internals.
        return {path: path for path in paths}

    async def create_broker_session(
        self,
        *,
        request: EvalRequest,
        allowed_commands: list[str],
        ttl_s: float,
    ) -> tuple[str, str]:
        broker = request.env_params.get("target_broker") or {}
        base_url = broker.get("base_url")
        token = broker.get("token")
        if not base_url or not token:
            raise ValueError("brokered_container requires env_params.target_broker.base_url and token")
        return str(base_url), str(token)

    async def cleanup_target_access(self, target: TargetAgentRef) -> None:
        return None


def _first_present(data: dict[str, Any], keys: list[str]) -> str | None:
    for key in keys:
        value = data.get(key)
        if value:
            return str(value)
    return None


def _build_exec_hint(alias: str, container: str, access_mode: str) -> str | None:
    if access_mode == "sandbox_proxy" and container:
        return f"Use the Sandbox proxy described in the structured target; its alias is {alias!r}."
    if access_mode != "direct_docker" or not container:
        return None
    return f"Use docker exec {container!r}; the task may refer to this target as {alias!r}."
