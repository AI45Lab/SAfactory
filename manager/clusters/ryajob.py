from __future__ import annotations

from typing import Any, List, Optional

import secrets
import sys

from rayjob_sdk import HeadConfig, RayJobClient, SDKException, WorkerGroupConfig


def _random_name_hint() -> str:
    """Return a short random name hint like 'AABBCC'."""
    return secrets.token_hex(3).upper()


def _extract_job_name(result: Any) -> str:
    """Best-effort extraction of the platform-created RayJob name from SDK response."""
    meta = getattr(result, "metadata", None)
    if meta is not None:
        name = getattr(meta, "name", None)
        if isinstance(name, str) and name.strip():
            return name.strip()

    for attr in ("jobName", "name"):
        name = getattr(result, attr, None)
        if isinstance(name, str) and name.strip():
            return name.strip()

    raise RuntimeError(f"Could not determine RayJob name from SDK response: {result!r}")


class RayJobManager:

    def __init__(
        self,
        domain: str,
        tenant: str,
        access_key: str,
        secret_key: str,
        token: Optional[str] = None,
        verify: bool = False,
    ) -> None:
        """
        Initialize RayJob manager.

        Args:
            domain: RayJob platform domain.
            tenant: Tenant name.
            access_key: Access key for authentication.
            secret_key: Secret key for authentication.
            token: Optional token for authentication.
            verify: Whether to verify HTTPS certificates.
        """
        self.client = RayJobClient(
            domain=domain,
            tenant=tenant,
            access_key=access_key,
            secret_key=secret_key,
            token=token,
            verify=verify,
        )
        self.tenant = tenant

    def create(
        self,
        project: str,
        name: Optional[str] = None,
        image: str = "",
        entrypoint: str = "",
        quotagroup: str = "",
        description: str = "",
        ray_version: str = "2.49.2",
        head_config: Optional[HeadConfig] = None,
        worker_group_config: Optional[List[WorkerGroupConfig]] = None,
        ttl_seconds: int = 604800,
        backoff_limit: int = 5,
        active_deadline_seconds: int = 86400,
    ) -> str:
        if not name or not str(name).strip():
            name = _random_name_hint()

        if head_config is None:
            head_config = HeadConfig(
                resources={
                    "cpu": "15",
                    "memory": "40Gi",
                    "nvidia.com/gpu": "0",
                }
            )
        if worker_group_config is None:
            # Default worker group
            worker_group_config = [
                WorkerGroupConfig(
                    groupName="worker-group-1",
                    positiveTags=[],
                    negativeTags=[],
                    localStorage="0",
                    privateMachine=self.tenant,
                    replicas=1,
                    resources={
                        "cpu": "3",
                        "memory": "10Gi",
                        "nvidia.com/gpu": "0",
                    },
                )
            ]

        try:
            result = self.client.create(
                project=project,
                name=name,
                image=image,
                entrypoint=entrypoint,
                quotagroup=quotagroup,
                description=description,
                rayVersion=ray_version,
                headConfg=head_config,
                workerGroupConfig=worker_group_config,
                ttlSecondsAfterFinished=ttl_seconds,
                backoffLimit=backoff_limit,
                activeDeadlineSeconds=active_deadline_seconds,
            )

            job_name = _extract_job_name(result)
            print(f"Created rayjob: {job_name} (name_hint={name})")
            return job_name

        except SDKException as e:
            print(f"Create failed: {getattr(e, 'code', 'UNKNOWN')} {e}", file=sys.stderr)
            raise

    def delete(self, project: str, name: str) -> Any:
        """Delete a RayJob (Ray cluster)."""
        try:
            result = self.client.delete(project=project, name=name)
            print(f"Deleted rayjob: {name}")
            return result
        except SDKException as e:
            print(f"Delete failed: {getattr(e, 'code', 'UNKNOWN')} {e}", file=sys.stderr)
            raise

    def list(self, project: str, verbose: bool = False) -> List[Any]:
        """List RayJobs under a project."""
        try:
            result = self.client.list(project=project)
            jobs = getattr(result, "data", result)
            if verbose:
                print(f"Found {getattr(result, 'total', len(jobs))} rayjobs in project={project}")
                for job in jobs:
                    print(" -", getattr(job, "jobName", getattr(job, "name", "UNKNOWN")))
            return list(jobs)
        except SDKException as e:
            print(f"List failed: {getattr(e, 'code', 'UNKNOWN')} {e}", file=sys.stderr)
            raise

    def get(self, project: str, name: str, verbose: bool = False) -> Any:
        """Get RayJob details."""
        try:
            result = self.client.get(project=project, name=name)
            if verbose:
                print(f"Rayjob {name} details:")
                print("  entrypoint:", getattr(result, "entrypoint", None))
                print("  creator:", getattr(result, "creatorid", None))
                worker_groups = getattr(result, "workerGroups", None)
                if worker_groups:
                    print("  worker replicas:", getattr(worker_groups[0], "replicas", None))
            return result
        except SDKException as e:
            print(f"Get failed: {getattr(e, 'code', 'UNKNOWN')} {e}", file=sys.stderr)
            raise

    def stop(self, project: str, name: str) -> Any:
        """Stop a running RayJob."""
        try:
            result = self.client.stop(project=project, name=name)
            print(f"Stopped rayjob: {name}")
            return result
        except SDKException as e:
            print(f"Stop failed: {getattr(e, 'code', 'UNKNOWN')} {e}", file=sys.stderr)
            raise

    def replicas(self, project: str, name: str, verbose: bool = False) -> List[Any]:
        """Get replica pod information for a RayJob."""
        try:
            result = self.client.replicas(project=project, name=name)
            pods = getattr(result, "data", result)
            if verbose:
                print(f"Replicas for {name} (total={getattr(result, 'total', len(pods))}):")
                for pod in pods:
                    print(
                        " -",
                        getattr(pod, "id", None),
                        getattr(pod, "nodeName", None),
                        getattr(pod, "podIP", None),
                    )
            return list(pods)
        except SDKException as e:
            print(f"Replicas failed: {getattr(e, 'code', 'UNKNOWN')} {e}", file=sys.stderr)
            raise

    def get_head_ip(self, project: str, name: str) -> Optional[str]:
        pods = self.replicas(project=project, name=name, verbose=False)
        if not pods:
            return None

        def _is_head(pod: Any) -> bool:
            # 1) Dedicated boolean flags
            for attr in ("isHead", "head"):
                v = getattr(pod, attr, None)
                if isinstance(v, bool) and v:
                    return True

            # 2) Role/type markers
            for attr in ("role", "type", "nodeType"):
                v = getattr(pod, attr, None)
                if isinstance(v, str) and v.lower() == "head":
                    return True

            # 3) Fallback: any name/id including the word 'head'
            for attr in ("name", "podName", "id"):
                v = getattr(pod, attr, None)
                if isinstance(v, str) and "head" in v.lower():
                    return True

            return False

        head_candidates = [pod for pod in pods if _is_head(pod)]
        target = head_candidates[0] if head_candidates else pods[0]

        ip = getattr(target, "podIP", None)
        return ip or None
