from __future__ import annotations

from typing import List, Optional, Any
import random
import sys

from rayjob_sdk import RayJobClient, HeadConfig, WorkerGroupConfig, SDKException


class RayJobManager:
    """Lightweight manager class for RayJob (Ray cluster) operations."""

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
            domain: RayJob platform domain
            tenant: Tenant name
            access_key: Access key for authentication
            secret_key: Secret key for authentication
            token: Optional token for authentication
            verify: Whether to verify HTTPS certificates
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
    ) -> Any:
        """
        Create a new RayJob (Ray cluster).

        Args:
            project: Project name.
            name: Job name. Auto-generated if None.
            image: Container image used for Ray cluster pods.
            entrypoint: Command executed inside the Ray head pod.
            quotagroup: Quota group name.
            description: Job description.
            ray_version: Ray version string.
            head_config: Head node configuration.
            worker_group_config: Worker group configuration list.
            ttl_seconds: TTL after job finishes.
            backoff_limit: Retry limit.
            active_deadline_seconds: Job active deadline.

        Returns:
            The created RayJob object (SDK model).

        Raises:
            SDKException: If creation fails.
        """
        if name is None:
            name = f"sdk{random.randint(10000, 99999)}"

        #TODO: how many resources is required should be set though the config.yaml
        if head_config is None:
            head_config = HeadConfig(
                resources={
                    "cpu": "15",
                    "memory": "40Gi",
                    "nvidia.com/gpu": "0",
                }
            )

        if worker_group_config is None:
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

            #TODO： replace this with log instead of print
            print(f"Created rayjob: {getattr(result, 'jobName', getattr(result, 'name', 'UNKNOWN'))}")
            return result.metadata.name
        except SDKException as e:
            print(f"Create failed: {e.code} {e}", file=sys.stderr)
            raise

    def delete(self, project: str, name: str) -> Any:
        """
        Delete a RayJob (Ray cluster).

        Args:
            project: Project name.
            name: Job name.

        Returns:
            Delete result (SDK model).

        Raises:
            SDKException: If deletion fails.
        """
        try:
            result = self.client.delete(project=project, name=name)
            print(f"Deleted rayjob: {name}")
            return result
        except SDKException as e:
            print(f"Delete failed: {e.code} {e}", file=sys.stderr)
            raise

    def list(self, project: str, verbose: bool = False) -> List[Any]:
        """
        List RayJobs (Ray clusters) in a project.

        Args:
            project: Project name.
            verbose: Print details if True.

        Returns:
            List of RayJob objects.
        """
        try:
            result = self.client.list(project=project)
            if verbose:
                print(f"Found {result.total} rayjobs:")
                for job in result.data:
                    print(f" - {job.jobName}")
            return result.data
        except SDKException as e:
            print(f"List failed: {e.code} {e}", file=sys.stderr)
            raise

    def get(self, project: str, name: str, verbose: bool = False) -> Any:
        """
        Get detailed information about a specific RayJob (Ray cluster).

        Args:
            project: Project name.
            name: Job name.
            verbose: Print details if True.

        Returns:
            RayJob object (SDK model).

        Raises:
            SDKException: If get operation fails.
        """
        try:
            result = self.client.get(project=project, name=name)
            if verbose:
                print(f"Rayjob {name} details:")
                print(f"  entrypoint: {result.entrypoint}")
                print(f"  creator: {result.creatorid}")
                if result.workerGroups:
                    print(f"  worker replicas: {result.workerGroups[0].replicas}")
            return result
        except SDKException as e:
            print(f"Get failed: {e.code} {e}", file=sys.stderr)
            raise

    def stop(self, project: str, name: str) -> Any:
        """
        Stop a running RayJob (Ray cluster).

        Args:
            project: Project name.
            name: Job name.

        Returns:
            Stop result (SDK model).

        Raises:
            SDKException: If stop operation fails.
        """
        try:
            result = self.client.stop(project=project, name=name)
            print(f"Stopped rayjob: {name}")
            return result
        except SDKException as e:
            print(f"Stop failed: {e.code} {e}", file=sys.stderr)
            raise

    def replicas(self, project: str, name: str, verbose: bool = False) -> List[Any]:
        """
        Get replica (pod) information for a RayJob (Ray cluster).

        Args:
            project: Project name.
            name: Job name.
            verbose: Print replica info if True.

        Returns:
            List of replica objects from SDK. Each pod usually has:
              - id
              - nodeName
              - podIP

        Raises:
            SDKException: If replicas operation fails.
        """
        try:
            result = self.client.replicas(project=project, name=name)
            if verbose:
                print(f"Replicas for {name} (total: {result.total}):")
                for pod in result.data:
                    print(f" - {pod.id}: {pod.nodeName} ({pod.podIP})")
            return result.data
        except SDKException as e:
            print(f"Replicas failed: {e.code} {e}", file=sys.stderr)
            raise

    def get_head_ip(self, project: str, name: str) -> Optional[str]:
        pods = self.replicas(project=project, name=name, verbose=False)
        if not pods:
            return None

        def _is_head(pod: Any) -> bool:
            """Best-effort detection of the head pod."""
            # 1) Dedicated boolean flags, e.g. isHead / head
            for attr in ("isHead", "head"):
                value = getattr(pod, attr, None)
                if isinstance(value, bool) and value:
                    return True

            # 2) Role / type markers such as role == "head"
            for attr in ("role", "type", "nodeType"):
                value = getattr(pod, attr, None)
                if isinstance(value, str) and value.lower() == "head":
                    return True

            # 3) Fallback: any name/id including the word 'head'
            for attr in ("name", "podName", "id"):
                value = getattr(pod, attr, None)
                if isinstance(value, str) and "head" in value.lower():
                    return True

            return False

        # Prefer an explicitly detected head pod; otherwise use the first pod.
        head_candidates = [pod for pod in pods if _is_head(pod)]
        target = head_candidates[0] if head_candidates else pods[0]

        ip = getattr(target, "podIP", None)
        return ip or None