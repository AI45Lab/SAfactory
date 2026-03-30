from __future__ import annotations

import logging
import re
import secrets
import string

from typing import Any, List, Optional

from rayjob_sdk import HeadConfig, RayJobClient, SDKException, WorkerGroupConfig, Volume


log = logging.getLogger("rayjob")

# Platform constraint (DNS-1123 label-like):
# must match: [a-z]([-a-z0-9]*[a-z0-9])?
_JOBNAME_PATTERN = re.compile(r"^[a-z]([-a-z0-9]*[a-z0-9])?$")
_INVALID_CHARS = re.compile(r"[^a-z0-9-]+")


def _random_jobname_hint(length: int = 6) -> str:
    if length < 1:
        length = 1
    first = secrets.choice(string.ascii_lowercase)  # must start with [a-z]
    if length == 1:
        return first
    rest_choices = string.ascii_lowercase + string.digits
    rest = "".join(secrets.choice(rest_choices) for _ in range(length - 1))
    return first + rest


def _sanitize_jobname_hint(name: Optional[str], *, fallback_len: int = 6, max_len: int = 63) -> str:
    if not name or not str(name).strip():
        return _random_jobname_hint(fallback_len)

    s = str(name).strip().lower()
    s = _INVALID_CHARS.sub("-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")

    if not s:
        return _random_jobname_hint(fallback_len)

    # Must start with a letter
    if not ("a" <= s[0] <= "z"):
        s = "r" + s  # prefix with a letter

    # Enforce max length
    if len(s) > max_len:
        s = s[:max_len]

    # Cannot end with '-'
    s = s.rstrip("-")

    # If after trimming it's empty or invalid, fallback to random
    if not s or not _JOBNAME_PATTERN.match(s):
        return _random_jobname_hint(fallback_len)

    return s


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
        volumes:Optional[List[Volume]] = None,
        worker_group_config: Optional[List[WorkerGroupConfig]] = None,
        ttl_seconds: int = 604800,
        backoff_limit: int = 5,
        active_deadline_seconds: int = 86400,
    ) -> str:
        """
        Create a new RayJob (Ray cluster).

        IMPORTANT:
          - The platform may ignore the provided `name` and generate the final job name itself.
          - BUT the request `jobName` field must be VALID. So we pass a valid *hint*.
          - This method returns the **platform-created job name**.

        Returns:
            platform-created job name (string).
        """
        if not str(project).strip():
            raise ValueError("project must be non-empty")
        if not str(image).strip():
            raise ValueError("image must be non-empty")
        if not str(entrypoint).strip():
            raise ValueError("entrypoint must be non-empty")

        name_hint = _sanitize_jobname_hint(name, fallback_len=6, max_len=63)

        if head_config is None:
            head_config = HeadConfig(
                resources={
                    "cpu": "10",
                    "memory": "20Gi",
                    "nvidia.com/gpu": "0",
                },
                privileged=True
            )

        if worker_group_config is None:
            worker_group_config = [
                WorkerGroupConfig(
                    groupName="worker-group-1",
                    positiveTags=[],
                    negativeTags=[],
                    localStorage="0",
                    privateMachine=self.tenant,
                    replicas=0,
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
                name=name_hint,  # hint ONLY, but must be valid
                image=image,
                entrypoint=entrypoint,
                quotagroup=quotagroup,
                description=description,
                rayVersion=ray_version,
                headConfg=head_config,
                volumes=volumes,
                workerGroupConfig=worker_group_config,
                ttlSecondsAfterFinished=ttl_seconds,
                backoffLimit=backoff_limit,
                activeDeadlineSeconds=active_deadline_seconds,
                privateMachine="yes",
            )

            job_name = _extract_job_name(result)
            log.info("Created rayjob: %s (jobName_hint=%s)", job_name, name_hint)
            return job_name

        except SDKException as e:
            log.error("Create failed: %s %s", getattr(e, 'code', 'UNKNOWN'), e)
            raise

    def delete(self, project: str, name: str) -> Any:
        try:
            result = self.client.delete(project=project, name=name)
            log.info("Deleted rayjob: %s", name)
            return result
        except SDKException as e:
            log.error("Delete failed: %s %s", getattr(e, 'code', 'UNKNOWN'), e)
            raise

    def list(self, project: str, verbose: bool = False) -> List[Any]:
        try:
            result = self.client.list(project=project)
            jobs = getattr(result, "data", result)
            if verbose:
                log.info("Found %d rayjobs in project=%s", getattr(result, 'total', len(jobs)), project)
                for job in jobs:
                    log.info(" - %s", getattr(job, "jobName", getattr(job, "name", "UNKNOWN")))
            return list(jobs)
        except SDKException as e:
            log.error("List failed: %s %s", getattr(e, 'code', 'UNKNOWN'), e)
            raise

    def get(self, project: str, name: str, verbose: bool = False) -> Any:
        try:
            result = self.client.get(project=project, name=name)
            if verbose:
                log.info("Rayjob %s details:", name)
                log.info("  entrypoint: %s", getattr(result, "entrypoint", None))
                log.info("  creator: %s", getattr(result, "creatorid", None))
                worker_groups = getattr(result, "workerGroups", None)
                if worker_groups:
                    log.info("  worker replicas: %s", getattr(worker_groups[0], "replicas", None))
            return result
        except SDKException as e:
            log.error("Get failed: %s %s", getattr(e, 'code', 'UNKNOWN'), e)
            raise

    def stop(self, project: str, name: str) -> Any:
        try:
            result = self.client.stop(project=project, name=name)
            log.info("Stopped rayjob: %s", name)
            return result
        except SDKException as e:
            log.error("Stop failed: %s %s", getattr(e, 'code', 'UNKNOWN'), e)
            raise

    def replicas(self, project: str, name: str, verbose: bool = False) -> List[Any]:
        try:
            result = self.client.replicas(project=project, name=name)
            pods = getattr(result, "data", result)
            if verbose:
                log.info("Replicas for %s (total=%d):", name, getattr(result, 'total', len(pods)))
                for pod in pods:
                    log.info(
                        " - %s %s %s",
                        getattr(pod, "id", None),
                        getattr(pod, "nodeName", None),
                        getattr(pod, "podIP", None),
                    )
            return list(pods)
        except SDKException as e:
            log.error("Replicas failed: %s %s", getattr(e, 'code', 'UNKNOWN'), e)
            raise

    def get_head_ip(self, project: str, name: str) -> Optional[str]:
        pods = self.replicas(project=project, name=name, verbose=False)
        if not pods:
            return None

        def _is_head(pod: Any) -> bool:
            for attr in ("isHead", "head"):
                v = getattr(pod, attr, None)
                if isinstance(v, bool) and v:
                    return True
            for attr in ("role", "type", "nodeType"):
                v = getattr(pod, attr, None)
                if isinstance(v, str) and v.lower() == "head":
                    return True
            for attr in ("name", "podName", "id"):
                v = getattr(pod, attr, None)
                if isinstance(v, str) and "head" in v.lower():
                    return True
            return False

        head_candidates = [pod for pod in pods if _is_head(pod)]
        target = head_candidates[0] if head_candidates else pods[0]

        ip = getattr(target, "podIP", None)
        return ip or None