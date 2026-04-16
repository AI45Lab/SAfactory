from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

ActorKey = Tuple[str, str]     # (env_name, env_id)
ActorRoute = Tuple[str, int]   # (host, port)


@dataclass(slots=True)
class RayClusterInfo:
    """
    Descriptor of one cluster.

    Remote mode: one image -> one RayJob; head_ip resolved by polling.
    Local mode: synthetic cluster info; head_ip is local host.
    """
    image: str
    project: str
    job_name: str
    head_ip: str


@dataclass(slots=True)
class EnvClusterBinding:
    """
    Binding between env_name and a cluster.
    """
    env_name: str
    image: str
    project: str
    job_name: str
    head_ip: str


@dataclass(slots=True)
class PoolEntry:
    """
    Local record of a ready actor, regardless of transport.
    """
    env_name: str
    env_id: str
    row_id: Optional[int]
    image: str
    job_name: str
    head_ip: str
    group_id: str = ""
    status: str = "ready"
    transport: str = "http"
    local_env: Optional[Any] = None


@dataclass(slots=True)
class ClusterRegistry:
    """
    Shared registry used by pool and manager.
    """
    clusters_by_id: Dict[str, RayClusterInfo]
    env_bindings: Dict[str, EnvClusterBinding]
