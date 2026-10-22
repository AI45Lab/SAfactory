from types import SimpleNamespace
from unittest.mock import patch

import pytest

from clusters.rjob_cluster import RJobClusterBackend


class _Struct:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def _symbols():
    return {
        name: _Struct
        for name in (
            "Job",
            "Metadata",
            "Spec",
            "Task",
            "Template",
            "Container",
            "Resources",
            "Affinity",
        )
    }


def test_build_job_passes_node_affinity_to_template():
    backend = RJobClusterBackend()
    backend.symbols = _symbols
    node_tag = "node/lg-cmc-h-cpu-0042.host.h.pjlab.org.cn"

    job = backend.build_job(
        cfg={
            "affinity": {"positive_tags": [node_tag]},
            "resources": {"cpu": 1, "memory_in_mb": 512},
        },
        lease=SimpleNamespace(
            image="busybox:latest",
            run_command="hostname",
            workdir="",
            agent_name="test-agent",
            agent_id="test-agent-id",
        ),
        request=SimpleNamespace(job_id="test-job", session_id="test-session"),
        rjob_name="test-rjob-affinity",
        env={},
    )

    template = job.spec.tasks["main"].template
    assert template.affinity.positive_tags == [node_tag]


def test_build_affinity_accepts_sdk_camel_case_fields():
    affinity = RJobClusterBackend._build_affinity(
        _symbols(),
        {"positiveTags": "node/lg-cmc-h-cpu-0097.host.h.pjlab.org.cn"},
    )
    assert affinity.positive_tags == ["node/lg-cmc-h-cpu-0097.host.h.pjlab.org.cn"]


def test_build_affinity_selects_one_random_positive_tag_per_job():
    candidates = [
        "node/lg-cmc-h-cpu-0042.host.h.pjlab.org.cn",
        "node/lg-cmc-h-cpu-0097.host.h.pjlab.org.cn",
    ]
    with patch("clusters.rjob_cluster.random.choice", return_value=candidates[1]) as choose:
        affinity = RJobClusterBackend._build_affinity(
            _symbols(),
            {
                "positive_tags": ["feature/runtime=docker"],
                "random_positive_tags": candidates,
            },
        )

    choose.assert_called_once_with(candidates)
    assert affinity.positive_tags == ["feature/runtime=docker", candidates[1]]


def test_build_affinity_rejects_unknown_fields():
    with pytest.raises(ValueError, match="Unsupported RJob affinity field"):
        RJobClusterBackend._build_affinity(_symbols(), {"node_name": "host-1"})
