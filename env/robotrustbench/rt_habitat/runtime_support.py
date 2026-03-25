import logging
import os
from typing import Any


LOGGER = logging.getLogger(__name__)

_RT_HAB_DIR = os.path.dirname(__file__)
_VERSIONED_DATA_CANDIDATES = [
    os.path.normpath(
        os.path.join(
            _RT_HAB_DIR,
            "..",
            "EmbodiedBench",
            "embodiedbench",
            "envs",
            "eb_habitat",
            "data",
            "versioned_data",
        )
    ),
    os.path.normpath(
        os.path.join(
            _RT_HAB_DIR,
            "..",
            "..",
            "embodiedgym",
            "EmbodiedBench",
            "embodiedbench",
            "envs",
            "eb_habitat",
            "data",
            "versioned_data",
        )
    ),
]
_LEGACY_EMBODIEDBENCH_DATA_CANDIDATES = [
    os.path.normpath(os.path.join(_RT_HAB_DIR, "..", "EmbodiedBench", "data")),
    os.path.normpath(
        os.path.join(_RT_HAB_DIR, "..", "..", "embodiedgym", "EmbodiedBench", "data")
    ),
]


def _prefix(path, trailing_slash=False):
    normalized = os.path.normpath(path).replace("\\", "/")
    if trailing_slash:
        return normalized + "/"
    return normalized


def _resolve_versioned_data_root():
    for candidate in _VERSIONED_DATA_CANDIDATES:
        replica_cfg = os.path.join(
            candidate,
            "replica_cad_dataset",
            "replicaCAD.scene_dataset_config.json",
        )
        ycb_cfg = os.path.join(candidate, "ycb", "configs")
        if os.path.isfile(replica_cfg) and os.path.isdir(ycb_cfg):
            return candidate
    return _VERSIONED_DATA_CANDIDATES[0]


RT_VERSIONED_DATA_ROOT = _resolve_versioned_data_root()
RT_REPLICA_CAD_ROOT = os.path.join(RT_VERSIONED_DATA_ROOT, "replica_cad_dataset")
RT_YCB_CONFIG_ROOT = os.path.join(RT_VERSIONED_DATA_ROOT, "ycb", "configs")
RT_HAB_FETCH_ROOT = os.path.join(RT_VERSIONED_DATA_ROOT, "hab_fetch")
RT_HAB_FETCH_OLD_ROOT = os.path.join(RT_VERSIONED_DATA_ROOT, "hab_fetch_1.0")


def _build_path_prefix_rewrites():
    rewrites = [
        (
            "/data/zxy/Stereotypes/EmbodiedBench/data/replica_cad/",
            RT_REPLICA_CAD_ROOT + "/",
        ),
        (
            "/data/zxy/EmbodiedBench/robotrustbench/envs/rt_habitat/data/versioned_data/replica_cad_dataset/",
            RT_REPLICA_CAD_ROOT + "/",
        ),
        ("./data/replica_cad/", RT_REPLICA_CAD_ROOT + "/"),
        ("data/replica_cad/", RT_REPLICA_CAD_ROOT + "/"),
        (
            "/data/zxy/Stereotypes/EmbodiedBench/data/objects/ycb/configs",
            RT_YCB_CONFIG_ROOT,
        ),
        (
            "/data/zxy/EmbodiedBench/robotrustbench/envs/rt_habitat/data/versioned_data/ycb/configs",
            RT_YCB_CONFIG_ROOT,
        ),
        ("./data/objects/ycb/configs/", RT_YCB_CONFIG_ROOT + "/"),
        ("./data/objects/ycb/configs", RT_YCB_CONFIG_ROOT),
        ("data/objects/ycb/configs/", RT_YCB_CONFIG_ROOT + "/"),
        ("data/objects/ycb/configs", RT_YCB_CONFIG_ROOT),
        (RT_HAB_FETCH_OLD_ROOT + "/", RT_HAB_FETCH_ROOT + "/"),
        ("./data/robots/hab_fetch_1.0/", RT_HAB_FETCH_ROOT + "/"),
        ("data/robots/hab_fetch_1.0/", RT_HAB_FETCH_ROOT + "/"),
        ("./data/robots/hab_fetch/", RT_HAB_FETCH_ROOT + "/"),
        ("data/robots/hab_fetch/", RT_HAB_FETCH_ROOT + "/"),
    ]

    for candidate in _LEGACY_EMBODIEDBENCH_DATA_CANDIDATES:
        rewrites.extend(
            [
                (
                    _prefix(os.path.join(candidate, "replica_cad"), trailing_slash=True),
                    RT_REPLICA_CAD_ROOT + "/",
                ),
                (
                    _prefix(os.path.join(candidate, "objects", "ycb", "configs")),
                    RT_YCB_CONFIG_ROOT,
                ),
            ]
        )

    return rewrites


_PATH_PREFIX_REWRITES = _build_path_prefix_rewrites()


def prepare_egl_runtime(logger, env_label):
    os.environ.pop("DISPLAY", None)
    os.environ.pop("WAYLAND_DISPLAY", None)
    os.environ.setdefault(
        "__EGL_VENDOR_LIBRARY_FILENAMES",
        "/usr/share/glvnd/egl_vendor.d/10_nvidia.json",
    )
    os.environ.setdefault("__GLX_VENDOR_LIBRARY_NAME", "nvidia")
    os.environ.setdefault("__GLVND_DISALLOW_PATCHING", "1")
    os.environ.setdefault("EGL_VISIBLE_DEVICES", "0")

    logger.info(
        "%s runtime: CUDA_VISIBLE_DEVICES=%s EGL_VISIBLE_DEVICES=%s "
        "__EGL_VENDOR_LIBRARY_FILENAMES=%s",
        env_label,
        os.environ.get("CUDA_VISIBLE_DEVICES"),
        os.environ.get("EGL_VISIBLE_DEVICES"),
        os.environ.get("__EGL_VENDOR_LIBRARY_FILENAMES"),
    )


def rewrite_rt_data_path(value):
    if not isinstance(value, str):
        return value
    normalized = value.replace("\\", "/")
    for old_prefix, new_prefix in _PATH_PREFIX_REWRITES:
        if normalized.startswith(old_prefix):
            return os.path.normpath(new_prefix + normalized[len(old_prefix) :])
    return value


def patch_dataset_episode_paths(dataset):
    episodes = getattr(dataset, "episodes", None)
    if not episodes:
        return 0

    changes = 0
    for episode in episodes:
        for attr_name in (
            "scene_id",
            "scene_dataset_config",
            "additional_obj_config_paths",
        ):
            if not hasattr(episode, attr_name):
                continue
            current_value = getattr(episode, attr_name)
            if isinstance(current_value, list):
                patched_value = [rewrite_rt_data_path(item) for item in current_value]
            else:
                patched_value = rewrite_rt_data_path(current_value)
            if patched_value != current_value:
                setattr(episode, attr_name, patched_value)
                changes += 1
    return changes


def patch_simulator_resource_paths(config):
    changes = 0
    simulator = config.habitat.simulator

    additional_paths = getattr(simulator, "additional_object_paths", None)
    if additional_paths:
        patched_paths = [rewrite_rt_data_path(path) for path in additional_paths]
        if list(patched_paths) != list(additional_paths):
            simulator.additional_object_paths = patched_paths
            changes += 1

    main_agent = simulator.agents.main_agent
    robot_urdf = getattr(main_agent, "articulated_agent_urdf", None)
    patched_urdf = rewrite_rt_data_path(robot_urdf)
    if patched_urdf != robot_urdf:
        main_agent.articulated_agent_urdf = patched_urdf
        changes += 1

    return changes
