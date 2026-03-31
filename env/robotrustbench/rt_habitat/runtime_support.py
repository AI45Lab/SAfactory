import logging
import os
import sys
import ctypes
from typing import Any


LOGGER = logging.getLogger(__name__)

_RT_HAB_DIR = os.path.dirname(__file__)
_RT_HAB_RESOURCE_ROOT_ENVVAR = "AIEVOBOX_RT_HABITAT_RESOURCE_ROOT"
_RT_HAB_RESOURCE_ROOT_CANDIDATES = [
    os.environ.get(_RT_HAB_RESOURCE_ROOT_ENVVAR, "").strip(),
    _RT_HAB_DIR,
    os.path.join(os.sep, "opt", "robotrustbench", "rt_habitat"),
]
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


def get_rt_resource_root():
    for candidate in _RT_HAB_RESOURCE_ROOT_CANDIDATES:
        if candidate and os.path.isdir(candidate):
            return os.path.normpath(candidate)
    return os.path.normpath(_RT_HAB_DIR)


def get_rt_resource_path(*parts):
    return os.path.join(get_rt_resource_root(), *parts)


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
_PATH_MARKER_REWRITES = [
    (
        "EmbodiedBench/data/replica_cad/",
        RT_REPLICA_CAD_ROOT + "/",
    ),
    (
        "robotrustbench/envs/rt_habitat/data/versioned_data/replica_cad_dataset/",
        RT_REPLICA_CAD_ROOT + "/",
    ),
    (
        "EmbodiedBench/data/objects/ycb/configs",
        RT_YCB_CONFIG_ROOT,
    ),
    (
        "robotrustbench/envs/rt_habitat/data/versioned_data/ycb/configs",
        RT_YCB_CONFIG_ROOT,
    ),
]


def _build_path_prefix_rewrites():
    rewrites = [
        ("./data/replica_cad/", RT_REPLICA_CAD_ROOT + "/"),
        ("data/replica_cad/", RT_REPLICA_CAD_ROOT + "/"),
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


def _ensure_nvidia_runtime_env():
    nvidia_lib_dirs = [
        os.path.join(os.sep, "usr", "local", "nvidia", "lib64"),
        os.path.join(os.sep, "usr", "local", "nvidia", "lib"),
    ]
    current_ld_paths = [
        path
        for path in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep)
        if path
    ]
    for lib_dir in reversed(nvidia_lib_dirs):
        if os.path.isdir(lib_dir) and lib_dir not in current_ld_paths:
            current_ld_paths.insert(0, lib_dir)
    if current_ld_paths:
        os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(current_ld_paths)

    os.environ.pop("DISPLAY", None)
    os.environ.pop("WAYLAND_DISPLAY", None)
    os.environ.setdefault(
        "__EGL_VENDOR_LIBRARY_FILENAMES",
        os.path.join(
            os.sep, "usr", "share", "glvnd", "egl_vendor.d", "10_nvidia.json"
        ),
    )
    os.environ.setdefault("__GLX_VENDOR_LIBRARY_NAME", "nvidia")
    os.environ.setdefault("__GLVND_DISALLOW_PATCHING", "1")

    preload_candidates = [
        "libcuda.so.1",
        "libnvidia-ml.so.1",
        "libEGL_nvidia.so.0",
    ]
    for lib_dir in nvidia_lib_dirs:
        if not os.path.isdir(lib_dir):
            continue
        for library_name in preload_candidates:
            library_path = os.path.join(lib_dir, library_name)
            if not os.path.exists(library_path):
                continue
            try:
                ctypes.CDLL(library_path, mode=ctypes.RTLD_GLOBAL)
            except Exception:
                continue


def bootstrap_gpu_runtime():
    nvidia_lib_dir = os.path.join(os.sep, "usr", "local", "nvidia", "lib64")
    current_ld = os.environ.get("LD_LIBRARY_PATH", "")
    current_ld_paths = [path for path in current_ld.split(os.pathsep) if path]
    can_reexec = bool(sys.argv) and sys.argv[0] not in ("-c", "-")
    if (
        os.path.isdir(nvidia_lib_dir)
        and nvidia_lib_dir not in current_ld_paths
        and not os.environ.get("_AIEVOBOX_RT_GPU_RUNTIME_BOOTSTRAPPED")
        and can_reexec
    ):
        new_env = os.environ.copy()
        new_env["_AIEVOBOX_RT_GPU_RUNTIME_BOOTSTRAPPED"] = "1"
        new_env["LD_LIBRARY_PATH"] = os.pathsep.join([nvidia_lib_dir] + current_ld_paths)
        os.execvpe(sys.executable, [sys.executable] + sys.argv, new_env)
    _ensure_nvidia_runtime_env()


def _rewrite_marker_path(normalized_value):
    for marker, new_prefix in _PATH_MARKER_REWRITES:
        marker_index = normalized_value.find(marker)
        if marker_index == -1:
            continue
        suffix = normalized_value[marker_index + len(marker) :]
        return os.path.normpath(new_prefix + suffix)
    return None


def prepare_egl_runtime(logger, env_label):
    _ensure_nvidia_runtime_env()

    selected_gpu_device_id = 0
    selected_egl_index = None
    selected_cuda_id = None

    try:
        libegl = ctypes.CDLL("libEGL.so.1")
        get_proc = libegl.eglGetProcAddress
        get_proc.argtypes = [ctypes.c_char_p]
        get_proc.restype = ctypes.c_void_p

        query_devices_ptr = get_proc(b"eglQueryDevicesEXT")
        query_device_attr_ptr = get_proc(b"eglQueryDeviceAttribEXT")
        if query_devices_ptr and query_device_attr_ptr:
            query_devices = ctypes.CFUNCTYPE(
                ctypes.c_uint32,
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_int),
            )(query_devices_ptr)
            query_device_attr = ctypes.CFUNCTYPE(
                ctypes.c_uint32,
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_longlong),
            )(query_device_attr_ptr)

            num_devices = ctypes.c_int()
            if query_devices(32, None, ctypes.byref(num_devices)) and num_devices.value > 0:
                devices = (ctypes.c_void_p * num_devices.value)()
                if query_devices(num_devices.value, devices, ctypes.byref(num_devices)):
                    egl_cuda_pairs = []
                    for egl_index in range(num_devices.value):
                        cuda_id = ctypes.c_longlong(-1)
                        ok = query_device_attr(
                            devices[egl_index], 0x323A, ctypes.byref(cuda_id)
                        )
                        if ok:
                            egl_cuda_pairs.append((egl_index, int(cuda_id.value)))

                    requested_cuda_ids = []
                    raw_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
                    if raw_visible_devices:
                        for item in raw_visible_devices.split(","):
                            item = item.strip()
                            if not item:
                                continue
                            try:
                                requested_cuda_ids.append(int(item))
                            except ValueError:
                                logger.warning(
                                    "%s runtime: ignored non-integer CUDA_VISIBLE_DEVICES entry %r",
                                    env_label,
                                    item,
                                )

                    selected_pair = None
                    if requested_cuda_ids:
                        for requested_cuda_id in requested_cuda_ids:
                            for pair in egl_cuda_pairs:
                                if pair[1] == requested_cuda_id:
                                    selected_pair = pair
                                    break
                            if selected_pair is not None:
                                break
                    elif egl_cuda_pairs:
                        selected_pair = egl_cuda_pairs[0]

                    if selected_pair is not None:
                        selected_egl_index, selected_cuda_id = selected_pair
                        os.environ["EGL_VISIBLE_DEVICES"] = str(selected_egl_index)
                        if not requested_cuda_ids:
                            os.environ["CUDA_VISIBLE_DEVICES"] = str(selected_cuda_id)
                            os.environ["NVIDIA_VISIBLE_DEVICES"] = str(selected_cuda_id)
                            selected_gpu_device_id = selected_cuda_id
                        else:
                            selected_gpu_device_id = requested_cuda_ids.index(
                                selected_cuda_id
                            )
                        logger.info(
                            "%s runtime: selected EGL device %s for CUDA device %s",
                            env_label,
                            selected_egl_index,
                            selected_cuda_id,
                        )
        else:
            logger.warning(
                "%s runtime: EGL query extensions unavailable (query_devices=%s query_device_attr=%s)",
                env_label,
                bool(query_devices_ptr),
                bool(query_device_attr_ptr),
            )
    except Exception as exc:
        logger.warning("%s runtime: EGL device probing failed: %s", env_label, exc)

    os.environ.setdefault("EGL_VISIBLE_DEVICES", "0")

    logger.info(
        "%s runtime: CUDA_VISIBLE_DEVICES=%s NVIDIA_VISIBLE_DEVICES=%s "
        "EGL_VISIBLE_DEVICES=%s gpu_device_id=%s __EGL_VENDOR_LIBRARY_FILENAMES=%s",
        env_label,
        os.environ.get("CUDA_VISIBLE_DEVICES"),
        os.environ.get("NVIDIA_VISIBLE_DEVICES"),
        os.environ.get("EGL_VISIBLE_DEVICES"),
        selected_gpu_device_id,
        os.environ.get("__EGL_VENDOR_LIBRARY_FILENAMES"),
    )
    return selected_gpu_device_id


def rewrite_rt_data_path(value):
    if not isinstance(value, str):
        return value
    normalized = value.replace("\\", "/")
    for old_prefix, new_prefix in _PATH_PREFIX_REWRITES:
        if normalized.startswith(old_prefix):
            return os.path.normpath(new_prefix + normalized[len(old_prefix) :])
    marker_rewrite = _rewrite_marker_path(normalized)
    if marker_rewrite is not None:
        return marker_rewrite
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
