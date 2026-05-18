try:
    from rl.utils import get_env, AggType, MetricsRecorder
except ImportError as exc:
    _RL_IMPORT_ERROR = exc

    def get_env(*args, **kwargs):
        raise ImportError("rl.utils is unavailable; install RL optional dependencies first") from _RL_IMPORT_ERROR

    class AggType:  # type: ignore[no-redef]
        pass

    class MetricsRecorder:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError("rl.utils is unavailable; install RL optional dependencies first") from _RL_IMPORT_ERROR

__all__ = [
    "get_env",
    "AggType",
    "MetricsRecorder",
]
