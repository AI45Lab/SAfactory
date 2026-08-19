"""OSGym package exports without eagerly loading desktop dependencies."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .os_env import OSGym

__all__ = ["OSGym"]


def __getattr__(name: str):
    if name == "OSGym":
        from .os_env import OSGym

        return OSGym
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
