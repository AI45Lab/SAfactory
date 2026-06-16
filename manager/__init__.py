from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .manager import AgentPoolManager

__all__ = ["AgentPoolManager"]


def __getattr__(name: str):
    if name == "AgentPoolManager":
        from .manager import AgentPoolManager

        return AgentPoolManager
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
