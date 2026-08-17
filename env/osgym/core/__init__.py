"""
OSGym Core Modules

This package contains core functionality modules extracted from os_env.py
for better code organization and maintainability.
"""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .action_flow import ActionFlow
    from .observation_processor import ObservationProcessor
    from .prompt_session import PromptSession
    from .repeated_action_detector import RepeatedActionDetector

__all__ = [
    "ActionFlow",
    "ObservationProcessor",
    "PromptSession",
    "RepeatedActionDetector",
]

_EXPORTS = {
    "ActionFlow": (".action_flow", "ActionFlow"),
    "ObservationProcessor": (".observation_processor", "ObservationProcessor"),
    "PromptSession": (".prompt_session", "PromptSession"),
    "RepeatedActionDetector": (".repeated_action_detector", "RepeatedActionDetector"),
}


def __getattr__(name: str):
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = target
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value
