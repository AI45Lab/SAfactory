"""
MCGym Environment Tests

This package contains test suites for the Minecraft Gym environment.
"""

from .test_mc_env import (
    test_environment_registration,
    test_initialization,
    test_reset,
    test_step,
    test_render,
    test_build_prompt,
    test_action_conversion,
    test_different_configs,
    test_fov_initialization,
)

__all__ = [
    'test_environment_registration',
    'test_initialization',
    'test_reset',
    'test_step',
    'test_render',
    'test_build_prompt',
    'test_action_conversion',
    'test_different_configs',
    'test_fov_initialization',
]

