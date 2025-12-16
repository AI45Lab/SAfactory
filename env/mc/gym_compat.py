"""
Gym Compatibility Shim for MineStudio

This module provides backward compatibility for MineStudio code that uses 'import gym'.
Since gym is deprecated, we use gymnasium as a drop-in replacement.
"""

import sys
import gymnasium

# Make gymnasium available as 'gym' module
sys.modules['gym'] = gymnasium

# Ensure gym.spaces is also available
if hasattr(gymnasium, 'spaces'):
    sys.modules['gym.spaces'] = gymnasium.spaces

