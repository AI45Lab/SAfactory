from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict

SAFACTORY_INTERNAL_ENV_KEY = "__safactory_internal__"


def strip_internal_env_params(env_params: Dict[str, Any] | None) -> Dict[str, Any]:
    """Return env params safe to expose to the target agent."""
    if not isinstance(env_params, dict):
        return {}
    public_params = deepcopy(env_params)
    public_params.pop(SAFACTORY_INTERNAL_ENV_KEY, None)
    return public_params
