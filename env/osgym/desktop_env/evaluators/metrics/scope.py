import re
from typing import Any, Dict


_REWARD_RE = re.compile(r"REWARD:\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+))", re.IGNORECASE)


def scope_reward(result: Dict[str, Any], **_: Any) -> float:
    """Parse a SCOPE reward.py execution result into an OSWorld score."""
    if result is None:
        return 0.0

    output = str(result.get("output") or "")
    error = str(result.get("error") or "")
    matches = _REWARD_RE.findall(output + "\n" + error)
    if not matches:
        return 0.0

    try:
        score = float(matches[-1])
    except ValueError:
        return 0.0

    return max(0.0, min(1.0, score))
