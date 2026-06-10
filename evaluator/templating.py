from __future__ import annotations

import json
import re
from typing import Any


_PLACEHOLDER_RE = re.compile(r"{{\s*([a-zA-Z_][\w.:-]*)\s*}}")


def render_template(template: str, variables: dict[str, Any]) -> str:
    """Render a small, dependency-free {{ dotted.path }} template."""

    def replace(match: re.Match[str]) -> str:
        key = match.group(1)
        value = resolve_path(variables, key)
        if isinstance(value, (dict, list, tuple)):
            return json.dumps(value, ensure_ascii=False, indent=2)
        if value is None:
            return ""
        return str(value)

    return _PLACEHOLDER_RE.sub(replace, template)


def resolve_path(data: dict[str, Any], dotted_path: str) -> Any:
    current: Any = data
    for part in dotted_path.split("."):
        if isinstance(current, dict):
            current = current.get(part)
        else:
            current = getattr(current, part, None)
        if current is None:
            return None
    return current
