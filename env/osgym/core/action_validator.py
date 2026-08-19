"""Action validation metadata for OSGym rollouts."""

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


TOOL_CALL_RE = re.compile(r"<tool_call>\s*(?P<payload>.*?)\s*</tool_call>", re.DOTALL)

COORDINATE_ACTIONS = {
    "left_click",
    "click",
    "right_click",
    "middle_click",
    "double_click",
    "triple_click",
    "mouse_move",
    "left_click_drag",
}

SUPPORTED_ACTIONS = COORDINATE_ACTIONS | {
    "key",
    "key_down",
    "key_up",
    "hold_key",
    "type",
    "cursor_position",
    "left_mouse_down",
    "left_mouse_up",
    "scroll",
    "hscroll",
    "wait",
    "screenshot",
    "terminate",
}

MODIFIER_KEYS = {"ctrl", "control", "alt", "shift", "command", "cmd", "win", "super"}
NAMED_KEYS = {
    "enter",
    "return",
    "esc",
    "escape",
    "tab",
    "space",
    "backspace",
    "delete",
    "del",
    "insert",
    "home",
    "end",
    "pageup",
    "pagedown",
    "up",
    "down",
    "left",
    "right",
    "capslock",
    "numlock",
    "scrolllock",
    "printscreen",
    "pause",
    "menu",
}
FUNCTION_KEYS = {f"f{i}" for i in range(1, 25)}
SINGLE_CHAR_KEYS = {chr(code) for code in range(ord("a"), ord("z") + 1)} | {
    str(i) for i in range(10)
} | set("`-=[]\\;',./")
VALID_KEYS = MODIFIER_KEYS | NAMED_KEYS | FUNCTION_KEYS | SINGLE_CHAR_KEYS

SHELL_LIKE_RE = re.compile(
    r"(\b(?:sudo|apt|pip|python|bash|sh|cd|ls|rm|mv|cp|cat|grep|curl|wget|git)\b|[;&|`$<>/])",
    re.IGNORECASE,
)


@dataclass
class ActionValidationResult:
    """Compact metadata used by RL masking; it does not block execution."""

    valid: bool = True
    syntax_invalid: bool = False
    semantic_invalid: bool = False
    parser_failed: bool = False
    fallback_wait: bool = False
    invalid_reasons: List[str] = field(default_factory=list)
    suspicious_reasons: List[str] = field(default_factory=list)
    action_names: List[str] = field(default_factory=list)
    terminate_statuses: List[str] = field(default_factory=list)
    tool_call_count: int = 0
    parsed_action_count: int = 0

    def add_syntax(self, reason: str) -> None:
        self.syntax_invalid = True
        self.invalid_reasons.append(reason)

    def add_semantic(self, reason: str) -> None:
        self.semantic_invalid = True
        self.invalid_reasons.append(reason)

    def add_suspicious(self, reason: str) -> None:
        self.suspicious_reasons.append(reason)

    def finalize(self) -> "ActionValidationResult":
        if self.parser_failed:
            self.syntax_invalid = True
            self.invalid_reasons.append("parser_failed")
        self.invalid_reasons = sorted(set(self.invalid_reasons))
        self.suspicious_reasons = sorted(set(self.suspicious_reasons))
        self.valid = not self.syntax_invalid and not self.semantic_invalid
        return self

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valid": self.valid,
            "syntax_invalid": self.syntax_invalid,
            "semantic_invalid": self.semantic_invalid,
            "parser_failed": self.parser_failed,
            "fallback_wait": self.fallback_wait,
            "invalid_reasons": list(self.invalid_reasons),
            "suspicious_reasons": list(self.suspicious_reasons),
            "action_names": list(self.action_names),
            "terminate_statuses": list(self.terminate_statuses),
            "tool_call_count": self.tool_call_count,
            "parsed_action_count": self.parsed_action_count,
        }


def validate_osgym_action(
    raw_action: str,
    parsed_actions: List[str],
    *,
    screen_width: int,
    screen_height: int,
    allowed_tool_names: Optional[Set[str]] = None,
    allow_multiple_tool_calls: bool = False,
    coordinate_denominator: float = 999.0,
) -> Dict[str, Any]:
    """Validate the model action format and coarse tool semantics.

    The validator is intentionally conservative: invalid syntax is masked by
    the RL side, while execution still follows the existing parser fallback.
    """

    result = ActionValidationResult(
        parsed_action_count=len(parsed_actions or []),
        parser_failed=not bool(parsed_actions),
        fallback_wait=not bool(parsed_actions),
    )

    if not raw_action or not raw_action.strip():
        result.add_syntax("empty_response")
        return result.finalize().to_dict()

    payloads = _extract_tool_payloads(raw_action)
    result.tool_call_count = len(payloads)

    if not payloads:
        if len(parsed_actions or []) == 1 and str(parsed_actions[0]).upper() in {"DONE", "FAIL"}:
            result.action_names.append(str(parsed_actions[0]).upper())
            return result.finalize().to_dict()
        result.add_syntax("missing_tool_call")
        return result.finalize().to_dict()
    if len(payloads) > 1 and not allow_multiple_tool_calls:
        result.add_semantic("multiple_tool_calls")

    supported_tool_names = allowed_tool_names or {"computer_use"}

    for payload, wrapped in payloads:
        if not wrapped:
            result.add_syntax("missing_tool_call_wrapper")
        parsed = _loads_json_object(payload)
        if not parsed:
            xml_calls = _loads_qwen_xml_calls(payload)
            if not xml_calls:
                result.add_syntax("invalid_json")
                continue
            for name, arguments in xml_calls:
                if name and name not in supported_tool_names:
                    result.add_syntax("unsupported_tool_name")
                    continue
                _validate_arguments(
                    arguments,
                    result,
                    screen_width,
                    screen_height,
                    coordinate_denominator,
                )
            continue

        name = parsed.get("name")
        if name and name not in supported_tool_names:
            result.add_syntax("unsupported_tool_name")
            continue

        arguments = parsed if "action" in parsed else parsed.get("arguments", {})
        if isinstance(arguments, str):
            arguments = _loads_json_object(arguments) or {}
        if not isinstance(arguments, dict):
            result.add_syntax("invalid_arguments")
            continue

        _validate_arguments(
            arguments,
            result,
            screen_width,
            screen_height,
            coordinate_denominator,
        )

    if result.parser_failed:
        result.add_syntax("parser_failed")

    return result.finalize().to_dict()


def validate_qwen35_action_response(
    raw_action: str,
    *,
    screen_width: int = 1920,
    screen_height: int = 1080,
) -> Dict[str, Any]:
    """Rebuild Qwen3.5 action validation from a stored assistant response.

    This uses the same protocol parser and validator as the live OSGym action
    path, but does not require an environment instance. The RL data path can
    therefore derive an ephemeral invalid-action mask without persisting
    ``StepOutput.info`` through the framework Interactor.
    """

    from ..mm_agents.model_protocols.qwen35 import Qwen35Protocol

    protocol = Qwen35Protocol(
        prompt_observation_type="screenshot",
        screen_width=screen_width,
        screen_height=screen_height,
    )
    try:
        parsed_actions = protocol.parse_actions(raw_action)
    except (TypeError, ValueError):
        parsed_actions = []

    return validate_osgym_action(
        raw_action,
        parsed_actions,
        screen_width=screen_width,
        screen_height=screen_height,
        allowed_tool_names=protocol.allowed_tool_names,
        allow_multiple_tool_calls=protocol.allow_multiple_tool_calls,
        coordinate_denominator=protocol.coordinate_denominator,
    )


def _extract_tool_payloads(raw_action: str) -> List[Tuple[str, bool]]:
    wrapped = [(match.group("payload").strip(), True) for match in TOOL_CALL_RE.finditer(raw_action)]
    if wrapped:
        return wrapped

    payloads: List[Tuple[str, bool]] = []
    for line in raw_action.splitlines():
        stripped = line.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            payloads.append((stripped, False))
    return payloads


def _loads_json_object(raw_value: Any) -> Optional[Dict[str, Any]]:
    if isinstance(raw_value, dict):
        return raw_value
    if not isinstance(raw_value, str):
        return None
    try:
        parsed = json.loads(raw_value)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _loads_qwen_xml_calls(payload: str) -> List[Tuple[str, Dict[str, str]]]:
    function_pattern = re.compile(
        r"<function=(?P<name>.*?)>(?P<body>.*?)</function>",
        re.DOTALL,
    )
    parameter_pattern = re.compile(
        r"<parameter=(?P<name>.*?)>(?P<value>.*?)</parameter>",
        re.DOTALL,
    )

    calls: List[Tuple[str, Dict[str, str]]] = []
    for func_match in function_pattern.finditer(payload or ""):
        arguments: Dict[str, str] = {}
        for param_match in parameter_pattern.finditer(func_match.group("body")):
            arguments[param_match.group("name").strip()] = param_match.group("value").strip()
        calls.append((func_match.group("name").strip(), arguments))
    return calls


def _validate_arguments(
    arguments: Dict[str, Any],
    result: ActionValidationResult,
    screen_width: int,
    screen_height: int,
    coordinate_denominator: float,
) -> None:
    action = str(arguments.get("action") or "").strip()
    if not action:
        result.add_syntax("missing_action")
        return

    result.action_names.append(action)
    if action not in SUPPORTED_ACTIONS:
        result.add_syntax("unsupported_action")
        return

    if action in COORDINATE_ACTIONS:
        coord = _parse_coordinate(arguments.get("coordinate"))
        if coord is None:
            result.add_semantic("missing_or_invalid_coordinate")
            return
        x, y = coord
        max_coordinate = coordinate_denominator
        if x < 0 or x > max_coordinate or y < 0 or y > max_coordinate:
            result.add_semantic("coordinate_out_of_protocol_range")
        scaled_x = int(x * (screen_width / coordinate_denominator))
        scaled_y = int(y * (screen_height / coordinate_denominator))
        if scaled_x < 0 or scaled_x >= screen_width or scaled_y < 0 or scaled_y >= screen_height:
            result.add_semantic("coordinate_out_of_screen")
        return

    if action in {"key", "key_down", "key_up", "hold_key"}:
        keys = _parse_keys(arguments.get("keys", []))
        if not keys and isinstance(arguments.get("text"), str):
            keys = [
                key.strip()
                for key in arguments["text"].split("+")
                if key.strip()
            ]
        if not keys:
            result.add_semantic("missing_keys")
            return
        for key in keys:
            normalized = key.strip().lower()
            if normalized in VALID_KEYS:
                continue
            if SHELL_LIKE_RE.search(key) or len(key) > 30 or " " in key:
                result.add_semantic("shell_like_key")
            else:
                result.add_semantic("unsupported_key")
        return

    if action in {"scroll", "hscroll"}:
        scroll_value = arguments.get("pixels", arguments.get("scroll_amount", 0))
        if not _is_number(scroll_value):
            result.add_semantic("invalid_scroll_pixels")
        return

    if action == "wait":
        raw_time = arguments.get("time")
        if raw_time is not None and (not _is_number(raw_time) or float(raw_time) < 0):
            result.add_semantic("invalid_wait_time")
        return

    if action == "terminate":
        if "status" not in arguments or not str(arguments.get("status") or "").strip():
            result.add_semantic("missing_terminate_status")
            return
        status = str(arguments["status"]).strip().lower()
        if status not in {"success", "failure"}:
            result.add_semantic("invalid_terminate_status")
            return
        result.terminate_statuses.append(status)


def _parse_coordinate(raw_coord: Any) -> Optional[Tuple[float, float]]:
    if raw_coord is None:
        return None
    try:
        if isinstance(raw_coord, str):
            cleaned = raw_coord.strip("[]() ")
            parts = [part.strip() for part in cleaned.split(",")]
        else:
            parts = list(raw_coord)
        if len(parts) < 2:
            return None
        return float(parts[0]), float(parts[1])
    except (TypeError, ValueError):
        return None


def _parse_keys(keys: Any) -> List[str]:
    if keys is None:
        return []
    if isinstance(keys, str):
        try:
            parsed = json.loads(keys) if keys.startswith(("[", "{")) else [keys]
        except json.JSONDecodeError:
            parsed = [keys]
    else:
        parsed = keys
    if isinstance(parsed, (list, tuple)):
        return [str(key).strip() for key in parsed if str(key).strip()]
    return [str(parsed).strip()] if parsed else []


def _is_number(value: Any) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False
