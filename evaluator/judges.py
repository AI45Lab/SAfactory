from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class JudgeDefinition:
    judge_id: str
    version: str
    model: str
    prompt_template: str
    input_builder: str = "trajectory_final_answer"
    output_parser: str = "json_score_reason"
    score_min: float = 0.0
    score_max: float = 10.0
    metadata: dict[str, Any] = field(default_factory=dict)


class JudgeRegistry:
    def __init__(self, *, definition_dirs: list[str] | None = None) -> None:
        self.definition_dirs = [Path(path) for path in (definition_dirs or [])]
        self._definitions: dict[tuple[str, str], JudgeDefinition] = {}
        self.reload()

    def get(self, judge_id: str, version: str | None = None) -> JudgeDefinition:
        matches = [
            definition
            for (known_id, _), definition in self._definitions.items()
            if known_id == judge_id and (version is None or definition.version == version)
        ]
        if not matches:
            suffix = f":{version}" if version else ""
            raise KeyError(f"judge definition not found: {judge_id}{suffix}")
        return sorted(matches, key=lambda item: item.version)[-1]

    def list(self) -> list[JudgeDefinition]:
        return list(self._definitions.values())

    def register(self, definition: JudgeDefinition) -> None:
        self._definitions[(definition.judge_id, definition.version)] = definition

    def reload(self) -> None:
        self._definitions.clear()
        for directory in self.definition_dirs:
            if not directory.exists():
                continue
            for path in sorted(directory.glob("*.y*ml")) + sorted(directory.glob("*.json")):
                self.register(_load_definition(path))
        self._register_builtin_defaults()

    def _register_builtin_defaults(self) -> None:
        if ("general_task_judge", "builtin") not in self._definitions:
            self.register(
                JudgeDefinition(
                    judge_id="general_task_judge",
                    version="builtin",
                    model="gpt-4o-mini",
                    input_builder="trajectory_final_answer",
                    output_parser="json_score_reason",
                    score_min=0,
                    score_max=10,
                    prompt_template=(
                        "任务:\n{{ task }}\n\n"
                        "评分标准:\n{{ rubric }}\n\n"
                        "执行轨迹:\n{{ trajectory }}\n\n"
                        "最终回答:\n{{ final_response }}\n\n"
                        "请只输出 JSON: {\"score\": 0-10, \"reason\": \"...\", \"passed\": true/false}"
                    ),
                )
            )


def _load_definition(path: Path) -> JudgeDefinition:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".json":
        data = json.loads(text)
    else:
        data = yaml.safe_load(text) or {}
    return JudgeDefinition(**data)
