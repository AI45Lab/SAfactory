from __future__ import annotations

import json
import re
from typing import Any, Protocol

from evaluator.eval_types import EvalRequest, EvalResult, EvalSpec, EvalStatus, normalize_score
from evaluator.judges import JudgeDefinition


class JudgeOutputParser(Protocol):
    def parse(
        self,
        *,
        text: str,
        request: EvalRequest,
        spec: EvalSpec,
        definition: JudgeDefinition | None = None,
    ) -> EvalResult:
        ...


class JsonScoreReasonParser:
    def parse(
        self,
        *,
        text: str,
        request: EvalRequest,
        spec: EvalSpec,
        definition: JudgeDefinition | None = None,
    ) -> EvalResult:
        data = extract_json_object(text)
        score = data.get("score")
        if score is None and "passed" in data:
            score = 1.0 if bool(data["passed"]) else 0.0
            data.setdefault("scale", {"min": 0, "max": 1})
        if score is None:
            raise ValueError("judge output did not include score")

        scale = data.get("scale") or {}
        min_score = float(scale.get("min", definition.score_min if definition else spec.score_min))
        max_score = float(scale.get("max", definition.score_max if definition else spec.score_max))
        normalized = normalize_score(float(score), min_score, max_score)
        return EvalResult(
            session_id=request.session_id,
            eval_id=spec.eval_id,
            method=str(spec.method.value),
            status=EvalStatus.SUCCEEDED.value,
            raw_score=float(score),
            normalized_score_10=normalized,
            reason=str(data.get("reason") or data.get("explanation") or ""),
            artifacts={
                "parsed": data,
                "score_scale": {"min": min_score, "max": max_score},
            },
        )


class LooseScoreRegexParser:
    SCORE_RE = re.compile(r"(?i)(?:score|分数)\s*[:=：]\s*(-?\d+(?:\.\d+)?)")
    BARE_NUMBER_RE = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*$")

    def parse(
        self,
        *,
        text: str,
        request: EvalRequest,
        spec: EvalSpec,
        definition: JudgeDefinition | None = None,
    ) -> EvalResult:
        match = self.SCORE_RE.search(text)
        if not match:
            match = self.BARE_NUMBER_RE.search(text)
        if not match:
            raise ValueError("could not extract score")
        score = float(match.group(1))
        min_score = definition.score_min if definition else spec.score_min
        max_score = definition.score_max if definition else spec.score_max
        return EvalResult(
            session_id=request.session_id,
            eval_id=spec.eval_id,
            method=str(spec.method.value),
            status=EvalStatus.SUCCEEDED.value,
            raw_score=score,
            normalized_score_10=normalize_score(score, min_score, max_score),
            reason="score extracted by loose regex parser",
            artifacts={"raw_text": text[:4000]},
        )


def default_output_parsers() -> dict[str, JudgeOutputParser]:
    json_parser = JsonScoreReasonParser()
    return {
        "json_score_reason": json_parser,
        "json_score_passed_reason": json_parser,
        "json_rubric_breakdown": json_parser,
        "loose_score_regex": LooseScoreRegexParser(),
    }


def extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        parsed = json.loads(stripped[start : end + 1])
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("text did not contain a JSON object")
