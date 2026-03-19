import os
import random
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import requests

try:
    from math_verify import parse as math_parse
    from math_verify import verify as math_verify
except Exception:  # pragma: no cover - optional dependency
    math_parse = None
    math_verify = None


VISUAL_JUDGE_TEMPLATE = """
Below are two answers to a question. Question is [Question], [Standard Answer] is the standard answer to the question, and [Model_answer] is the answer extracted from a model's output to this question. Determine whether these two answers are consistent.
Note that [Model Answer] is consistent with [Standard Answer] whenever they are essentially the same. If the meaning is expressed in the same way, it is considered consistent, for example, 'pink' and 'it is pink'.
If they are consistent, Judgement is 1; if they are different, Judgement is 0. Just output Judgement and don't output anything else.
"""


VISUAL_JUDGE_EXAMPLES = [
    """
[Question]: Is the countertop tan or blue?
[Standard Answer]: The countertop is tan.
[Model_answer] : tan
Judgement: 1
""",
    """
[Question]: On which side of the picture is the barrier?
[Standard Answer]: The barrier is on the left side of the picture.
[Model_answer] : left
Judgement: 1
""",
    """
[Question]: Is the kite brown and large?
[Standard Answer]: Yes, the kite is brown and large.
[Model_answer] : Yes
Judgement: 1
""",
    """
[Question]: Are the spots on a giraffe?
[Standard Answer]: No, the spots are on a banana.
[Model_answer] : no
Judgement: 1
""",
    """
[Question]: Who is wearing pants?
[Standard Answer]: The boy is wearing pants.
[Model_answer] : The person in the picture is wearing pants.
Judgement: 1
""",
    """
[Question]: Is the man phone both blue and closed?
[Standard Answer]: Yes, the man phone is both blue and closed.
[Model_answer] : No.
Judgement: 0
""",
    """
[Question]: What color is the towel in the center of the picture?
[Standard Answer]: The towel in the center of the picture is blue.
[Model_answer] : The towel in the center of the picture is pink.
Judgement: 0
""",
]


COMMON_VERIFY_PROMPT = """# CONTEXT #
I am a teacher, and I have some high-level reasoning problems. I am tasked with evaluating the correctness of a student's answer.
Below, I am provided with a problem and a reference answer. Additionally, a student's answer is provided. My job is to assess whether the student's answer captures the same meaning as the reference answer, even when expressed with different wording or format.

# OBJECTIVE #
I need you to judge whether the student's answer is correct given the ground truth answer.

Your tasks include:
1. Identify Semantic Equivalence: Carefully examine the expression in both answers. Confirm whether the semantic meaning of student's final answer is equivalent to the reference answer, even when expressed with different wording or format.

# TONE #
Professional, scientific.

# RESPONSE: MARKDOWN REPORT #
## Equivalence Judgement
[Whether the student's answer share the same meaning with the reference answer. (TRUE or FALSE)]

# ATTENTION #
 - The reference answer is ALWAYS correct. You should carefully judge whether the student gives the same answer as reference answer.
 - The Equivalence Judgement is only TRUE or FALSE. The answer is FALSE even if the student's final answer almost correct with a minor mistakes.
 - Don't give extra explanation.

**Question**:
{query}

**Reference Answer**
{gold_ans}

## Student Final Answer
{pred_ans}"""


MATH_VERIFY_PROMPT = """# CONTEXT #
I am a teacher, and I have some high-level math problems. I am tasked with evaluating the correctness of a student's answer.
Below, I am provided with a problem and a reference answer. Additionally, a student's answer is provided. My job is to assess whether the student's answer captures the same meaning as the reference answer, even when expressed with different wording or format.

# OBJECTIVE #
I need you to judge whether the student's answer is correct given the ground truth answer.

Your tasks include:
1. Identify Mathematical or Notational Equivalence: Pay special attention to any LaTeX expressions in both answers. Confirm that the mathematical relationships, variables, and operations conveyed are equivalent.

# TONE #
Professional, scientific.

# RESPONSE: MARKDOWN REPORT #
## Equivalence Judgement
[Whether the student's answer share the same meaning with the reference answer. (TRUE or FALSE)]

# ATTENTION #
 - The reference answer is ALWAYS correct. You should carefully judge whether the student gives the same answer as reference answer.
 - The Equivalence Judgement is only TRUE or FALSE. The answer is FALSE even if the student's final answer almost correct with a minor mistakes.
 - Don't give extra explanation.

**Question**:
{query}

**Reference Answer**
{gold_ans}

## Student Final Answer
{pred_ans}"""

JUDGE_MAX_RETRIES = 8


@dataclass
class RewardResult:
    score: float
    answer_text: str
    acc_reward: float
    format_reward: float
    tool_reward: float
    judge_used: bool
    judge_available: bool
    judge_response: str
    reward_type: str


class JudgeClient:
    def __init__(
        self,
        *,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        timeout: float = 60.0,
        api_key: str = "EMPTY",
    ) -> None:
        self.base_url = (base_url or os.environ.get("LLM_AS_A_JUDGE_BASE", "")).strip().rstrip("/")
        self.model_name = (model_name or "").strip() or None
        self.timeout = float(timeout)
        self.api_key = api_key

    @property
    def available(self) -> bool:
        return bool(self.base_url)

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _resolve_model(self) -> Optional[str]:
        if not self.available:
            return None
        if self.model_name:
            return self.model_name
        try:
            response = requests.get(
                f"{self.base_url}/models",
                headers=self._headers(),
                timeout=self.timeout,
            )
            response.raise_for_status()
            payload = response.json()
            data = payload.get("data") or []
            if data:
                model = str(data[0].get("id", "")).strip()
                if model:
                    self.model_name = model
                    return model
        except Exception:
            return None
        return None

    def chat(self, messages: list[dict[str, str]], *, temperature: float) -> Optional[str]:
        model = self._resolve_model()
        if not model:
            return None
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "seed": random.randint(0, 1_000_000),
        }
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self._headers(),
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
            return str(data["choices"][0]["message"]["content"]).strip()
        except Exception:
            return None


def _parse_binary_judgement(response: str) -> Optional[bool]:
    stripped = response.strip()
    if "Judgement:" in stripped:
        stripped = stripped.split("Judgement:")[-1].strip()
    match = re.search(r"\b([01])\b", stripped)
    if not match:
        return None
    return match.group(1) == "1"


def _parse_true_false(response: str) -> Optional[bool]:
    lower = response.strip().lower()
    if "## equivalence judgement" in lower:
        lower = lower.split("## equivalence judgement")[-1].strip()
    if "true" in lower and "false" not in lower:
        return True
    if "false" in lower and "true" not in lower:
        return False
    return None


def _extract_last_answer_tag(text: str) -> str:
    matches = re.findall(r"<answer>(.*?)</answer>", text, flags=re.DOTALL)
    return matches[-1].strip() if matches else ""


def _extract_last_boxed_answer(text: str) -> str:
    matches = re.findall(r"\\boxed{([^}]+)}", text, flags=re.DOTALL)
    return matches[-1].strip() if matches else ""


def _visual_judge_prompt(predict_str: str, ground_truth: str, question: str) -> str:
    examples = "\n\n".join(VISUAL_JUDGE_EXAMPLES)
    return (
        f"{VISUAL_JUDGE_TEMPLATE}{examples}\n\n"
        f"[Question]: {question}\n"
        f"[Standard Answer]: {ground_truth}\n"
        f"[Model_answer] : {predict_str}\n"
        "Judgement:"
    )


def _compute_visual_acc(
    answer_text: str,
    ground_truth: str,
    question: str,
    judge_client: JudgeClient,
) -> Tuple[float, bool, str]:
    if len(answer_text) >= 1000:
        return 0.0, False, ""
    if not judge_client.available:
        return 0.0, False, ""
    judge_response = ""
    judge_response = judge_client.chat(
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": _visual_judge_prompt(answer_text, ground_truth, question)},
        ],
        temperature=0.3,
    ) or ""
    judgement = _parse_binary_judgement(judge_response)
    if judgement is not None:
        return (1.0 if judgement else 0.0), True, judge_response
    return 0.0, False, judge_response


def _compute_common_acc(
    answer_text: str,
    ground_truth: str,
    question: str,
    judge_client: JudgeClient,
) -> Tuple[float, bool, str]:
    if len(answer_text) >= 1000:
        return 0.0, False, ""
    if not judge_client.available:
        return 0.0, False, ""
    messages = [
        {
            "role": "user",
            "content": COMMON_VERIFY_PROMPT.format(
                query=question,
                gold_ans=ground_truth,
                pred_ans=answer_text,
            ),
        }
    ]
    judge_response = ""
    for _ in range(JUDGE_MAX_RETRIES):
        judge_response = judge_client.chat(messages, temperature=0.5) or ""
        judgement = _parse_true_false(judge_response)
        if judgement is not None:
            return (1.0 if judgement else 0.0), True, judge_response
    return 0.0, False, judge_response


def _rule_math_verify(ground_truth: str, model_answer: str) -> Optional[bool]:
    if math_parse is None or math_verify is None:
        return None
    try:
        gold = math_parse(ground_truth)
        answer = math_parse(model_answer)
        return bool(math_verify(gold, answer))
    except Exception:
        return None


def _compute_math_acc(
    model_answer: str,
    ground_truth: str,
    question: str,
    judge_client: JudgeClient,
) -> Tuple[float, bool, str]:
    verified = _rule_math_verify(ground_truth, model_answer)
    if verified is True:
        return 1.0, False, ""
    if not judge_client.available:
        return 0.0, False, ""
    messages = [
        {
            "role": "user",
            "content": MATH_VERIFY_PROMPT.format(
                query=question,
                gold_ans=ground_truth,
                pred_ans=model_answer,
            ),
        }
    ]
    judge_response = ""
    for _ in range(JUDGE_MAX_RETRIES):
        response = judge_client.chat(messages, temperature=0.0)
        if response is None:
            continue
        judge_response = response
        break
    if not judge_response:
        return 0.0, False, ""
    judgement = _parse_true_false(judge_response)
    if judgement is not None:
        return (1.0 if judgement else 0.0), True, judge_response
    return 0.0, False, judge_response


def compute_visual_score(
    predict_str: str,
    ground_truth: str,
    *,
    question: str,
    tool_used: bool,
    judge_client: JudgeClient,
) -> RewardResult:
    is_format_error = False
    if predict_str.count("<think>") != predict_str.count("</think>"):
        is_format_error = True

    predict_no_think = predict_str.split("</think>")[-1].strip()
    if predict_no_think.count("<answer>") != predict_no_think.count("</answer>"):
        is_format_error = True

    answer_text = _extract_last_answer_tag(predict_str)
    if not answer_text:
        is_format_error = True

    acc_reward, judge_used, judge_response = _compute_visual_acc(
        answer_text,
        ground_truth,
        question,
        judge_client,
    )

    if len(answer_text) >= 1000:
        acc_reward = 0.0
        is_format_error = True

    tool_reward = 1.0 if tool_used and acc_reward > 0.5 else 0.0
    format_reward = -1.0 if is_format_error else 0.0
    score = 0.8 * acc_reward + 0.2 * format_reward + 1.2 * tool_reward
    return RewardResult(
        score=score,
        answer_text=answer_text,
        acc_reward=acc_reward,
        format_reward=format_reward,
        tool_reward=tool_reward,
        judge_used=judge_used,
        judge_available=judge_client.available,
        judge_response=judge_response,
        reward_type="visual",
    )


def compute_common_reasoning_score(
    predict_str: str,
    ground_truth: str,
    *,
    question: str,
    tool_used: bool,
    judge_client: JudgeClient,
) -> RewardResult:
    is_format_error = False
    if predict_str.count("<think>") != predict_str.count("</think>"):
        is_format_error = True

    predict_no_think = predict_str.split("</think>")[-1].strip()
    if predict_no_think.count("<answer>") != predict_no_think.count("</answer>"):
        is_format_error = True

    answer_text = _extract_last_answer_tag(predict_no_think)
    if not answer_text:
        is_format_error = True

    acc_reward, judge_used, judge_response = _compute_common_acc(
        answer_text,
        ground_truth,
        question,
        judge_client,
    )

    if len(answer_text) >= 1000:
        acc_reward = 0.0
        is_format_error = True

    tool_reward = 1.0 if tool_used and acc_reward > 0.5 else 0.0
    format_reward = -1.0 if is_format_error else 0.0
    score = 0.8 * acc_reward + 0.2 * format_reward + 1.2 * tool_reward
    return RewardResult(
        score=score,
        answer_text=answer_text,
        acc_reward=acc_reward,
        format_reward=format_reward,
        tool_reward=tool_reward,
        judge_used=judge_used,
        judge_available=judge_client.available,
        judge_response=judge_response,
        reward_type="common_reasoning",
    )


def compute_math_score(
    predict_str: str,
    ground_truth: str,
    *,
    question: str,
    judge_client: JudgeClient,
) -> RewardResult:
    is_format_error = False
    if predict_str.count("<think>") != predict_str.count("</think>"):
        is_format_error = True

    predict_no_think = predict_str.split("</think>")[-1].strip()
    model_answer = _extract_last_boxed_answer(predict_no_think)
    if not model_answer:
        is_format_error = True
        acc_reward = 0.0
        judge_used = False
        judge_response = ""
    else:
        if len(re.findall(r"\\boxed{([^}]+)}", predict_no_think, flags=re.DOTALL)) > 1:
            is_format_error = True
        acc_reward, judge_used, judge_response = _compute_math_acc(
            model_answer,
            ground_truth,
            question,
            judge_client,
        )

    format_reward = -1.0 if is_format_error else 0.0
    score = 1.2 * acc_reward + 0.4 * format_reward
    return RewardResult(
        score=score,
        answer_text=model_answer,
        acc_reward=acc_reward,
        format_reward=format_reward,
        tool_reward=0.0,
        judge_used=judge_used,
        judge_available=judge_client.available,
        judge_response=judge_response,
        reward_type="math",
    )


def compute_reward(
    data_source: str,
    predict_str: str,
    ground_truth: str,
    *,
    extra_info: Optional[Dict[str, Any]] = None,
    tool_used: bool,
    judge_client: JudgeClient,
) -> RewardResult:
    info = extra_info or {}
    question = str(info.get("question", "")).strip()
    if data_source in {"geoguessr"}:
        return compute_common_reasoning_score(
            predict_str,
            ground_truth,
            question=question,
            tool_used=tool_used,
            judge_client=judge_client,
        )
    if data_source in {"thinklite_eureka", "xince"}:
        return compute_math_score(
            predict_str,
            ground_truth,
            question=question,
            judge_client=judge_client,
        )
    return compute_visual_score(
        predict_str,
        ground_truth,
        question=question,
        tool_used=tool_used,
        judge_client=judge_client,
    )
