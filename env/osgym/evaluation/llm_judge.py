"""
LLM Judge Evaluator Module

Provides LLM-based task completion evaluation as an alternative to rule-based evaluators.
This is useful for tasks without explicit evaluator functions or for complex tasks where
writing rule-based evaluators is impractical.

Usage:
    judge = LLMJudge(api_key="...", base_url="...", model="gpt-4o")
    result = await judge.evaluate_task_completion(
        instruction="Open Firefox and navigate to google.com",
        trajectory=[...],
        final_screenshot=screenshot_bytes
    )
"""

import base64
import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger("osgym.llm_judge")


@dataclass
class JudgeResult:
    """Result from LLM Judge evaluation."""
    score: float  # 0.0 or 1.0
    reasoning: str
    confidence: float  # 0.0 - 1.0
    key_observations: List[str] = None

    def __post_init__(self):
        if self.key_observations is None:
            self.key_observations = []


# System prompts for different evaluation scenarios
TASK_COMPLETION_SYSTEM_PROMPT = """You are an expert evaluator for GUI automation tasks on Ubuntu desktop.
Your job is to determine if a task was completed successfully based on:
1. The original instruction (what the user wanted to accomplish)
2. The sequence of actions taken by the agent
3. The final screenshot showing the current state

Scoring rules (binary):
- 1: Task completed. The goal is achieved (fully or mostly completed with minor deviations).
- 0: Task not completed. The goal is not achieved or only partially completed.

Important guidelines:
- Focus on the end result, not the exact path taken. If the goal is achieved, score 1 even if the method differed.
- Consider whether the instruction's intent was fulfilled, not just literal interpretation.
- If the screenshot clearly shows the expected outcome, that's strong evidence of completion.
- Be lenient with minor UI differences (e.g., different window positions) if the core task is done.
- If the task is mostly completed with only minor issues, score 1.
- If the task is only partially completed or not completed at all, score 0.

You MUST respond in valid JSON format:
{
    "score": <integer, either 0 or 1>,
    "reasoning": "<brief explanation of your evaluation>",
    "confidence": <float between 0.0 and 1.0>,
    "key_observations": ["<observation 1>", "<observation 2>", ...]
}"""


class LLMJudge:
    """
    LLM-based task completion evaluator.

    Uses a vision-language model to assess whether GUI automation tasks
    were completed successfully based on screenshots and action sequences.
    """

    def __init__(
        self,
        api_key: str = None,
        base_url: str = None,
        model: str = "gpt-4o",
        max_retries: int = 3,
        timeout: float = 60.0
    ):
        """
        Initialize LLM Judge.

        Args:
            api_key: API key for the LLM service
            base_url: Base URL for the API (for custom endpoints)
            model: Model identifier to use
            max_retries: Maximum retry attempts for API calls
            timeout: Request timeout in seconds
        """
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.max_retries = max_retries
        self.timeout = timeout
        self._client = None

    def _get_client(self):
        """Lazily initialize the OpenAI client."""
        if self._client is None:
            try:
                from openai import AsyncOpenAI
                self._client = AsyncOpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url,
                    timeout=self.timeout
                )
            except ImportError:
                raise ImportError(
                    "openai package is required for LLM Judge. "
                    "Install with: pip install openai"
                )
        return self._client

    def _encode_image(self, image_data: bytes) -> str:
        """Encode image bytes to base64 string."""
        return base64.b64encode(image_data).decode("utf-8")

    def _format_trajectory(self, trajectory: List[Dict]) -> str:
        """Format trajectory into a readable string."""
        if not trajectory:
            return "No actions recorded."

        lines = []
        for i, step in enumerate(trajectory, 1):
            action = step.get("action", "unknown action")
            # Truncate very long actions
            # if len(action) > 500:
            #     action = action[:500] + "..."
            lines.append(f"Step {i}: {action}")

        return "\n".join(lines)

    def _parse_response(self, content: str) -> JudgeResult:
        """Parse LLM response into JudgeResult."""
        try:
            # Try to extract JSON from the response
            # Handle cases where JSON might be wrapped in markdown code blocks
            if "```json" in content:
                start = content.find("```json") + 7
                end = content.find("```", start)
                content = content[start:end].strip()
            elif "```" in content:
                start = content.find("```") + 3
                end = content.find("```", start)
                content = content[start:end].strip()

            data = json.loads(content)

            return JudgeResult(
                score=1.0 if float(data.get("score", 0)) > 0.5 else 0.0,
                reasoning=str(data.get("reasoning", "")),
                confidence=float(data.get("confidence", 0.5)),
                key_observations=data.get("key_observations", [])
            )
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(f"Failed to parse LLM response: {e}")
            # Return a default result indicating parsing failure
            return JudgeResult(
                score=0.0,
                reasoning=f"Failed to parse LLM response: {content[:200]}",
                confidence=0.0,
                key_observations=[]
            )

    async def evaluate_task_completion(
        self,
        instruction: str,
        trajectory: List[Dict] = None,
        final_screenshot: bytes = None,
        expected_outcome: str = None
    ) -> JudgeResult:
        """
        Evaluate if a task was completed successfully.

        Args:
            instruction: The original task instruction
            trajectory: List of action steps taken (each with 'action' key)
            final_screenshot: Screenshot bytes of the final state
            expected_outcome: Optional description of what success looks like

        Returns:
            JudgeResult with score, reasoning, and confidence
        """
        client = self._get_client()

        # Build user message content
        user_content = []

        # Add text description
        text_parts = [f"## Task Instruction\n{instruction}"]

        if expected_outcome:
            text_parts.append(f"\n## Expected Outcome\n{expected_outcome}")

        if trajectory:
            text_parts.append(f"\n## Actions Taken\n{self._format_trajectory(trajectory)}")

        text_parts.append("\n## Your Task\nEvaluate whether the task was completed successfully based on the above information and the final screenshot.")

        user_content.append({
            "type": "text",
            "text": "\n".join(text_parts)
        })

        # Add screenshot if provided
        if final_screenshot:
            user_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{self._encode_image(final_screenshot)}",
                    "detail": "high"
                }
            })

        # Make API call with retries
        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = await client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": TASK_COMPLETION_SYSTEM_PROMPT},
                        {"role": "user", "content": user_content}
                    ],
                    max_tokens=1024,
                    temperature=0.1  # Low temperature for consistent evaluation
                )

                content = response.choices[0].message.content
                return self._parse_response(content)

            except Exception as e:
                last_error = e
                logger.warning(f"LLM Judge attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    import asyncio
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff

        # All retries failed
        logger.error(f"LLM Judge failed after {self.max_retries} attempts: {last_error}")
        return JudgeResult(
            score=0.0,
            reasoning=f"Evaluation failed: {last_error}",
            confidence=0.0,
            key_observations=[]
        )


class SyncLLMJudge:
    """
    Synchronous wrapper for LLMJudge.

    Provides blocking methods for use in non-async contexts.
    """

    def __init__(self, *args, **kwargs):
        """Initialize with same arguments as LLMJudge."""
        self._async_judge = LLMJudge(*args, **kwargs)

    def _run_async(self, coro):
        """Run async coroutine in sync context."""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        if loop.is_running():
            # We're inside an async context, use nest_asyncio
            try:
                import nest_asyncio
                nest_asyncio.apply()
            except ImportError:
                raise RuntimeError(
                    "Cannot run sync LLM Judge inside async context. "
                    "Either use async version or install nest_asyncio."
                )

        return loop.run_until_complete(coro)

    def evaluate_task_completion(self, *args, **kwargs) -> JudgeResult:
        """Synchronous version of evaluate_task_completion."""
        return self._run_async(
            self._async_judge.evaluate_task_completion(*args, **kwargs)
        )

def create_llm_judge_evaluator_config(
    expected_outcome: str = None
) -> Dict[str, Any]:
    """
    Create evaluator config for LLM Judge.

    This helper function generates the evaluator configuration that should
    be included in task JSONL files for tasks using LLM Judge evaluation.

    Args:
        expected_outcome: Description of what successful completion looks like

    Returns:
        Evaluator configuration dictionary

    Example:
        >>> config = create_llm_judge_evaluator_config(
        ...     expected_outcome="Firefox window is open with google.com loaded"
        ... )
    """
    evaluator_config = {
        "func": "llm_judge",
        "result": {"type": "trajectory_and_screenshot"},
        "expected": {
            "type": "llm_judge"
        }
    }

    if expected_outcome:
        evaluator_config["expected"]["expected_outcome"] = expected_outcome

    return evaluator_config
