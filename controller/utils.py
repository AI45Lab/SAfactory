import json
import re
from typing import Optional, Sequence

import numpy as np
from transformers import GenerationConfig

from . import Agent, APIAgent, BaseTask
from .types import (
    ActionFormat,
    ActionWithTought,
    ConversationMessage,
    EvaluationOutput,
    ExperienceOutput,
    APIExperienceOutput,
)

INVOKING_FUNCTION_PROMPT = """

If you want to invoke a provided function or tool, please reply in the following *JSON* format:
```json
{
    "thought": "I think ...",
    "function_name": "function_name",
    "arguments": <valid json object of args>
}
```
Only reply the *JSON* object, no other text should be present.
"""

WRITE_CODE_PROMPT = """

If you want to call these functions, please reply the python code block:
```python
# Write you thought in the code comment before you call any function.
<write valid python code here.>
```
Only reply the code block with "```python" and "```",  no other text should be present.
"""

def format_function_call_prompt(function_description: Sequence) -> str:
    prompt = "You have the following functions available:\n\n"

    tool_descs = [{"type": "function", "function": f} for f in function_description]
    prompt += "\n".join(
        [json.dumps(f, ensure_ascii=False, indent=2) for f in tool_descs]
    )
    prompt += INVOKING_FUNCTION_PROMPT

    return prompt


def generate_function_signatures(function_descriptions: Sequence):
    function_strings = []
    for func in function_descriptions:
        name = func["name"]
        description = func["description"]
        params = func["parameters"]["properties"]
        required_params = func["parameters"].get("required", [])

        # Generate function signature
        signature_params = ", ".join(
            [
                f"{param}='{param}'" if param not in required_params else param
                for param in params
            ]
        )
        function_signature = f"def {name}({signature_params}):"

        # Generate docstring
        docstring = f'    """\n    {description}\n\n'
        for param, details in params.items():
            docstring += (
                f"    :param {param} ({details['type']}): {details['description']}\n"
            )
        docstring += '    """'

        # Combine signature and docstring
        function_strings.append(f"{function_signature}\n{docstring}\n")

    return "\n".join(function_strings)


def format_code_as_action_prompt(function_description: Sequence) -> str:
    prompt = "Here are the signatures and docstrings of these functions:\n\n```python\n"
    prompt += generate_function_signatures(function_description)
    prompt += "\n```"
    prompt += WRITE_CODE_PROMPT

    return prompt


_python_comment_pattern = re.compile(r"#.*")


def parse_python_code_comments(code: str) -> str:
    comments = _python_comment_pattern.findall(code)
    comments = [c.strip() for c in comments]
    comments = [c if c else "\n" for c in comments]
    return " ".join(comments)


def extract_python_code_blocks(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
    if text.endswith("```"):
        text = text.rsplit("\n", 1)[0]
    return text


class BaseAdapter:
    conversation_start_dict: dict[
        ActionFormat, tuple[ConversationMessage, ConversationMessage]
    ]

    @staticmethod
    def parse_react(text: str) -> ActionWithTought:
        """
        ReAct format:
        ```
        Thought:
        I think ...

        Action:
        action
        ```
        """
        invalid_format_flg = False
        _split = text.rsplit("Action:", 1)
        if len(_split) == 0:
            _thought, _action = text
            invalid_format_flg = True
        elif len(_split) == 1:
            if "search[" in text or "click[" in text:
                _thought, _action = "", _split[0]
            else:
                _thought, _action = _split[0], ""
            invalid_format_flg = True
        else:
            assert len(_split) == 2
            _thought, _action = _split

        thought = _thought.split("Thought:")
        if len(thought) == 1:
            thought = thought[0]
            invalid_format_flg = True
        else:
            thought = thought[1].strip()
        action = _action.strip()
        if invalid_format_flg:
            print(
                "The text is not in the correct format. Parsing result may not be accurate."
            )
            print("###RAW TEXT:\n", text)
            print("\n###PARSED THOUGHT:\n", thought)
            print("\n###PARSED ACTION:\n", action)
        return ActionWithTought(thought, action)

    @staticmethod
    def to_react(action_with_thought: ActionWithTought) -> str:
        return f"Thought:\n{action_with_thought.thought}\n\nAction:\n{action_with_thought.action}"

    @staticmethod
    def parse_function_calling(text: str) -> ActionWithTought:
        """
        Function Calling format:
        ```json
        {
            "function_name": "function_name",
            "args": {"kwarg1": "value1", "kwarg2": "value2"}
        }
        ```
        """
        raise NotImplementedError

    @staticmethod
    def to_function_calling(action_with_thought: ActionWithTought) -> str:
        raise NotImplementedError

    @staticmethod
    def parse_code_as_action(text: str) -> ActionWithTought:
        """
        Code as Action format:
        ```
        code
        ```
        """
        raise NotImplementedError

    @staticmethod
    def to_code_as_action(action_with_thought: ActionWithTought) -> str:
        raise NotImplementedError

    @classmethod
    def action_parser(cls, action: str, action_format: ActionFormat) -> str:
        if action_format == ActionFormat.REACT:
            return cls.parse_react(action).action
        elif action_format == ActionFormat.FUNCTION_CALLING:
            return cls.parse_function_calling(action).action
        elif action_format == ActionFormat.CODE_AS_ACTION:
            return cls.parse_code_as_action(action).action
        else:
            raise NotImplementedError


class BaseAgentEnvController:
    def __init__(self, agent: Agent | APIAgent, tasks: Sequence[BaseTask]) -> None:
        self.agent = agent
        self.tasks = tasks

    def generate_experience(
        self,
        idxs: Sequence[int] | Sequence[Sequence[int]] | None = None,
        generation_config: Optional[GenerationConfig] = None,
        max_rounds: Optional[int] = None,
    ) -> list[ExperienceOutput | APIExperienceOutput]:
        experience = []
        if isinstance(idxs[0], int):
            experience += self.tasks[0].generate_experience(
                self.agent,
                idxs,
                generation_config,
                max_rounds,
            )
        elif isinstance(idxs[0], Sequence):
            for idx, task in enumerate(self.tasks):
                experience += task.generate_experience(
                    self.agent,
                    idxs[idx],
                    generation_config,
                    max_rounds,
                )
        else:
            raise ValueError("Incorrect Format for idxs")

        return experience


class Evaluator(BaseAgentEnvController):
    def eval(
        self,
        generation_config: Optional[GenerationConfig] = None,
        max_rounds: Optional[int] = None,
        idxs: Sequence[int] | Sequence[Sequence[int]] | None = None,
    ) -> EvaluationOutput:
        exps = self.generate_experience(
            idxs=(
                idxs
                if idxs is not None
                else [list(range(len(task.clients[0]))) for task in self.tasks]
            ),
            generation_config=generation_config,
            max_rounds=max_rounds,
        )
        rewards = np.array([exp.reward for exp in exps])
        return EvaluationOutput(
            experiences=exps, score=rewards.mean(), success=(rewards == 1 or rewards == 100).mean()
        )


class BaseTrainer(BaseAgentEnvController):
    # def __init__(self, agent: Agent, tasks: Sequence[BaseTask]) -> None:
    #     super().__init__(agent, tasks)

    def train(self):
        pass

    def eval(
        self,
        generation_config: Optional[GenerationConfig] = None,
        max_rounds: Optional[int] = None,
        idxs: Sequence[int] | Sequence[Sequence[int]] = None,
    ) -> EvaluationOutput:
        exps = self.generate_experience(
            idxs=idxs,
            generation_config=generation_config,
            max_rounds=max_rounds,
        )
        rewards = np.array([exp.reward for exp in exps])
        return EvaluationOutput(
            experiences=exps, score=rewards.mean(), success=(rewards == 1 or rewards == 100).mean()
        )

    def save_model(self):
        pass
