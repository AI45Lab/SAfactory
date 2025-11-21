import copy
from dataclasses import dataclass
from hmac import new
import json
import logging
from multiprocessing.managers import ValueProxy
from typing import List, Any, Tuple, Optional
import re

from matplotlib.pyplot import hist
from numpy import isin
from recoma.models.core.base_model import BaseModel
from recoma.models.core.base_react_controller import BaseReactController
from recoma.search.state import SearchState, SearchNode

from agents.recoma.discoveryworld_env_models import task_completed, num_interactions

logger = logging.getLogger(__name__)


@dataclass
class Action:
    action_str: str
    action_json: dict[Any, Any]


@dataclass
class Observation:
    raw_observation: str


@BaseModel.register("discoveryworld_react_controller")
class DiscoveryWorldReactController(BaseReactController):
    """
    React Controller for DiscoveryWorld environment
    """
    def __init__(self, eoq="[EOQ]", max_history=-1, termination_suggestion=-1,
                 ignore_multiple_calls = True, **kwargs):
        super().__init__(**kwargs)
        self.eoq = eoq
        self.ignore_multiple_jsons = ignore_multiple_calls
        self.max_history = max_history
        self.termination_suggestion = termination_suggestion
        # The stop token may be "```" and wont appear here. So need to consider it.
        self.partial_code_regex = r".*```json\n(.*)"
        self.full_code_regex = r"```json\n(.*?)```"

    def summarize_history(self, history: List[Any]) -> str:
        """Summarize the history of the conversation"""
        summary = ""
        for message in history:
            if isinstance(message, Action):
                summary += "Action:\n + ```\n" + message.action_str + "```\n"
            else:
                summary += "Observation:\n + ```\n" + message.raw_observation + "```\n"
        return summary

    def next_step_and_observation_input(self, state: SearchState,
                                        last_child: SearchNode) -> Tuple[int, str]:
        """
        This method returns the next step and observation input for the controller.
        """
        last_message = self.get_history(self.get_react_node(state))[-1]
        if not isinstance(last_message, Action):
            raise ValueError("Last message is not an action but calling observation model!")
        if last_message.action_json is None:
            raise ValueError("No JSON extracted from last action!")
        return self.OBSERVATION, last_message.action_str

    def next_step_and_action_input(self, state: SearchState,
                                   last_child: SearchNode) -> Tuple[int, str]:
        """
        This function determines the next step and action input based on the current state and the
          last child node.
        """
        current_node = self.get_react_node(state)
        history = self.get_history(current_node)
        new_history = copy.deepcopy(history)
        if self.max_history > 0 and len(new_history) > self.max_history:
            new_history = new_history[-self.max_history:]
            output_str = current_node.input_str + "\n\nHistory of action-observations:\n"
            output_str += "[TRIMMED HISTORY]\n\n"
            output_str += self.build_message_thread2(new_history, self.max_output_length-len(output_str))
        else:
            output_str = current_node.input_str + "\n\nHistory of action-observations:\n"
            output_str += self.build_message_thread2(history, self.max_output_length-len(output_str))

        while len(output_str) > self.max_output_length:
            output_str = current_node.input_str + "\n\nHistory of action-observations:\n"
            output_str += "[TRIMMED HISTORY]\n\n"
            new_history = new_history[1:]
            output_str += self.build_message_thread2(new_history, self.max_output_length-len(output_str))
            print(len(output_str), self.max_output_length)

        if self.termination_suggestion > 0  and len(history) > 2 * self.termination_suggestion:
            output_str += "\n**ARE YOU SURE YOU WANT TO CONTINUE? CONSIDER SUBMITTING WITH FAILURE**\n"
        return self.ACTION, output_str

    def terminate_with_output(self, state: SearchState, last_child: SearchNode) -> Optional[str]:
        """
        This function determines if the controller should terminate with output. Set to None
        if the controller should not terminate.
        """

        if last_child is not None:
            last_message = self.get_history(self.get_react_node(state))[-1]
            if isinstance(last_message, Action):
                if "action" in last_message.action_json and last_message.action_json["action"] == "SUBMIT":
                    return last_message.action_json["arg1"] + last_message.action_json.get("thought", "")

        if task_completed():
            return last_child.output
        return None

    def append_message_to_history(self, current_history: List[Any], last_child: SearchNode) -> None:
        """
        Append the message from the last child to the current history
        """
        step_type = self.get_step(last_child)
        if step_type == self.OBSERVATION:
            current_history.append(Observation(raw_observation=last_child.output))
        else:
            action_json = self.extract_json_output(last_child)
            try:
                formatted_json = json.loads(action_json)
                current_history.append(Action(action_str=last_child.output, action_json=formatted_json))
            except json.JSONDecodeError:
                raise ValueError("Failed to decode JSON from action output: {}".format(last_child.output))


    def build_message_thread(self, history):
        output_str = ""
        for message in history:
            if isinstance(message, Action):
                # Small hack to make sure that "Task: " always matches the input task.
                output_str +="Action:\n```json\n" + \
                    message.action_str.replace("Task:", "Task :") + "\n```\n\n"
            elif isinstance(message, Observation):
                output_str += "Observation:\n```json\n" + message.raw_observation + "```\n\n"

            else:
                raise ValueError("Unknown message type: {}".format(message))
        return output_str


    def build_message_thread2(self, history, max_output_length):
        output = []
        char_count = 0
        for message in history[::-1]:
            if isinstance(message, Action):
                # Small hack to make sure that "Task: " always matches the input task.
                output.append("Action:\n```json\n" + message.action_str.replace("Task:", "Task :") + "\n```\n\n")
            elif isinstance(message, Observation):
                output.append("Observation:\n```json\n" + message.raw_observation + "```\n\n")
            else:
                raise ValueError("Unknown message type: {}".format(message))

            char_count += len(output[-1])
            if char_count > max_output_length:
                output = output[:-1]
                break

        output_str = "".join(output[::-1])
        return output_str

    def extract_json_output(self, last_child: SearchNode) -> str:
        action_output = last_child.output
        output_code = ""
        match_end = 0
        # Handle multiple JSONs
        for re_match in re.finditer(self.full_code_regex, action_output, flags=re.DOTALL):
            code = re_match.group(1).strip()
            if self.ignore_multiple_jsons:
                last_child.output = code
                print(code)
                return code
            output_code += code + "\n"
            match_end = re_match.end()

        # check for partial code match at end (no terminating ```)  following the last match
        partial_m = re.match(self.partial_code_regex, action_output[match_end:], flags=re.DOTALL)
        if partial_m:
            output_code += partial_m.group(1).strip()
            # terminated due to stop condition. Add stop condition to output.
            if not last_child.output.endswith("\n"):
                last_child.output = last_child.output + "\n"
            last_child.output = output_code
        if len(output_code) == 0:
            # print("No JSON found in action: {}".format(action_output))
            try:
                action_json = json.loads(action_output)
                return action_output
            except json.JSONDecodeError:
                print("Could not decode JSON from {}".format(action_output))
                return None
        else:
            return output_code

    def next_step_and_input(self, state: SearchState, last_child: SearchNode) -> Tuple[int, str]:
        """
        This function determines the next step type and input based on the current state and the
        last child node.
        """
        # Observation always follows an action step
        if last_child is not None and self.get_step(last_child) == self.ACTION:
            # Assuming code was extracted from the last action
            last_message = self.get_history(self.get_react_node(state))[-1]
            if not isinstance(last_message, Action):
                raise ValueError("Last message is not an action but last step was an action!")
            if last_message.action_json is not None:
                return self.next_step_and_observation_input(state, last_child)

        return self.next_step_and_action_input(state, last_child)