import copy
import logging
from re import S
from typing import Any
import json
from agents.recoma.discoveryworld_env_models import SingletonEnvironment
from agents.HypothesizerAgent import mkShortInteractableObjectList
import yaml
from recoma.models.core.base_model import BaseModel
from recoma.models.core.prompted_lm_model import PromptedLMModel
from recoma.models.core.generator import GenerationOutputs
from recoma.search.state import SearchState

logger = logging.getLogger(__name__)


@BaseModel.register("discoveryworld_promptedlm")
class DiscoveryWorldPromptedLMModel(PromptedLMModel):

    def __init__(self, max_prompt_length: int = 24000, **kwargs):
        super().__init__(**kwargs)
        self.max_prompt_length = max_prompt_length

    # def truncate_input(self, input_str):
    #     # last mention of goal
    #     max_prompt_length = self.max_prompt_length
    #     goal_index = input_str.rfind("Task:")
    #     if goal_index == -1:
    #         raise ValueError("No goal found in input string:\n{}".format(input_str))
    #     next_new_line_index = input_str.find("\n", goal_index) + 1
    #     init_prompt = input_str[:next_new_line_index]
    #     prompt = input_str[next_new_line_index:]
    #     if len(init_prompt) > max_prompt_length:
    #         print("="*40)
    #         print(input_str[next_new_line_index-50:next_new_line_index+50])
    #         print("*"*40)
    #         print(init_prompt, str(len(init_prompt)))
    #         raise ValueError("Input prompt longer than max allowed length")
    #     if len(prompt) > max_prompt_length - len(init_prompt):
    #         new_prompt =  prompt[-(max_prompt_length-len(init_prompt)):]
    #         cmd_index = new_prompt.find("ASSISTANT:") if "ASSISTANT:" in new_prompt else 0
    #         prompt = "\n[TRIMMED HISTORY]\n\n" + new_prompt[cmd_index:]
    #     return init_prompt + prompt

    # def generate_output(self, state) -> GenerationOutputs:
    #     """
    #     Generate the output string using this prompted LM by first building the LM input prompt and
    #     calling the generator to produce the output
    #     :return: generator outputs
    #     """
    #     open_node = state.get_open_node()
    #     if open_node is None:
    #         raise ValueError("Model called without any open node!!")

    #     lm_input = self.build_lm_input(self.prompt, open_node.input_str, state)
    #     lm_input = self.truncate_input(lm_input)
    #     output = self.generator.generate(lm_input, state)
    #     logger.debug("Input: ..." + lm_input[-200:])
    #     logger.debug("Output: " + output.outputs[0])
    #     open_node.add_input_output_prompt(lm_input, output)
    #     return output

    def populate_template_dictionary(self, input_str: str, state: SearchState) -> dict[str, Any]:
        param_dict = super().populate_template_dictionary(input_str, state)
        env = SingletonEnvironment().env
        observation = env.getAgentObservation(agentIdx=0)
        param_dict["facing_direction"] = observation["ui"]["agentLocation"]["faceDirection"]
        param_dict["valid_dirs"]= observation["ui"]["agentLocation"]["directions_you_can_move"]
        param_dict["additional_instructions"] = env.additionalActionDescriptionString()

        task_description = ""
        for task in observation["ui"]["taskProgress"]:
            task_description += task["description"] + "\n"
            task_description += " Progress: " + str(task["completed"]) + "\n\n"
        param_dict["task_str"] = task_description

        # Deep copy the observation dictionary
        observationNoVision = copy.deepcopy(observation)
        # Remove the 'vision' key from the observation
        observationNoVision.pop("vision", None)
        # Not needed since we already have task status above
        observationNoVision["ui"].pop("taskProgress", None)
        param_dict["observation"] = json.dumps(observationNoVision, indent=4, sort_keys=True)

        param_dict["known_actions"] = json.dumps(env.listKnownActions(limited=False), indent=4, sort_keys=True)
        param_dict["teleport_destinations"] = json.dumps(env.listTeleportLocationsDict(), indent=4, sort_keys=True)
        param_dict["interactable_objects"] = mkShortInteractableObjectList(observation=observation)

        # For dialog
        param_dict["in_dialog"] = env.isAgentInDialog(agentIdx=0)
        param_dict["dialog_box"] = json.dumps(observationNoVision["ui"]["dialog_box"], indent=4, sort_keys=True)
        return param_dict