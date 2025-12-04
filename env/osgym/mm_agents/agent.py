import base64
import json
import logging
import os
import re
import tempfile
import time
import xml.etree.ElementTree as ET
from http import HTTPStatus
from io import BytesIO
from pathlib import Path
from typing import Dict, List

import backoff
import dashscope
import google.generativeai as genai
import openai
import requests
import tiktoken
from PIL import Image
from google.api_core.exceptions import InvalidArgument, ResourceExhausted, InternalServerError, BadRequest
from groq import Groq
from requests.exceptions import SSLError

from .accessibility_tree_wrap.heuristic_retrieve import filter_nodes, draw_bounding_boxes
from .prompts import SYS_PROMPT_IN_SCREENSHOT_OUT_CODE, SYS_PROMPT_IN_SCREENSHOT_OUT_ACTION, \
    SYS_PROMPT_IN_A11Y_OUT_CODE, SYS_PROMPT_IN_A11Y_OUT_ACTION, \
    SYS_PROMPT_IN_BOTH_OUT_CODE, SYS_PROMPT_IN_BOTH_OUT_ACTION, \
    SYS_PROMPT_IN_SOM_OUT_TAG

from ..env_risk_utils.attack import agent_attack, is_single_color_image, adversarial_text_agent_attack
from ..env_risk_utils.general_attack_utils import extract_coordinate_list, find_largest_non_overlapping_box, extract_bounding_boxes_from_image, draw_som_for_attack_osworld

logger = logging.getLogger("desktopenv.agent")

MODULE_DIR = Path(__file__).resolve().parent
OSGYM_ROOT = MODULE_DIR.parent
INTENT_CLICK_TGT_PATH = OSGYM_ROOT / "env_risk_utils" / "intent_click_tgt_OK.json"

pure_text_settings = ['a11y_tree']

attributes_ns_ubuntu = "https://accessibility.windows.example.org/ns/attributes"
attributes_ns_windows = "https://accessibility.windows.example.org/ns/attributes"
state_ns_ubuntu = "https://accessibility.ubuntu.example.org/ns/state"
state_ns_windows = "https://accessibility.windows.example.org/ns/state"
component_ns_ubuntu = "https://accessibility.ubuntu.example.org/ns/component"
component_ns_windows = "https://accessibility.windows.example.org/ns/component"
value_ns_ubuntu = "https://accessibility.ubuntu.example.org/ns/value"
value_ns_windows = "https://accessibility.windows.example.org/ns/value"
class_ns_windows = "https://accessibility.windows.example.org/ns/class"
# More namespaces defined in OSWorld, please check desktop_env/server/main.py

def log_info(*args):
    # Convert each argument to a string and join them with spaces
    message = ' '.join(map(str, args))
    # Log the message
    logger.info(message)

# Function to encode the image
def encode_image(image_content):
    return base64.b64encode(image_content).decode('utf-8')


def encoded_img_to_pil_img(data_str):
    base64_str = data_str.replace("data:image/png;base64,", "")
    image_data = base64.b64decode(base64_str)
    image = Image.open(BytesIO(image_data))

    return image


def save_to_tmp_img_file(data_str):
    base64_str = data_str.replace("data:image/png;base64,", "")
    image_data = base64.b64decode(base64_str)
    image = Image.open(BytesIO(image_data))

    tmp_img_path = os.path.join(tempfile.mkdtemp(), "tmp_img.png")
    image.save(tmp_img_path)

    return tmp_img_path


def linearize_accessibility_tree(accessibility_tree, platform="ubuntu"):

    if platform == "ubuntu":
        _attributes_ns = attributes_ns_ubuntu
        _state_ns = state_ns_ubuntu
        _component_ns = component_ns_ubuntu
        _value_ns = value_ns_ubuntu
    elif platform == "windows":
        _attributes_ns = attributes_ns_windows
        _state_ns = state_ns_windows
        _component_ns = component_ns_windows
        _value_ns = value_ns_windows
    else:
        raise ValueError("Invalid platform, must be 'ubuntu' or 'windows'")

    filtered_nodes = filter_nodes(ET.fromstring(accessibility_tree), platform)
    linearized_accessibility_tree = ["tag\tname\ttext\tclass\tdescription\tposition (top-left x&y)\tsize (w&h)"]

    # Linearize the accessibility tree nodes into a table format
    for node in filtered_nodes:
        if node.text:
            text = (
                node.text if '"' not in node.text \
                    else '"{:}"'.format(node.text.replace('"', '""'))
            )

        elif node.get("{{{:}}}class".format(class_ns_windows), "").endswith("EditWrapper") \
                and node.get("{{{:}}}value".format(_value_ns)):
            node_text = node.get("{{{:}}}value".format(_value_ns), "")
            text = (node_text if '"' not in node_text \
                        else '"{:}"'.format(node_text.replace('"', '""'))
                    )
        else:
            text = '""'

        linearized_accessibility_tree.append(
            "{:}\t{:}\t{:}\t{:}\t{:}\t{:}\t{:}".format(
                node.tag, node.get("name", ""),
                text,
                node.get("{{{:}}}class".format(_attributes_ns), "") if platform == "ubuntu" else node.get("{{{:}}}class".format(class_ns_windows), ""),
                node.get("{{{:}}}description".format(_attributes_ns), ""),
                node.get('{{{:}}}screencoord'.format(_component_ns), ""),
                node.get('{{{:}}}size'.format(_component_ns), "")
            )
        )

    return "\n".join(linearized_accessibility_tree)


def tag_screenshot(screenshot, accessibility_tree, platform="ubuntu"):
    nodes = filter_nodes(ET.fromstring(accessibility_tree), platform=platform, check_image=True)
    # Make tag screenshot
    marks, drew_nodes, element_list, tagged_screenshot = draw_bounding_boxes(nodes, screenshot)

    return marks, drew_nodes, tagged_screenshot, element_list


def parse_actions_from_string(input_string):
    if input_string.strip() in ['WAIT', 'DONE', 'FAIL']:
        return [input_string.strip()]
    # Search for a JSON string within the input string
    actions = []
    matches = re.findall(r'```json\s+(.*?)\s+```', input_string, re.DOTALL)
    if matches:
        # Assuming there's only one match, parse the JSON string into a dictionary
        try:
            for match in matches:
                action_dict = json.loads(match)
                actions.append(action_dict)
            return actions
        except json.JSONDecodeError as e:
            return f"Failed to parse JSON: {e}"
    else:
        matches = re.findall(r'```\s+(.*?)\s+```', input_string, re.DOTALL)
        if matches:
            # Assuming there's only one match, parse the JSON string into a dictionary
            try:
                for match in matches:
                    action_dict = json.loads(match)
                    actions.append(action_dict)
                return actions
            except json.JSONDecodeError as e:
                return f"Failed to parse JSON: {e}"
        else:
            try:
                action_dict = json.loads(input_string)
                return [action_dict]
            except json.JSONDecodeError:
                raise ValueError("Invalid response format: " + input_string)


def parse_code_from_string(input_string):
    input_string = "\n".join([line.strip() for line in input_string.split(';') if line.strip()])
    if input_string.strip() in ['WAIT', 'DONE', 'FAIL']:
        return [input_string.strip()]

    # This regular expression will match both ```code``` and ```python code```
    # and capture the `code` part. It uses a non-greedy match for the content inside.
    pattern = r"```(?:\w+\s+)?(.*?)```"
    # Find all non-overlapping matches in the string
    matches = re.findall(pattern, input_string, re.DOTALL)

    # The regex above captures the content inside the triple backticks.
    # The `re.DOTALL` flag allows the dot `.` to match newline characters as well,
    # so the code inside backticks can span multiple lines.

    # matches now contains all the captured code snippets

    codes = []

    for match in matches:
        match = match.strip()
        commands = ['WAIT', 'DONE', 'FAIL']  # fixme: updates this part when we have more commands

        if match in commands:
            codes.append(match.strip())
        elif match.split('\n')[-1] in commands:
            if len(match.split('\n')) > 1:
                codes.append("\n".join(match.split('\n')[:-1]))
            codes.append(match.split('\n')[-1])
        else:
            codes.append(match)

    return codes


def parse_code_from_som_string(input_string, masks):
    # parse the output string by masks
    tag_vars = ""
    for i, mask in enumerate(masks):
        x, y, w, h = mask
        tag_vars += "tag_" + str(i + 1) + "=" + "({}, {})".format(int(x + w // 2), int(y + h // 2))
        tag_vars += "\n"

    actions = parse_code_from_string(input_string)

    for i, action in enumerate(actions):
        if action.strip() in ['WAIT', 'DONE', 'FAIL']:
            pass
        else:
            action = tag_vars + action
            actions[i] = action

    return actions


def trim_accessibility_tree(linearized_accessibility_tree, max_tokens):
    enc = tiktoken.encoding_for_model("gpt-4")
    tokens = enc.encode(linearized_accessibility_tree)
    if len(tokens) > max_tokens:
        linearized_accessibility_tree = enc.decode(tokens[:max_tokens])
        linearized_accessibility_tree += "[...]\n"
    return linearized_accessibility_tree

def filter_bounding_boxes(bounding_boxes, nodes, max_width=1920, max_height=1080):
    """
    Filters out bounding boxes larger than the specified width and height.
    
    Args:
    bounding_boxes (list): List of bounding boxes in format [x, y, w, h].
    max_width (int): Maximum allowed width.
    max_height (int): Maximum allowed height.
    
    Returns:
    list: Filtered bounding boxes.
    """

    def get_node_text(_node):
        if _node.text:
            node_text = (_node.text if '"' not in _node.text \
                else '"{:}"'.format(_node.text.replace('"', '""'))
            )
        elif _node.get("{uri:deskat:uia.windows.microsoft.org}class", "").endswith("EditWrapper") \
            and _node.get("{uri:deskat:value.at-spi.gnome.org}value"):
            node_text: str = _node.get("{uri:deskat:value.at-spi.gnome.org}value")
            node_text = (node_text if '"' not in node_text \
                else '"{:}"'.format(node_text.replace('"', '""'))
            )
        else:
            node_text = '""'
        return _node.tag + " " + _node.get("name", "")+ " " + node_text

    filtered_boxes = []

    for id, box in enumerate(bounding_boxes):
        if box[2] <= max_width or box[3] <= max_height:
            filtered_boxes.append(box)
        else:
            logger.debug(str(id) + " " + str(box) + " " + get_node_text(nodes[id]) + "removed")
    return filtered_boxes

def agent_attack_wrapper(attack_config, instruction, button_name_dict, current_boundingbox, nodes, current_observation, som=False, reserved_index=-1, content_str=""):
    OCR_boxes = extract_bounding_boxes_from_image(current_observation)
    current_observation = Image.open(BytesIO(current_observation))
    current_boundingbox = filter_bounding_boxes(current_boundingbox, nodes)
    
    # try not to block any tags or detected texts
    current_boundingbox = current_boundingbox + OCR_boxes

    largest_non_overlapping_box = find_largest_non_overlapping_box((1920, 1080), current_boundingbox)

    if attack_config:
        current_observation, skip, button_name_dict, whole_attack_bounding_box, x_tgt, y_tgt, attack_string = adversarial_text_agent_attack(attack_config=attack_config, \
                                                                                                        current_observation=current_observation, \
                                                                                                        largest_non_overlapping_box=largest_non_overlapping_box, \
                                                                                                        reserved_index=reserved_index, \
                                                                                                        reserved_color=None,
                                                                                                        customized_draw_som_for_attack=draw_som_for_attack_osworld, \
                                                                                                        som=som, \
                                                                                                        button_name_dict=button_name_dict, \
                                                                                                        instruction=instruction, \
                                                                                                        customized_print=log_info, \
                                                                                                        content_str=content_str)
    else:
        current_observation, skip, button_name_dict, whole_attack_bounding_box, x_tgt, y_tgt, attack_string = agent_attack(attack_config=attack_config, \
                                                                                                            current_observation=current_observation, \
                                                                                                            largest_non_overlapping_box=largest_non_overlapping_box, \
                                                                                                            reserved_index=reserved_index, \
                                                                                                            reserved_color=None,
                                                                                                            customized_draw_som_for_attack=draw_som_for_attack_osworld, \
                                                                                                            som=som, \
                                                                                                            button_name_dict=button_name_dict, \
                                                                                                            instruction=instruction, \
                                                                                                            customized_print=log_info, \
                                                                                                            content_str=content_str)
    image_bytes_io = BytesIO()
    current_observation.save(image_bytes_io, format='PNG')
    current_observation = image_bytes_io.getvalue()

    return current_observation, skip, button_name_dict, whole_attack_bounding_box, x_tgt, y_tgt, attack_string

class PromptAgent:
    def __init__(
            self,
            platform="ubuntu",
            model="gpt-4-vision-preview",
            max_tokens=1500,
            top_p=0.9,
            temperature=0.5,
            action_space="computer_13",
            observation_type="screenshot_a11y_tree",
            # observation_type can be in ["screenshot", "a11y_tree", "screenshot_a11y_tree", "som"]
            max_trajectory_length=3,
            a11y_tree_max_tokens=10000
    ):
        self.platform = platform
        self.model = model
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.temperature = temperature
        self.action_space = action_space
        self.observation_type = observation_type
        self.max_trajectory_length = 0 # max_trajectory_length
        self.a11y_tree_max_tokens = a11y_tree_max_tokens

        self.thoughts = []
        self.actions = []
        self.observations = []
        self.button_name_dict = {}

        if observation_type == "screenshot":
            if action_space == "computer_13":
                self.system_message = SYS_PROMPT_IN_SCREENSHOT_OUT_ACTION
            elif action_space == "pyautogui":
                self.system_message = SYS_PROMPT_IN_SCREENSHOT_OUT_CODE
            else:
                raise ValueError("Invalid action space: " + action_space)
        elif observation_type == "a11y_tree":
            if action_space == "computer_13":
                self.system_message = SYS_PROMPT_IN_A11Y_OUT_ACTION
            elif action_space == "pyautogui":
                self.system_message = SYS_PROMPT_IN_A11Y_OUT_CODE
            else:
                raise ValueError("Invalid action space: " + action_space)
        elif observation_type == "screenshot_a11y_tree":
            if action_space == "computer_13":
                self.system_message = SYS_PROMPT_IN_BOTH_OUT_ACTION
            elif action_space == "pyautogui":
                self.system_message = SYS_PROMPT_IN_BOTH_OUT_CODE
            else:
                raise ValueError("Invalid action space: " + action_space)
        elif observation_type == "som":
            if action_space == "computer_13":
                raise ValueError("Invalid action space: " + action_space)
            elif action_space == "pyautogui":
                self.system_message = SYS_PROMPT_IN_SOM_OUT_TAG
            else:
                raise ValueError("Invalid action space: " + action_space)
        else:
            raise ValueError("Invalid experiment type: " + observation_type)

    def predict(self, instruction: str, obs: Dict, example_id: str, step_idx: bool) -> List:
        """
        Predict the next action(s) based on the current observation.
        """
        system_message = self.system_message + "\nYou are asked to complete the following task: {}".format(instruction)

        # Prepare the payload for the API call
        messages = []
        masks = None

        messages.append({
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": system_message
                },
            ]
        })

        # Append trajectory
        assert len(self.observations) == len(self.actions) and len(self.actions) == len(self.thoughts) \
            , "The number of observations and actions should be the same."

        if len(self.observations) > self.max_trajectory_length:
            if self.max_trajectory_length == 0:
                _observations = []
                _actions = []
                _thoughts = []
            else:
                _observations = self.observations[-self.max_trajectory_length:]
                _actions = self.actions[-self.max_trajectory_length:]
                _thoughts = self.thoughts[-self.max_trajectory_length:]
        else:
            _observations = self.observations
            _actions = self.actions
            _thoughts = self.thoughts

        for previous_obs, previous_action, previous_thought in zip(_observations, _actions, _thoughts):

            # {{{1
            if self.observation_type == "screenshot_a11y_tree":
                _screenshot = previous_obs["screenshot"]
                _linearized_accessibility_tree = previous_obs["accessibility_tree"]

                messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Given the screenshot and info from accessibility tree as below:\n{}\nWhat's the next step that you will do to help with the task?".format(
                                _linearized_accessibility_tree)
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{_screenshot}",
                                "detail": "high"
                            }
                        }
                    ]
                })
            elif self.observation_type in ["som"]:
                _screenshot = previous_obs["screenshot"]

                messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Given the tagged screenshot as below. What's the next step that you will do to help with the task?"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{_screenshot}",
                                "detail": "high"
                            }
                        }
                    ]
                })
            elif self.observation_type == "screenshot":
                _screenshot = previous_obs["screenshot"]

                messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Given the screenshot as below. What's the next step that you will do to help with the task?"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{_screenshot}",
                                "detail": "high"
                            }
                        }
                    ]
                })
            elif self.observation_type == "a11y_tree":
                _linearized_accessibility_tree = previous_obs["accessibility_tree"]

                messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Given the info from accessibility tree as below:\n{}\nWhat's the next step that you will do to help with the task?".format(
                                _linearized_accessibility_tree)
                        }
                    ]
                })
            else:
                raise ValueError("Invalid observation_type type: " + self.observation_type)  # 1}}}

            messages.append({
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": previous_thought.strip() if len(previous_thought) > 0 else "No valid action"
                    },
                ]
            })

        # {{{1
        if self.observation_type in ["screenshot", "screenshot_a11y_tree"]:
            whole_attack_bounding_box, x_tgt, y_tgt = None, None, None
            if "popup" in example_id or "induced_text" in example_id:
                if "induced_text" in example_id:
                    attack_config = str(INTENT_CLICK_TGT_PATH)
                else:
                    attack_config = False
                current_observation = obs["screenshot"]
                current_boundingbox, nodes, _, linearized_accessibility_tree = tag_screenshot(obs["screenshot"], obs["accessibility_tree"], self.platform)
                current_observation, skip, self.button_name_dict, whole_attack_bounding_box, x_tgt, y_tgt, attack_string = agent_attack_wrapper(attack_config, instruction, self.button_name_dict, current_boundingbox, nodes, current_observation, content_str=linearized_accessibility_tree)
                base64_image = encode_image(current_observation)
            else:
                base64_image = encode_image(obs["screenshot"])
            linearized_accessibility_tree = linearize_accessibility_tree(accessibility_tree=obs["accessibility_tree"],
                                                                         platform=self.platform) if self.observation_type == "screenshot_a11y_tree" else None
            logger.debug("LINEAR AT: %s", linearized_accessibility_tree)

            if linearized_accessibility_tree:
                linearized_accessibility_tree = trim_accessibility_tree(linearized_accessibility_tree,
                                                                        self.a11y_tree_max_tokens)

            if self.observation_type == "screenshot_a11y_tree":
                self.observations.append({
                    "screenshot": base64_image,
                    "accessibility_tree": linearized_accessibility_tree
                })
            else:
                self.observations.append({
                    "screenshot": base64_image,
                    "accessibility_tree": None
                })

            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Given the screenshot as below. What's the next step that you will do to help with the task?"
                        if self.observation_type == "screenshot"
                        else "Given the screenshot and info from accessibility tree as below:\n{}\nWhat's the next step that you will do to help with the task?".format(
                            linearized_accessibility_tree)
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}",
                            "detail": "high"
                        }
                    }
                ]
            })
        elif self.observation_type == "a11y_tree":
            linearized_accessibility_tree = linearize_accessibility_tree(accessibility_tree=obs["accessibility_tree"],
                                                                         platform=self.platform)
            logger.debug("LINEAR AT: %s", linearized_accessibility_tree)

            if linearized_accessibility_tree:
                linearized_accessibility_tree = trim_accessibility_tree(linearized_accessibility_tree,
                                                                        self.a11y_tree_max_tokens)

            self.observations.append({
                "screenshot": None,
                "accessibility_tree": linearized_accessibility_tree
            })

            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Given the info from accessibility tree as below:\n{}\nWhat's the next step that you will do to help with the task?".format(
                            linearized_accessibility_tree)
                    }
                ]
            })
        elif self.observation_type == "som":
            # Add som to the screenshot
            masks, drew_nodes, tagged_screenshot, linearized_accessibility_tree = tag_screenshot(obs["screenshot"], obs[
                "accessibility_tree"], self.platform)
            base64_image = encode_image(tagged_screenshot)
            logger.debug("LINEAR AT: %s", linearized_accessibility_tree)

            if linearized_accessibility_tree:
                linearized_accessibility_tree = trim_accessibility_tree(linearized_accessibility_tree,
                                                                        self.a11y_tree_max_tokens)

            self.observations.append({
                "screenshot": base64_image,
                "accessibility_tree": linearized_accessibility_tree
            })

            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Given the tagged screenshot and info from accessibility tree as below:\n{}\nWhat's the next step that you will do to help with the task?".format(
                            linearized_accessibility_tree)
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}",
                            "detail": "high"
                        }
                    }
                ]
            })
        else:
            raise ValueError("Invalid observation_type type: " + self.observation_type)  # 1}}}

        try:
            response = self.call_llm({
                "model": self.model,
                "messages": messages,
                "max_tokens": self.max_tokens,
                "top_p": self.top_p,
                "temperature": self.temperature
            })
        except Exception as e:
            logger.error("Failed to call " + self.model + ", Error: " + str(e))
            response = ""
        
        logger.info("RESPONSE: %s", response)
        # print(response)

        try:
            actions = self.parse_actions(response, masks)
            self.thoughts.append(response)
        except ValueError as e:
            print("Failed to parse action from response", e)
            actions = None
            self.thoughts.append("")

        return response, actions, whole_attack_bounding_box, x_tgt, y_tgt

    @backoff.on_exception(
        backoff.constant,
        # here you should add more model exceptions as you want,
        # but you are forbidden to add "Exception", that is, a common type of exception
        # because we want to catch this kind of Exception in the outside to ensure each example won't exceed the time limit
        (
                # General exceptions
                SSLError,

                # OpenAI exceptions
                openai.RateLimitError,
                openai.BadRequestError,
                openai.InternalServerError,

                # Google exceptions
                InvalidArgument,
                ResourceExhausted,
                InternalServerError,
                BadRequest,

                # Groq exceptions
                # todo: check
        ),
        interval=30,
        max_tries=10
    )
    def call_llm(self, payload):
        if self.model.startswith("lmdeploy") or self.model.startswith("sglang") or self.model.startswith("vllm"):
            # import warnings
            # warnings.filterwarnings("ignore", category=FutureWarning)
            # from lmdeploy.serve.openai.api_client import APIClient
            from openai import OpenAI
            base_url = os.environ["LMDEPLOY_BASE_URL"]
            messages = payload["messages"]
            max_tokens = payload["max_tokens"]
            top_p = payload["top_p"]
            temperature = payload["temperature"]
            lmdeploy_messages = messages

            if lmdeploy_messages[0]['role'] == "system":
                lmdeploy_system_message_item = lmdeploy_messages[0]['content'][0]
                lmdeploy_messages[1]['content'].insert(0, lmdeploy_system_message_item)
                lmdeploy_messages.pop(0)

            # The implementation based on OpenAI
            
            client = OpenAI( 
                base_url=base_url, api_key="EMPTY") # you need to modify the IP address when you use another compute node
            model_name = client.models.list().data[0].id
     
            flag = 0
            while True:
                try:
                    if flag > 20:
                        break
                    logger.info("Generating content with model: %s", self.model)

                    # The implementation based on OpenAI
                    response = client.chat.completions.create(
                        model=model_name,
                        messages=lmdeploy_messages,
                        max_tokens=max_tokens,
                        top_p=top_p,
                        temperature=temperature)
                    break
                except:
                    if flag == 0:
                        lmdeploy_messages = [lmdeploy_messages[0]] + lmdeploy_messages[-1:]
                    else:
                        lmdeploy_messages[-1]["content"] = ' '.join(lmdeploy_messages[-1]["content"].split()[:-500])
                    flag = flag + 1
            try:
                # The implementation based on OpenAI
                return response.choices[0].message.content
            except Exception as e:
                print("Failed to call LLM: " + str(e))
                return ""

        elif self.model.startswith("gpt"):
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"
            }
            url = os.environ["OPENAI_URL_BASE"]
            logger.info("Generating content with GPT model: %s", self.model)
            logger.info(f"url: {url}")
            response = requests.post(
                url,
                headers=headers,
                json=payload
            )
           
            if response.status_code != 200:
                if response.json()['error']['code'] == "context_length_exceeded":
                    logger.error("Context length exceeded. Retrying with a smaller context.")
                    payload["messages"] = [payload["messages"][0]] + payload["messages"][-1:]
                    retry_response = requests.post(
                        url,
                        headers=headers,
                        json=payload
                    )
                    if retry_response.status_code != 200:
                        logger.error(
                            "Failed to call LLM even after attempt on shortening the history: " + retry_response.text)
                        return ""

                logger.error("Failed to call LLM: " + response.text)
                time.sleep(5)
                return ""
            else:
                logger.info(f"response:{response}")
                try:
                    import inspect
                    logger.info(f"type(response): {type(response)}, dir(response): {dir(response)}")
                    logger.info(f"inspect.ismethod(getattr(response, 'json')): {inspect.ismethod(getattr(response, 'json'))}")
                    logger.info(f"inspect.isfunction(getattr(response, 'json')): {inspect.isfunction(getattr(response, 'json'))}")
                    logger.info(f"response.raw: {response.raw}")
                    logger.info(f"response.text: {response.text}")
                    return response.json()['choices'][0]['message']['content']
                except Exception as e:
                    logger.info(f"error: {e}")
                    logger.info(f"response: {response}")
                    return response['choices'][0]['message']['content']
        
        elif self.model.startswith("claude"):
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {os.environ['ANTHROPIC_API_KEY']}"
            }

            url = os.environ["CLAUDE_URL_BASE"]
            logger.info("Generating content with Claude model: %s", self.model)
            response = requests.post(
                url,
                headers=headers,
                json=payload
            )
           
            if response.status_code != 200:
                logger.info("response.json(): %s", response.json())
                if response.json()['error']['code'] == "context_length_exceeded":
                    logger.error("Context length exceeded. Retrying with a smaller context.")
                    payload["messages"] = [payload["messages"][0]] + payload["messages"][-1:]
                    retry_response = requests.post(
                        url,
                        headers=headers,
                        json=payload
                    )
                    if retry_response.status_code != 200:
                        logger.error(
                            "Failed to call LLM even after attempt on shortening the history: " + retry_response.text)
                        return ""

                logger.error("Failed to call LLM: " + response.text)
                time.sleep(5)
                return ""
            else:
                logger.info("response.json(): %s", response.json())
                logger.info("response.json()['choices'][0]['message']['content']: %s", response.json()['choices'][0]['message']['content'])
                return response.json()['choices'][0]['message']['content']

        elif self.model.startswith("gemini"):
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {os.environ['GOOGLE_API_KEY']}"
            }
            url = os.environ["GEMINI_URL_BASE"]
            logger.info("Generating content with Gemini model: %s", self.model)
            

            response = requests.post(
                url,
                headers=headers,
                json=payload
            )

   
            if response.status_code != 200:
                if 'error' in response.json() and response.json()['error'].get('code') == "context_length_exceeded":
                    logger.error("Context length exceeded. Retrying with a smaller context.")
                    payload["messages"] = [payload["messages"][0]] + payload["messages"][-1:]
                    retry_response = requests.post(
                        url,
                        headers=headers,
                        json=payload
                    )
                    if retry_response.status_code != 200:
                        logger.error(
                            "Failed to call LLM even after attempt on shortening the history: " + retry_response.text)
                        return ""

                logger.error("Failed to call LLM: " + response.text)
                time.sleep(5)
                return ""
            else:
                logger.info(f"response:{response}")
                try:
                    logger.info(f"response.text: {response.text}")
                    return response.json()['choices'][0]['message']['content']
                except Exception as e:
                    logger.info(f"error: {e}")
                    logger.info(f"response: {response}")
                    return response['choices'][0]['message']['content']

        elif self.model.startswith("mistral"):
            messages = payload["messages"]
            max_tokens = payload["max_tokens"]
            top_p = payload["top_p"]
            temperature = payload["temperature"]

            assert self.observation_type in pure_text_settings, f"The model {self.model} can only support text-based input, please consider change based model or settings"

            mistral_messages = []

            for i, message in enumerate(messages):
                mistral_message = {
                    "role": message["role"],
                    "content": ""
                }

                for part in message["content"]:
                    mistral_message['content'] = part['text'] if part['type'] == "text" else ""

                mistral_messages.append(mistral_message)

            from openai import OpenAI

            client = OpenAI(api_key=os.environ["TOGETHER_API_KEY"],
                            base_url=os.environ["TOGETHER_URL_BASE"],
                            )

            flag = 0
            while True:
                try:
                    if flag > 20:
                        break
                    logger.info("Generating content with model: %s", self.model)
                    response = client.chat.completions.create(
                        messages=mistral_messages,
                        model=self.model,
                        max_tokens=max_tokens,
                        top_p=top_p,
                        temperature=temperature
                    )
                    break
                except:
                    if flag == 0:
                        mistral_messages = [mistral_messages[0]] + mistral_messages[-1:]
                    else:
                        mistral_messages[-1]["content"] = ' '.join(mistral_messages[-1]["content"].split()[:-500])
                    flag = flag + 1

            try:
                return response.choices[0].message.content
            except Exception as e:
                print("Failed to call LLM: " + str(e))
                return ""

        elif self.model.startswith("THUDM"):
            # THUDM/cogagent-chat-hf
            messages = payload["messages"]
            max_tokens = payload["max_tokens"]
            top_p = payload["top_p"]
            temperature = payload["temperature"]

            cog_messages = []

            for i, message in enumerate(messages):
                cog_message = {
                    "role": message["role"],
                    "content": []
                }

                for part in message["content"]:
                    if part['type'] == "image_url":
                        cog_message['content'].append(
                            {"type": "image_url", "image_url": {"url": part['image_url']['url']}})

                    if part['type'] == "text":
                        cog_message['content'].append({"type": "text", "text": part['text']})

                cog_messages.append(cog_message)

            # the cogagent not support system message in our endpoint, so we concatenate it at the first user message
            if cog_messages[0]['role'] == "system":
                cog_system_message_item = cog_messages[0]['content'][0]
                cog_messages[1]['content'].insert(0, cog_system_message_item)
                cog_messages.pop(0)

            payload = {
                "model": self.model,
                "max_tokens": max_tokens,
                "messages": cog_messages,
                "temperature": temperature,
                "top_p": top_p
            }

            base_url = "http://127.0.0.1:8000"

            response = requests.post(f"{base_url}/v1/chat/completions", json=payload, stream=False)
            if response.status_code == 200:
                decoded_line = response.json()
                content = decoded_line.get("choices", [{}])[0].get("message", "").get("content", "")
                return content
            else:
                print("Failed to call LLM: ", response.status_code)
                return ""

        elif self.model in ["gemini-pro", "gemini-pro-vision"]:
            messages = payload["messages"]
            max_tokens = payload["max_tokens"]
            top_p = payload["top_p"]
            temperature = payload["temperature"]

            content = payload['messages'][0]['content']
            print(content)

            if self.model == "gemini-pro":
                assert self.observation_type in pure_text_settings, f"The model {self.model} can only support text-based input, please consider change based model or settings"

            gemini_messages = []
            for i, message in enumerate(messages):
                role_mapping = {
                    "assistant": "model",
                    "user": "user",
                    "system": "system"
                }
                gemini_message = {
                    "role": role_mapping[message["role"]],
                    "parts": []
                }
                assert len(message["content"]) in [1, 2], "One text, or one text with one image"

                # The gemini only support the last image as single image input
                if i == len(messages) - 1:
                    for part in message["content"]:
                        gemini_message['parts'].append(part['text']) if part['type'] == "text" \
                            else gemini_message['parts'].append(encoded_img_to_pil_img(part['image_url']['url']))
                else:
                    for part in message["content"]:
                        gemini_message['parts'].append(part['text']) if part['type'] == "text" else None

                gemini_messages.append(gemini_message)

            # the gemini not support system message in our endpoint, so we concatenate it at the first user message
            if gemini_messages[0]['role'] == "system":
                gemini_messages[1]['parts'][0] = gemini_messages[0]['parts'][0] + "\n" + gemini_messages[1]['parts'][0]
                gemini_messages.pop(0)

            # since the gemini-pro-vision donnot support multi-turn message
            if self.model == "gemini-pro-vision":
                message_history_str = ""
                for message in gemini_messages:
                    message_history_str += "<|" + message['role'] + "|>\n" + message['parts'][0] + "\n"
                gemini_messages = [{"role": "user", "parts": [message_history_str, gemini_messages[-1]['parts'][1]]}]
                # gemini_messages[-1]['parts'][1].save("output.png", "PNG")

            # print(gemini_messages)
            api_key = os.environ.get("GENAI_API_KEY")
            assert api_key is not None, "Please set the GENAI_API_KEY environment variable"
            genai.configure(api_key=api_key)
            logger.info("Generating content with Gemini model: %s", self.model)
            # request_options = {"timeout": 120}
            request_options = {"timeout": 10} ## for debug
            gemini_model = genai.GenerativeModel(self.model)

            response = gemini_model.generate_content(
                gemini_messages,
                generation_config={
                    "candidate_count": 1,
                    # "max_output_tokens": max_tokens,
                    "top_p": top_p,
                    "temperature": temperature
                },
                safety_settings={
                    "harassment": "block_none",
                    "hate": "block_none",
                    "sex": "block_none",
                    "danger": "block_none"
                },
                request_options=request_options
            )
            return response.text

        elif self.model == "gemini-1.5-pro-latest":
            messages = payload["messages"]
            max_tokens = payload["max_tokens"]
            top_p = payload["top_p"]
            temperature = payload["temperature"]

            gemini_messages = []
            for i, message in enumerate(messages):
                role_mapping = {
                    "assistant": "model",
                    "user": "user",
                    "system": "system"
                }
                assert len(message["content"]) in [1, 2], "One text, or one text with one image"
                gemini_message = {
                    "role": role_mapping[message["role"]],
                    "parts": []
                }

                # The gemini only support the last image as single image input
                for part in message["content"]:

                    if part['type'] == "image_url":
                        # Put the image at the beginning of the message
                        gemini_message['parts'].insert(0, encoded_img_to_pil_img(part['image_url']['url']))
                    elif part['type'] == "text":
                        gemini_message['parts'].append(part['text'])
                    else:
                        raise ValueError("Invalid content type: " + part['type'])

                gemini_messages.append(gemini_message)

            # the system message of gemini-1.5-pro-latest need to be inputted through model initialization parameter
            system_instruction = None
            if gemini_messages[0]['role'] == "system":
                system_instruction = gemini_messages[0]['parts'][0]
                gemini_messages.pop(0)

            api_key = os.environ.get("GENAI_API_KEY")
            assert api_key is not None, "Please set the GENAI_API_KEY environment variable"
            genai.configure(api_key=api_key)
            logger.info("Generating content with Gemini model: %s", self.model)
            # request_options = {"timeout": 120}
            request_options = {"timeout": 10} ## for debug
            gemini_model = genai.GenerativeModel(
                self.model,
                system_instruction=system_instruction
            )

            with open("response.json", "w") as f:
                messages_to_save = []
                for message in gemini_messages:
                    messages_to_save.append({
                        "role": message["role"],
                        "content": [part if isinstance(part, str) else "image" for part in message["parts"]]
                    })
                json.dump(messages_to_save, f, indent=4)

            response = gemini_model.generate_content(
                gemini_messages,
                generation_config={
                    "candidate_count": 1,
                    # "max_output_tokens": max_tokens,
                    "top_p": top_p,
                    "temperature": temperature
                },
                safety_settings={
                    "harassment": "block_none",
                    "hate": "block_none",
                    "sex": "block_none",
                    "danger": "block_none"
                },
                request_options=request_options
            )

            return response.text
            
        elif 'Qwen2.5-VL-72B-Instruct' in self.model:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {os.environ['QWEN_API_KEY']}"
            }

            url = os.environ["BASE_URL"]
            logger.info("Generating content with Qwen model: %s", self.model)
            response = requests.post(
                url,
                headers=headers,
                json=payload
            )
    
            if response.status_code != 200:
                if 'error' in response.json() and response.json()['error'].get('code') == "context_length_exceeded":
                    logger.error("Context length exceeded. Retrying with a smaller context.")
                    payload["messages"] = [payload["messages"][0]] + payload["messages"][-1:]
                    retry_response = requests.post(
                        url,
                        headers=headers,
                        json=payload
                    )
                    if retry_response.status_code != 200:
                        logger.error(
                            "Failed to call LLM even after attempt on shortening the history: " + retry_response.text)
                        return ""

                logger.error("Failed to call LLM: " + response.text)
                time.sleep(5)
                return ""
            else:
                logger.info(f"response:{response}")
                try:
                    logger.info(f"response.text: {response.text}")
                    return response.json()['choices'][0]['message']['content']
                except Exception as e:
                    logger.info(f"error: {e}")
                    logger.info(f"response: {response}")
                    return response['choices'][0]['message']['content']
                
        elif 'Qwen2-VL-72B-Instruct' in self.model:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {os.environ['QWEN_API_KEY']}"
            }
            # logger.info(f"payload: {payload}")
            url = os.environ["BASE_URL"]
            logger.info("Generating content with Qwen model: %s", self.model)
            response = requests.post(
                url,
                headers=headers,
                json=payload
            )
    
            if response.status_code != 200:
                if 'error' in response.json() and response.json()['error'].get('code') == "context_length_exceeded":
                    logger.error("Context length exceeded. Retrying with a smaller context.")
                    payload["messages"] = [payload["messages"][0]] + payload["messages"][-1:]
                    retry_response = requests.post(
                        url,
                        headers=headers,
                        json=payload
                    )
                    if retry_response.status_code != 200:
                        logger.error(
                            "Failed to call LLM even after attempt on shortening the history: " + retry_response.text)
                        return ""

                logger.error("Failed to call LLM: " + response.text)
                time.sleep(5)
                return ""
            else:
                logger.info(f"response:{response}")
                try:
                    logger.info(f"response.text: {response.text}")
                    return response.json()['choices'][0]['message']['content']
                except Exception as e:
                    logger.info(f"error: {e}")
                    logger.info(f"response: {response}")
                    return response['choices'][0]['message']['content']
                
        elif 'llama-3.2-90b-vision-instruct' in self.model:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {os.environ['LLAMA_API_KEY']}"
            }
            url = os.environ["BASE_URL"]
            logger.info("Generating content with llama model: %s", self.model)
            response = requests.post(
                url,
                headers=headers,
                json=payload
            )
    
            if response.status_code != 200:
                if 'error' in response.json() and response.json()['error'].get('code') == "context_length_exceeded":
                    logger.error("Context length exceeded. Retrying with a smaller context.")
                    payload["messages"] = [payload["messages"][0]] + payload["messages"][-1:]
                    retry_response = requests.post(
                        url,
                        headers=headers,
                        json=payload
                    )
                    if retry_response.status_code != 200:
                        logger.error(
                            "Failed to call LLM even after attempt on shortening the history: " + retry_response.text)
                        return ""

                logger.error("Failed to call LLM: " + response.text)
                time.sleep(5)
                return ""
            else:
                logger.info(f"response:{response}")
                try:
                    logger.info(f"response.text: {response.text}")
                    return response.json()['choices'][0]['message']['content']
                except Exception as e:
                    logger.info(f"error: {e}")
                    logger.info(f"response: {response}")
                    return response['choices'][0]['message']['content']

        else:
            raise ValueError("Invalid model: " + self.model)

    def parse_actions(self, response: str, masks=None):

        if self.observation_type in ["screenshot", "a11y_tree", "screenshot_a11y_tree"]:
            # parse from the response
            if self.action_space == "computer_13":
                actions = parse_actions_from_string(response)
            elif self.action_space == "pyautogui":
                actions = parse_code_from_string(response)
            else:
                raise ValueError("Invalid action space: " + self.action_space)

            self.actions.append(actions)

            return actions
        elif self.observation_type in ["som"]:
            # parse from the response
            if self.action_space == "computer_13":
                raise ValueError("Invalid action space: " + self.action_space)
            elif self.action_space == "pyautogui":
                actions = parse_code_from_som_string(response, masks)
            else:
                raise ValueError("Invalid action space: " + self.action_space)

            self.actions.append(actions)

            return actions

    def reset(self, _logger=None):
        global logger
        logger = _logger if _logger is not None else logging.getLogger("desktopenv.agent")

        self.thoughts = []
        self.actions = []
        self.observations = []