'''
Author: Muyao 2350076251@qq.com
Date: 2025-02-18 15:57:29
LastEditors: Muyao 2350076251@qq.com
LastEditTime: 2025-03-05 22:48:27
'''

import re
import numpy as np
import copy
import pickle
import json
from collections import OrderedDict
from typing import Union, List, Dict
import torch
from tqdm import tqdm
from pathlib import Path
from rich import console
from abc import ABC, abstractmethod

from minestudio.utils.vpt_lib.actions import ActionTransformer, Buttons
from minestudio.utils.vpt_lib.action_mapping import CameraHierarchicalMapping
from minestudio.simulator.entry import CameraConfig


def get_special_token(model_id: str, bases: list = [10, 3, 3, 3, 2, 2, 2, 2, 2, 2, 11, 11]) -> list:
    """
    Generate a list of all unknown tokens to mark unknown tokens.

    Args:
        model_id (str): Model identifier used to load the corresponding tokenizer.
        bases (list): List of bases for buttons and camera.

    Returns:
        list: A list containing all unknown tokens.

    Note:
        It is assumed that the number 8641 will never appear.
    """
    from transformers import AutoTokenizer
    # Load the pretrained tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # Calculate the required extra tokens based on the sum of bases plus an additional 30 tokens
    token_num = sum(bases) + 30
    # Sort and extract the last token_num special tokens from the vocabulary
    special_tokens = sorted(list(tokenizer.vocab.items()), key=lambda x: x[-1])[-token_num:]
    return special_tokens


def prepare_for_remap_control_token(tokenizer_type: str,
                                    bases: list = [10, 3, 3, 3, 2, 2, 2, 2, 2, 2, 21, 21],
                                    not_text=True) -> dict:
    """
    Prepare a dictionary for remapping control tokens.

    Args:
        tokenizer_type (str): Type of tokenizer.
        bases (list): List of bases for each action group.
        not_text (bool): Flag to indicate whether to return non-text tokens.

    Returns:
        dict: A dictionary where keys are tokens and values are corresponding (group index, number) tuples.
    """
    tokens = {}
    # Iterate over each action group
    for i, base in enumerate(bases):
        for j in range(base):
            token = map_control_token(j, i, tokenizer_type, not_text=not_text)
            tokens[token] = (i, j)
    return tokens


def map_control_token(num: int, place: int, tokenizer_type: str, not_text: bool = False) -> str:
    """
    Map a number at a specific position to a control token.

    Args:
        num (int): The current number (index within the action group).
        place (int): The action group index.
        tokenizer_type (str): Tokenizer type;.
        not_text (bool): Determines whether to return text or a numerical identifier.

    Returns:
        str: The corresponding control token string.

    Raises:
        ValueError: If the specified tokenizer type is not supported.
    """
    if tokenizer_type == "qwen2_vl":
        # Define the list of special tokens, organized by action groups and indices
        special_tokens = [
            # Group 1: hotbar
            [["<|reserved_special_token_180|>", 151837],
             ["<|reserved_special_token_181|>", 151838],
             ["<|reserved_special_token_182|>", 151839],
             ["<|reserved_special_token_183|>", 151840],
             ["<|reserved_special_token_184|>", 151841],
             ["<|reserved_special_token_185|>", 151842],
             ["<|reserved_special_token_186|>", 151843],
             ["<|reserved_special_token_187|>", 151844],
             ["<|reserved_special_token_188|>", 151845],
             ["<|reserved_special_token_189|>", 151846]],
            # Group 2: 3 tokens "forward", "back”
            [["<|reserved_special_token_190|>", 151847],
             ["<|reserved_special_token_191|>", 151848],
             ["<|reserved_special_token_192|>", 151849]],
            # Group 3: 3 tokens "left", "right"
            [["<|reserved_special_token_193|>", 151850],
             ["<|reserved_special_token_194|>", 151851],
             ["<|reserved_special_token_195|>", 151852]],
            # Group 4: 3 tokens, "sprint" "sneak"
            [["<|reserved_special_token_196|>", 151853],
             ["<|reserved_special_token_197|>", 151854],
             ["<|reserved_special_token_198|>", 151855]],
            # Group 5: 2 tokens, representing "use"
            [["<|reserved_special_token_199|>", 151856],
             ["<|reserved_special_token_200|>", 151857]],
            # Group 6: 2 tokens, representing "drop"
            [["<|reserved_special_token_201|>", 151858],
             ["<|reserved_special_token_202|>", 151859]],
            # Group 7: 2 tokens, representing "attack"
            [["<|reserved_special_token_203|>", 151860],
             ["<|reserved_special_token_204|>", 151861]],
            # Group 8: 2 tokens, representing "jump"
            [["<|reserved_special_token_205|>", 151862],
             ["<|reserved_special_token_206|>", 151863]],
            # Group 9: 2 tokens, representing "camera"
            [["<|reserved_special_token_207|>", 151864],
             ["<|reserved_special_token_208|>", 151865]],
            # Group 10: 2 tokens, representing "inventory"
            [["<|reserved_special_token_176|>", 151833],
             ["<|reserved_special_token_177|>", 151834]],
            # Group 11: camera
            [["<|reserved_special_token_209|>", 151866],
             ["<|reserved_special_token_210|>", 151867],
             ["<|reserved_special_token_211|>", 151868],
             ["<|reserved_special_token_212|>", 151869],
             ["<|reserved_special_token_213|>", 151870],
             ["<|reserved_special_token_214|>", 151871],
             ["<|reserved_special_token_215|>", 151872],
             ["<|reserved_special_token_216|>", 151873],
             ["<|reserved_special_token_217|>", 151874],
             ["<|reserved_special_token_218|>", 151875],
             ["<|reserved_special_token_219|>", 151876],
             ["<|reserved_special_token_220|>", 151877],
             ["<|reserved_special_token_221|>", 151878],
             ["<|reserved_special_token_222|>", 151879],
             ["<|reserved_special_token_223|>", 151880],
             ["<|reserved_special_token_224|>", 151881],
             ["<|reserved_special_token_225|>", 151882],
             ["<|reserved_special_token_226|>", 151883],
             ["<|reserved_special_token_227|>", 151884],
             ["<|reserved_special_token_228|>", 151885],
             ["<|reserved_special_token_229|>", 151886]],
            # Group 12: camera
            [["<|reserved_special_token_230|>", 151887],
             ["<|reserved_special_token_231|>", 151888],
             ["<|reserved_special_token_232|>", 151889],
             ["<|reserved_special_token_233|>", 151890],
             ["<|reserved_special_token_234|>", 151891],
             ["<|reserved_special_token_235|>", 151892],
             ["<|reserved_special_token_236|>", 151893],
             ["<|reserved_special_token_237|>", 151894],
             ["<|reserved_special_token_238|>", 151895],
             ["<|reserved_special_token_239|>", 151896],
             ["<|reserved_special_token_240|>", 151897],
             ["<|reserved_special_token_241|>", 151898],
             ["<|reserved_special_token_242|>", 151899],
             ["<|reserved_special_token_243|>", 151900],
             ["<|reserved_special_token_244|>", 151901],
             ["<|reserved_special_token_245|>", 151902],
             ["<|reserved_special_token_246|>", 151903],
             ["<|reserved_special_token_247|>", 151904],
             ["<|reserved_special_token_248|>", 151905],
             ["<|reserved_special_token_249|>", 151906],
             ["<|reserved_special_token_250|>", 151907]],
        ]
    else:
        raise ValueError(f"The tokenizer type {tokenizer_type} is not supported in control tokens.")
    
    try:
        # Return either the text token or numeric identifier based on not_text flag (index 0 or 1)
        token = special_tokens[place][num][not_text]
    except Exception as e:
        print("place:", place, "num:", num, "not_text:", not_text, e)
    return token


def remap_control_token(token: str, use_num: bool = True, tokenizer_type: str = "qwen2_vl") -> tuple:
    """
    Map a control token back to its corresponding action information.

    Args:
        token (str): The control token.
        use_num (bool): Whether to use numeric mapping.
        tokenizer_type (str): Tokenizer type; currently supports only "qwen2_vl".

    Returns:
        tuple: (action group index, number) tuple. Returns (-1, -1) if token not found.
    """
    re_tokens = {}
    if tokenizer_type == "qwen2_vl":
        # Define the mapping dictionary from token to action (when use_num is True)
        if use_num:
            re_tokens = {
                151837: [0, 0], 151838: [0, 1], 151839: [0, 2], 151840: [0, 3], 151841: [0, 4],
                151842: [0, 5], 151843: [0, 6], 151844: [0, 7], 151845: [0, 8], 151846: [0, 9],
                151847: [1, 0], 151848: [1, 1], 151849: [1, 2],
                151850: [2, 0], 151851: [2, 1], 151852: [2, 2],
                151853: [3, 0], 151854: [3, 1], 151855: [3, 2],
                151856: [4, 0], 151857: [4, 1],
                151858: [5, 0], 151859: [5, 1],
                151860: [6, 0], 151861: [6, 1],
                151862: [7, 0], 151863: [7, 1],
                151864: [8, 0], 151865: [8, 1],
                151833: (9, 0), 151834: (9, 1),
                151866: [10, 0], 151867: [10, 1], 151868: [10, 2], 151869: [10, 3], 151870: [10, 4],
                151871: [10, 5], 151872: [10, 6], 151873: [10, 7], 151874: [10, 8], 151875: [10, 9],
                151876: [10, 10], 151877: [10, 11], 151878: [10, 12], 151879: [10, 13], 151880: [10, 14],
                151881: [10, 15], 151882: [10, 16], 151883: [10, 17], 151884: [10, 18], 151885: [10, 19],
                151886: [10, 20],
                151887: [11, 0], 151888: [11, 1], 151889: [11, 2], 151890: [11, 3],
                151891: [11, 4], 151892: [11, 5], 151893: [11, 6], 151894: [11, 7], 151895: [11, 8],
                151896: [11, 9], 151897: [11, 10], 151898: [11, 11], 151899: [11, 12], 151900: [11, 13],
                151901: [11, 14], 151902: [11, 15], 151903: [11, 16], 151904: [11, 17], 151905: [11, 18],
                151906: [11, 19], 151907: [11, 20],
            }
        else:
            raise ValueError("can't use text as tokens")
    else:
        raise ValueError(f"The tokenizer type {tokenizer_type} is not supported in control tokens.")
    # Return (-1, -1) if token is not found
    return re_tokens.get(token, (-1, -1))


def tag_token(place: int, tokenizer_type: str, return_type: int = 0):
    """
    Return the start or end tag token based on the position.

    Args:
        place (int): 0 for the start tag, 1 for the end tag.
        tokenizer_type (str): Tokenizer type;.
        return_type (int): Specifies which part of the token to return: 0 for token text, 1 for numeric identifier.

    Returns:
        tuple: (token text, token numeric identifier)
    """
    assert place in {0, 1}
    if tokenizer_type == "qwen2_vl":
        special_tokens = [
            ('<|reserved_special_token_178|>', 151835),
            ('<|reserved_special_token_179|>', 151836),
        ]
    else:
        raise ValueError(f"The tokenizer type {tokenizer_type} is not supported in control tokens.")
    return special_tokens[place][return_type]


class ActionTokenizer(ABC):
    """
    Base class for action tokenizers, used to encode and decode actions to and from tokens.
    """
    # Define common movement and operation actions
    movements = ('forward', 'back', 'left', 'right', 'sprint', 'sneak')
    operations = ('use', 'drop', 'attack', 'jump')

    def __init__(self,
                 tokenizer_type="qwen2_vl",
                 camera_quantization_scheme="mu_law",
                 camera_mu=20,
                 camera_binsize=1,
                 camera_maxval=10):
        self.tokenizer_type = tokenizer_type

        # Retrieve the start and end tag tokens and their IDs
        self.act_beg_id = tag_token(0, self.tokenizer_type, return_type=1)
        self.act_end_id = tag_token(1, self.tokenizer_type, return_type=1)
        self.act_beg_token = tag_token(0, self.tokenizer_type, return_type=0)
        self.act_end_token = tag_token(1, self.tokenizer_type, return_type=0)

        # Initialize camera configuration
        camera_config = CameraConfig(
            camera_maxval=camera_maxval,
            camera_binsize=camera_binsize,
            camera_quantization_scheme=camera_quantization_scheme,
            camera_mu=camera_mu,
        )
        self.n_camera_bins = camera_config.n_camera_bins

        # Define the null action with default values (False for buttons, (0.0, 0.0) for camera)
        self.null_action = {
            'forward': False, 'back': False, 'left': False, 'right': False,
            'sprint': False, 'sneak': False,
            'hotbar.1': False, 'hotbar.2': False, 'hotbar.3': False, 'hotbar.4': False,
            'hotbar.5': False, 'hotbar.6': False, 'hotbar.7': False, 'hotbar.8': False, 'hotbar.9': False,
            'use': False, 'drop': False, 'attack': False, 'jump': False,
            'inventory': False,
            'camera': (0.0, 0.0)
        }

        # Initialize action transformer and action mapper
        self.action_transformer = ActionTransformer(**camera_config.action_transformer_kwargs)
        self.action_mapper = CameraHierarchicalMapping(n_camera_bins=camera_config.n_camera_bins)

    @abstractmethod
    def encode(self, actions: Dict) -> Union[torch.Tensor, list, str]:
        """
        Abstract method: Encode actions into tokens.

        Args:
            actions (Dict): Dictionary of actions.

        Returns:
            Union[torch.Tensor, list, str]: Encoded token representation.
        """
        pass

    @abstractmethod
    def decode(self, tokens: Union[torch.Tensor, list]) -> List[OrderedDict]:
        """
        Abstract method: Decode tokens into actions.

        Args:
            tokens (Union[torch.Tensor, list]): Sequence of tokens (string type is not allowed).

        Returns:
            List[OrderedDict]: A list of decoded actions as OrderedDict objects.
        """
        pass


class OneActionTokenizer(ActionTokenizer):
    """
    Single action tokenizer that implements the specific encoding and decoding logic.

    BUTTONS_GROUPS:
        Names of different action groups.
    """
    BUTTONS_GROUPS = [
        "hotbar", "fore or back", "left or right", "sprint or sneak", "use",
        "drop", "attack", "jump", "camera"
    ]

    def __init__(self,
                 tokenizer_type="llama-2",
                 bases: list = [10, 3, 3, 3, 2, 2, 2, 2, 2, 2, 21, 21],
                 camera_quantization_scheme="mu_law",
                 camera_mu=20,
                 camera_binsize=1):
        # Call the parent constructor to initialize common configurations
        super().__init__(tokenizer_type=tokenizer_type,
                         camera_quantization_scheme=camera_quantization_scheme,
                         camera_mu=camera_mu,
                         camera_binsize=camera_binsize)
        # Log related information using rich console
        console.Console().log(f"tokenizer_type: {tokenizer_type}")
        console.Console().log(f"bases: {bases}, camera_mu: {camera_mu}, n_camera_bins: {self.n_camera_bins}, camera_binsize: {camera_binsize}")
        self.bases = bases
        # NULL_ACTION is the default null action; its encoding uses the middle values of the last two elements of bases
        self.NULL_ACTION = [0, (bases[-2] // 2) * bases[-2] + (bases[-1] // 2)]
    
    def decode(self,tokens:Union[torch.Tensor,List]):
        """decode the tokens to action
        """
        group_actions = self.token_2_group_action(tokens,)
        
        actions = [self.group_action_2_decimal_action(group_action) for group_action in group_actions ]
        action_dicts = []
        for action in  actions:
            action_dict = {
                "buttons":np.array([action[0]]),
                "camera":np.array([action[1]]),  #返回一个工作
            }
            action_dict = OrderedDict({key: value[0] for key, value in action_dict.items()})
            action_dicts.append(action_dict)
        return action_dicts

    def encode(self, trajectory: dict) -> list[tuple[int]]:
        """
        Encode the action trajectory into tokens.

        Args:
            trajectory (dict): Dictionary containing actions, observations, frame IDs, and UUIDs.

        Returns:
            list: A list of encoded trajectories, each containing control token, observations, UUID, and frame information.
        """
        minerl_actions = trajectory['actions']
        traj_len = len(minerl_actions['attack'])
        # Retrieve additional trajectory information (observations, frame IDs, UUIDs)
        observations = trajectory.get('observations', [""] * traj_len)
        frame_ids = trajectory.get('frame_ids', range(0, traj_len))
        uuids = trajectory.get('uuids', [""] * traj_len)

        # Convert action values for buttons and camera into numpy arrays
        minerl_action_transformed = {key: np.array(val)
                                     for key, val in minerl_actions.items()
                                     if key in Buttons.ALL or key == "camera"}
        # Convert environment actions to policy-friendly action format
        minerl_action = self.action_transformer.env2policy(minerl_action_transformed)
        # Convert to factorized action representation using the mapper
        actions = self.action_mapper.from_factored(minerl_action)

        action_list = []
        # Store each frame's action as a tuple (buttons, camera)
        for idx in range(traj_len):
            action_list.append((actions["buttons"][idx][0], actions["camera"][idx][0]))

        encoded_trajectory = []
        # Generate control tokens for each action and combine with additional information into a dictionary
        for idx, action in enumerate(action_list):
            control_token = self.encode_action(action)
            encoded_trajectory.append({
                "action_token": control_token,
                "observations": [observations[idx]],
                'uuid': uuids[idx],
                'frames': (frame_ids[idx], 1, frame_ids[idx]),
            })
        return encoded_trajectory

    def encode_action(self, action: tuple) -> str:
        """
        Encode a single action into a control token string.

        Args:
            action (tuple): A tuple (buttons, camera).

        Returns:
            str: The encoded control token string.
        """
        # Ensure the action has two parts
        assert len(action) == 2
        # Convert decimal action to group action representation
        group_action = self.decimal_action_2_group_action(action)
        # Convert group action representation to token string
        tokens = self.group_action_2_token(group_action)
        return tokens

    def group_action_2_token(self, group_action):
        """
        Convert a group action representation into a control token string.

        Args:
            group_action: A list of numbers representing each part of the action.

        Returns:
            str: The concatenated control token string (with start and end tags).
        """
        # Map each group number to its corresponding control token
        zero_include_token_list = [map_control_token(num, i, self.tokenizer_type)
                                     for i, num in enumerate(group_action)]
        # Concatenate tokens for non-zero actions (excluding the last 4 tokens; camera tokens are handled separately)
        control_token = ''.join((s for x, s in zip(group_action[:-4], zero_include_token_list[:-4]) if x != 0))
        # Append camera-related tokens (ensure camera action information is preserved)
        control_token = control_token + "".join((s for s in zero_include_token_list[-2:]))
        # Add start and end tag tokens around the control token
        tag_control_token = self.act_beg_token + control_token + self.act_end_token
        return tag_control_token

    def token_2_group_action(self, tokens: Union[torch.Tensor, list]):
        """
        Convert a token sequence into a group action representation.

        Args:
            tokens (Union[torch.Tensor, list]): Sequence of tokens representing actions.

        Returns:
            list: A list of group action representations (each as a list of numbers).
        """
        actions = []
        # Initialize a default group action with zeros; for camera parts, use the midpoint values
        action_base = [0] * len(self.bases)
        camera_null = [self.bases[-1] // 2, self.bases[-2] // 2]
        action_base[-2:] = camera_null

        # Convert torch.Tensor tokens to list if necessary
        if isinstance(tokens, torch.Tensor):
            if tokens.ndim == 2:
                tokens = tokens.squeeze()
            tokens = tokens.tolist()
        elif not isinstance(tokens, list):
            raise ValueError("wrong type!")

        start_idx = 0
        # Split the token sequence based on start and end tag tokens; each segment represents one action
        while start_idx < len(tokens):
            try:
                first_index_n1 = tokens.index(self.act_beg_id, start_idx)
                first_index_n2 = tokens.index(self.act_end_id, first_index_n1 + 1)
            except ValueError:
                break

            # Extract control tokens between the start and end tags
            control_tokens = tokens[first_index_n1 + 1:first_index_n2]
            action = copy.copy(action_base)
            # Map each control token back to its corresponding group number and update the action
            for token in control_tokens:
                place, num = remap_control_token(token, use_num=True, tokenizer_type=self.tokenizer_type)
                if place != -1:
                    action[place] = num

            # If camera part is not equal to the default, set the inventory flag (set the fourth-last element to 1)
            if action[-2:] != camera_null:
                action[-4] = 1

            actions.append(copy.copy(action))
            start_idx = first_index_n2 + 1

        # If no actions are parsed, return the default null action
        if len(actions) == 0:
            actions.append(action_base)

        return actions

    def decimal_action_2_group_action(self, inputs: tuple):
        """
        Convert a decimal action representation into a group action representation with varying bases.

        Args:
            inputs (tuple): A tuple of two decimal integers representing button and camera actions.

        Returns:
            tuple: Each element represents the value for one action group.

        Description:
            - For button actions, perform successive modulo and integer division operations according to the bases.
            - If the button part equals 8640, mark it as inventory mode and set it to 0.
            - For camera actions, process the last two parts separately.
        """
        decimals = list(inputs)
        result = [0] * len(self.bases)
        inventory_flag = False

        # Check if the button part is 8640; if so, enable the inventory flag and set it to 0
        if decimals[0] == 8640:
            inventory_flag = True
            decimals[0] = 0
        else:
            # Convert the button part from lower to higher digits
            for i in range(len(self.bases) - 4, -1, -1):
                result[i] = decimals[0] % self.bases[i]
                decimals[0] //= self.bases[i]

        # Process the camera part: first the last digit, then the second last
        result[-1] = decimals[1] % self.bases[-1]
        decimals[1] //= self.bases[-1]
        result[-2] = decimals[1] % self.bases[-2]
        decimals[1] //= self.bases[-2]

        # If inventory flag is True, set the third-last element to 1
        if inventory_flag:
            result[-3] = 1
        if decimals != [0, 0]:
            print(decimals)
            raise ValueError("The decimal number is too large for the custom base system.")
        return tuple(result)

    def group_action_2_decimal_action(self, inputs):
        """
        Convert a group action representation with varying bases into a decimal action representation.

        Args:
            inputs: A list of numbers, with the length matching the bases.

        Returns:
            tuple: The converted decimal action representation, including button and camera parts.

        Raises:
            ValueError: If the input length does not match the expected number of digits or exceeds base limits.
        """
        if len(inputs) != len(self.bases):
            raise ValueError("The input number does not match the expected number of digits.")
        decimal_results = [0, 0]
        mid = len(inputs) - 3  # Boundary between button and camera parts

        # Calculate the decimal value for the button part
        for i, digit in enumerate(inputs):
            if digit >= self.bases[i]:
                raise ValueError(f"Digit at position {i} exceeds the base limit of {self.bases[i]-1}.")
            if i < mid:
                decimal_results[0] = decimal_results[0] * self.bases[i] + digit
            elif i == mid and digit:
                decimal_results[0] = 8640  # Special inventory flag
            else:
                decimal_results[1] = decimal_results[1] * self.bases[i] + digit
        return tuple(decimal_results)

    def null_token(self) -> str:
        """
        Get the token corresponding to the null action.

        Returns:
            str: The control token string for the null action.
        """
        return self.encode_action(self.NULL_ACTION)
    