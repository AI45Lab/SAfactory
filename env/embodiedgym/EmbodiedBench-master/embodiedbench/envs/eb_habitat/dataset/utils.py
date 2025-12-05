#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
import inspect
import os
import os.path as osp
from typing import Dict, List, Tuple

import yaml

# import llarp.dataset

ALL_CATS_NAME = "all_cats"
INSTRUCTION_FILE = "instructions.yaml"

# Standard buckets.
LOCAL_DATASETS_PATH = os.path.join(os.path.dirname(__file__), "data/datasets")


def get_instruct_data():
    instructs_path = osp.dirname(inspect.getfile(__file__))
    instructs_cfg = osp.join(instructs_path, "configs", INSTRUCTION_FILE)
    with open(instructs_cfg, "r") as f:
        instructs = yaml.load(f, Loader=yaml.FullLoader)
    return instructs


def get_name_mappings() -> Dict[str, str]:
    """
    Gets the friendly name mappings from the instruction file.
    """
    instructs = get_instruct_data()
    return instructs["name_mappings"]


def get_all_instruct_ids():
    instructs = get_instruct_data()
    return sorted([str(x) for x in instructs["instructions"].keys()])


def get_category_info(skip_load_receps=False):
    """
    Get the list of all categories and a mapping from object name to category.
    """
    dataset_path = osp.dirname(__file__)
    dataset_cfg = osp.join(dataset_path, "configs", "dataset.yaml")

    # Load dataset_cfg as a dict
    with open(dataset_cfg, "r") as f:
        dataset = yaml.load(f, Loader=yaml.FullLoader)
    cat_groups = dataset["category_groups"]
    all_receps_cat = dataset["receptacle_sets"][0]
    assert all_receps_cat["name"] == "all_receps"
    all_obj_cats = dataset["category_groups"][ALL_CATS_NAME]["included"]

    all_cats = []
    if not skip_load_receps:
        all_cats.extend(all_receps_cat["included_receptacle_substrings"])
    all_cats.extend(all_obj_cats)

    obj_to_cls = {}
    for oset in dataset["object_sets"]:
        if oset["name"] == "CLUTTER_OBJECTS":
            continue
        for oname in oset["included_substrings"]:
            if oname in obj_to_cls:
                raise ValueError(f"Object {oname} is in multiple sets")
            obj_to_cls[oname] = oset["name"]
    return all_cats, all_obj_cats, obj_to_cls
