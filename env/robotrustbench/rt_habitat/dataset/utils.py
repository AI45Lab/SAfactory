#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
import os
import os.path as osp
from typing import Dict, List, Tuple

import yaml
from env.robotrustbench.rt_habitat.runtime_support import get_rt_resource_path

# import llarp.dataset

ALL_CATS_NAME = "all_cats"
INSTRUCTION_FILE = "instructions.yaml"

# Standard buckets.
LOCAL_DATASETS_PATH = os.path.join(os.path.dirname(__file__), "data/datasets")


def get_instruct_data():
    instructs_cfg = get_rt_resource_path("dataset", "configs", INSTRUCTION_FILE)
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


def get_category_info(skip_load_receps=False, dataset_name="dataset.yaml"):
    """
    Get the list of all categories and a mapping from object name to category.
    """
    dataset_cfg = get_rt_resource_path("dataset", "configs", dataset_name)

    # Load dataset_cfg as a dict
    with open(dataset_cfg, "r") as f:
            dataset = yaml.load(f, Loader=yaml.FullLoader)
    cat_groups = dataset["category_groups"]
    all_receps_cat = None
    for recep_set in dataset["receptacle_sets"]:
        if recep_set.get("name") == "all_receps":
            all_receps_cat = recep_set
            break
    if all_receps_cat is None:
        raise KeyError("Missing receptacle set 'all_receps' in %s" % dataset_name)
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
