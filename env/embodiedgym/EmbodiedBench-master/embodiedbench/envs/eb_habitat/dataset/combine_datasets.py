#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
import argparse
import gzip
import os.path as osp
import pickle
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from threading import Thread
from typing import Dict, List, Optional, Tuple

import habitat
import hydra
import magnum as mn
import numpy as np
import torch
from habitat.datasets import make_dataset
from habitat.config.default_structured_configs import register_hydra_plugin
from habitat.tasks.rearrange.multi_task.pddl_action import PddlAction
from habitat.tasks.rearrange.multi_task.pddl_predicate import Predicate
from habitat.tasks.rearrange.multi_task.rearrange_pddl import (
    PddlEntity, SimulatorObjectType)
# from habitat_baselines.config.default_structured_configs import \
#     HabitatBaselinesConfigPlugin
from PIL import Image
from torch import multiprocessing as mp
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, T5Model

import sys
sys.path.append('../')
import config
import dataset
from create_episodes import summarize_episodes
from demo_dataset import DemoDataset, cat_demo_datasets
from episodes import LangRearrangeDatasetV0
from ..utils import PLACABLE_RECEP_TYPE, get_allowed_actions


def start(args):
    initial_paths = args.paths.split(",")
    paths = []
    for path in initial_paths:
        if "X" in path:
            for i in range(args.num_splits):
                paths.append(path.replace("X", str(i)))
        else:
            paths.append(path)

    print("Combining ")
    all_eps = []
    for path in paths:
        print(path)
        path = osp.join("data/datasets", path)
        if not osp.exists(path):
            print(f"Could not find {path}")
            continue
        config = habitat.get_config(args.cfg, [f"habitat.dataset.data_path={path}"])
        dataset = make_dataset(
            config.habitat.dataset.type, config=config.habitat.dataset
        )
        all_eps.extend(dataset.episodes)
    print("Total eps", len(all_eps))
    summarize_episodes(all_eps)

    combined_dataset = LangRearrangeDatasetV0(config, all_eps)
    with open(args.out_path, "wb") as f:
        pickle.dump(combined_dataset.to_binary(), f)
    print(f"Dumped to {args.out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True, type=str)
    parser.add_argument("--paths", required=True, type=str)
    parser.add_argument("--out-path", required=True, type=str)
    parser.add_argument("--num-splits", required=True, type=int)
    args = parser.parse_args()
    start(args)
