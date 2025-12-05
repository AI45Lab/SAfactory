#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
import gzip
import os
import os.path as osp
import pickle
import random
from collections import defaultdict
from dataclasses import dataclass, field
from threading import Thread
from typing import TYPE_CHECKING, Any, Dict, List

import numpy as np
import pandas as pd
from habitat.core.logging import logger
from habitat.datasets.rearrange.rearrange_generator import \
    RearrangeEpisodeGenerator
from habitat.datasets.rearrange.run_episode_generator import (
    RearrangeEpisodeGeneratorConfig, get_arg_parser, get_config_defaults,
    print_metadata_mediator)
from habitat.datasets.rearrange.samplers.receptacle import \
    get_all_scenedataset_receptacles
from habitat.utils.common import cull_string_list_by_substrings
from omegaconf import OmegaConf
from torch import multiprocessing as mp


from episodes import LangRearrangeDatasetV0
from generator import (LangRearrangeEpisodeGenerator,
                                     generate_all_instructions,
                                     get_flat_eps_split)
from utils import get_category_info
from ..utils import get_parser


@dataclass
class LangEpisodeGeneratorConfig(RearrangeEpisodeGeneratorConfig):
    category_groups: Dict[str, Any] = field(default_factory=dict)
    recep_cat_groups: Dict[str, Any] = field(default_factory=dict)


def generate_episodes(args, cfg, conn, iter_eps, proc_idx):
    with LangRearrangeEpisodeGenerator(
        cfg=cfg,
        instruct_path=args.instruct_path,
        iter_eps=iter_eps,
        debug_visualization=args.debug,
        limit_scene_set=args.limit_scene_set,
        proc_idx=proc_idx,
    ) as ep_gen:
        if not osp.isdir(args.db_output):
            os.makedirs(args.db_output)
        ep_gen.vdb.output_path = osp.abspath(args.db_output)
        conn.send(ep_gen.generate_episodes(args.num_episodes, args.verbose))
        conn.close()


def summarize_episodes(episodes, show_examples=False, tokenizer_name=None):
    if tokenizer_name is not None:
        tokenizer = get_parser(tokenizer_name)
        instructs = [ep.instruction for ep in episodes]
        token_lens = []
        for ep in episodes:
            # I intentionally didn't batch this to be sure I am matching the
            # way the tokenizer is used in the actual simulation.
            tokens = tokenizer(
                ep.instruction,
                return_tensors="np",
            )[
                "input_ids"
            ][0]
            token_lens.append(tokens.shape[0])
        print("max token len ", max(token_lens))
    instructs = defaultdict(int)
    instruct_ids = defaultdict(int)
    instruct_id_instructs = defaultdict(set)
    scene_ids = defaultdict(int)
    for ep in episodes:
        instructs[ep.instruction] += 1
        instruct_ids[ep.instruct_id] += 1
        scene_ids[ep.scene_id] += 1
        instruct_id_instructs[ep.instruct_id].add(ep.instruction)
    if show_examples:
        # Show 25 of the instructions.
        for k, ex_instructs in instruct_id_instructs.items():
            ex_instructs = list(ex_instructs)
            np.random.shuffle(ex_instructs)
            print(k)
            print("\n".join(instruct for instruct in ex_instructs[:25]))
            print("")
    print(f"Total instructs: {len(episodes)}")
    print(f"Num distinct instructs: {len(instructs)}")
    print(f"Num instruct IDS: {len(instruct_ids)}")
    print(f"Num scene IDS: {len(scene_ids)}")

    for k, v in instruct_ids.items():
        print(f"    {k}: {len(instruct_id_instructs[k])} distinct, {v} total")

    return instructs


def get_base_info(cfg, args, conn):
    print("Starting rearrange gen")
    tmp_ep_gen = RearrangeEpisodeGenerator(
        cfg=cfg,
        debug_visualization=args.debug,
        limit_scene_set=args.limit_scene_set,
    )
    print("Done rearrange gen")
    tmp_ep_gen.sim.close(destroy=True)
    del tmp_ep_gen.sim
    conn.send((tmp_ep_gen._obj_sets, tmp_ep_gen._receptacle_sets))
    del tmp_ep_gen
    conn.close()


def start(args):
    # Verify the dataset
    get_category_info()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    cfg = OmegaConf.create(LangEpisodeGeneratorConfig())
    if args.config is not None:
        assert osp.exists(
            args.config
        ), f"Provided config, '{args.config}', does not exist."
        override_config = OmegaConf.load(args.config)
        cfg = OmegaConf.merge(cfg, override_config)  # type: ignore[assignment]

    dataset = LangRearrangeDatasetV0()
    if args.procs_per_gpu is None:
        # Default to all processes being on the same GPU.
        procs_per_gpu = args.n_procs
    else:
        procs_per_gpu = args.procs_per_gpu

    if args.instruct_lim is not None:
        instruct_lim_id = args.instruct_lim.split(",")
    else:
        instruct_lim_id = None

    mp_ctx = mp.get_context("forkserver")

    parent_conn, child_conn = mp_ctx.Pipe()
    # We need to fetch this info in another process because it is creating a new habsim.
    proc = mp_ctx.Process(
        target=get_base_info,
        args=(cfg, args, child_conn),
    )
    proc.daemon = True
    proc.start()
    obj_sets, recep_sets = parent_conn.recv()
    proc.join()

    assert args.seed is not None
    rng = np.random.RandomState(args.seed)

    # By default, these are the Hydra list type.
    cat_groups = {k: list(v["included"]) for k, v in cfg.category_groups.items()}
    recep_sets = {
        k: list(v.included_receptacle_substrings) for k, v in recep_sets.items()
    }
    all_eps, instruct_samples = generate_all_instructions(
        args.instruct_path,
        obj_sets,
        recep_sets,
        instruct_lim_id,
        args.phrase,
        args.tag,
        args.obj_sample_split,
        cat_groups,
        args.per_instruct_count_lim,
        rng,
    )
    rng.shuffle(instruct_samples)
    ep_keys = sorted(list(all_eps.keys()))
    for k in ep_keys:
        rng.shuffle(all_eps[k])

    proc_infos = []
    to_gen_distinct_instructs = defaultdict(lambda: [set(), 0])

    for i in range(args.n_procs):
        use_cfg = cfg.copy()
        use_cfg.gpu_device_id = i // procs_per_gpu
        parent_conn, child_conn = mp_ctx.Pipe()
        iter_eps = get_flat_eps_split(
            all_eps,
            i,
            args.total_take,
            args.cur_gen_idx,
            args.n_procs,
            args.num_episodes,
            instruct_samples,
        )
        for ep in iter_eps:
            to_gen_distinct_instructs[ep.instruct_info.instruct_id][0].add(ep.instruct)
            to_gen_distinct_instructs[ep.instruct_info.instruct_id][1] += 1

        if args.proc_debug:
            p = Thread(
                target=generate_episodes,
                args=(args, use_cfg, child_conn, iter_eps, i),
            )
        else:
            p = mp_ctx.Process(
                target=generate_episodes,
                args=(args, use_cfg, child_conn, iter_eps, i),
            )
        print(f"Starting worker {i}")
        p.start()
        proc_infos.append((parent_conn, p))

    total_distinct = sum(len(x[0]) for x in to_gen_distinct_instructs.values())
    total_instructs = sum(x[1] for x in to_gen_distinct_instructs.values())
    print(f"Plan for generation: {total_distinct} distinct, {total_instructs} total.")
    for k, v in to_gen_distinct_instructs.items():
        print(f"    {k}: {len(v[0])} distinct, {v[1]} total")
    print()

    not_collected = list(range((len(proc_infos))))
    for i, (conn, proc) in enumerate(proc_infos):
        try:
            result = conn.recv()
        except EOFError as e:
            logger.warning(f"Problem in worker {i}. Could not generate any episodes.")
            continue
        proc.join()
        if result is None:
            print("Result is none skipping")
            continue
        n_eps_collected = len(result)
        not_collected.pop(not_collected.index(i))
        print(
            f"Collected {n_eps_collected} episodes from worker {i}. Waiting for {not_collected}"
        )
        dataset.episodes.extend(result)
        if n_eps_collected != args.num_episodes:
            logger.warning(
                f"Problem collecting episodes from worker {i}. Expected {args.num_episodes}, got {n_eps_collected}"
            )
        print("Done extending episodes list.")
    print("Summarizing the episodes")
    summarize_episodes(dataset.episodes)

    output_path = args.out
    if not osp.exists(osp.dirname(output_path)) and len(osp.dirname(output_path)) > 0:
        os.makedirs(osp.dirname(output_path))
    with open(output_path, "wb") as f:
        pickle.dump(dataset.to_binary(), f)

    logger.warning("==============================================================")
    logger.warning(f"RearrangeDatasetV0 saved to '{osp.abspath(output_path)}'")
    logger.warning("==============================================================")


if __name__ == "__main__":
    parser = get_arg_parser()
    parser.add_argument("--instruct-path", required=True, type=str)
    parser.add_argument("--get-instructs", action="store_true")
    parser.add_argument(
        "--instruct-lim",
        type=str,
        default=None,
        help="Comma seperated unique instruction IDs to limit sampling to",
    )
    parser.add_argument("--tag", type=str, default=None, required=True)
    parser.add_argument("--phrase", type=str, default="train")
    parser.add_argument(
        "--obj-sample-split",
        type=str,
        default="train",
        help="Either 'train' or 'eval'",
    )
    parser.add_argument("--n-procs", type=int, default=1)
    parser.add_argument("--procs-per-gpu", type=int, default=None)
    parser.add_argument("--cur-gen-idx", type=int, default=0)
    parser.add_argument("--total-take", type=int, default=1)
    parser.add_argument(
        "--per-instruct-count-lim",
        type=int,
        default=200_000,
        help="The maximum number of instruction allowed per instruction type.",
    )
    parser.add_argument("--proc-debug", action="store_true")
    parser.add_argument(
        "--instruct-dir", default="interactive_and_embodied/projects/llarp/instructs"
    )
    args, _ = parser.parse_known_args()

    start(args)
