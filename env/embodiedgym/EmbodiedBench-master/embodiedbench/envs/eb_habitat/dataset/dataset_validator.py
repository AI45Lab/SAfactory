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
import magnum as mn
import numpy as np
import torch
from habitat.dataset import make_dataset
from habitat.tasks.rearrange.multi_task.pddl_action import PddlAction
from habitat.tasks.rearrange.multi_task.pddl_predicate import Predicate
from habitat.tasks.rearrange.multi_task.rearrange_pddl import (
    PddlEntity, SimulatorObjectType)
from PIL import Image
from torch import multiprocessing as mp
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, T5Model

import sys
sys.path.append('../')
import config
import dataset
# import llarp.task
from create_episodes import summarize_episodes
from utils import get_instruct_data
from utils import PLACABLE_RECEP_TYPE, get_allowed_actions

RECEP_ACTIONS = [
    "open_fridge",
    "open_cab",
    "close_fridge",
    "close_cab",
]
RECEP_PREDS = [
    "opened_fridge",
    "opened_cab",
    "closed_fridge",
    "closed_cab",
]

ALLOWED_ACTIONS = ["nav", "pick", "place", *RECEP_ACTIONS]

SEARCH_DEPTH_TIMEOUT = 15
LOG_INTERVAL = 20


def stack_obs(obs: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    ks = obs[0].keys()
    return {k: np.stack([o[k] for o in obs], axis=0) for k in ks}


def pred_to_str(pred):
    args = ",".join([x.name for x in pred._arg_values])
    return f"{pred.name}({args})"


def entity_matches(entity: PddlEntity, match_entities: List[str]) -> bool:
    if entity.expr_type.name == PLACABLE_RECEP_TYPE:
        return entity.name in match_entities
    else:
        return entity.expr_type.name in match_entities


def get_pred_hash(preds: List[Predicate]):
    """
    Hashes to return an ID based only on RELEVANT predicates
    """

    pred_strs = [pred_to_str(pred) for pred in preds]
    pred_strs.sort()
    return ",".join(pred_strs)


@dataclass(frozen=True)
class SearchNode:
    pred_state: List[Predicate]
    prev_action: PddlAction
    parent: "SearchNode"
    sim_state: Dict
    depth: int
    obs: Dict[str, np.ndarray]


@dataclass
class EpisodeInfo:
    actions: np.ndarray
    subgoals: List[List[str]]
    obs: np.ndarray


def find_action(search, all_actions):
    for action in all_actions:
        if all(sub_search in str(action) for sub_search in search):
            return action
    return None


def extract_entities(expr):
    if isinstance(expr, Predicate):
        return set(expr._arg_values)
    entities = set()
    for sub_expr in expr.sub_exprs:
        entities.update(extract_entities(sub_expr))
    return entities


def printpreds(preds):
    """
    Used for to help interactive debugging.
    """

    for pred in preds:
        print(pred.compact_str)


def printacs(actions):
    """
    Used for to help interactive debugging.
    """

    for ac in actions:
        print(ac.compact_str)


def get_obs(env):
    # NECESSARY TO UPDATE THE VISUAL OBSERVATION
    env.task._sim.step(0)
    return env.task._get_observations(env.current_episode)


class DataValidator:
    def __init__(self):
        self._bad_ep_ids = []
        self._good_idxs = []
        self._bad_causes = defaultdict(int)
        self._ac_lens = deque(maxlen=100)
        self._subgoal_lens = deque(maxlen=100)
        self._bad_lang_ids = defaultdict(int)
        self._good_lang_ids = defaultdict(int)
        self._avg_times = deque(maxlen=100)

        self._instruct_sols = self._get_instruct_sols()

    def _get_instruct_sols(self):
        instructs = get_instruct_data()
        ret = {}
        for instruct_id, instruct_data in instructs["instructions"].items():
            if "solution_template" in instruct_data:
                ret[instruct_id] = instruct_data["solution_template"]
        return ret

    def _filter_relevant_actions(self, actions):
        recep_type = self._pddl.expr_types[
            SimulatorObjectType.ARTICULATED_RECEPTACLE_ENTITY.value
        ]
        relevant_obj_names = set(x.expr_type.name for x in self._relevant_entities)
        for action in actions:
            if action.name in RECEP_ACTIONS and not self._has_recep_entity:
                continue
            # If the task has nothing to do with articulated receptacles, don't navigate to them.
            if (
                action.name == "nav"
                and action.param_values[0].expr_type.is_subtype_of(recep_type)
                and not self._has_recep_entity
            ):
                continue
            if action.name.startswith("pick_"):
                pick_obj_name = action.name[len("pick_") :]
                if pick_obj_name in relevant_obj_names:
                    yield action
            elif action.name == "place":
                if (
                    self._sampler_info["place_search_all_receps"]
                    or action.param_values[0] in self._relevant_entities
                ):
                    yield action
            else:
                yield action

    def _get_preds(self):
        recep_type = self._pddl.expr_types[
            SimulatorObjectType.ARTICULATED_RECEPTACLE_ENTITY.value
        ]
        new_preds = self._pddl.get_true_predicates()
        ret_preds = []
        for pred in new_preds:
            if pred.name not in self._core_preds:
                continue

            if pred.name == "on_top":
                obj_entity = pred._arg_values[0]
                if obj_entity not in self._relevant_entities:
                    continue

            # Ignore the object in robot_at if it is not of interest
            if (
                pred.name in ["robot_at", "robot_at_obj"]
                and pred._arg_values[0].expr_type.parent.name
                == SimulatorObjectType.MOVABLE_ENTITY.value
                and pred._arg_values[0] not in self._relevant_entities
            ):
                continue

            if (
                pred.name in ["robot_at", "robot_at_obj"]
                and pred._arg_values[0].expr_type.is_subtype_of(recep_type)
                and not self._has_recep_entity
            ):
                continue

            ret_preds.append(pred)
        return ret_preds

    def _setup_core_preds(self):
        recep_type = self._pddl.expr_types[
            SimulatorObjectType.ARTICULATED_RECEPTACLE_ENTITY.value
        ]
        self._has_recep_entity = any(
            x.expr_type.is_subtype_of(recep_type) for x in self._relevant_entities
        )
        self._core_preds = [
            "holding",
            "not_holding",
            "in",
            "on_top",
            "robot_at",
            "robot_at_obj",
        ]
        if self._has_recep_entity:
            self._core_preds.extend(
                [
                    "opened_cab",
                    "closed_cab",
                    "opened_fridge",
                    "closed_fridge",
                ]
            )

    def _get_cur_actions(self, node):
        is_holding = any(x.name == "holding" for x in node.pred_state)
        ret_acs = self._all_actions

        if node.prev_action is not None and node.prev_action.name == "nav":
            # No need to navigate again because we could have just navigated directly to the target.
            ret_acs = [
                ac for ac in self._all_actions if ac.name != node.prev_action.name
            ]

        if is_holding:
            # If we are holding an object, no picking actions.
            ret_acs = [ac for ac in self._all_actions if not ac.name.startswith("pick")]
        else:
            # If we are not holding an object, not placing actions.
            ret_acs = [ac for ac in self._all_actions if ac.name != "place"]
        return ret_acs

    def _compute_subgoals(self, env, ordered_actions):
        """
        Performs BFS to find a path from the start to the predicate goal state.
        Returns the predicate goals and the number of bfs iterations required to
        find the goal.

        Returns None if the episode is bad.
        """

        self._pddl = env.task.pddl
        sim_info = self._pddl.sim_info
        sim = sim_info.sim
        robot = sim.get_agent_data(None).articulated_agent

        # Capture the starting observation. Need to capture here before the
        # robot is respawned far away.
        start_obs = get_obs(env)

        # Spawn the robot far away so no robot location predicates are activated
        # with default the random robot placement.
        robot.base_pos = mn.Vector3(-1000, -1000, -1000)
        goal_expr = env.task.goal_expr

        self._relevant_entities = extract_entities(goal_expr)
        for sampled_entity in env.task._sim.ep_info.sampled_entities.values():
            # If it is not in the entities list, then this is an object class and it should be expressed in the object goal.
            if sampled_entity in self._pddl.all_entities:
                self._relevant_entities.add(self._pddl.get_entity(sampled_entity))

        self._setup_core_preds()

        start_preds = self._get_preds()
        for start_pred in start_preds:
            if start_pred.name in RECEP_PREDS:
                continue
            for entity in extract_entities(start_pred):
                self._relevant_entities.add(entity)

        self._sampler_info = {"place_search_all_receps": False}
        self._sampler_info.update(env.task.sampler_info)

        self._all_actions = get_allowed_actions(self._pddl, ALLOWED_ACTIONS)
        self._all_actions = list(self._filter_relevant_actions(self._all_actions))

        if goal_expr.is_true_from_predicates(start_preds):
            return None, "sat_at_start"

        start_state = sim.capture_state()

        Q = deque([SearchNode(start_preds, None, None, start_state, 0, start_obs)])
        visited = set([get_pred_hash(start_preds)])

        prev_depth = 0
        not_allowed = 0
        already_in = 0
        goal_node = None
        while len(Q) != 0:
            node = Q.popleft()
            if node.depth > SEARCH_DEPTH_TIMEOUT:
                break
            use_actions = self._get_cur_actions(node)

            for action in use_actions:
                if not action.is_precond_satisfied_from_predicates(node.pred_state):
                    continue

                # Set sim state to capture the observation
                sim.set_state(node.sim_state, True)
                action.apply(sim_info)
                sim_state = sim.capture_state(True)

                new_preds = self._get_preds()

                im_obs = {k: np.copy(v) for k, v in get_obs(env).items()}
                new_node = SearchNode(
                    new_preds, action, node, sim_state, node.depth + 1, im_obs
                )
                new_pred_hash = get_pred_hash(new_preds)

                if goal_expr.is_true_from_predicates(new_preds):
                    goal_node = new_node
                    break

                if new_pred_hash not in visited:
                    visited.add(new_pred_hash)
                    Q.append(new_node)

            if goal_node is not None:
                break

        sim.set_state(start_state)

        if goal_node is None:
            return None, "no_path"

        # Extract the intermediate predicate states.
        pred_subgoals = []
        actions = []
        obs = []
        node = goal_node
        action_names = []
        while node is not None:
            if node.prev_action is not None:
                action_names.insert(0, node.prev_action.compact_str)
                ac_idx = ordered_actions.index(node.prev_action.compact_str)
                actions.insert(0, ac_idx)
            obs.insert(0, node.obs)
            subgoal_preds = [
                pred_to_str(pred) for pred in node.pred_state if pred not in start_preds
            ]
            if len(subgoal_preds) != 0:
                pred_subgoals.insert(0, subgoal_preds)
            node = node.parent

        all_obs = stack_obs(obs)
        head_rgb = all_obs["head_rgb"]
        sums = head_rgb.reshape(head_rgb.shape[0], -1).sum(1)
        if np.all(sums == sums[0]) and len(actions) > 2:
            print(f"Got static demo (from planned)")
            # If the demo consists of only the same observation, there is a problem.
            return None, "static_demo"

        return (
            EpisodeInfo(np.array(actions), pred_subgoals, all_obs),
            "good_episode",
        )

    def _print_stats(self):
        print(f"Average search time: {np.mean(self._avg_times)} seconds")
        print(
            f"Ac len {np.mean(self._ac_lens)}, Subgoal len {np.mean(self._subgoal_lens)}"
        )
        print(f"Bad/Good episodes: {len(self._bad_ep_ids)}/{len(self._good_idxs)}")
        for k, v in self._bad_causes.items():
            print(f"{k}: {v}")
        all_instruct_ks = list(
            set(list(self._bad_lang_ids.keys()) + list(self._good_lang_ids.keys()))
        )
        print("Bad/Good Instruct Ratio:")
        for k in all_instruct_ks:
            print(f"{k}: {self._bad_lang_ids[k]}/{self._good_lang_ids[k]}")

    def _compute_subgoals_from_sol(self, env, ordered_actions, sols, instruct_id):
        self._pddl = env.task.pddl
        sim_info = self._pddl.sim_info
        sim = sim_info.sim
        goal_expr = env.task.goal_expr

        self._relevant_entities = extract_entities(goal_expr)
        for sampled_entity in env.task._sim.ep_info.sampled_entities.values():
            # If it is not in the entities list, then this is an object class and it should be expressed in the object goal.
            if sampled_entity in self._pddl.all_entities:
                self._relevant_entities.add(self._pddl.get_entity(sampled_entity))
        self._setup_core_preds()

        all_actions = get_allowed_actions(self._pddl, ALLOWED_ACTIONS)
        all_action_strs = [compact_str_no_robot(x) for x in all_actions]
        # sub in the sampled entities into the solution
        use_actions = []
        for sol_entry in sols:
            for from_k, to_k in env.current_episode.sampled_entities.items():
                sol_entry = sol_entry.replace(f"`{from_k}`", to_k)
            if sol_entry not in all_action_strs:
                return None, "Couldn't find action"
            use_actions.append(all_actions[all_action_strs.index(sol_entry)])

        start_preds = self._get_preds()
        pred_state = start_preds
        pred_subgoals = []
        all_obs = [{k: np.copy(v) for k, v in get_obs(env).items()}]

        for action in use_actions:
            if not action.is_precond_satisfied_from_predicates(pred_state):
                return None, "Couldn't apply action"

            # Set sim state to capture the observation
            action.apply(sim_info)

            pred_state = self._get_preds()

            all_obs.append({k: np.copy(v) for k, v in get_obs(env).items()})

            subgoal_preds = [
                pred_to_str(pred) for pred in pred_state if pred not in start_preds
            ]
            if len(subgoal_preds) != 0:
                pred_subgoals.append(subgoal_preds)

        if not goal_expr.is_true_from_predicates(pred_state):
            return None, "Solution didn't satisfy goal"

        all_obs = stack_obs(all_obs)

        head_rgb = all_obs["head_rgb"]
        sums = head_rgb.reshape(head_rgb.shape[0], -1).sum(1)
        if np.all(sums == sums[0]) and len(use_actions) > 2:
            print(f"Got static demo", use_actions)
            # If the demo consists of only the same observation, there is a problem.
            return None, "static_demo"
        action_idxs = [ordered_actions.index(ac.compact_str) for ac in use_actions]

        return (
            EpisodeInfo(np.array(action_idxs), pred_subgoals, all_obs),
            "good_episode",
        )

    def validate_eps(self, env):
        all_hxs = []
        ep_idx_to_hxs_idx = {}

        all_actions = []
        all_obs = defaultdict(list)
        max_num_tokens = -1

        ep_id_to_idx = {ep.episode_id: i for i, ep in enumerate(env.episodes)}
        ordered_actions = [
            x.compact_str for x in get_allowed_actions(env.task.pddl, ALLOWED_ACTIONS)
        ]
        possible_preds: List[str] = [
            x.compact_str for x in env.task.pddl.get_possible_predicates()
        ]

        n_eps = env.number_of_episodes
        ret_eps = []
        for i in tqdm(range(n_eps)):
            if i % LOG_INTERVAL == 0:
                self._print_stats()

            try:
                env.reset()
            except Exception as e:
                self._bad_causes[f"Exception {e}"] += 1
                self._bad_ep_ids.append(env.current_episode.episode_id)
                self._bad_lang_ids[env.task.instruct_id] += 1
                continue

            use_ep_i = ep_id_to_idx[env.current_episode.episode_id]
            instruct = env.current_episode.instruction

            start_t = time.time()
            # Get without the semantic ID since the same solution will apply even with semantic info.
            base_instruct_id = env.current_episode.instruct_id.split("_")[0]
            if base_instruct_id in self._instruct_sols:
                ep_info, msg = self._compute_subgoals_from_sol(
                    env,
                    ordered_actions,
                    self._instruct_sols[base_instruct_id],
                    env.current_episode.instruct_id,
                )
            else:
                ep_info, msg = self._compute_subgoals(env, ordered_actions)
            search_time = time.time() - start_t

            if ep_info is None:
                self._bad_causes[msg] += 1
                self._bad_ep_ids.append(env.current_episode.episode_id)
                self._bad_lang_ids[env.task.instruct_id] += 1
                continue
            else:
                self._good_idxs.append(use_ep_i)
                self._ac_lens.append(len(ep_info.actions))
                self._subgoal_lens.append(len(ep_info.subgoals))

            self._avg_times.append(search_time)
            env.episodes[use_ep_i].subgoals = ep_info.subgoals
            self._good_lang_ids[env.task.instruct_id] += 1

        self._print_stats()

        ret_eps = [ep for ep in env.episodes if ep.episode_id not in self._bad_ep_ids]

        return ret_eps


def validate_eps(config, eps, conn):
    dataset = make_dataset(
        config.habitat.dataset.type, config=config.habitat.dataset, preset_eps=eps
    )
    data_validator = DataValidator()
    with habitat.Env(config=config, dataset=dataset) as env:
        print("Starting validation")
        conn.send(data_validator.validate_eps(env))
    conn.close()


def start(args):
    config = habitat.get_config(args.cfg, args.opts)
    dataset = make_dataset(config.habitat.dataset.type, config=config.habitat.dataset)
    split_datasets = []
    eps = dataset.episodes
    if args.limit_count is not None:
        eps = eps[: args.limit_count]

    num_distinct_instructs = len(set(ep.instruction for ep in eps))
    summarize_episodes(
        eps,
        show_examples=args.only_summarize,
        tokenizer_name="data/hf_llama_7B/",
    )
    if args.only_summarize:
        return

    split_size = len(eps) // args.n_procs
    for i in range(args.n_procs):
        split_datasets.append(eps[split_size * i : (i + 1) * split_size])

    proc_infos = []
    mp_ctx = mp.get_context("forkserver")

    for split_dataset in split_datasets:
        parent_conn, child_conn = mp_ctx.Pipe()
        if args.proc_debug:
            p = Thread(
                target=validate_eps,
                args=(config, split_dataset, child_conn),
            )
        else:
            p = mp_ctx.Process(
                target=validate_eps,
                args=(config, split_dataset, child_conn),
            )
        p.start()
        proc_infos.append((parent_conn, p))

    dataset.episodes = []
    all_actions = []
    all_obs = defaultdict(list)
    all_masks = []
    all_instructs = []

    orig_ordered_instruct = None
    for i, (conn, proc) in enumerate(proc_infos):
        try:
            ret_eps = conn.recv()
        except EOFError as e:
            print(e)
            print(f"Process {i} crashed")
            continue
        dataset.episodes.extend(ret_eps)

        proc.join()
        print(f"Process {i} finished")

    summarize_episodes(dataset.episodes)

    save_prefix = config.habitat.dataset.data_path.split(".")[0]

    new_ep_path = save_prefix + "_val.pickle"
    num_distinct_instructs = len(set(ep.instruction for ep in dataset.episodes))
    print(
        f"Saving {len(dataset.episodes)} episodes with {num_distinct_instructs} distinct instructions."
    )

    with open(new_ep_path, "wb") as f:
        pickle.dump(dataset.to_binary(), f)
    print(f"Saved validated episodes to {new_ep_path}")


def compact_str_no_robot(action: PddlAction) -> str:
    params = ",".join(x.name for x in action.param_values if "robot" not in x.name)
    return f"{action.name}({params})"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True, type=str)
    parser.add_argument("--limit-count", default=None, type=int)
    parser.add_argument("--n-procs", default=1, type=int)
    parser.add_argument("--proc-debug", action="store_true")
    parser.add_argument("--only-summarize", action="store_true")
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    start(args)
