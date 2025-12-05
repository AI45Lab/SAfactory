#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
import inspect
import os.path as osp
import random
import time
from typing import Any, Dict, List

import hydra
import magnum as mn
import numpy as np
from habitat.core.registry import registry
from habitat.datasets.rearrange.rearrange_dataset import RearrangeEpisode
from habitat.tasks.rearrange.multi_task.pddl_domain import PddlDomain
from habitat.tasks.rearrange.multi_task.pddl_logical_expr import LogicalExpr
from habitat.tasks.rearrange.multi_task.pddl_predicate import Predicate
from habitat.tasks.rearrange.multi_task.rearrange_pddl import (
    ExprType, PddlEntity, SimulatorObjectType)
from habitat.tasks.rearrange.rearrange_task import RearrangeTask
from habitat.tasks.rearrange.utils import add_perf_timing_func
from omegaconf import DictConfig, ListConfig
from PIL import Image
from transformers import AutoTokenizer

import embodiedbench.envs.eb_habitat.config
from embodiedbench.envs.eb_habitat.dataset.episodes import LangRearrangeEpisode
from embodiedbench.envs.eb_habitat.dataset.utils import get_category_info
from embodiedbench.envs.eb_habitat.actions import KinematicArmEEAction
from embodiedbench.envs.eb_habitat.utils import PLACABLE_RECEP_TYPE, get_pddl 


@registry.register_task(name="RearrangePredicateTask-v0")
class RearrangePredicateTask(RearrangeTask):
    def __init__(self, *args, sim, config, dataset, **kwargs):
        print(f"Num episodes {len(dataset.episodes)}")

        self._all_cls, obj_cats, self._name_to_cls = get_category_info(
            config.skip_load_receps
        )

        self.pddl = get_pddl(config, self._all_cls, obj_cats)
        self._fix_agent_pos = config.fix_agent_pos

        super().__init__(
            *args,
            sim=sim,
            config=config,
            dataset=dataset,
            should_place_articulated_agent=not self._fix_agent_pos,
            **kwargs,
        )
        # self._tokenizer = get_parser(config.tokenizer_name)

        self._start_template = self._config.start_template
        self._goal_template = self._config.goal_template
        self._sample_entities = self._config.sample_entities
        self._sample_entities_use_constant_sampling = (
            self._config.sample_entities_use_constant_sampling
        )
        self._force_scene_per_worker = self._config.force_scene_per_worker
        self._goal_expr = None
        self._is_first_reset = True
        self._is_freeform = False

    # @property
    # def tokenizer(self):
    #     return self._tokenizer

    @add_perf_timing_func()
    def _load_start_goal(self, episode):
        """
        Setup the start and goal PDDL conditions. Will change the simulator
        state to set the start state.
        """
        # print(self.pddl.all_entities)
        # print(self._sim)
        # print(self._dataset)
        # print(episode)
        pddl_entities = self.pddl.all_entities
        self.pddl.bind_to_instance(self._sim, self._dataset, self, episode)

        self._setup_pddl_entities(episode)

        if self._is_first_reset or not self._force_scene_per_worker:
            self.pddl.bind_actions()
            self._is_first_reset = False

        self._sim.internal_step(-1)
        self._load_sampled_names()

        self._goal_expr = self._load_goal_preds(episode)
        if self._goal_expr is not None:
            t_start = time.time()
            self._goal_expr, _ = self.pddl.expand_quantifiers(self._goal_expr)
            self._sim.add_perf_timing("goal_expand_quantifiers", t_start)
        self._load_start_preds(episode)

    def _load_sampled_names(self):
        t_start = time.time()
        self.new_entities: Dict[str, PddlEntity] = {}
        for entity_name, entity_conds in self._sample_entities.items():
            match_type = self.pddl.expr_types[entity_conds["type"]]
            matches = list(self.pddl.find_entities(match_type))

            if entity_conds.get("ignore_articulated_receptacles", False):
                matches = [
                    m
                    for m in matches
                    if m.name
                    not in [
                        "receptacle_aabb_middle_topfrl_apartment_refrigerator",
                        "receptacle_aabb_drawer_left_top_frl_apartment_kitchen_counter",
                        "receptacle_aabb_drawer_right_top_frl_apartment_kitchen_counter",
                    ]
                ]

            pred_conds = entity_conds.get("pred_conds", [])
            matches = _filter_matching_pred_conds(
                matches, pred_conds, self.pddl, self.new_entities, entity_name
            )

            if len(matches) == 0:
                obj_ref, obj_name = list(self.new_entities.items())[0]
                on_top_pred = f"on_top({obj_ref},{entity_name})"
                cmp_pred_cond = pred_conds[0].replace(" ", "")
                if len(pred_conds) == 1 and cmp_pred_cond == on_top_pred:
                    use_entity_name: str = self._sim.ep_info.name_to_receptacle[
                        obj_name.name
                    ]
                    entity = self.pddl.all_entities[use_entity_name]
                else:
                    raise ValueError(
                        f"Could not find match for {entity_name}: {entity_conds}"
                    )
            elif self._sample_entities_use_constant_sampling:
                entity = matches[0]
            else:
                entity = random.choice(matches)

            self.new_entities[entity_name] = entity
        self._sampled_names = list(self.new_entities.keys())
        self._sim.add_perf_timing("find_entities", t_start)

    @property
    def pddl_problem(self):
        return self.pddl

    @property
    def all_cls(self):
        return self._all_cls

    def _get_cls_for_obj_name(self, obj_name, episode):
        return self._name_to_cls[obj_name]

    @add_perf_timing_func()
    def _setup_pddl_entities(self, episode):
        # Register the specific objects in this scene as simulator entities.
        for obj_name in self.pddl.sim_info.obj_ids:
            asset_name = _strip_instance_id(obj_name)
            if asset_name not in self._name_to_cls:
                # This is a goal or target object indicator from a Hab2 dataset.
                # We don't use these geo-goals, so we can ignore.
                continue
            cls_name = self._name_to_cls[asset_name]

            entity_type = self.pddl.expr_types[cls_name]
            self.pddl.register_episode_entity(PddlEntity(obj_name, entity_type))

    @add_perf_timing_func()
    def _load_goal_preds(self, episode):
        if self._goal_template is None:
            # Load from the episode.
            return self.pddl.parse_only_logical_expr(
                episode.goal_preds, self.pddl.all_entities
            )
        else:
            # Load from the config.
            goal_d = dict(self._goal_template)
            goal_d = _recur_dict_replace(goal_d, self.new_entities)
            return self.pddl.parse_only_logical_expr(goal_d, self.pddl.all_entities)

    @add_perf_timing_func()
    def _load_start_preds(self, episode):
        if self._start_template is None:
            # Load form the episode data.
            for pred in episode.start_preds:
                pred = self.pddl.parse_predicate(pred, self.pddl.all_entities)
                pred.set_state(self.pddl.sim_info)
        else:
            # Load from the config.
            start_preds = self._start_template[:]
            for pred in start_preds:
                for k, entity in self.new_entities.items():
                    pred = pred.replace(k, entity.name)
                pred = self.pddl.parse_predicate(pred, self.pddl.all_entities)
                pred.set_state(self.pddl.sim_info)

    @property
    def goal_expr(self) -> LogicalExpr:
        return self._goal_expr

    def set_is_freeform(self, is_freeform):
        self._is_freeform = is_freeform

    @property
    def subgoals(self):
        if self._is_freeform:
            return []
        else:
            return self._subgoals

    def is_goal_satisfied(self):
        if self._is_freeform:
            return False
        if self._goal_expr is None:
            return False
        ret = self.pddl.is_expr_true(self._goal_expr)
        return ret

    @add_perf_timing_func()
    def _get_subgoals(self, episode) -> List[List[Predicate]]:
        if episode.subgoals is None:
            return []
        # start_preds = self.pddl.get_true_predicates()
        ret_subgoals = []
        for subgoal in episode.subgoals:
            subgoal_preds = []
            for subgoal_predicate in subgoal:
                pred = self.pddl.parse_predicate(
                    subgoal_predicate, self.pddl.all_entities
                )
                subgoal_preds.append(pred)
            if len(subgoal_preds) != 0:
                ret_subgoals.append(subgoal_preds)
        return ret_subgoals

    @add_perf_timing_func()
    def step(self, *args, action, **kwargs):
        self.pddl.sim_info.reset_pred_truth_cache()
        fix_top_down_cam_pos(self._sim)
        self.num_steps += 1
        if "action_args" not in action:
            # This won't be added to discrete action spaces, but RearrangeTask
            # expects it.
            action["action_args"] = {"sel": action["action"]}
            action["action"] = 0

        return super().step(*args, action=action, **kwargs)

    @add_perf_timing_func()
    def reset(self, episode):
        self.num_steps = 0

        # Generate a random episode ID that can be used for debugging purpose. This should be different than the dataset episode ID, because multiple workers could be operating on the same episode.
        self.episode_rollout_id = random.randint(0, 9999999)

        super().reset(episode, fetch_observations=False)
        if type(episode) == RearrangeEpisode:
            # Hab2 style episode.
            episode = convert_rearrange_ep_to_lang_rearrange(episode)
        self.lang_goal = episode.instruction
        self.instruct_id = episode.instruct_id
        self.sampler_info = episode.sampler_info
        self._load_start_goal(episode)

        if self._fix_agent_pos:
            # Fix the agent position to 0,0 and the rotation to 0.
            agent = self._sim.get_agent_data(0).articulated_agent
            agent.base_pos = self._sim.pathfinder.snap_point(
                mn.Vector3(0.0, agent.base_pos[1], 0.0)
            )
            agent.base_rot = 0.0

        self._subgoals = list(self._get_subgoals(episode))

        self._sim.draw_bb_objs.clear()
        fix_top_down_cam_pos(self._sim)

        self._sim.maybe_update_articulated_agent()
        return self._get_observations(episode)

    def get_sampled(self) -> List[PddlEntity]:
        return [self.new_entities[k] for k in self._sampled_names]


def _strip_instance_id(instance_id: str) -> str:
    # Strip off the unique instance ID of the object and only return the asset
    # name.
    return "_".join(instance_id.split("_")[:-1])


def _recur_dict_replace(d: Any, replaces: Dict[str, PddlEntity]) -> Any:
    """
    Replace all string entries in `d` with the replace name to PDDL entity
    mapping in replaces.
    """
    if isinstance(d, ListConfig):
        d = list(d)
    if isinstance(d, DictConfig):
        d = dict(d)

    if isinstance(d, str):
        for name, entity in replaces.items():
            d = d.replace(f"{name}.type", entity.expr_type.name)
            d = d.replace(name, entity.name)
    elif isinstance(d, list):
        for i, v in enumerate(d):
            d[i] = _recur_dict_replace(v, replaces)
    elif isinstance(d, dict):
        for k, v in d.items():
            d[k] = _recur_dict_replace(d[k], replaces)
    return d


def _filter_matching_pred_conds(
    matches: List[PddlEntity],
    pred_conds: List[str],
    pddl: PddlDomain,
    other_entities: Dict[str, PddlEntity],
    entity_name: str,
) -> List[PddlEntity]:
    """
    Filters elements from matches based on a set of PDDL predicate strings.
    """

    ret = []
    for match in matches:
        should_include = True
        for pred_cond in pred_conds:
            subbed_pred_cond = pred_cond.replace(entity_name, match.name)
            for other_entity_name, other_entity in other_entities.items():
                subbed_pred_cond = subbed_pred_cond.replace(
                    other_entity_name, other_entity.name
                )
            pred = pddl.parse_predicate(subbed_pred_cond, pddl.all_entities)

            if not pred.is_true(pddl.sim_info):
                should_include = False
                break
        if not should_include:
            continue
        ret.append(match)

    return ret


def fix_top_down_cam_pos(sim):
    if "top_down_rgb" not in sim._sensors:
        return
    middle_of_scene = np.array(sim.pathfinder.get_bounds()).mean(0)
    look_at_mat = mn.Matrix4.look_at(
        mn.Vector3(middle_of_scene[0], 6.5, middle_of_scene[0]),
        middle_of_scene,
        mn.Vector3(0.0, 1.0, 0.0),
    )
    rot_mat = mn.Matrix4.from_(
        mn.Matrix4.rotation(mn.Rad(np.pi / 2), mn.Vector3(0.0, 1.0, 0.0)).rotation(),
        # mn.Vector3(2.5, 0.0, 1.0),
        mn.Vector3(0.0, 0.0, 4.0),
    )
    sim._sensors["top_down_rgb"]._sensor_object.node.transformation = (
        rot_mat @ look_at_mat
    )


def convert_rearrange_ep_to_lang_rearrange(
    ep: RearrangeEpisode,
) -> LangRearrangeEpisode:
    return LangRearrangeEpisode(
        instruction="",
        sampled_entities={},
        start_preds=[],
        goal_preds={},
        instruct_id="",
        sampler_info={},
        subgoals=[],
        **ep.__getstate__(),
    )
