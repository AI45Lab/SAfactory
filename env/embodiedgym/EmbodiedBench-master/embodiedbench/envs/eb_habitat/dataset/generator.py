#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
import copy
import os.path as osp
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import habitat.datasets.rearrange.samplers as samplers
import numpy as np
import yaml
from habitat.core.logging import logger
from habitat.datasets.rearrange.rearrange_generator import \
    RearrangeEpisodeGenerator
from habitat.datasets.rearrange.samplers.receptacle import (OnTopOfReceptacle,
                                                            ReceptacleSet,
                                                            ReceptacleTracker,
                                                            find_receptacles)
from tqdm import tqdm

from .episodes import LangRearrangeEpisode


@dataclass(frozen=True)
class ReceptacleSampleInfo:
    type: str
    not_equal: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class ObjectSampleInfo:
    types: Dict[str, str]
    not_recep: Optional[str] = None
    recep: Optional[str] = None
    not_equal: List[str] = field(default_factory=list)


@dataclass
class SemanticInstructInfo:
    match_objs: Dict[str, List[str]]
    lang: Dict[str, List[str]]


@dataclass
class InstructionInfo:
    """
    :propery overweight_freq: How much more frequently is this instruction
        sampled during training?
    """

    instruct_id: str
    lang: Dict[str, List[str]]
    tags: List[str]
    goal_expr: Any
    start_preds: List[Any]
    sampled_objects: Dict[str, ObjectSampleInfo]
    sampled_receps: Dict[str, ReceptacleSampleInfo]
    sampler_info: Dict[str, Any]
    overweight_freq: int
    solution_template: List[str]
    semantics: Dict[str, SemanticInstructInfo]
    subsample_entity_ratio: float


@dataclass
class EpisodeData:
    instruct_info: InstructionInfo
    # The active language paraphrasing.
    instruct: Optional[str] = ""
    recep_sampler_names: Dict[str, ReceptacleSet] = field(default_factory=dict)
    sampled_entities: Dict[str, str] = field(default_factory=dict)
    object_spawn_recep: Dict[str, str] = field(default_factory=dict)
    object_samples: Dict[str, str] = field(default_factory=dict)

    def clone(self):
        new_instruct_data = copy.deepcopy(self.instruct_info)
        return EpisodeData(
            new_instruct_data,
            self.instruct,
            dict(self.recep_sampler_names),
            dict(self.sampled_entities),
            dict(self.object_spawn_recep),
            dict(self.object_samples),
        )


class LangRearrangeEpisodeGenerator(RearrangeEpisodeGenerator):
    _valid_instructs: List[str]
    _instructs: Dict[str, InstructionInfo]

    def __init__(
        self,
        cfg,
        instruct_path,
        iter_eps,
        debug_visualization,
        limit_scene_set,
        proc_idx,
    ):
        super().__init__(cfg, debug_visualization, limit_scene_set)
        with open(instruct_path, "r") as f:
            self._instruct_data = yaml.safe_load(f)
        self._max_per_obj_count = self._instruct_data["max_per_obj_count"]

        self._cur_ep_idx = 0
        self._proc_idx = proc_idx

        self._iter_eps = iter_eps

    def generate_scene(self) -> str:
        """
        Gets the scene ID for the current episode. Is fixed by the process ID.
        """

        cur_scene_name = self._scene_sampler.scenes[
            self._proc_idx % len(self._scene_sampler.scenes)
        ]
        # print(f"Worker # {self._proc_idx} processing scene {cur_scene_name}")
        logger.info(f"Initializing scene {cur_scene_name}")
        self.initialize_sim(cur_scene_name, self.cfg.dataset_path)

        return cur_scene_name

    def generate_episodes(self, num_episodes: int = 1, verbose: bool = False):
        # Compute the total distinct objects.
        self._scene_obj_handles = defaultdict(list)
        self._scene_obj_cls_to_handle_idxs = {}
        for scene in self._scene_sampler.scenes:
            # We need to fix the random state so different threads will sample
            # the same set of objects per scene.
            scene_rnd = np.random.RandomState(hash(scene) % (2**32))

            obj_cls_to_handle_idxs = {}
            # Sample distinct object sets per scene.
            for obj_set_name, obj_set in self._obj_sets.items():
                if obj_set_name == "CLUTTER":
                    continue

                use_extra_counts = scene_rnd.randint(1, self._max_per_obj_count + 1)

                new_objs = scene_rnd.choice(obj_set, size=use_extra_counts)
                self._scene_obj_handles[scene].extend(list(new_objs))
                total_num_objs = len(self._scene_obj_handles[scene])
                obj_cls_to_handle_idxs[obj_set_name] = [
                    total_num_objs - i - 1 for i in range(len(new_objs))
                ]
            self._scene_obj_cls_to_handle_idxs[scene] = obj_cls_to_handle_idxs

        generated_episodes = []
        failed_episodes = 0
        if verbose:
            pbar = tqdm(total=num_episodes)

        num_failures_per_instruct = 0
        num_failures_per_scene = 0
        PER_INSTRUCT_LIM = 25
        PER_SCENE_LIM = 200

        while len(generated_episodes) < num_episodes:
            new_episode = self.generate_single_episode()
            if new_episode is None:
                num_failures_per_instruct += 1
                num_failures_per_scene += 1
                if num_failures_per_instruct > PER_INSTRUCT_LIM:
                    num_failures_per_instruct = 0
                    logger.warning(
                        f"Cannot generate for instruct `{self._get_cur_ep().instruct}`"
                    )
                    # Advance to the next instruction. For some reason we couldn't generate this one.
                    self._on_successful_sample()
                if num_failures_per_scene > PER_SCENE_LIM:
                    logger.warning("Cannot generate any episodes. Breaking early")
                    break
                failed_episodes += 1
                continue
            else:
                num_failures_per_instruct = 0
                num_failures_per_scene = 0

            generated_episodes.append(new_episode)
            if verbose:
                pbar.update(1)
        if verbose:
            pbar.close()

        logger.info(
            f"Generated {num_episodes} episodes in {num_episodes+failed_episodes} tries."
        )

        return generated_episodes

    def _generate_ao_states(self):
        # sample AO states for objects in the scene
        # ao_instance_handle -> [ (link_ix, state), ... ]
        ao_states: Dict[str, Dict[int, float]] = {}
        for _sampler_name, ao_state_sampler in self._ao_state_samplers.items():
            sampler_states = ao_state_sampler.sample(
                self.sim,
                [],
            )
            if sampler_states is None:
                return None
            for sampled_instance, link_states in sampler_states.items():
                if sampled_instance.handle not in ao_states:
                    ao_states[sampled_instance.handle] = {}
                for link_ix, joint_state in link_states.items():
                    ao_states[sampled_instance.handle][link_ix] = joint_state
        return ao_states

    def _generate_objects(self, ep_scene_handle, cur_ep):
        # Create this after all the new trackers have been added.
        recep_tracker = ReceptacleTracker(
            {k: v for k, v in self.cfg.max_objects_per_receptacle},
            self._receptacle_sets,
        )

        spawn_objs = copy.deepcopy(self._scene_obj_handles[ep_scene_handle])
        obj_type_to_idx = defaultdict(list)

        # Sample object placements
        self.object_to_containing_receptacle = {}
        obj_sampler = self._obj_samplers["CLUTTER"]

        # Override the object placements to be on the requested receptacles.
        orig_recep_sets = obj_sampler._allowed_recep_set_names[:]
        object_idx_to_recep = {}
        obj_cls_to_handle_idxs = copy.deepcopy(
            self._scene_obj_cls_to_handle_idxs[ep_scene_handle]
        )
        for entity_name, obj_name in cur_ep.object_samples.items():
            targ_recep_sampler = cur_ep.object_spawn_recep[entity_name]
            obj_sampler._allowed_recep_set_names = [targ_recep_sampler]
            recep = obj_sampler.sample_receptacle(self.sim, recep_tracker)
            obj_sampler.receptacle_candidates = None
            if len(obj_cls_to_handle_idxs[obj_name]) == 0:
                raise ValueError(
                    f"Cannot find instance to spawn for {entity_name} with sampled {obj_name}"
                )
            obj_idx = obj_cls_to_handle_idxs[obj_name].pop()
            object_idx_to_recep[obj_idx] = recep
        obj_sampler._allowed_recep_set_names = orig_recep_sets

        obj_sampler.target_objects_number = len(
            self._scene_obj_handles[ep_scene_handle]
        )
        object_sample_data = obj_sampler.sample(
            self.sim,
            recep_tracker,
            [],
            snap_down=True,
            vdb=(self.vdb if self._render_debug_obs else None),
            target_object_handles=spawn_objs,
            object_idx_to_recep=object_idx_to_recep,
        )
        if len(object_sample_data) == 0:
            return None
        new_objects, receptacles = zip(*object_sample_data)

        for obj, rec in zip(new_objects, receptacles):
            self.object_to_containing_receptacle[obj.handle] = rec
        self.episode_data["sampled_objects"]["CLUTTER"] = new_objects
        self.ep_sampled_objects += new_objects
        logger.info(f"Sampler generated {len(new_objects)} new object placements.")
        # debug visualization showing each newly added object
        if self._render_debug_obs:
            for new_object in new_objects:
                self.vdb.look_at(new_object.translation)
                self.vdb.get_observation()

    def _get_cur_ep(self) -> EpisodeData:
        return self._iter_eps[self._cur_ep_idx]

    def generate_single_episode(self):
        # Reset the number of allowed objects per receptacle.
        self._reset_samplers()
        self.episode_data: Dict[str, Dict[str, Any]] = {
            "sampled_objects": {},
        }

        ep_scene_handle = self.generate_scene()
        scene_base_dir = osp.dirname(osp.dirname(ep_scene_handle))

        orig_recep_sets_keys = list(self._receptacle_sets.keys())

        cur_ep = self._get_cur_ep()
        instruct = cur_ep.instruct_info

        # This is reset every episode.
        for k, v in cur_ep.recep_sampler_names.items():
            self._receptacle_sets[k] = v

        scene_name = ep_scene_handle.split(".")[0]
        navmesh_path = osp.join(scene_base_dir, "navmeshes", scene_name + ".navmesh")
        self.sim.pathfinder.load_nav_mesh(navmesh_path)

        ao_states = self._generate_ao_states()

        self._generate_objects(ep_scene_handle, cur_ep)

        # Simulate the world for a few seconds to validate the placements
        if not self.settle_sim([]):
            logger.debug("Aborting episode generation due to unstable state.")
            return None

        handles = [x.creation_attributes.handle for x in self.ep_sampled_objects]
        if self._scene_obj_handles[ep_scene_handle] != handles:
            logger.error(
                f"Requsted {self._scene_obj_handles[ep_scene_handle]} but got {handles}. Throwing out episode."
            )
            return None

        sampled_rigid_object_states = []
        for i, sampled_obj in enumerate(self.ep_sampled_objects):
            creation_attrib = sampled_obj.creation_attributes
            file_handle = creation_attrib.handle.split(creation_attrib.file_directory)[
                -1
            ].split("/")[-1]
            sampled_rigid_object_states.append(
                (
                    file_handle,
                    np.array(sampled_obj.transformation),
                )
            )

        self.num_ep_generated += 1

        name_to_receptacle = {
            k: v.name for k, v in self.object_to_containing_receptacle.items()
        }

        # Substitute the grounded entities into the goal predicate.
        goal_preds = _recur_dict_replace(instruct.goal_expr, cur_ep.sampled_entities)
        self._on_successful_sample()

        # Clear the added samplers so they don't carry to the next episode.
        self._receptacle_sets = {
            k: v for k, v in self._receptacle_sets.items() if k in orig_recep_sets_keys
        }

        return LangRearrangeEpisode(
            scene_dataset_config=self.cfg.dataset_path,
            additional_obj_config_paths=self.cfg.additional_object_paths,
            episode_id=str(self.num_ep_generated - 1),
            instruct_id=instruct.instruct_id,
            start_position=[0, 0, 0],
            sampled_entities=cur_ep.sampled_entities,
            sampler_info=instruct.sampler_info,
            start_rotation=[
                0,
                0,
                0,
                1,
            ],
            scene_id=ep_scene_handle,
            ao_states=ao_states,
            rigid_objs=sampled_rigid_object_states,
            targets={},
            target_receptacles=[],
            goal_receptacles=[],
            markers=self.cfg.markers,
            name_to_receptacle=name_to_receptacle,
            info={"object_labels": {}},
            instruction=cur_ep.instruct,
            start_preds=instruct.start_preds,
            goal_preds=goal_preds,
        )

    def _on_successful_sample(self):
        # Move onto generating the next episode.
        self._cur_ep_idx += 1
        if self._cur_ep_idx >= len(self._iter_eps):
            self._cur_ep_idx = 0


def _process_instruct(phrasing: str, new_entities, name_remap) -> str:
    # Sub in the display names
    # name_remap = self._instruct_data["name_mappings"]
    renamed_new_entities = {k: name_remap.get(v, v) for k, v in new_entities.items()}
    subbed_instruct = phrasing
    for k, v in renamed_new_entities.items():
        subbed_instruct = subbed_instruct.replace(f"`{k}`", v)
    assert (
        "`" not in subbed_instruct
    ), f"Instruction {subbed_instruct} still has unsubbed tokens"

    # Make sure instruction ends in punctuation.
    if subbed_instruct[-1] not in [".", "?", "!"]:
        subbed_instruct += "."
    # Remove any starting or trailing white space.
    subbed_instruct = subbed_instruct.strip()
    return subbed_instruct


def _sample_receps(
    sampled_receps: List[Tuple[str, ReceptacleSampleInfo]],
    ep_dats: List[EpisodeData],
    receptacle_sets,
) -> List[EpisodeData]:
    if len(sampled_receps) == 0:
        return ep_dats
    entity_name, recep = sampled_receps[0]
    recep_set = receptacle_sets[recep.type]

    new_ep_dats = []
    for recep_name in recep_set:
        # Add the receptacle name where applicable.
        for ep_dat in ep_dats:
            sampled_entities = ep_dat.sampled_entities
            is_match = True
            for ne_entity in recep.not_equal:
                ne_entity = sampled_entities[ne_entity]
                is_match = is_match and ne_entity != recep_name
            if not is_match:
                continue
            ep_dat = ep_dat.clone()
            ep_dat.sampled_entities[entity_name] = recep_name
            new_ep_dats.append(ep_dat)

    return _sample_receps(sampled_receps[1:], new_ep_dats, receptacle_sets)


def _create_recep_sampler(
    recep_sampler_name,
    restrict_recep,
    exclude_recep,
    sampled_entities,
    receptacle_sets,
) -> ReceptacleSet:
    include_receps = receptacle_sets["open_air_receps"]
    if restrict_recep is not None:
        if restrict_recep in sampled_entities:
            restrict_recep = sampled_entities[restrict_recep]
        include_receps = [restrict_recep]
    elif exclude_recep is not None:
        if exclude_recep in sampled_entities:
            not_recep_name = sampled_entities[exclude_recep]
        else:
            not_recep_name = exclude_recep
        include_receps = [x for x in include_receps if not_recep_name != x]

    return ReceptacleSet(recep_sampler_name, [""], [], include_receps, [])


def _sample_objects(
    obj_sample_split,
    sampled_objs: List[Tuple[str, ObjectSampleInfo]],
    ep_dats: List[EpisodeData],
    category_groups,
    receptacle_sets,
) -> List[EpisodeData]:
    if len(sampled_objs) == 0:
        return ep_dats
    entity_name, obj_info = sampled_objs[0]

    # Setup the receptacle for this object.
    recep_sampler_name = f"{entity_name}_recep_sampler"

    etype = obj_info.types[obj_sample_split]
    # if etype in cfg.category_groups:
    if etype in category_groups:
        # Refers to group
        obj_set_names = category_groups[etype]
    else:
        # No choice, we are not sampling, but just referring to a single
        # category.
        obj_set_names = [etype]

    new_ep_dats = []
    for obj_set_name in obj_set_names:
        for episode_dat in ep_dats:
            sampled_entities = episode_dat.sampled_entities
            is_match = True
            for ne_entity in obj_info.not_equal:
                ne_entity = sampled_entities[ne_entity]
                is_match = is_match and ne_entity != obj_set_name
            if not is_match:
                continue

            recep_sampler = _create_recep_sampler(
                recep_sampler_name,
                obj_info.recep,
                obj_info.not_recep,
                sampled_entities,
                receptacle_sets,
            )

            ep = episode_dat.clone()
            ep.object_spawn_recep[entity_name] = recep_sampler_name
            ep.object_samples[entity_name] = obj_set_name
            ep.recep_sampler_names[recep_sampler_name] = recep_sampler
            ep.sampled_entities[entity_name] = obj_set_name

            new_ep_dats.append(ep)

    return _sample_objects(
        obj_sample_split,
        sampled_objs[1:],
        new_ep_dats,
        category_groups,
        receptacle_sets,
    )


def _shuffle_select_subset(
    entity_set: Dict[str, List[str]], ratio_sel: float
) -> Dict[str, List[str]]:
    use_set = {}
    for k, v in entity_set.items():
        # Make shuffled copy.
        v = random.sample(v, len(v))
        take_count = int(len(v) * ratio_sel)
        if take_count != 0:
            v = v[:take_count]
        use_set[k] = v
    return use_set


def _get_all_instructs(
    valid_instructs,
    instruct_source: Dict[str, InstructionInfo],
    phrase,
    category_groups,
    receptacle_sets,
    obj_sample_split: str,
    name_remap,
    per_instruct_count_lim,
    rng,
) -> Dict[str, List[EpisodeData]]:
    """
    Gets the set of all possible distinct instructions grouped per instruct worker.

    """

    instructs = defaultdict(list)
    for instruct_id in valid_instructs:
        print(f"Generating instructions for {instruct_id}")
        instruct = instruct_source[instruct_id]
        ep_dats = [EpisodeData(instruct)]
        if instruct.subsample_entity_ratio != 1.0:
            use_recep_sets = _shuffle_select_subset(
                receptacle_sets, instruct.subsample_entity_ratio
            )
            use_obj_sets = _shuffle_select_subset(
                category_groups, instruct.subsample_entity_ratio
            )
        else:
            use_recep_sets = receptacle_sets
            use_obj_sets = category_groups

        ep_dats = _sample_receps(
            [(k, v) for k, v in instruct.sampled_receps.items()],
            ep_dats,
            use_recep_sets,
        )
        ep_dats = _sample_objects(
            obj_sample_split,
            [(k, v) for k, v in instruct.sampled_objects.items()],
            ep_dats,
            use_obj_sets,
            use_recep_sets,
        )
        if len(ep_dats) > per_instruct_count_lim:
            idxs = np.arange(per_instruct_count_lim)
            rng.shuffle(idxs)
            print(f"Subsampled instruction ID {instruct_id}")
            ep_dats = [ep_dats[i] for i in idxs]

        for ep_dat in tqdm(ep_dats):
            # Gather the semantic instructions
            for sem_name, sem_info in instruct.semantics.items():
                if not _is_semantic_match(ep_dat, sem_info):
                    continue
                # sem_is_matched[sem_name] = True
                if phrase not in sem_info.lang:
                    continue
                for instruct_phrasing in sem_info.lang[phrase]:
                    new_ep_dat = ep_dat.clone()
                    new_ep_dat.instruct = _process_instruct(
                        instruct_phrasing, new_ep_dat.sampled_entities, name_remap
                    )
                    sem_instruct_id = f"{instruct_id}_{sem_name}"
                    new_ep_dat.instruct_info.instruct_id = sem_instruct_id
                    instructs[sem_instruct_id].append(new_ep_dat)

            # Gather the base instructions
            if phrase not in instruct.lang:
                continue
            for instruct_phrasing in instruct.lang[phrase]:
                new_ep_dat = ep_dat.clone()
                new_ep_dat.instruct = _process_instruct(
                    instruct_phrasing, new_ep_dat.sampled_entities, name_remap
                )
                instructs[instruct_id].append(new_ep_dat)

    return instructs


def _get_sample_instructs(all_eps, instruct_source) -> List[str]:
    """
    List of instruction IDs t sample. There will be repeats depending on the
    overweight_freq.
    """
    sample_instructs = []
    for instruct_id in all_eps.keys():
        if "_" in instruct_id:
            # Semantic instructions are not overweighted now.
            overweight_freq = 1
        else:
            instruct = instruct_source[instruct_id]
            overweight_freq = instruct.overweight_freq
        sample_instructs.extend([instruct_id for _ in range(overweight_freq)])
    return sample_instructs


def generate_all_instructions(
    instruct_path: str,
    obj_sets: Dict[str, List[str]],
    recep_sets: Dict[str, List[str]],
    instruct_lim_id: List[str],
    phrase: str,
    tag: str,
    obj_sample_split,
    category_groups,
    per_instruct_count_lim,
    rng,
) -> Dict[str, EpisodeData]:
    with open(instruct_path, "r") as f:
        instruct_data = yaml.safe_load(f)

    instruct_source = {}
    valid_obj_cls_names = list(obj_sets.keys())
    valid_recep_sets = []
    for recep_set in recep_sets.values():
        valid_recep_sets.extend(recep_set)
    valid_entity_cls_names = valid_recep_sets + valid_obj_cls_names

    for k, v in instruct_data["instructions"].items():
        try:
            instruct_source[k] = _parse_instruct(v, k, valid_entity_cls_names)
        except Exception as e:
            raise ValueError(f"Could not parse {k}") from e

    # Get `valid_instructs`
    if instruct_lim_id is not None:
        valid_instructs = instruct_lim_id
    else:
        valid_instructs = [
            k for k, v in instruct_source.items() if tag in v.tags and phrase in v.lang
        ]

    logger.warning(f"Sampling from {len(valid_instructs)} instructions.")
    name_remap = instruct_data["name_mappings"]
    all_eps = _get_all_instructs(
        valid_instructs,
        instruct_source,
        phrase,
        category_groups,
        recep_sets,
        obj_sample_split,
        name_remap,
        per_instruct_count_lim,
        rng,
    )
    instruct_samples = _get_sample_instructs(all_eps, instruct_source)
    return all_eps, instruct_samples


def _recur_dict_replace(d: Any, replaces: Dict[str, str]) -> Any:
    """
    Replace all string entries in `d` with the replace name to PDDL entity
    mapping in replaces.
    """
    if isinstance(d, str):
        for name, entity in replaces.items():
            d = d.replace(name, entity)
    elif isinstance(d, list):
        for i, v in enumerate(d):
            d[i] = _recur_dict_replace(v, replaces)
    elif isinstance(d, dict):
        for k, v in d.items():
            d[k] = _recur_dict_replace(d[k], replaces)
    return d


def _parse_instruct(
    instruct: Dict[str, Any], instruct_id: str, valid_obj_cls_names: List[str]
) -> InstructionInfo:
    if "semantics" in instruct:
        semantics = {
            k: SemanticInstructInfo(v["match_objs"], v["lang"])
            for k, v in instruct["semantics"].items()
        }
    else:
        semantics = {}

    # # Verify the semantic instructions refer to valid object classes.
    for sem_k, sem in semantics.items():
        for obj_k, match_vals in sem.match_objs.items():
            for k in match_vals:
                assert k in valid_obj_cls_names, f"From {sem_k}, {obj_k}, {k} not found"

    return InstructionInfo(
        instruct_id=instruct_id,
        lang=instruct["lang"],
        tags=instruct["tags"],
        goal_expr=instruct["goal_expr"],
        sampled_objects={
            k: ObjectSampleInfo(**v) for k, v in instruct["sampled_objects"].items()
        },
        sampled_receps={
            k: ReceptacleSampleInfo(**v) for k, v in instruct["sampled_receps"].items()
        },
        start_preds=instruct.get("start_preds", []),
        sampler_info=instruct.get("sampler_info", {}),
        overweight_freq=instruct.get("overweight_freq", 1),
        solution_template=instruct.get("instruction_template", []),
        subsample_entity_ratio=instruct.get("subsample_entity_ratio", 1.0),
        semantics=semantics,
    )


def log_ep_counts(eps, did_split_subsample=None, did_proc_subsample=None):
    """
    Used for debugging.
    """

    if did_split_subsample is None:
        for k, v in eps.items():
            print(f"{k}: {len(v)}")
    else:
        for k, v in eps.items():
            print(
                f"{k}: {len(v)} ({did_split_subsample[k]}), ({did_proc_subsample[k]})"
            )


def _flatten_eps(all_eps, num_eps, instruct_samples) -> List[EpisodeData]:
    ep_flat = []
    cur_instruct_idx = 0
    cur_instruct_ep_idx = defaultdict(int)
    while len(ep_flat) < num_eps:
        cur_instruct_id = instruct_samples[cur_instruct_idx % len(instruct_samples)]
        cur_instruct_idx += 1

        instruct_eps = all_eps[cur_instruct_id]
        ep_flat.append(
            instruct_eps[cur_instruct_ep_idx[cur_instruct_id] % len(instruct_eps)]
        )
        cur_instruct_ep_idx[cur_instruct_id] += 1
    return ep_flat


def get_flat_eps_split(
    all_eps,
    proc_idx,
    total_take,
    cur_gen_idx,
    total_procs,
    num_eps,
    instruct_samples,
):
    """
    Deterministically splits the episodes and selects the episode split for `proc_idx`.
    """

    subsel_eps = {}
    for k, use_eps in all_eps.items():
        n = len(use_eps)
        if total_take > 1 and n > total_take:
            portion_size = n / total_take
            use_eps = use_eps[
                int(cur_gen_idx * portion_size) : int((cur_gen_idx + 1) * portion_size)
            ]

        n = len(use_eps)
        if n > total_procs:
            portion_size = n / total_procs
            use_eps = use_eps[
                int(proc_idx * portion_size) : int((proc_idx + 1) * portion_size)
            ]

        assert len(use_eps) > 0
        subsel_eps[k] = use_eps

    return _flatten_eps(subsel_eps, num_eps, instruct_samples)


def _is_semantic_match(ep_dat, sem_info: SemanticInstructInfo):
    for k, poss_matches in sem_info.match_objs.items():
        for poss_match in poss_matches:
            real_v = ep_dat.sampled_entities[k]
            if real_v != poss_match:
                return False
    return True
