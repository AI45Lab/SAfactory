#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
import os
import json
import pickle
import random
from itertools import groupby
from typing import Any, Dict, List, Optional

import attr
import numpy as np
from habitat.core.dataset import EpisodeIterator
from habitat.core.logging import logger
from habitat.core.registry import registry
from habitat.core.utils import DatasetFloatJSONEncoder
from habitat.datasets.rearrange.rearrange_dataset import (RearrangeDatasetV0,
                                                          RearrangeEpisode)
# from habitat.datasets.utils import check_and_gen_physics_config
from habitat.tasks.rearrange.multi_task.pddl_predicate import Predicate

DEFAULT_PHYSICS_CONFIG_PATH = os.path.join(os.path.dirname(__file__), '../data/default.physics_config.json')

def check_and_gen_physics_config():
    if os.path.exists(DEFAULT_PHYSICS_CONFIG_PATH):
        return
    #  Config is sourced from
    #  https://github.com/facebookresearch/habitat-sim/blob/main/data/default.physics_config.json
    physics_config = {
        "physics_simulator": "bullet",
        "timestep": 0.008,
        "gravity": [0, -9.8, 0],
        "friction_coefficient": 0.4,
        "restitution_coefficient": 0.1,
        "rigid object paths": ["objects"],
    }
    with open(DEFAULT_PHYSICS_CONFIG_PATH, "w") as f:
        json.dump(physics_config, f)


@attr.s(auto_attribs=True, kw_only=True)
class LangRearrangeEpisode(RearrangeEpisode):
    instruction: str
    sampled_entities: Dict[str, str]
    # str form of predicates
    start_preds: List[str]
    goal_preds: Dict[str, Any]
    instruct_id: str
    sampler_info: Dict[str, str]
    # str form of predicates
    subgoals: List[List[str]] = None


@registry.register_dataset(name="LangRearrangeDataset-v0")
class LangRearrangeDatasetV0(RearrangeDatasetV0):
    def __init__(self, config=None, preset_eps=None) -> None:
        self.config = config

        check_and_gen_physics_config()

        self.episodes = []

        if config is None:
            return

        if preset_eps is None:
            datasetfile_path = config.data_path.format(split=config.split)
            logger.info(f"Loading from {datasetfile_path}")
            with open(datasetfile_path, "rb") as f:
                self.from_binary(pickle.load(f), scenes_dir=config.scenes_dir)

            self.episodes = list(
                filter(self.build_content_scenes_filter(config), self.episodes)
            )
        else:
            self.episodes = preset_eps

    def to_json(self) -> str:
        result = DatasetFloatJSONEncoder().encode(self)
        return result

    def to_binary(self) -> str:
        def access_idx(k, name_to_idx):
            if len(name_to_idx) == 0:
                name_to_idx[k] = 0
            if k not in name_to_idx:
                name_to_idx[k] = max(name_to_idx.values()) + 1
            return name_to_idx[k]

        def encode_name_dict(d, name_to_idx):
            ret_d = {}
            for k, v in d.items():
                ret_d[access_idx(k, name_to_idx)] = v
            return ret_d

        all_transforms = []
        name_to_idx = {}
        all_eps = []

        for ep in self.episodes:
            new_ep_data = attr.asdict(ep)
            rigid_objs = []
            for name, T in ep.rigid_objs:
                rigid_objs.append([access_idx(name, name_to_idx), len(all_transforms)])
                all_transforms.append(T)

            name_to_recep = []
            for name, recep in ep.name_to_receptacle.items():
                name_to_recep.append(
                    [access_idx(name, name_to_idx), access_idx(recep, name_to_idx)]
                )
            new_ep_data["rigid_objs"] = np.array(rigid_objs)
            new_ep_data["ao_states"] = encode_name_dict(ep.ao_states, name_to_idx)
            new_ep_data["name_to_receptacle"] = np.array(name_to_recep)
            new_ep_data["additional_obj_config_paths"] = list(
                new_ep_data["additional_obj_config_paths"]
            )
            del new_ep_data["_shortest_path_cache"]

            new_markers = []
            for marker_data in ep.markers:
                new_markers.append(
                    [
                        access_idx(marker_data["name"], name_to_idx),
                        access_idx(marker_data["type"], name_to_idx),
                        np.array(marker_data["params"]["offset"]),
                        access_idx(marker_data["params"]["link"], name_to_idx),
                        access_idx(marker_data["params"]["object"], name_to_idx),
                    ]
                )

            new_ep_data["markers"] = new_markers

            all_eps.append(new_ep_data)

        idx_to_name = {}
        for k, v in name_to_idx.items():
            # idx_to_name should define a 1-1 mapping between the name and the
            # name index.
            assert v not in idx_to_name
            idx_to_name[v] = k

        return {
            "all_transforms": np.array(all_transforms),
            "idx_to_name": idx_to_name,
            "all_eps": all_eps,
        }

    def from_binary(
        self, data_dict: Dict[str, Any], scenes_dir: Optional[str] = None
    ) -> None:
        all_T = data_dict["all_transforms"]
        idx_to_name = data_dict["idx_to_name"]
        for i, ep in enumerate(data_dict["all_eps"]):
            ep["rigid_objs"] = [
                [idx_to_name[ni], all_T[ti]] for ni, ti in ep["rigid_objs"]
            ]
            ep["ao_states"] = {idx_to_name[ni]: v for ni, v in ep["ao_states"].items()}
            ep["name_to_receptacle"] = {
                idx_to_name[k]: idx_to_name[v] for k, v in ep["name_to_receptacle"]
            }

            new_markers = []
            for name, mtype, offset, link, obj in ep["markers"]:
                new_markers.append(
                    {
                        "name": idx_to_name[name],
                        "type": idx_to_name[mtype],
                        "params": {
                            "offset": offset,
                            "link": idx_to_name[link],
                            "object": idx_to_name[obj],
                        },
                    }
                )
            ep["markers"] = new_markers

            rearrangement_episode = LangRearrangeEpisode(**ep)
            rearrangement_episode.episode_id = str(i)
            self.episodes.append(rearrangement_episode)

    def from_json(self, json_str: str, scenes_dir: Optional[str] = None) -> None:
        deserialized = json.loads(json_str)

        for i, episode in enumerate(deserialized["episodes"]):
            rearrangement_episode = LangRearrangeEpisode(**episode)
            rearrangement_episode.episode_id = str(i)

            self.episodes.append(rearrangement_episode)

    def get_episode_iterator(self, *args, **kwargs):
        return CustomEpisodeIterator(self.episodes, *args, **kwargs)


class CustomEpisodeIterator(EpisodeIterator):
    def __init__(
        self,
        episodes,
        cycle: bool = True,
        shuffle: bool = True,
        group_by_scene: bool = False,
        max_scene_repeat_episodes: int = -1,
        max_scene_repeat_steps: int = -1,
        num_episode_sample: int = -1,
        step_repetition_range: float = 0.2,
        seed: int = None,
    ) -> None:
        if seed:
            random.seed(seed)
            np.random.seed(seed)

        # sample episodes
        if num_episode_sample >= 0:
            episodes = np.random.choice(  # type: ignore[assignment]
                episodes, num_episode_sample, replace=False  # type: ignore[arg-type]
            )

        if not isinstance(episodes, list):
            episodes = list(episodes)

        self.episodes = episodes
        self.cycle = cycle
        self.group_by_scene = group_by_scene
        self.shuffle = shuffle

        if shuffle:
            random.shuffle(self.episodes)

        if group_by_scene:
            self.episodes = self._group_scenes(self.episodes)

        self.max_scene_repetition_episodes = max_scene_repeat_episodes
        self.max_scene_repetition_steps = max_scene_repeat_steps

        self._rep_count = -1  # 0 corresponds to first episode already returned
        self._step_count = 0
        self._prev_scene_id: Optional[str] = None

        self._iterator = iter(self.episodes)

        self.step_repetition_range = step_repetition_range
        self._set_shuffle_intervals()

    def __iter__(self):
        return self

    def __next__(self):
        self._forced_scene_switch_if()
        next_episode = next(self._iterator, None)
        if next_episode is None:
            if not self.cycle:
                raise StopIteration

            self._iterator = iter(self.episodes)

            if self.shuffle:
                self._shuffle()

            next_episode = next(self._iterator)

        if (
            self._prev_scene_id != next_episode.scene_id
            and self._prev_scene_id is not None
        ):
            self._rep_count = 0
            self._step_count = 0

        self._prev_scene_id = next_episode.scene_id
        return next_episode

    def _forced_scene_switch(self) -> None:
        grouped_episodes = [
            list(g) for k, g in groupby(self._iterator, key=lambda x: x.scene_id)
        ]

        if len(grouped_episodes) > 1:
            # Ensure we swap by moving the current group to the end
            grouped_episodes = grouped_episodes[1:] + grouped_episodes[0:1]

        self._iterator = iter(sum(grouped_episodes, []))

    def _shuffle(self) -> None:
        assert self.shuffle
        episodes = list(self._iterator)

        random.shuffle(episodes)

        if self.group_by_scene:
            episodes = self._group_scenes(episodes)

        self._iterator = iter(episodes)

    def _group_scenes(self, episodes):
        assert self.group_by_scene

        scene_sort_keys: Dict[str, int] = {}
        for e in episodes:
            if e.scene_id not in scene_sort_keys:
                scene_sort_keys[e.scene_id] = len(scene_sort_keys)

        return sorted(episodes, key=lambda e: scene_sort_keys[e.scene_id])

    def step_taken(self) -> None:
        self._step_count += 1

    @staticmethod
    def _randomize_value(value: int, value_range: float) -> int:
        return random.randint(
            int(value * (1 - value_range)), int(value * (1 + value_range))
        )

    def _set_shuffle_intervals(self) -> None:
        if self.max_scene_repetition_episodes > 0:
            self._max_rep_episode = self.max_scene_repetition_episodes
        else:
            self._max_rep_episode = None

        if self.max_scene_repetition_steps > 0:
            self._max_rep_step = self._randomize_value(
                self.max_scene_repetition_steps, self.step_repetition_range
            )
        else:
            self._max_rep_step = None

    def _forced_scene_switch_if(self) -> None:
        do_switch = False
        self._rep_count += 1

        # Shuffle if a scene has been selected more than _max_rep_episode times in a row
        if (
            self._max_rep_episode is not None
            and self._rep_count >= self._max_rep_episode
        ):
            do_switch = True

        # Shuffle if a scene has been used for more than _max_rep_step steps in a row
        if self._max_rep_step is not None and self._step_count >= self._max_rep_step:
            do_switch = True

        if do_switch:
            self._forced_scene_switch()
            self._set_shuffle_intervals()
