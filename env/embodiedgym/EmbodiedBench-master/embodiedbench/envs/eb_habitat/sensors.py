#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
import os.path as osp
from typing import List

import gym.spaces as spaces
import numpy as np
import torch
from habitat.core.registry import registry
from habitat.core.simulator import Sensor, SensorTypes
from habitat.tasks.rearrange.multi_task.pddl_predicate import Predicate
from habitat.tasks.rearrange.multi_task.rearrange_pddl import \
    SimulatorObjectType
from habitat.tasks.rearrange.utils import batch_transform_point

from embodiedbench.envs.eb_habitat.dataset.utils import get_category_info, get_name_mappings
from embodiedbench.envs.eb_habitat.utils import get_parser


@registry.register_sensor
class OneHotTargetSensor(Sensor):
    def __init__(self, *args, task, **kwargs):
        self._task = task
        self._n_cls = len(task.all_cls)
        self._obs = np.zeros((self._n_cls,))
        super().__init__(*args, **kwargs)

    def _get_uuid(self, *args, **kwargs):
        return "one_hot_target_sensor"

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args, config, **kwargs):
        return spaces.Box(shape=(self._n_cls,), low=0, high=1, dtype=np.float32)

    def get_observation(self, *args, **kwargs):
        cur_target = self._task.get_sampled()[self.config.sampled_idx]

        # For receptacles the name will not be a class but the name directly.
        use_name = cur_target.expr_type.name
        if cur_target.name in self._task.all_cls:
            use_name = cur_target.name
        set_i = self._task.all_cls.index(use_name)
        self._obs *= 0.0
        self._obs[set_i] = 1.0

        return self._obs


@registry.register_sensor
class SimpleTargetSensor(Sensor):
    uuid = "simple_lang_goal"

    def __init__(self, *args, config, **kwargs):
        self._max_len = config.max_len
        self._add_special_tokens = config.add_special_tokens
        self._name_mappings = get_name_mappings()
        self.cur_lang_goal = ""
        super().__init__(*args, config=config, **kwargs)

    def _get_uuid(self, *args, **kwargs):
        return SimpleTargetSensor.uuid

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args, config, **kwargs):
        return spaces.Box(
            shape=(self._max_len,),
            low=0,
            high=1,
            dtype=np.int64,
        )

    def get_observation(self, *args, task, **kwargs):
        cur_target = task.get_sampled()[0]
        if cur_target.expr_type.parent.name == SimulatorObjectType.MOVABLE_ENTITY.value:
            # This is an object.
            target_name = cur_target.expr_type.name
        else:
            # This is a receptacle.
            target_name = cur_target.name

        target_name = self._name_mappings.get(target_name, target_name)

        tokens = task.tokenizer(
            target_name,
            return_tensors="np",
            padding="max_length",
            max_length=self._max_len,
            add_special_tokens=self._add_special_tokens,
        )["input_ids"][0]

        assert (
            tokens.shape[0] == self._max_len
        ), f"Instruction is too many tokens at {tokens.shape[0]} but the max is {self._max_len} for instruction {target_name}"
        self.cur_lang_goal = target_name
        return tokens


@registry.register_sensor
class WindowDebugSensor(Sensor):
    uuid = "debug_info"

    def _get_uuid(self, *args, **kwargs):
        return WindowDebugSensor.uuid

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args, config, **kwargs):
        return spaces.Box(
            shape=(2,),
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            dtype=np.float32,
        )

    def get_observation(self, *args, task, **kwargs):
        return np.array([task.num_steps + 1, task.episode_rollout_id], dtype=np.float32)


@registry.register_sensor
class StepCountSensor(Sensor):
    uuid = "step_count"

    def _get_uuid(self, *args, **kwargs):
        return StepCountSensor.uuid

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args, config, **kwargs):
        return spaces.Box(
            shape=(1,),
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            dtype=np.float32,
        )

    def get_observation(self, *args, task, **kwargs):
        return np.array([task.num_steps], dtype=np.float32)


@registry.register_sensor
class VocabLangGoalSensor(Sensor):
    uuid = "vocab_lang_goal"

    def __init__(self, *args, config, **kwargs):
        self._max_len = config.max_len
        self._add_special_tokens = config.add_special_tokens
        super().__init__(*args, config=config, **kwargs)

    def _get_uuid(self, *args, **kwargs):
        return VocabLangGoalSensor.uuid

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args, config, **kwargs):
        return spaces.Box(
            shape=(self._max_len,),
            low=0,
            high=1,
            dtype=np.int64,
        )

    def get_observation(self, *args, task, **kwargs):
        tokens = task.tokenizer(
            task.lang_goal,
            return_tensors="np",
            padding="max_length",
            max_length=self._max_len,
            add_special_tokens=self._add_special_tokens,
        )["input_ids"][0]
        assert (
            tokens.shape[0] == self._max_len
        ), f"Instruction is too many tokens at {tokens.shape[0]} but the max is {self._max_len}"
        return tokens


@registry.register_sensor
class T5VocabLangGoalSensor(Sensor):
    """
    Always outputs the T5 tokenization.
    """

    uuid = "t5_vocab_lang_goal"

    def __init__(self, *args, config, **kwargs):
        self._max_len = config.max_len
        self._tokenizer = get_parser("google/flan-t5-small")
        super().__init__(*args, config=config, **kwargs)

    def _get_uuid(self, *args, **kwargs):
        return T5VocabLangGoalSensor.uuid

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args, **kwargs):
        return spaces.Box(
            shape=(self._max_len,),
            low=0,
            high=1,
            dtype=np.int64,
        )

    def get_observation(self, *args, task, **kwargs):
        tokens = self._tokenizer(
            task.lang_goal,
            return_tensors="np",
            padding="max_length",
            max_length=self._max_len,
        )["input_ids"][0]
        assert tokens.shape[0] == self._max_len
        return tokens


@registry.register_sensor
class LlamaVocabLangGoalSensor(Sensor):
    """
    Always outputs the llama tokenization.
    """

    uuid = "llama_vocab_lang_goal"

    def __init__(self, *args, config, **kwargs):
        self._max_len = config.max_len
        self._tokenizer = get_parser(config.tokenizer_name)
        super().__init__(*args, config=config, **kwargs)

    def _get_uuid(self, *args, **kwargs):
        return LlamaVocabLangGoalSensor.uuid

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args, **kwargs):
        return spaces.Box(
            shape=(self._max_len,),
            low=0,
            high=1,
            dtype=np.int64,
        )

    def get_observation(self, *args, task, **kwargs):
        tokens = self._tokenizer(
            task.lang_goal,
            return_tensors="np",
            padding="max_length",
            max_length=self._max_len,
        )["input_ids"][0]
        assert tokens.shape[0] == self._max_len
        return tokens


@registry.register_sensor
class ObsLangSensor(Sensor):
    uuid = "obs_lang"

    def __init__(self, *args, config, task, **kwargs):
        self._max_len = config.max_len
        self._task = task
        self._predicates_list = None
        super().__init__(*args, config=config, task=task, **kwargs)

    def _get_uuid(self, *args, **kwargs):
        return ObsLangSensor.uuid

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TENSOR

    @property
    def predicates_list(self) -> List[Predicate]:
        """
        Returns all possible predicate combinations in the environment.
        """

        if self._predicates_list is None:
            self._predicates_list = self._task.pddl_problem.get_possible_predicates()
        return self._predicates_list

    def _get_observation_space(self, *args, config, **kwargs):
        return spaces.Box(
            shape=(self._max_len,),
            low=0,
            high=1,
            dtype=np.int64,
        )

    def get_observation(self, *args, **kwargs):
        # Fetch the predicates that are true in the current simulator step.
        sim_info = self._task.pddl_problem.sim_info
        true_preds: List[Predicate] = [
            p for p in self.predicates_list if p.is_true(sim_info)
        ]

        # Conver the predicates to a string representation.
        true_preds_s: List[str] = [p.compact_str for p in true_preds]

        # Join all the predicate strings by sentences.
        state_s = ". ".join(true_preds_s)

        # Return the tokenized version.
        return self._task.tokenizer(
            state_s,
            return_tensors="np",
            padding="max_length",
            max_length=self._max_len,
            truncation=True,
        )["input_ids"][0]


@registry.register_sensor
class VocabEmbedSensor(Sensor):
    def __init__(self, *args, config, **kwargs):
        embed_dat = torch.load(config.embed_path)
        self._embeddings = embed_dat["hxs"]
        self._ep_idx_to_idx = embed_dat["ep_idx_to_hxs_idx"]
        super().__init__(*args, config=config, **kwargs)

    def _get_uuid(self, *args, **kwargs):
        return "vocab_embed_goal"

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args, config, **kwargs):
        return spaces.Box(
            shape=(self.config.hidden_dim,),
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            dtype=np.float32,
        )

    def get_observation(self, *args, task, **kwargs):
        pt_idx = self._ep_idx_to_idx[task._episode_id]
        return self._embeddings[pt_idx]


@registry.register_sensor
class ClosestTargetObjectPosSensor(Sensor):
    """
    Gets the distance between the EE and the closest target object.
    """

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return "closest_targ_obj_pos_sensor"

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args, config, **kwargs):
        return spaces.Box(
            shape=(3,),
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            dtype=np.float32,
        )

    def get_observation(self, *args, task, **kwargs):
        targ_type = task.get_sampled()[0]
        matches = list(task.pddl.find_entities(targ_type.expr_type))
        obj_pos = np.array(
            [task.pddl.sim_info.get_entity_pos(match) for match in matches]
        )

        ee_pos = task._sim.articulated_agent.ee_transform().translation

        dists = np.linalg.norm(obj_pos - ee_pos, ord=2, axis=-1)
        closest_id = np.argmin(dists)
        return obj_pos[closest_id]


@registry.register_sensor
class AllObjectPositionsSensor(Sensor):
    def __init__(self, *args, task, **kwargs):
        self._all_cats, _, _ = get_category_info()
        super().__init__(*args, **kwargs)

    def _get_uuid(self, *args, **kwargs):
        return "all_obj_pos_sensor"

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args, config, **kwargs):
        return spaces.Box(
            shape=(3,),
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            dtype=np.float32,
        )

    def get_observation(self, *args, task, **kwargs):
        pddl = task.pddl
        entities = pddl.get_ordered_entities_list()

        entity_pos = [pddl.sim_info.get_entity_pos(entity) for entity in entities]

        base_T = task._sim.articulated_agent.base_transformation
        entity_pos = batch_transform_point(entity_pos, base_T.inverted(), np.float32)

        entity_type_idxs = []
        for entity in entities:
            if entity.name in self._all_cats:
                entity_idx = self._all_cats.index(entity.name)
            elif entity.expr_type.name in self._all_cats:
                entity_idx = self._all_cats.index(entity.expr_type.name)
            else:
                entity_idx = -1
            entity_type_idxs.append(entity_idx)

        return np.array(
            [[*pos, idx] for pos, idx in zip(entity_pos, entity_type_idxs)],
            dtype=np.float32,
        ).reshape(-1)
