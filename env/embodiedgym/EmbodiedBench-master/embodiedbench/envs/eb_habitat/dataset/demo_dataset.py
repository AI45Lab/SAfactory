#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
import os
import os.path as osp
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from habitat.core.logging import logger
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from .utils import LOCAL_DATASETS_PATH, get_name_mappings


@dataclass
class TrajInfo:
    """
    Let `H` be the length of the trajectory (meaning the number of actions the
    agent takes).

    :property actions: Tensor of shape [H, action_dim]
    :property obs: Dictionary of shape {k: [H, obs_dim...]}. Where step `t` is
        the observation in which the agent made action at `actions[t]`.
    """

    actions: torch.Tensor
    obs: Dict[str, torch.Tensor]
    infos: Dict[str, Any]
    instruction: Optional[str] = None


@dataclass
class DemoDataset:
    trajs: List[TrajInfo]

    def save(self, filepath):
        with open(filepath, "wb") as f:
            torch.save(asdict(self), f)

    def __len__(self):
        return sum(x.actions.shape[0] for x in self.trajs)

    @staticmethod
    def load(filepath):
        # We don't have space to move these to GPU. They will be moved to GPU
        # on demand.
        demos = torch.load(filepath, map_location="cpu")

        trajs = [
            TrajInfo(
                actions=d["actions"],
                obs=d["obs"],
                infos=d["infos"],
                instruction=d.get("instruction", None),
            )
            for d in demos["trajs"]
        ]
        logger.info(f"Loaded demo dataset from {filepath}")
        np.random.shuffle(trajs)
        return DemoDataset(
            trajs=trajs,
        )


def cat_demo_datasets(demos: List[DemoDataset]) -> DemoDataset:
    trajs = []
    for demo in demos:
        trajs.extend(demo.trajs)

    return DemoDataset(
        trajs=trajs,
    )


class DatasetCollector:
    def __init__(self, n_envs, flush_interval, dataset_name):
        self._n_envs = n_envs
        self._cur_traj_actions = [[] for _ in range(n_envs)]
        self._cur_traj_infos = [defaultdict(list) for _ in range(n_envs)]
        self._cur_traj_obs = [defaultdict(list) for _ in range(n_envs)]
        self._all_trajs = []
        self._flush_interval = flush_interval
        self._dataset_name = dataset_name
        self._save_dir = osp.join(LOCAL_DATASETS_PATH, dataset_name)
        self._cur_block = 0

    def collect_action(self, action, infos):
        for env_i in range(self._n_envs):
            # Only save primitive types. Otherwise serialization will get messed up.
            filtered_infos = {
                k: v
                for k, v in infos[env_i].items()
                if isinstance(v, (int, float, str))
            }
            self._cur_traj_actions[env_i].append(action[env_i])
            for k, v in filtered_infos.items():
                self._cur_traj_infos[env_i][k].append(v)

    def collect_obs(self, obs):
        """
        Add observation to all workers.
        """

        for env_i in range(self._n_envs):
            for k, obs_k in obs.items():
                self._cur_traj_obs[env_i][k].append(obs_k[env_i].cpu())

    def on_ep_done(self, env_i, env_infos, instruction: Optional[str] = None):
        # Only save the episode if it was successful.
        if env_infos["env_success"]:
            traj_info = TrajInfo(
                actions=torch.stack(self._cur_traj_actions[env_i], dim=0),
                obs={
                    k: torch.stack(v, dim=0)
                    for k, v in self._cur_traj_obs[env_i].items()
                },
                infos=dict(self._cur_traj_infos[env_i]),
                instruction=instruction,
            )
            for k, v in traj_info.obs.items():
                assert (
                    v.shape[0] == traj_info.actions.shape[0]
                ), f"Traj info shapes don't match {k}: {v.shape}, {traj_info.actions.shape}"
            self._all_trajs.append(traj_info)

        n_trajs = len(self._all_trajs)
        if n_trajs != 0 and n_trajs % self._flush_interval == 0:
            self.flush()

        # Restart demo collection for this worker.
        self._cur_traj_obs[env_i] = defaultdict(list)
        self._cur_traj_actions[env_i] = []
        self._cur_traj_infos[env_i] = defaultdict(list)

    def flush(self):
        if len(self._all_trajs) == 0:
            # Nothing to write.
            return
        os.makedirs(self._save_dir, exist_ok=True)
        block_name = f"block{self._cur_block}.pt"
        save_path = osp.join(self._save_dir, block_name)
        dataset = DemoDataset(trajs=self._all_trajs)
        dataset.save(save_path)
        logger.info(
            f"Saved {len(dataset)} transitions ({len(self._all_trajs)} trajs) to {save_path}"
        )
        self._all_trajs = []
        self._cur_block += 1
