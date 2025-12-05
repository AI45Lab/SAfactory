#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
from typing import Any, Optional
import numpy as np
from gym import spaces
from habitat.core.registry import registry
from habitat.tasks.rearrange.actions.articulated_agent_action import \
    ArticulatedAgentAction
from habitat.tasks.rearrange.actions.grip_actions import (
    GazeGraspAction, GripSimulatorTaskAction, MagicGraspAction,
    SuctionGraspAction)
from habitat.tasks.rearrange.rearrange_sim import RearrangeSim

from embodiedbench.envs.eb_habitat.utils import get_allowed_actions


@registry.register_task_action
class CustomArmAbsJointAction(ArticulatedAgentAction):
    @property
    def action_space(self):
        return spaces.Dict(
            {
                "arm_joint_action": spaces.Box(
                    shape=(self._config.arm_joint_dimensionality,),
                    low=-255.0,
                    high=255.0,
                    dtype=np.float32,
                )
            }
        )

    def step(self, *args, arm_joint_action, **kwargs):
        if np.sum(arm_joint_action) < 1e-3:
            # Don't change the arm position if this action wasn't invoked (set to 0).
            return
        # No clipping because the arm is being set to exactly where it needs to
        # go.
        self.cur_articulated_agent.arm_joint_pos = arm_joint_action


@registry.register_task_action
class SafeSuctionGraspAction(MagicGraspAction):
    def __init__(self, *args, config, sim: RearrangeSim, **kwargs):
        super().__init__(*args, config=config, sim=sim, **kwargs)
        self._sim: RearrangeSim = sim
        self._prevent_change_duration = config.prevent_gripper_change_duration

    def reset(self, *args: Any, **kwargs: Any) -> None:
        self._step_count = 0
        super().reset(*args, **kwargs)

    def step(self, grip_action, should_step=True, *args, **kwargs):
        self._step_count += 1
        if grip_action is None:
            return

        if self._step_count < self._prevent_change_duration:
            return

        if grip_action >= 0 and not self.cur_grasp_mgr.is_grasped:
            self._grasp()
        elif grip_action < 0 and self.cur_grasp_mgr.is_grasped:
            self._ungrasp()


@registry.register_task_action
class KinematicArmEEAction(ArticulatedAgentAction):
    """Uses inverse kinematics (requires pybullet) to apply end-effector position control for the articulated_agent's arm."""

    def __init__(self, *args, sim: RearrangeSim, **kwargs):
        self.ee_target: Optional[np.ndarray] = None
        self.ee_index: Optional[int] = 0
        super().__init__(*args, sim=sim, **kwargs)
        self._sim: RearrangeSim = sim
        self._render_ee_target = False  # self._config.get("render_ee_target", False)
        self._ee_ctrl_lim = self._config.ee_ctrl_lim

    def reset(self, *args, **kwargs):
        super().reset()
        cur_ee = self._ik_helper.calc_fk(
            np.array(self._sim.articulated_agent.arm_joint_pos)
        )

        self.ee_target = cur_ee

    @property
    def action_space(self):
        return spaces.Box(shape=(3,), low=-255.0, high=255.0, dtype=np.float32)

    def apply_ee_constraints(self):
        self.ee_target = np.clip(
            self.ee_target,
            self._sim.articulated_agent.params.ee_constraint[self.ee_index, :, 0],
            self._sim.articulated_agent.params.ee_constraint[self.ee_index, :, 1],
        )

    def set_desired_ee_pos(self, ee_delta: np.ndarray) -> None:
        self.ee_target += np.array(ee_delta)

        self.apply_ee_constraints()

        joint_pos = np.array(self._sim.articulated_agent.arm_joint_pos)
        joint_vel = np.zeros(joint_pos.shape)

        self._ik_helper.set_arm_state(joint_pos, joint_vel)

        des_joint_pos = self._ik_helper.calc_ik(self.ee_target)
        des_joint_pos = list(des_joint_pos)
        self._sim.articulated_agent.arm_joint_pos = des_joint_pos

    def step(self, ee_delta, **kwargs):
        speed = np.linalg.norm(ee_delta)
        if speed == 0.0:
            # Only act when called.
            return
        if speed > self._ee_ctrl_lim:
            # Clip norm.
            ee_delta *= self._ee_ctrl_lim / speed
        self.set_desired_ee_pos(ee_delta)

        if self._render_ee_target:
            global_pos = (
                self._sim.articulated_agent.base_transformation.transform_point(
                    self.ee_target
                )
            )
            self._sim.viz_ids["ee_target"] = self._sim.visualize_position(
                global_pos, self._sim.viz_ids["ee_target"]
            )


@registry.register_task_action
class PddlHlAction(ArticulatedAgentAction):
    def __init__(self, *args, config, task, **kwargs):
        actions = get_allowed_actions(task.pddl_problem, config.allowed_actions)

        self._action_datas = []
        for action in actions:
            self._action_datas.append(
                (action.name, [p.name for p in action.param_values])
            )
        super().__init__(*args, config=config, task=task, **kwargs)

    @property
    def action_space(self):
        return spaces.Discrete(len(self._action_datas))

    @property
    def was_prev_action_invalid(self):
        return self._was_prev_action_invalid

    def reset(self, *args, task, **kwargs):
        self._was_prev_action_invalid = False
        self._prev_action = None

    def step(self, *args, sel, task, **kwargs):
        pddl = task.pddl_problem
        action_name, param_names = self._action_datas[sel]

        # Get the current up to date PDDL action.
        param_values = []
        missing_entity = False
        for name in param_names:
            if name not in pddl.all_entities:
                missing_entity = True
                break
            param_values.append(pddl.all_entities[name])

        if missing_entity:
            # self._was_prev_action_invalid = True
            # return
            raise ValueError("MISSING ENTITY. THIS SHOULDNT HAPPEN")

        apply_action = task.pddl_problem.actions[action_name].clone()
        apply_action.set_param_values(param_values)

        self._was_prev_action_invalid = not apply_action.apply_if_true(
            task.pddl_problem.sim_info
        )
        self._prev_action = apply_action
