#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
from collections import defaultdict

import numpy as np
from habitat.core.embodied_task import Measure
from habitat.core.registry import registry
from habitat.tasks.rearrange.multi_task.pddl_logical_expr import \
    LogicalExprType
from habitat.tasks.rearrange.multi_task.pddl_predicate import Predicate
from habitat.tasks.rearrange.rearrange_sensors import (
    DoesWantTerminate, EndEffectorToRestDistance, RearrangeReward)

from embodiedbench.envs.eb_habitat.dataset.utils import get_all_instruct_ids
from embodiedbench.envs.eb_habitat.sensors import SimpleTargetSensor


@registry.register_measure
class TargetNameMeasure(Measure):
    def __init__(self, config, *args, **kwargs):
        super().__init__(**kwargs)
        self._config = config

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return "target"

    def reset_metric(self, *args, **kwargs):
        self.update_metric(*args, **kwargs)

    def update_metric(self, *args, task, **kwargs):
        self._metric = ",".join([str(x) for x in task.get_sampled()])


@registry.register_measure
class LangGoalMeasure(Measure):
    cls_uuid = "lang_goal"

    def __init__(self, config, *args, **kwargs):
        super().__init__(**kwargs)
        self._config = config

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return LangGoalMeasure.cls_uuid

    def reset_metric(self, *args, **kwargs):
        self.update_metric(*args, **kwargs)

    def update_metric(self, *args, task, **kwargs):
        self._metric = task.lang_goal


@registry.register_measure
class SimpleLangGoalMeasure(Measure):
    cls_uuid = "simple_lang_goal"

    def __init__(self, config, *args, **kwargs):
        super().__init__(**kwargs)
        self._config = config

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return SimpleLangGoalMeasure.cls_uuid

    def reset_metric(self, *args, **kwargs):
        self.update_metric(*args, **kwargs)

    def update_metric(self, *args, task, **kwargs):
        target_sensor = task.sensor_suite.sensors[SimpleTargetSensor.uuid]
        self._metric = target_sensor.cur_lang_goal


@registry.register_measure
class TaskProgressMonitor(Measure):
    def __init__(self, config, *args, **kwargs):
        super().__init__(**kwargs)
        self._config = config

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return "task_progress"

    def reset_metric(self, *args, task, **kwargs):
        self._achieved = defaultdict(lambda: False)
        self._total_count = len(task.subgoals)
        self.update_metric(*args, task=task, **kwargs)

    def update_metric(self, *args, task, **kwargs):
        for i, subgoal in enumerate(task.subgoals):
            if self._achieved[i]:
                continue
            self._achieved[i] = all(
                pred.is_true(task.pddl.sim_info) for pred in subgoal
            )

        self._metric = sum(self._achieved.values()) / max(self._total_count, 1)


@registry.register_measure
class WasPrevActionInvalid(Measure):
    cls_uuid = "was_prev_action_invalid"

    def __init__(self, config, *args, **kwargs):
        self._pddl_action_name = config.pddl_action_name
        super().__init__(**kwargs)

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return WasPrevActionInvalid.cls_uuid

    def reset_metric(self, *args, task, **kwargs):
        self.update_metric(*args, task=task, **kwargs)

    def update_metric(self, *args, task, **kwargs):
        self._metric = task.actions[self._pddl_action_name].was_prev_action_invalid


@registry.register_measure
class PrevActionName(Measure):
    cls_uuid = "action"

    def __init__(self, config, *args, **kwargs):
        self._pddl_action_name = config.pddl_action_name
        super().__init__(**kwargs)

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return PrevActionName.cls_uuid

    def reset_metric(self, *args, task, **kwargs):
        self.update_metric(*args, task=task, **kwargs)

    def update_metric(self, *args, task, **kwargs):
        if task.actions[self._pddl_action_name]._prev_action is None:
            self._metric = ""
        else:
            self._metric = task.actions[self._pddl_action_name]._prev_action.compact_str


@registry.register_measure
class NumInvalidActions(Measure):
    @staticmethod
    def _get_uuid(*args, **kwargs):
        return "num_invalid_actions"

    def reset_metric(self, *args, task, **kwargs):
        task.measurements.check_measure_dependencies(
            self.uuid, [WasPrevActionInvalid._get_uuid()]
        )
        self._metric = 0
        self.update_metric(*args, task=task, **kwargs)

    def update_metric(self, *args, task, **kwargs):
        was_prev_invalid = task.measurements.measures[
            WasPrevActionInvalid._get_uuid()
        ].get_metric()
        self._metric += int(was_prev_invalid)


@registry.register_measure
class SubgoalReward(Measure):
    def __init__(self, config, *args, **kwargs):
        super().__init__(**kwargs)
        self._config = config
        self.invalid_ac_pen = config.invalid_ac_pen
        self.progress_reward_factor = config.progress_reward_factor

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return "subgoal_reward"

    def reset_metric(self, *args, task, **kwargs):
        task.measurements.check_measure_dependencies(
            self.uuid,
            [TaskProgressMonitor._get_uuid(), WasPrevActionInvalid._get_uuid()],
        )
        self._prev_progress = 0.0
        self.update_metric(*args, task=task, **kwargs)

    def update_metric(self, *args, task, **kwargs):
        self._metric = 0.0
        was_prev_invalid = task.measurements.measures[
            WasPrevActionInvalid._get_uuid()
        ].get_metric()

        progress = task.measurements.measures[
            TaskProgressMonitor._get_uuid()
        ].get_metric()
        if was_prev_invalid:
            self._metric -= self.invalid_ac_pen

        self._metric = (progress - self._prev_progress) * self.progress_reward_factor
        self._prev_progress = progress


@registry.register_measure
class PredicateTaskSuccess(Measure):
    def __init__(self, config, *args, **kwargs):
        super().__init__(**kwargs)
        self._must_call_stop = config.must_call_stop
        self._allows_invalid_actions = config.allows_invalid_actions
        self._sanity_end_task = config.sanity_end_task
        if self._sanity_end_task:
            print("SANITY END TASK SET TO TRUE. TASK IS ONLY IN DEBUG MODE!!!")

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return "predicate_task_success"

    def reset_metric(self, *args, task, **kwargs):
        if self._must_call_stop:
            task.measurements.check_measure_dependencies(
                self.uuid, [DoesWantTerminate.cls_uuid]
            )
        if not self._allows_invalid_actions:
            task.measurements.check_measure_dependencies(
                self.uuid, [WasPrevActionInvalid.cls_uuid]
            )

        self.update_metric(*args, task=task, **kwargs)

    def update_metric(self, *args, task, **kwargs):
        self._metric = task.is_goal_satisfied()
        if not self._allows_invalid_actions:
            was_prev_action_invalid = task.measurements.measures[
                WasPrevActionInvalid.cls_uuid
            ].get_metric()
            if was_prev_action_invalid:
                task.should_end = True
                self._metric = False
                return

        if self._must_call_stop:
            does_action_want_stop = task.measurements.measures[
                DoesWantTerminate.cls_uuid
            ].get_metric()
            self._metric = self._metric and does_action_want_stop

            if does_action_want_stop:
                task.should_end = True

        if self._sanity_end_task and np.random.random() < 0.2:
            self._metric = True


@registry.register_measure
class WaitPredicateTaskSuccess(Measure):
    def __init__(self, config, *args, **kwargs):
        super().__init__(**kwargs)
        self._must_call_stop = config.must_call_stop
        self._allows_invalid_actions = config.allows_invalid_actions
        self._wait_after_steps = config.wait_after_steps
        self._steps_after_succ = 0

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return "predicate_task_success"

    def reset_metric(self, *args, task, **kwargs):
        self._steps_after_succ = 0
        if self._must_call_stop:
            task.measurements.check_measure_dependencies(
                self.uuid, [DoesWantTerminate.cls_uuid]
            )
        if not self._allows_invalid_actions:
            task.measurements.check_measure_dependencies(
                self.uuid, [WasPrevActionInvalid.cls_uuid]
            )

        self.update_metric(*args, task=task, **kwargs)

    def update_metric(self, *args, task, **kwargs):
        self._metric = task.is_goal_satisfied()
        if not self._allows_invalid_actions:
            was_prev_action_invalid = task.measurements.measures[
                WasPrevActionInvalid.cls_uuid
            ].get_metric()
            if was_prev_action_invalid:
                task.should_end = True
                self._metric = False
                return

        if self._must_call_stop:
            does_action_want_stop = task.measurements.measures[
                DoesWantTerminate.cls_uuid
            ].get_metric()
            self._metric = self._metric and does_action_want_stop

            if does_action_want_stop:
                task.should_end = True

        if self._metric:
            self._steps_after_succ += 1
            if self._wait_after_steps > self._steps_after_succ:
                self._metric = False


@registry.register_measure
class TaskCondSuccess(Measure):
    def __init__(self, config, *args, **kwargs):
        super().__init__(**kwargs)
        self._config = config

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return "cond_success"

    def reset_metric(self, *args, task, **kwargs):
        task.measurements.check_measure_dependencies(
            self.uuid, [PredicateTaskSuccess._get_uuid()]
        )
        self.update_metric(*args, task=task, **kwargs)

    def update_metric(self, *args, task, **kwargs):
        self._metric = task.is_goal_satisfied()
        # Get the predicate task success measure
        pred_task_success = task.measurements.measures[
            PredicateTaskSuccess._get_uuid()
        ].get_metric()

        sampled_target = task.get_sampled()[0]
        self._metric = {sampled_target.expr_type.name: float(pred_task_success)}


@registry.register_measure
class EEToClosestTargetDist(Measure):
    """
    Gets the distance between the EE and the closest target object.
    """

    def __init__(self, sim, config, *args, **kwargs):
        self._sim = sim
        self._config = config
        super().__init__(**kwargs)

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return "ee_obj_dist"

    def reset_metric(self, *args, episode, **kwargs):
        self.update_metric(*args, episode=episode, **kwargs)

    def update_metric(self, *args, task, **kwargs):
        targ_type = task.get_sampled()[0]
        matches = list(task.pddl.find_entities(targ_type.expr_type))
        obj_pos = np.array(
            [task.pddl.sim_info.get_entity_pos(match) for match in matches]
        )

        ee_pos = self._sim.articulated_agent.ee_transform().translation

        dists = np.linalg.norm(obj_pos - ee_pos, ord=2, axis=-1)
        closest_id = np.argmin(dists)
        self.closest_entity = matches[closest_id]
        self._metric = dists[closest_id]


@registry.register_measure
class ObjPlaceReward(RearrangeReward):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return "obj_place_reward"

    def reset_metric(self, *args, **kwargs):
        super().reset_metric(*args, **kwargs)

    def update_metric(self, *args, **kwargs):
        super().update_metric(*args, **kwargs)
        # Using a sparse reward for now
        self._metric = 0.0


@registry.register_measure
class ObjNavDistToGoal(Measure):
    cls_uuid = "obj_nav_dist"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return ObjNavDistToGoal.cls_uuid

    def reset_metric(self, *args, **kwargs):
        self.update_metric(*args, **kwargs)

    def update_metric(self, *args, task, **kwargs):
        cur_target = task.get_sampled()[0]
        target_pos = task.pddl.sim_info.get_entity_pos(cur_target)
        agent_pos = np.array(task._sim.articulated_agent.base_pos)[[0, 2]]
        self._metric = np.linalg.norm(agent_pos - target_pos[[0, 2]])


@registry.register_measure
class ObjNavReward(RearrangeReward):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return "obj_nav_reward"

    def reset_metric(self, *args, **kwargs):
        self._prev_dist = -1.0
        super().reset_metric(*args, **kwargs)

    def update_metric(self, *args, task, **kwargs):
        super().update_metric(*args, task=task, **kwargs)
        cur_dist = task.measurements.measures[ObjNavDistToGoal.cls_uuid].get_metric()

        # Using a sparse reward for now
        self._metric = 0.0
        if self._prev_dist < 0.0:
            dist_diff = 0.0
        else:
            dist_diff = self._prev_dist - cur_dist

        self._metric += self._config.dist_reward * dist_diff
        self._prev_dist = cur_dist


@registry.register_measure
class ZeroMeasure(Measure):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return "zero_measure"

    def reset_metric(self, *args, **kwargs):
        self.update_metric(*args, **kwargs)

    def update_metric(self, *args, **kwargs):
        self._metric = 0.0


@registry.register_measure
class OneMeasure(Measure):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return "one_measure"

    def reset_metric(self, *args, **kwargs):
        self.update_metric(*args, **kwargs)

    def update_metric(self, *args, **kwargs):
        self._metric = 1.0


@registry.register_measure
class BreakdownMeasure(Measure):
    def __init__(self, *args, config, **kwargs):
        super().__init__(*args, config, **kwargs)
        self._other_measures = config.other_measures
        self._log_counts = config.log_counts
        if self._log_counts:
            self._all_instruct_ids = get_all_instruct_ids()

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return "breakdown"

    def reset_metric(self, *args, task, **kwargs):
        task.measurements.check_measure_dependencies(
            self.uuid,
            self._other_measures,
        )
        use_instruct_id = self._parse_instruct_id(task.instruct_id)
        if self._log_counts:
            self._cur_counts = {k: 0 for k in self._all_instruct_ids}
            self._cur_counts[use_instruct_id] = 1
        self.update_metric(*args, task=task, **kwargs)

    def _parse_instruct_id(self, instruct_id):
        return instruct_id.split("_")[0]

    def update_metric(self, *args, task, **kwargs):
        self._metric = {}
        use_instruct_id = self._parse_instruct_id(task.instruct_id)
        for name in self._other_measures:
            self._metric[name] = {
                use_instruct_id: task.measurements.measures[name].get_metric()
            }
        if self._log_counts:
            self._metric["count"] = self._cur_counts


def _extract_pred_name(expr, i):
    if isinstance(expr, Predicate):
        return f"{expr.name}|{i}"
    else:
        return f"{expr.sub_exprs[0].sub_exprs[0].name}|{i}"


@registry.register_measure
class PddlGoalDistanceMeasure(Measure):
    @staticmethod
    def _get_uuid(*args, **kwargs):
        return "pddl_subgoal_distances"

    def reset_metric(self, *args, task, **kwargs):
        self.update_metric(
            *args,
            task=task,
            **kwargs,
        )

    def _get_pred_distance(self, pred, sim_info):
        if pred.name == "on_top":
            obj = pred._arg_values[0]
            recep = pred._arg_values[1]
            # Get the distance from the object COM to the surface bounding box COM.
            recep_bb = sim_info.search_for_entity(recep)

            rom = sim_info.sim.get_rigid_object_manager()
            obj_idx = sim_info.search_for_entity(obj)
            abs_obj_id = sim_info.sim.scene_obj_ids[obj_idx]
            entity_obj = rom.get_object_by_id(abs_obj_id)

            return np.linalg.norm(entity_obj.translation - recep_bb.center(), ord=2)
        elif pred.name == "holding":
            obj = pred._arg_values[0]
            obj_pos = sim_info.get_entity_pos(obj)
            ee_pos = sim_info.sim.articulated_agent.ee_transform().translation

            return np.linalg.norm(obj_pos - ee_pos, ord=2)

        # no sense of distance.
        return 0.0

    def update_metric(self, *args, task, **kwargs):
        assert task.goal_expr.expr_type == LogicalExprType.AND
        self._metric = {}

        for i, expr in enumerate(task.goal_expr.sub_exprs):
            if isinstance(expr, Predicate):
                dist = self._get_pred_distance(expr, task.pddl.sim_info)
            else:
                assert expr.expr_type == LogicalExprType.OR
                dist = None
                for sub_expr in expr.sub_exprs:
                    assert len(sub_expr.sub_exprs) == 1
                    assert isinstance(sub_expr.sub_exprs[0], Predicate)
                    pred_dist = self._get_pred_distance(
                        sub_expr.sub_exprs[0], task.pddl.sim_info
                    )
                    if dist is None:
                        dist = pred_dist
                    else:
                        dist = min(pred_dist, dist)
                assert dist is not None

            expr_name = _extract_pred_name(expr, i)
            self._metric[expr_name] = dist


@registry.register_measure
class PddlGoalBreakdownMeasure(Measure):
    @staticmethod
    def _get_uuid(*args, **kwargs):
        return "pddl_subgoal_breakdown"

    def reset_metric(self, *args, task, **kwargs):
        self.update_metric(
            *args,
            task=task,
            **kwargs,
        )

    def update_metric(self, *args, task, **kwargs):
        assert task.goal_expr.expr_type == LogicalExprType.AND
        self._metric = {}

        for i, expr in enumerate(task.goal_expr.sub_exprs):
            expr_name = _extract_pred_name(expr, i)
            self._metric[expr_name] = expr.is_true(task.pddl.sim_info)


@registry.register_measure
class PddlDenseReward(RearrangeReward):
    def __init__(self, *args, config, **kwargs):
        super().__init__(*args, config=config, **kwargs)
        self._subgoal_rewards = config.subgoal_rewards
        self._subgoal_dist_scale = config.subgoal_dist_scale

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return "pddl_dense_reward"

    def reset_metric(self, *args, episode, task, observations, **kwargs):
        self._last_subgoal_state = {}
        self._last_dist_state = {}

        super().reset_metric(
            *args,
            episode=episode,
            task=task,
            observations=observations,
            **kwargs,
        )

    def update_metric(self, *args, task, **kwargs):
        super().update_metric(
            *args,
            task=task,
            **kwargs,
        )

        goal_breakdowns = task.measurements.measures[
            PddlGoalBreakdownMeasure._get_uuid()
        ].get_metric()
        goal_distances = task.measurements.measures[
            PddlGoalDistanceMeasure._get_uuid()
        ].get_metric()
        assert task.goal_expr.expr_type == LogicalExprType.AND
        for i, expr_name in enumerate(task.goal_expr.sub_exprs):
            expr_name = _extract_pred_name(expr_name, i)

            goal_distance = goal_distances[expr_name]
            goal_breakdown = goal_breakdowns[expr_name]

            # If 1st step.
            if expr_name not in self._last_subgoal_state:
                self._last_subgoal_state[expr_name] = goal_breakdown
            if expr_name not in self._last_dist_state:
                self._last_dist_state[expr_name] = goal_distance

            # Give subgoal reward
            if (not self._last_subgoal_state[expr_name]) and goal_breakdown:
                self._metric += self._subgoal_rewards.get(expr_name, 0.0)

            # Give distance reward
            dist_diff = self._last_dist_state[expr_name] - goal_distance
            # Filter out small fluctuations
            dist_diff = round(dist_diff, 3)
            self._metric += self._subgoal_dist_scale.get(expr_name, 0.0) * dist_diff

            # Set last step vars.
            self._last_subgoal_state[expr_name] = goal_breakdown
            self._last_dist_state[expr_name] = goal_distance


@registry.register_measure
class ObjPickReward(RearrangeReward):
    """
    Reward function for picking an object up.
    """

    def __init__(self, *args, sim, config, task, **kwargs):
        self.cur_dist = -1.0
        self._prev_picked = False
        self._metric = None
        self._dist_reward = config.dist_reward
        self._drop_pen = config.drop_pen
        self._use_diff = config.use_diff
        self._pick_reward = config.pick_reward
        self._wrong_pick_pen = config.wrong_pick_pen
        self._wrong_pick_should_end = config.wrong_pick_should_end
        self._drop_obj_should_end = config.drop_obj_should_end

        super().__init__(*args, sim=sim, config=config, task=task, **kwargs)

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return "obj_pick_reward"

    def reset_metric(self, *args, episode, task, observations, **kwargs):
        task.measurements.check_measure_dependencies(
            self.uuid,
            [
                EEToClosestTargetDist._get_uuid(),
                EndEffectorToRestDistance.cls_uuid,
            ],
        )
        self.cur_dist = -1.0
        self._prev_picked = self._sim.grasp_mgr.snap_idx is not None

        super().reset_metric(
            *args,
            episode=episode,
            task=task,
            observations=observations,
            **kwargs,
        )

    def update_metric(self, *args, episode, task, observations, **kwargs):
        super().update_metric(
            *args,
            episode=episode,
            task=task,
            observations=observations,
            **kwargs,
        )
        ee_to_closest_meas = task.measurements.measures[
            EEToClosestTargetDist._get_uuid()
        ]
        ee_to_object_distance = ee_to_closest_meas.get_metric()
        ee_to_rest_distance = task.measurements.measures[
            EndEffectorToRestDistance.cls_uuid
        ].get_metric()

        snapped_id = self._sim.grasp_mgr.snap_idx
        cur_picked = snapped_id is not None

        if cur_picked:
            dist_to_goal = ee_to_rest_distance
        else:
            dist_to_goal = ee_to_object_distance

        closest_name = ee_to_closest_meas.closest_entity.name
        rel_obj_idx = task.pddl.sim_info.obj_ids[closest_name]
        abs_targ_obj_idx = self._sim.scene_obj_ids[rel_obj_idx]

        did_pick = cur_picked and (not self._prev_picked)
        if did_pick:
            if snapped_id == abs_targ_obj_idx:
                self._metric += self._pick_reward
                # If we just transitioned to the next stage our current
                # distance is stale.
                self.cur_dist = -1
            else:
                # picked the wrong object
                self._metric -= self._wrong_pick_pen
                if self._wrong_pick_should_end:
                    self._task.should_end = True
                self._prev_picked = cur_picked
                self.cur_dist = -1
                return

        if self._use_diff:
            if self.cur_dist < 0:
                dist_diff = 0.0
            else:
                dist_diff = self.cur_dist - dist_to_goal

            # Filter out the small fluctuations
            dist_diff = round(dist_diff, 3)
            self._metric += self._dist_reward * dist_diff
        else:
            self._metric -= self._dist_reward * dist_to_goal
        self.cur_dist = dist_to_goal

        if not cur_picked and self._prev_picked:
            # Dropped the object
            self._metric -= self._drop_pen
            if self._drop_obj_should_end:
                # End the episode if it is not the correct object.
                self._task.should_end = True
            self._prev_picked = cur_picked
            self.cur_dist = -1
            return

        self._prev_picked = cur_picked
