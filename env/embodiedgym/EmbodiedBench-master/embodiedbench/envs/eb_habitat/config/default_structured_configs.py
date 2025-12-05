"""
Contains the structured config definitions.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import attr
from habitat.config.default_structured_configs import (
    ActionConfig, ArmActionConfig, HabitatSimRGBSensorConfig, LabSensorConfig,
    MeasurementConfig, TaskConfig)
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

cs = ConfigStore.instance()


##########################################################################
# Tasks
##########################################################################
@dataclass
class RearrangePredTaskConfig(TaskConfig):
    start_template: Optional[List[str]] = MISSING
    goal_template: Optional[Dict[str, Any]] = MISSING
    sample_entities: Dict[str, Any] = MISSING
    # Filters out instructions from the trainer dataset.
    filter_instructs: Optional[List[str]] = None
    # How many episodes per env worker to limit to.
    filter_down_num: Optional[int] = None
    force_scene_per_worker: bool = False
    fix_agent_pos: bool = False
    # Tokenizer used for a variety of sensors and measurements.
    tokenizer_name: str = "google/flan-t5-small"
    skip_load_receps: bool = False
    force_gym_option: str = "nav_sanity"
    force_gym_overrides: Any = None
    gym_options: Any = None
    # If true, this will only log the shared keys between all environments.
    gym_safe_mode: bool = False
    # If true, this will fix the action space to be the same dimension for all tasks.
    fix_action_space: bool = False
    # If true, will disable adding any prompt to the instruction.
    no_prompt: bool = False
    # If False, this will disable language action space for discrete action
    # envs.
    use_lang_actions: bool = True
    sample_entities_use_constant_sampling: bool = False


@dataclass
class CustomArmActionConfig(ArmActionConfig):
    prevent_gripper_change_duration: int = 0


@dataclass
class CustomArmAbsJointActionConfig(ActionConfig):
    type: str = "CustomArmAbsJointAction"
    arm_joint_dimensionality: int = 7


@dataclass
class PddlHlActionConfig(ActionConfig):
    type: str = "PddlHlAction"
    allowed_actions: List[str] = field(default_factory=list)


cs.store(
    package="habitat.task",
    group="habitat/task",
    name="llarp_task_config_base",
    node=RearrangePredTaskConfig,
)
cs.store(
    package="habitat.task.actions.arm_joint_action",
    group="habitat/task/actions",
    name="arm_joint_action",
    node=CustomArmAbsJointActionConfig,
)
cs.store(
    package="habitat.task.actions.pddl_hl_action",
    group="habitat/task/actions",
    name="pddl_hl_action",
    node=PddlHlActionConfig,
)
cs.store(
    package="habitat.task.actions.arm_action",
    group="habitat/task/actions",
    name="arm_action",
    node=CustomArmActionConfig,
)

##########################################################################


##########################################################################
# Sensors
##########################################################################
@dataclass
class OneHotTargetSensorConfig(LabSensorConfig):
    type: str = "OneHotTargetSensor"
    sampled_idx: int = 0


@dataclass
class AllObjectPositionsSensorConfig(LabSensorConfig):
    type: str = "AllObjectPositionsSensor"


@dataclass
class StepCountSensorConfig(LabSensorConfig):
    type: str = "StepCountSensor"


@dataclass
class WindowDebugSensorConfig(LabSensorConfig):
    type: str = "WindowDebugSensor"


@dataclass
class ClosestTargetObjectPosSensorConfig(LabSensorConfig):
    type: str = "ClosestTargetObjectPosSensor"


@dataclass
class VocabLangGoalSensorConfig(LabSensorConfig):
    type: str = "VocabLangGoalSensor"
    max_len: int = 30
    add_special_tokens: bool = True


@dataclass
class SimpleTargetSensorConfig(LabSensorConfig):
    type: str = "SimpleTargetSensor"
    # Without the special start token, this is 4.
    max_len: int = 5
    add_special_tokens: bool = True


@dataclass
class LlamaVocabLangGoalSensorConfig(LabSensorConfig):
    type: str = "LlamaVocabLangGoalSensor"
    max_len: int = 30
    tokenizer_name: str = "data/hf_llama_7B/"


@dataclass
class T5VocabLangGoalSensorConfig(LabSensorConfig):
    type: str = "T5VocabLangGoalSensor"
    max_len: int = 30


@dataclass
class ObsLangSensorConfig(LabSensorConfig):
    type: str = "ObsLangSensor"
    max_len: int = 30


@dataclass
class VocabEmbedSensorConfig(LabSensorConfig):
    type: str = "VocabEmbedSensor"
    hidden_dim: int = 512
    embed_path: str = "minidata/llm_hxs.pt"


@dataclass
class TopDownRGBSensorConfig(HabitatSimRGBSensorConfig):
    uuid: str = "top_down_rgb"
    width: int = 1024
    height: int = 1024


@dataclass
class ThirdRGBSensorConfig(HabitatSimRGBSensorConfig):
    type: str = "HabitatSimRGBSensor"
    uuid: str = "third_rgb"
    width: int = 500 
    height: int = 500 


cs.store(
    group="habitat/simulator/sim_sensors",
    name="third_rgb_sensor",
    node=ThirdRGBSensorConfig,
)

cs.store(
    group="habitat/simulator/sim_sensors",
    name="top_down_rgb",
    node=TopDownRGBSensorConfig,
)
cs.store(
    package="habitat.task.lab_sensors.one_hot_target_sensor",
    group="habitat/task/lab_sensors",
    name="one_hot_target_sensor",
    node=OneHotTargetSensorConfig,
)
cs.store(
    package="habitat.task.lab_sensors.all_obj_pos_sensor",
    group="habitat/task/lab_sensors",
    name="all_obj_pos_sensor",
    node=AllObjectPositionsSensorConfig,
)
cs.store(
    package="habitat.task.lab_sensors.step_count",
    group="habitat/task/lab_sensors",
    name="step_count",
    node=StepCountSensorConfig,
)
cs.store(
    package="habitat.task.lab_sensors.debug_info",
    group="habitat/task/lab_sensors",
    name="debug_info",
    node=WindowDebugSensorConfig,
)
cs.store(
    package="habitat.task.lab_sensors.closest_targ_obj_pos_sensor",
    group="habitat/task/lab_sensors",
    name="closest_targ_obj_pos_sensor",
    node=ClosestTargetObjectPosSensorConfig,
)
cs.store(
    package="habitat.task.lab_sensors.vocab_lang_goal",
    group="habitat/task/lab_sensors",
    name="vocab_lang_goal",
    node=VocabLangGoalSensorConfig,
)
cs.store(
    package="habitat.task.lab_sensors.simple_lang_goal",
    group="habitat/task/lab_sensors",
    name="simple_lang_goal",
    node=SimpleTargetSensorConfig,
)
cs.store(
    package="habitat.task.lab_sensors.llama_vocab_lang_goal",
    group="habitat/task/lab_sensors",
    name="llama_vocab_lang_goal",
    node=LlamaVocabLangGoalSensorConfig,
)
cs.store(
    package="habitat.task.lab_sensors.t5_vocab_lang_goal",
    group="habitat/task/lab_sensors",
    name="t5_vocab_lang_goal",
    node=T5VocabLangGoalSensorConfig,
)
cs.store(
    package="habitat.task.lab_sensors.obs_lang",
    group="habitat/task/lab_sensors",
    name="obs_lang",
    node=ObsLangSensorConfig,
)
cs.store(
    package="habitat.task.lab_sensors.vocab_embed_sensor",
    group="habitat/task/lab_sensors",
    name="vocab_embed_sensor",
    node=VocabEmbedSensorConfig,
)
##########################################################################


##########################################################################
# Measures
##########################################################################
@dataclass
class PredicateTaskSuccessConfig(MeasurementConfig):
    type: str = "PredicateTaskSuccess"
    must_call_stop: bool = True
    allows_invalid_actions: bool = True
    sanity_end_task: bool = False


@dataclass
class WaitPredicateTaskSuccessConfig(MeasurementConfig):
    type: str = "WaitPredicateTaskSuccess"
    must_call_stop: bool = True
    allows_invalid_actions: bool = True
    wait_after_steps: int = 0


@dataclass
class SimpleLangGoalConfig(MeasurementConfig):
    type: str = "SimpleLangGoalMeasure"


@dataclass
class ZeroMeasureConfig(MeasurementConfig):
    type: str = "ZeroMeasure"


@dataclass
class OneMeasureConfig(MeasurementConfig):
    type: str = "OneMeasure"


@dataclass
class TaskCondSuccessConfig(MeasurementConfig):
    type: str = "TaskCondSuccess"


@dataclass
class TargetNameMeasureConfig(MeasurementConfig):
    type: str = "TargetNameMeasure"


@dataclass
class LangGoalConfig(MeasurementConfig):
    type: str = "LangGoalMeasure"


@dataclass
class WasPrevActionInvalidConfig(MeasurementConfig):
    type: str = "WasPrevActionInvalid"
    pddl_action_name: str = "pddl_apply_action"


@dataclass
class PrevActionNameConfig(MeasurementConfig):
    type: str = "PrevActionName"
    pddl_action_name: str = "pddl_apply_action"


@dataclass
class NumInvalidActionsConfig(MeasurementConfig):
    type: str = "NumInvalidActions"


@dataclass
class EEToClosestTargetDistConfig(MeasurementConfig):
    type: str = "EEToClosestTargetDist"


@dataclass
class TaskProgressMonitor(MeasurementConfig):
    type: str = "TaskProgressMonitor"


@dataclass
class SubgoalRewardConfig(MeasurementConfig):
    type: str = "SubgoalReward"
    invalid_ac_pen: float = 0.01
    progress_reward_factor: float = 1.0


@dataclass
class BreakdownMeasureConfig(MeasurementConfig):
    type: str = "BreakdownMeasure"
    other_measures: List[str] = field(default_factory=list)
    log_counts: bool = True


@dataclass
class ObjPickRewardConfig(MeasurementConfig):
    type: str = "ObjPickReward"
    dist_reward: float = 2.0
    pick_reward: float = 2.0
    constraint_violate_pen: float = 1.0
    drop_pen: float = 0.5
    wrong_pick_pen: float = 0.5
    force_pen: float = 0.0001
    max_force_pen: float = 0.01
    force_end_pen: float = 1.0
    use_diff: bool = True
    drop_obj_should_end: bool = True
    wrong_pick_should_end: bool = True


@dataclass
class PddlGoalBreakdownConfig(MeasurementConfig):
    type: str = "PddlGoalBreakdownMeasure"


@dataclass
class PddlGoalDistanceConfig(MeasurementConfig):
    type: str = "PddlGoalDistanceMeasure"


@dataclass
class PddlDenseRewardConfig(MeasurementConfig):
    type: str = "PddlDenseReward"
    constraint_violate_pen: float = 1.0
    force_pen: float = 0.0
    max_force_pen: float = 0.0
    force_end_pen: float = 1.0
    subgoal_dist_scale: Dict[str, float] = field(default_factory=dict)
    subgoal_rewards: Dict[str, float] = field(default_factory=dict)


@dataclass
class ObjPlaceRewardConfig(MeasurementConfig):
    type: str = "ObjPlaceReward"
    constraint_violate_pen: float = 1.0
    force_pen: float = 0.0001
    max_force_pen: float = 0.01
    force_end_pen: float = 1.0


@dataclass
class ObjNavDistToGoalConfig(MeasurementConfig):
    type: str = "ObjNavDistToGoal"


@dataclass
class ObjNavRewardConfig(MeasurementConfig):
    type: str = "ObjNavReward"
    constraint_violate_pen: float = 1.0
    force_pen: float = 0.0001
    max_force_pen: float = 0.01
    force_end_pen: float = 1.0
    dist_reward: float = 1.0


cs.store(
    package="habitat.task.measurements.ee_obj_dist",
    group="habitat/task/measurements",
    name="ee_obj_dist",
    node=EEToClosestTargetDistConfig,
)
cs.store(
    package="habitat.task.measurements.task_progress",
    group="habitat/task/measurements",
    name="task_progress",
    node=TaskProgressMonitor,
)
cs.store(
    package="habitat.task.measurements.breakdown",
    group="habitat/task/measurements",
    name="breakdown",
    node=BreakdownMeasureConfig,
)


cs.store(
    package="habitat.task.measurements.subgoal_reward",
    group="habitat/task/measurements",
    name="subgoal_reward",
    node=SubgoalRewardConfig,
)

cs.store(
    package="habitat.task.measurements.pddl_dense_reward",
    group="habitat/task/measurements",
    name="pddl_dense_reward",
    node=PddlDenseRewardConfig,
)
cs.store(
    package="habitat.task.measurements.pddl_subgoal_breakdown",
    group="habitat/task/measurements",
    name="pddl_subgoal_breakdown",
    node=PddlGoalBreakdownConfig,
)
cs.store(
    package="habitat.task.measurements.pddl_subgoal_distances",
    group="habitat/task/measurements",
    name="pddl_subgoal_distances",
    node=PddlGoalDistanceConfig,
)


cs.store(
    package="habitat.task.measurements.obj_pick_reward",
    group="habitat/task/measurements",
    name="obj_pick_reward",
    node=ObjPickRewardConfig,
)

cs.store(
    package="habitat.task.measurements.obj_place_reward",
    group="habitat/task/measurements",
    name="obj_place_reward",
    node=ObjPlaceRewardConfig,
)

cs.store(
    package="habitat.task.measurements.simple_lang_goal",
    group="habitat/task/measurements",
    name="simple_lang_goal",
    node=SimpleLangGoalConfig,
)

cs.store(
    package="habitat.task.measurements.zero_measure",
    group="habitat/task/measurements",
    name="zero_measure",
    node=ZeroMeasureConfig,
)

cs.store(
    package="habitat.task.measurements.one_measure",
    group="habitat/task/measurements",
    name="one_measure",
    node=OneMeasureConfig,
)

cs.store(
    package="habitat.task.measurements.obj_nav_dist_to_goal",
    group="habitat/task/measurements",
    name="obj_nav_dist_to_goal",
    node=ObjNavDistToGoalConfig,
)

cs.store(
    package="habitat.task.measurements.obj_nav_reward",
    group="habitat/task/measurements",
    name="obj_nav_reward",
    node=ObjNavRewardConfig,
)

cs.store(
    package="habitat.task.measurements.target_name",
    group="habitat/task/measurements",
    name="target_name",
    node=TargetNameMeasureConfig,
)
cs.store(
    package="habitat.task.measurements.was_prev_action_invalid",
    group="habitat/task/measurements",
    name="was_prev_action_invalid",
    node=WasPrevActionInvalidConfig,
)
cs.store(
    package="habitat.task.measurements.prev_action_name",
    group="habitat/task/measurements",
    name="prev_action_name",
    node=PrevActionNameConfig,
)
cs.store(
    package="habitat.task.measurements.num_invalid_actions",
    group="habitat/task/measurements",
    name="num_invalid_actions",
    node=NumInvalidActionsConfig,
)
cs.store(
    package="habitat.task.measurements.lang_goal",
    group="habitat/task/measurements",
    name="lang_goal",
    node=LangGoalConfig,
)
cs.store(
    package="habitat.task.measurements.predicate_task_success",
    group="habitat/task/measurements",
    name="predicate_task_success",
    node=PredicateTaskSuccessConfig,
)
cs.store(
    package="habitat.task.measurements.wait_predicate_task_success",
    group="habitat/task/measurements",
    name="wait_predicate_task_success",
    node=WaitPredicateTaskSuccessConfig,
)

cs.store(
    package="habitat.task.measurements.task_cond_success",
    group="habitat/task/measurements",
    name="task_cond_success",
    node=TaskCondSuccessConfig,
)
##########################################################################
