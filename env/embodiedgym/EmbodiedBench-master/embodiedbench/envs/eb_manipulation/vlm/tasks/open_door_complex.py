from typing import List
import numpy as np
import os
from amsolver.backend.task import Task
from pyrep.objects.shape import Shape
from pyrep.objects.dummy import Dummy
from pyrep.objects.proximity_sensor import ProximitySensor
from amsolver.backend.unit_tasks import T0_ObtainControl, T2_MoveObjectConstraints, TargetSpace, VLM_Object
from amsolver.backend.conditions import JointCondition
from amsolver.const import colors, complex_door_list
from amsolver.backend.conditions import DetectedCondition
from amsolver.backend.spawn_boundary import SpawnBoundary
from pyrep.objects.joint import Joint

from vlm.tasks.open_door import OpenDoor

door_states = {
    "open":["Fully close", "Slightly close"],
    "close":["Fully open", "Slightly open"]
}
class OpenDoorComplex(OpenDoor):

    def init_task(self) -> None:
        model_dir = os.path.dirname(os.path.realpath(__file__)).replace("tasks","object_models/")
        model_path = model_dir+complex_door_list[0]['path']
        self.spawn_space = SpawnBoundary([Shape('workspace')])
        self.boundary_root_ori = Shape("boundary_root").get_orientation()
        self.attach_point = Dummy.create()
        self.attach_point.set_name("door_attach_point")
        self.attach_point.set_parent(Shape("boundary_root"))
        self.door = VLM_Object(self.pyrep, model_path, 0)
        root_pose = Shape("boundary_root").get_pose()
        root_pose[:2] = self.door.get_position()[:2]
        self.attach_point.set_pose(root_pose)
        self.door.set_parent(self.attach_point)
        self.door_joint = self.door.constraints[0]
        self.door_joint_name = self.door_joint.get_name()
        self.door_joint.set_joint_interval(False, [np.deg2rad(0), np.deg2rad(30)])
        self.handle_joint = self.door.constraints[1]
        self.door_unlock_cond = JointCondition(self.handle_joint, np.deg2rad(20))
        self.pyrep.step()
        self.temporary_waypoints = []
        self.task_base = self.get_base()

    def init_episode(self, index: int) -> List[str]:
        self.variation_index = index
        goal_angel, goal_description = self.door_setting(index)
        try_times = 0
        self.door_joint.set_motor_locked_at_zero_velocity(False)
        while len(self.temporary_waypoints)==0 and try_times<200:
            self.sample_method()
            init_states = self.get_state()[0]
        
            self.door.manipulated_part.descriptions = "the handle of door"
            door_target_description = "the door"
            if "Slightly" in goal_description:
                door_target_description += " slightly"
            door_target = TargetSpace(self.door_joint, None,
                                            np.deg2rad(goal_angel), np.deg2rad(goal_angel), door_target_description, self.door.manipulated_part.visual)
            door_target.set_target(self.door.parts[1])
            door_task = T2_MoveObjectConstraints(self.robot, self.pyrep, door_target, self.task_base, fail_times=2)
            post_grasp_task = door_task
            if not self.door_unlocked:
                handle_target = TargetSpace(self.handle_joint, None,
                                                np.deg2rad(25), np.deg2rad(25), "the handle of door", self.door.manipulated_part.visual)
                handle_target.set_target(self.door.manipulated_part)
                handle_task = T2_MoveObjectConstraints(self.robot, self.pyrep, handle_target, self.task_base, fail_times=2, next_task_fuc=door_task.get_path_with_constraints)
                post_grasp_task = handle_task
            grasp_task = T0_ObtainControl(self.robot, self.pyrep, self.door.manipulated_part, self.task_base, try_times=200,
                    need_post_grasp=False, grasp_sort_key="horizontal", next_task_fuc=post_grasp_task.get_path_with_constraints)
            if try_times>100:
                waypoints = grasp_task.get_path(try_ik_sampling=True, ignore_collisions=False)
            else:
                waypoints = grasp_task.get_path(try_ik_sampling=False, ignore_collisions=False)
            if waypoints is not None:
                self.temporary_waypoints += waypoints
            try_times += 1
            self.reset_robot()
            self.pyrep.set_configuration_tree(init_states)
            self.pyrep.step()
            for i,waypoint in enumerate(self.temporary_waypoints):
                waypoint.set_name('waypoint{}'.format(i))
        if not self.door_unlocked:
            self.door_joint.set_motor_locked_at_zero_velocity(True)
        return [f'{goal_description} the door.']

    def variation_count(self) -> int:
        # TODO: The number of variations for this task.
        return 4

    def step(self) -> None:
        # Called during each sim step. Remove this if not using.
        door_joint = Joint(self.door_joint_name)
        if not self.door_unlocked:
            door_joint.set_motor_locked_at_zero_velocity(True)
            self.door_unlocked = self.door_unlock_cond.condition_met()[0]
            if self.door_unlocked:
                door_joint.set_motor_locked_at_zero_velocity(False)

    def is_static_workspace(self) -> bool:
        return True
    
    def sample_method(self):
        Shape("boundary_root").set_orientation(self.boundary_root_ori)
        self.spawn_space.clear()
        self.spawn_space.sample(Shape("boundary_root"), 
            min_rotation=[0, 0, -3.14 / 8.], max_rotation=[0, 0, 3.14 / 8.])
        # SpawnBoundary([Shape('boundary_root')]).sample(self.attach_point, 
        #         min_rotation=[0, 0, 0], max_rotation=[0, 0, 0])

    def door_setting(self, index):
        if index in [0,1]:
            init_state = "open"
            self.door_joint.set_joint_position(np.deg2rad(30), True)
        else:
            init_state = "close"
            self.door_joint.set_joint_position(np.deg2rad(0), True)
        sub_index = index%2
        goal_state = door_states[init_state][sub_index]
        if init_state == "close":
            self.door_unlocked = False
        else:
            self.door_unlocked = True
        if "Slightly" in goal_state:
            detect_angle = 10
            goal_angel = 15
            detect_bound = 20
        else:
            detect_angle = 25
            detect_bound = 35
            if "open" in goal_state:
                goal_angel = 30
            else:
                goal_angel = 0
        self.register_success_conditions([JointCondition(self.door_joint, np.deg2rad(detect_angle), np.deg2rad(detect_bound))])
        return goal_angel, goal_state
