import random
from typing import List
import numpy as np
import os
from amsolver.backend.task import Task
from pyrep.objects.shape import Shape
from pyrep.objects.dummy import Dummy
from pyrep.objects.proximity_sensor import ProximitySensor
from amsolver.backend.unit_tasks import T0_ObtainControl, T2_MoveObjectConstraints, TargetSpace, VLM_Object
from amsolver.backend.conditions import JointCondition
from amsolver.const import door_list
from amsolver.backend.conditions import DetectedCondition
from amsolver.backend.spawn_boundary import SpawnBoundary
from pyrep.objects.joint import Joint

door_states = {
    "open":["Fully close", "Slightly close"],
    "close":["Fully open", "Slightly open"]
}
class OpenDoor(Task):

    def init_task(self) -> None:
        self.model_dir = os.path.dirname(os.path.realpath(__file__)).replace("tasks","object_models/")

        self.spawn_space = SpawnBoundary([Shape('workspace')])
        self.boundary_root_ori = Shape("boundary_root").get_orientation()
        
        self.temporary_waypoints = []
        self.task_base = self.get_base()
        if not hasattr(self, "_ignore_collisions"):
            self._ignore_collisions = False

    def init_episode(self, index: int) -> List[str]:
        self.import_objects()
        self.variation_index = index
        goal_angel, goal_description = self.door_setting(index)
        try_times = 0
        while len(self.temporary_waypoints)==0 and try_times<100:
            self.sample_method()
            init_states = self.get_state()[0]
        
            self.door.manipulated_part.descriptions = f"the handle of the {self.door.manipulated_part.property['shape']}"
            door_target_description = f"the {self.door.manipulated_part.property['shape']}"
            if "Slightly" in goal_description:
                door_target_description += " slightly"
            door_target = TargetSpace(self.door_joint, None,
                                                np.deg2rad(goal_angel), np.deg2rad(goal_angel), door_target_description, self.door.manipulated_part.visual)
            door_target.set_target(self.door.parts[1])
            door_task = T2_MoveObjectConstraints(self.robot, self.pyrep, door_target, self.task_base, fail_times=2)
            post_grasp_task = door_task
            grasp_task = T0_ObtainControl(self.robot, self.pyrep, self.door.manipulated_part, self.task_base, try_times=20,
                    need_post_grasp=False, grasp_sort_key="horizontal", next_task_fuc=post_grasp_task.get_path_with_constraints)
            if try_times>50:
                waypoints = grasp_task.get_path(try_ik_sampling=True, ignore_collisions=self._ignore_collisions)
            else:
                waypoints = grasp_task.get_path(try_ik_sampling=False, ignore_collisions=self._ignore_collisions)
            if waypoints is not None:
                self.temporary_waypoints += waypoints
            try_times += 1
            self.reset_robot()
            self.pyrep.set_configuration_tree(init_states)
            self.pyrep.step()
            for i,waypoint in enumerate(self.temporary_waypoints):
                waypoint.set_name('waypoint{}'.format(i))
        return [f"{goal_description} the {self.door.manipulated_part.property['shape']}."]

    def import_objects(self):
        self._selected_door = random.choice(door_list)
        model_path = self.model_dir+self._selected_door["path"]
        self.door = VLM_Object(self.pyrep, model_path, 0)
        self.door.set_parent(Shape("boundary_root"))
        self.door_joint = self.door.constraints[0]
        self.door_joint_name = self.door_joint.get_name()
        self._need_remove_objects.append(self.door)

    def variation_count(self) -> int:
        # TODO: The number of variations for this task.
        return 4

    def is_static_workspace(self) -> bool:
        return True
    
    def sample_method(self):
        Shape("boundary_root").set_orientation(self.boundary_root_ori)
        self.spawn_space.clear()
        self.spawn_space.sample(Shape("boundary_root"), 
            min_rotation=[0, 0, -3.14 / 4.], max_rotation=[0, 0, 3.14 / 4.])
        # SpawnBoundary([Shape('boundary_root')]).sample(self.attach_point, 
        #         min_rotation=[0, 0, 0], max_rotation=[0, 0, 0])

    def door_setting(self, index):
        joint_range = self.door_joint.get_joint_interval()[1]
        real_max_angle = joint_range[0]+joint_range[1]
        real_max_angle = np.rad2deg(real_max_angle)
        try:
            max_angle = np.random.uniform(30, real_max_angle*0.8)
        except:
            max_angle = 30
        # max_angle = 30
        if index in [2,3]:
            init_state = "open"
            self.door_joint.set_joint_position(np.deg2rad(max_angle), True)
        else:
            init_state = "close"
            self.door_joint.set_joint_position(0, True)
        sub_index = index%2
        goal_state = door_states[init_state][sub_index]
        if "Slightly" in goal_state:
            detect_distance = max_angle*0.4
            goal_distance = max_angle*np.random.uniform(0.4, 0.5)
            detect_bound = max_angle*0.7
        else:
            detect_distance = max_angle*0.8
            detect_bound = max_angle*1.1
            if "open" in goal_state:
                goal_distance = max_angle*np.random.uniform(0.8, 1.0)
            else:
                goal_distance = np.random.uniform(joint_range[0], joint_range[0]+joint_range[1]*0.1)
        self.register_success_conditions([JointCondition(self.door_joint, np.deg2rad(detect_distance), np.deg2rad(detect_bound))])
        return goal_distance, goal_state

    def load(self, ttms_folder=None):
        if Shape.exists('open_door'):
            return Dummy('open_door')
        ttm_file = os.path.join(ttms_folder, 'open_door.ttm')
        self._base_object = self.pyrep.import_model(ttm_file)
        return self._base_object
    
    def get_base(self) -> Dummy:
        self._base_object = Dummy('open_door')
        return self._base_object