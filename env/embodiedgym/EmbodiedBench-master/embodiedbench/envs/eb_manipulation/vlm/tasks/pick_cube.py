from turtle import shape
from typing import List
import numpy as np
import os
import random
from pyrep.objects.shape import Shape
from pyrep.objects.dummy import Dummy
from pyrep.objects.proximity_sensor import ProximitySensor
from amsolver.backend.task import Task
from amsolver.backend.unit_tasks import T0_ObtainControl, T1_MoveObjectGoal, TargetSpace, VLM_Object
from amsolver.backend.utils import get_relative_position_xy
from amsolver.const import colors, object_shapes
from amsolver.backend.conditions import DetectedCondition
from amsolver.backend.spawn_boundary import SpawnBoundary
        
class PickCube(Task):
    def init_task(self) -> None:
        self.spawn_space = SpawnBoundary([Shape('workspace')])
        self.target_space0 = TargetSpace(SpawnBoundary([Shape('target_space0')]), ProximitySensor('success0'),
                                        (-3.14,-3.14,-3.14), (3.14,3.14,3.14), None, Shape("small_container0").get_handle())
        self.target_space1 = TargetSpace(SpawnBoundary([Shape('target_space1')]), ProximitySensor('success1'),
                                        (-3.14,-3.14,-3.14), (3.14,3.14,3.14), None, Shape("small_container1").get_handle())
        self.target_spaces = [self.target_space0, self.target_space1]
        self.model_dir = os.path.dirname(os.path.realpath(__file__)).replace("tasks","object_models/")
        self.temporary_waypoints = []
        self.taks_base = self.get_base()
        if not hasattr(self, "model_num"):
            self.model_num = 2
        # self.import_objects(self.model_num)
    
    def init_episode(self, index: int) -> List[str]:
        self.import_objects(self.model_num)
        self.variation_index = index
        self.modified_init_episode(index)
        # ind = np.random.randint(0, len(self.target_spaces))
        self.target_space = self.target_spaces[0]
        self.manipulated_obj = self.object_list[0].manipulated_part
        try_times = 0
        while len(self.temporary_waypoints)==0:
            self.sample_method()
            for _ in range(5):
                self.pyrep.step()
            self.register_success_conditions([DetectedCondition(self.manipulated_obj, self.target_space.successor)])
            init_states = self.get_state()[0]
            self.target_space.set_target(self.manipulated_obj, try_ik_sampling=True, release=True)
            MoveTask0 = T1_MoveObjectGoal(self.robot, self.pyrep, self.target_space, self.taks_base, fail_times=10)
            GraspTask0 = T0_ObtainControl(self.robot, self.pyrep, self.manipulated_obj,self.taks_base,\
                        next_task_fuc=MoveTask0.get_path, try_times=20)
            if try_times<100:
                waypoints = GraspTask0.get_path(try_ik_sampling=False)
            else:
                waypoints = GraspTask0.get_path(try_ik_sampling=True)
            if waypoints is not None:
                self.temporary_waypoints += waypoints
            self.reset_robot()
            self.pyrep.set_configuration_tree(init_states)
            try_times+= 1
        for i,waypoint in enumerate(self.temporary_waypoints):
            waypoint.set_name('waypoint{}'.format(i))
        description = 'Pick up {} and place it into {}.'.format(self.manipulated_obj.descriptions, self.target_space.target_space_descriptions)
        return [description]
    
    def is_static_workspace(self) -> bool:
        return True

    def load(self, ttms_folder=None):
        if Shape.exists('pick_cube'):
            return Dummy('pick_cube')
        ttm_file = os.path.join(ttms_folder, 'pick_cube.ttm')
        self._base_object = self.pyrep.import_model(ttm_file)
        return self._base_object
    
    def get_base(self) -> Dummy:
        self._base_object = Dummy('pick_cube')
        return self._base_object
    
    def import_objects(self, num=2):
        self.object_list = []
        if not hasattr(self, "model_path"):
            selected_obj = random.choice(list(object_shapes.keys()))
            model_path = self.model_dir+object_shapes[selected_obj]['path']
            # model_path = self.model_dir+"cube/cube_normal/cube_normal.ttm"
        else:
            model_path = self.model_dir+self.model_path
        for i in range(num):
            cube = VLM_Object(self.pyrep, model_path, i)
            cube.set_parent(self.taks_base)
            self.object_list.append(cube)
        self.register_graspable_objects(self.object_list)
        self._need_remove_objects+=self.object_list
    
    def sample_method(self):
        satisfied = False
        while not satisfied:
            self.spawn_space.clear()
            for space in self.target_spaces:
                self.spawn_space.sample(Shape(space.focus_obj_id), min_distance=0.25)
            for obj in self.object_list:
                self.spawn_space.sample(obj, min_distance=0.1)
            satisfied = True
            self.pyrep.step()