from typing import List
import numpy as np
import itertools
from amsolver.backend.task import Task
from pyrep.objects.shape import Shape
from pyrep.objects.proximity_sensor import ProximitySensor
from amsolver.backend.unit_tasks import VLM_Object
from amsolver.const import colors, object_shapes
from amsolver.backend.conditions import DetectedCondition
from amsolver.backend.spawn_boundary import SpawnBoundary
from amsolver.backend.task import Task
from amsolver.backend.utils import get_relative_position_xy, get_sorted_grasp_pose, scale_object, select_color
from vlm.tasks.stack_cubes import StackCubes
from pyrep.const import ObjectType, PrimitiveShape

relative_pos_list = list(itertools.product(["left", "right", "front", "rear"], repeat=2))
class StackCubesRelative(StackCubes):

    def init_task(self) -> None:
        self.model_num = 2
        self.class_num = 2
        return super().init_task()

    def modified_init_episode(self, index: int) -> List[str]:
        self.object_target_relative0, self.object_target_relative1 = relative_pos_list[index]
        self.select_obj0 = self.cube_list[:self.model_num]
        self.select_obj1 = self.cube_list[self.model_num:2*self.model_num]
        self.cube_list = [self.select_obj0[0], self.select_obj1[0]]
        self.cube_num = len(self.cube_list)
        color_index = np.random.choice(len(colors), len(self.cube_list), replace=True)
        for i, obj in enumerate(self.cube_list):
            Shape(obj.manipulated_part.visual).set_color(colors[color_index[i]][1])

        return None

    def is_static_workspace(self) -> bool:
        return True

    def variation_count(self) -> int:
        # TODO: The number of variations for this task.
        return len(relative_pos_list)
    
    def sample_method(self):
        satisfied = False
        while not satisfied:
            self.spawn_space.clear()
            for obj in self.select_obj0:
                self.spawn_space.sample(obj, min_distance=0.1)
            
            object_relative_pose = get_relative_position_xy(self.select_obj0[1], self.select_obj0[0], self.robot.arm)
            if object_relative_pose != self.object_target_relative0:
                continue
            obj = self.select_obj0[0].manipulated_part
            obj.descriptions = "the {} {}".format(object_relative_pose, obj.property["shape"]) 

            for obj in self.select_obj1:
                self.spawn_space.sample(obj, min_distance=0.1)
            object_relative_pose = get_relative_position_xy(self.select_obj1[1], self.select_obj1[0], self.robot.arm)
            if object_relative_pose != self.object_target_relative1:
                continue
            obj = self.select_obj1[0].manipulated_part
            obj.descriptions = "the {} {}".format(object_relative_pose, obj.property["shape"]) 
            satisfied = True
            self.pyrep.step()
