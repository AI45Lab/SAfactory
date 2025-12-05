from typing import List
import numpy as np
import os
from amsolver.backend.task import Task
from pyrep.objects.shape import Shape
from pyrep.objects.proximity_sensor import ProximitySensor
from amsolver.backend.unit_tasks import VLM_Object
from amsolver.const import colors
from amsolver.backend.conditions import DetectedCondition
from amsolver.backend.spawn_boundary import SpawnBoundary
from amsolver.backend.task import Task
from amsolver.backend.utils import scale_object, select_color
from vlm.tasks.stack_cubes import StackCubes
from pyrep.const import ObjectType, PrimitiveShape

class StackCubesColor(StackCubes):

    def modified_init_episode(self, index: int) -> List[str]:
        for obj in self.cube_list:
            scale_factor = np.random.uniform(0.8, 1.2)
            relative_factor = scale_object(obj, scale_factor)
            if abs(relative_factor-1)>1e-2:
                local_grasp_pose = obj.manipulated_part.local_grasp
                local_grasp_pose[:, :3, 3] *= relative_factor
                obj.manipulated_part.local_grasp = local_grasp_pose

        # color_index = np.random.choice(len(colors), len(self.cube_list), replace=False)
        # for i, obj in enumerate(self.cube_list):
        #     Shape(obj.manipulated_part.visual).set_color(colors[color_index[i]][1])
        #     obj.manipulated_part.descriptions = f"the {colors[color_index[i]][0]} cube"
        color_names, rgbs = select_color(index, len(self.cube_list)-1, replace=False)
        for i, cube in enumerate(self.cube_list):
            Shape(cube.manipulated_part.visual).set_color(rgbs[i])
            cube.manipulated_part.descriptions = f"the {color_names[i]} {obj.manipulated_part.property['shape']}"
        return None

    def variation_count(self) -> int:
        # TODO: The number of variations for this task.
        return len(colors)
