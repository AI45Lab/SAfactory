from typing import List
import numpy as np
from amsolver.backend.unit_tasks import VLM_Object, TargetSpace
from pyrep.objects.shape import Shape
from amsolver.const import colors, planes
from amsolver.backend.utils import scale_object, select_color
from vlm.tasks.wipe_table import WipeTable
from pyrep.const import ObjectType, PrimitiveShape

shape_list = list(planes.keys())
DIRT_POINTS = 50
class WipeTableShape(WipeTable):
    def init_task(self) -> None:
        self.area_num = 1
        self.area_class_num = len(shape_list)
        return super().init_task()
        
    def modified_init_episode(self, index: int):
        selected_shapes = shape_list[index]
        self.target_list = [self.shape_lib[selected_shapes][0]]
        other_shapes_index = list(range(len(shape_list)))
        other_shapes_index.remove(index)
        distractor_number = np.random.randint(1,min(len(shape_list),3))
        distractor_index = np.random.choice(other_shapes_index, distractor_number, replace=False)
        for i in distractor_index:
            selected_shapes = shape_list[i]
            self.target_list.append(self.shape_lib[selected_shapes][0])

        target_space_colors = np.random.choice(len(colors), len(self.target_list), replace=True)
        for i, target in enumerate(self.target_list):
            target.set_color(colors[target_space_colors[i]][1])

        return None
    
    def variation_count(self) -> int:
        return len(planes)