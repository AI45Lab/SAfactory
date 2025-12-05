from typing import List
import numpy as np
from pyrep.objects.shape import Shape
from amsolver.const import colors, sorter_objects
from amsolver.backend.task import Task
from amsolver.backend.utils import select_color
from vlm.tasks.place_into_shape_sorter import PlaceIntoShapeSorter

shape_list = list(sorter_objects.keys())
class PlaceIntoShapeSorterShape(PlaceIntoShapeSorter):
    def init_task(self) -> None:
        self.need_distractors = True
        return super().init_task()
        
    def init_episode(self, index: int) -> List[str]:
        self.need_shape_resample = False
        self.pick_objs = [shape_list[index]]
        for obj_name in self.objects:
            obj_lens = len(self.objects[obj_name])
            color_index = np.random.choice(len(colors), obj_lens, replace=True)
            for i, obj in enumerate(self.objects[obj_name]):
                Shape(obj.manipulated_part.visual).set_color(colors[color_index[i]][1])
                obj.manipulated_part.descriptions = "the {}".format(obj.manipulated_part.property["shape"])
        return super().init_episode(index)

    def variation_count(self) -> int:
        # TODO: The number of variations for this task.
        return len(shape_list)
