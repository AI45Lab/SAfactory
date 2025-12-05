from typing import List
import numpy as np
from pyrep.objects.shape import Shape
from amsolver.const import colors
from amsolver.backend.task import Task
from amsolver.backend.utils import select_color
from vlm.tasks.place_into_shape_sorter import PlaceIntoShapeSorter

class PlaceIntoShapeSorterColor(PlaceIntoShapeSorter):

    def init_task(self) -> None:
        self.object_numbers = [3 for _ in self.object_dict]
        return super().init_task()
        
    def init_episode(self, index: int) -> List[str]:
        for obj_name in self.objects:
            obj_lens = len(self.objects[obj_name])
            color_names, rgbs = select_color(index, obj_lens-1)
            for i, obj in enumerate(self.objects[obj_name]):
                Shape(obj.manipulated_part.visual).set_color(rgbs[i])
                obj.manipulated_part.descriptions = "the {} {}".format(color_names[i], obj.manipulated_part.property["shape"])
        return super().init_episode(index)

    def variation_count(self) -> int:
        # TODO: The number of variations for this task.
        return len(colors)
