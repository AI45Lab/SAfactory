from typing import List
import numpy as np
from amsolver.backend.unit_tasks import VLM_Object, TargetSpace
from pyrep.objects.shape import Shape
from amsolver.const import colors
from amsolver.backend.utils import select_color
from vlm.tasks.pour_demo import PourDemo

class PourDemoColor(PourDemo):
    def init_task(self) -> None:
        self.model_num = 3
        return super().init_task()
        
    def modified_init_episode(self, index: int):
        color_names, rgbs = select_color(index, len(self.object_list)-1, replace=False)
        for i, obj in enumerate(self.object_list):
            Shape(obj.manipulated_part.visual).set_color(rgbs[i])
            obj.manipulated_part.property["color"] = color_names[i]
            obj.manipulated_part.descriptions = "the {} {}".format(color_names[i], obj.manipulated_part.property["shape"])

        return None
    
    def variation_count(self) -> int:
        return len(colors)