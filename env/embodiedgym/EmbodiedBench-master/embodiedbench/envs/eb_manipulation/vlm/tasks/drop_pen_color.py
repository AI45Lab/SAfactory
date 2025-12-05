from typing import List
import numpy as np
from amsolver.backend.unit_tasks import VLM_Object, TargetSpace
from pyrep.objects.shape import Shape
from amsolver.const import colors
from amsolver.backend.utils import select_color
from vlm.tasks.drop_pen import DropPen

class DropPenColor(DropPen):
    def init_task(self) -> None:
        # self.model_num = 3
        return super().init_task()
        
    def init_episode(self, index: int) -> List[str]:
        color_names, rgbs = select_color(index, len(self.object_list)-1)
        for i, obj in enumerate(self.object_list):
            Shape(obj.manipulated_part.visual).set_color(rgbs[i])
            obj.manipulated_part.property["color"] = color_names[i]
            obj.manipulated_part.descriptions = "the {} {}".format(color_names[i], obj.manipulated_part.property["shape"])

        target_space_colors = np.random.choice(len(colors), len(self.target_list), replace=False)
        for i, target_space in enumerate(self.target_list):
            target_space.target_space_descriptions = "the {} container".format(colors[target_space_colors[i]][0])
            target_space.visual.set_color(colors[target_space_colors[i]][1])

        return super().init_episode(index)
    
    def variation_count(self) -> int:
        return len(colors)