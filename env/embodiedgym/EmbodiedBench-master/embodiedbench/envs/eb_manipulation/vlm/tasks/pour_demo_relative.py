import itertools
from typing import List
import numpy as np
from pyrep.objects.shape import Shape
from amsolver.const import colors
from amsolver.backend.utils import get_relative_position_xy, scale_object, select_color
from vlm.tasks.pour_demo import PourDemo

relative_pos_list = [["left","right"], 
                        ["right", "left"], 
                        ["front", "rear"],
                        ["rear", "front"]]
class PourDemoRelative(PourDemo):
    def init_task(self) -> None:
        return super().init_task()
        
    def modified_init_episode(self, index: int):
        self.target_relative, self.compared_relative = relative_pos_list[index]

        color_index = np.random.choice(len(colors), len(self.object_list), replace=True)
        for i, obj in enumerate(self.object_list):
            Shape(obj.manipulated_part.visual).set_color(colors[color_index[i]][1])

        return None
    
    def variation_count(self) -> int:
        return len(relative_pos_list)
    
    def sample_method(self):
        while True:
            self.spawn_space.clear()
            for obj in self.object_list:
                self.spawn_space.sample(obj, min_distance=0.1)
            comparted_obj = self.object_list[1]
            object_relative_pose = get_relative_position_xy(comparted_obj.manipulated_part, self.manipulated_obj, self.robot.arm)
            if object_relative_pose != self.target_relative:
                continue
            self.manipulated_obj.property["relative_pos"] = object_relative_pose
            self.manipulated_obj.descriptions = "the {} {}".format(object_relative_pose, self.manipulated_obj.property["shape"])
            comparted_obj.manipulated_part.descriptions = "the {} {}".format(self.compared_relative, comparted_obj.manipulated_part.property["shape"])
            break
        for _ in range(5):
            self.pyrep.step()