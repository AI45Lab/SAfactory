import itertools
from typing import List
import numpy as np
from amsolver.backend.unit_tasks import VLM_Object, TargetSpace
from pyrep.objects.shape import Shape
from amsolver.const import colors
from amsolver.backend.utils import get_relative_position_xy, scale_object, select_color
from vlm.tasks.drop_pen import DropPen

relative_pos_list = list(itertools.product(["left", "right", "front", "rear"], repeat=2))
class DropPenRelative(DropPen):
    def init_task(self) -> None:
        # self.model_num = 3
        return super().init_task()
        
    def init_episode(self, index: int) -> List[str]:
        self.object_target_relative, self.destination_target_relative = relative_pos_list[index]
        color_index = np.random.choice(len(colors), len(self.object_list), replace=True)
        for i, obj in enumerate(self.object_list):
            Shape(obj.manipulated_part.visual).set_color(colors[color_index[i]][1])

        target_colors = np.random.choice(len(colors), len(self.target_list), replace=True)
        for i, target in enumerate(self.target_list):
            target.visual.set_color(colors[target_colors[i]][1])

        return super().init_episode(index)
    
    def sample_method(self):
        while True:
            self.spawn_space.clear()
            for tg in self.target_list:
                self.spawn_space.sample(tg, min_distance=0.1)
            object_relative_pose = get_relative_position_xy(self.target_list[1], self.target_list[0], self.robot.arm)
            if object_relative_pose != self.destination_target_relative:
                continue
            self.target_list[0].target_space_descriptions = f"the {object_relative_pose} container"

            for obj in self.object_list:
                self.spawn_space.sample(obj, min_distance=0.2)
            object_relative_pose = get_relative_position_xy(self.object_list[1], self.object_list[0], self.robot.arm)
            if object_relative_pose != self.object_target_relative:
                continue
            obj = self.object_list[0].manipulated_part
            obj.descriptions = "the {} {}".format(object_relative_pose, obj.property["shape"]) 
            break
        for _ in range(5):
            self.pyrep.step()

    def variation_count(self) -> int:
        return len(relative_pos_list)