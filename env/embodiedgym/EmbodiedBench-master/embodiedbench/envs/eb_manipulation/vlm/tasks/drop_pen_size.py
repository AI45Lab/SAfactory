import itertools
from typing import List
import numpy as np
from amsolver.backend.unit_tasks import VLM_Object, TargetSpace
from pyrep.objects.shape import Shape
from amsolver.const import colors
from amsolver.backend.utils import scale_object, select_color
from vlm.tasks.drop_pen import DropPen

size_permutations = list(itertools.product(["small", "large"], repeat=2))
class DropPenSize(DropPen):
    def init_task(self) -> None:
        # self.model_num = 3
        return super().init_task()
        
    def init_episode(self, index: int) -> List[str]:
        obj_size, target_size = size_permutations[index]
        # samll_scale_factor = np.random.uniform(0.6, 0.9, 2)
        # large_scale_factor = np.random.uniform(1.1, 1.5, 2)
        if obj_size == "small":
            small_obj = self.object_list[0]
            large_obj = self.object_list[1]
        else:
            small_obj = self.object_list[1]
            large_obj = self.object_list[0]
        small_obj.manipulated_part.descriptions = "the {} {}".format("smaller", small_obj.manipulated_part.property["shape"])
        large_obj.manipulated_part.descriptions = "the {} {}".format("larger", large_obj.manipulated_part.property["shape"])
        for obj, scale_factor in zip([small_obj, large_obj],[np.random.uniform(0.6, 0.9), np.random.uniform(1.0, 1.1)]):
            relative_factor = scale_object(obj, scale_factor)
            if abs(relative_factor-1)>1e-2:
                local_grasp_pose = obj.manipulated_part.local_grasp
                local_grasp_pose[:, :3, 3] *= relative_factor
                obj.manipulated_part.local_grasp = local_grasp_pose

        if target_size == "small":
            small_target = self.target_list[0]
            large_target = self.target_list[1]
        else:
            small_target = self.target_list[1]
            large_target = self.target_list[0]
        small_target.target_space_descriptions = "the smaller container"
        large_target.target_space_descriptions = "the larger container"
        for target, scale_factor in zip([small_target, large_target],[np.random.uniform(0.75, 0.9), np.random.uniform(1.0, 1.1)]):
            relative_factor = scale_object(target, scale_factor)

        color_index = np.random.choice(len(colors), len(self.object_list), replace=True)
        for i, obj in enumerate(self.object_list):
            Shape(obj.manipulated_part.visual).set_color(colors[color_index[i]][1])

        target_colors = np.random.choice(len(colors), len(self.target_list), replace=True)
        for i, target in enumerate(self.target_list):
            target.visual.set_color(colors[target_colors[i]][1])

        return super().init_episode(index)
    
    def variation_count(self) -> int:
        return len(size_permutations)