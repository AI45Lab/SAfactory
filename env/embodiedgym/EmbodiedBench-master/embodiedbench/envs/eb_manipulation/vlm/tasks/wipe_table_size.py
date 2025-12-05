from typing import List
import numpy as np
from amsolver.backend.unit_tasks import VLM_Object, TargetSpace
from pyrep.objects.shape import Shape
from amsolver.const import colors
from amsolver.backend.utils import scale_object, select_color
from vlm.tasks.wipe_table import WipeTable

size_list = ["small", "large"]
class WipeTableSize(WipeTable):
    def init_task(self) -> None:
        # self.model_num = 3
        return super().init_task()
        
    def modified_init_episode(self, index: int):

        target_size = size_list[index]
        if target_size == "small":
            small_target = self.target_list[0]
            large_target = self.target_list[1]
        else:
            small_target = self.target_list[1]
            large_target = self.target_list[0]
        small_target.target_space_descriptions = "the smaller area"
        large_target.target_space_descriptions = "the larger area"
        for target, scale_factor in zip([small_target, large_target],[np.random.uniform(0.75, 0.9), np.random.uniform(1.0, 1.2)]):
            scale_object(target, scale_factor)

        target_space_colors = np.random.choice(len(colors), len(self.target_list), replace=True)
        for i, target in enumerate(self.target_list):
            target.set_color(colors[target_space_colors[i]][1])

        return None
    
    def variation_count(self) -> int:
        return len(size_list)