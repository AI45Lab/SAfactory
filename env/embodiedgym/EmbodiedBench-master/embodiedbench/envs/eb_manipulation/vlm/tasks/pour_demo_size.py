import itertools
from typing import List
import numpy as np
from pyrep.objects.shape import Shape
from amsolver.const import colors
from amsolver.backend.utils import scale_object, select_color
from vlm.tasks.pour_demo import PourDemo

size_list = ["small", "large"]
class PourDemoSize(PourDemo):
    # def init_task(self) -> None:
    #     super().init_task()
    #     self.obj_init_pose = []
    #     for obj in self.object_list:
    #         self.obj_init_pose.append(obj.get_pose())
    #     self.ignore_collisions = True
        
    def modified_init_episode(self, index: int):
        self.ignore_collisions = True
        obj_size = size_list[index]
        if obj_size == "small":
            small_obj = self.object_list[0]
            large_obj = self.object_list[1]
        else:
            small_obj = self.object_list[1]
            large_obj = self.object_list[0]
        small_obj.manipulated_part.descriptions = "the smaller {}".format(small_obj.manipulated_part.property["shape"])
        large_obj.manipulated_part.descriptions = "the larger {}".format(large_obj.manipulated_part.property["shape"])
        for obj, scale_factor in zip([small_obj, large_obj],[np.random.uniform(0.75, 0.9), np.random.uniform(1.0, 1.1)]):
            relative_factor = scale_object(obj, scale_factor)
            if abs(relative_factor-1)>1e-2:
                local_grasp_pose = obj.manipulated_part.local_grasp
                local_grasp_pose[:, :3, 3] *= relative_factor
                obj.manipulated_part.local_grasp = local_grasp_pose

        color_index = np.random.choice(len(colors), len(self.object_list), replace=True)
        for i, obj in enumerate(self.object_list):
            Shape(obj.manipulated_part.visual).set_color(colors[color_index[i]][1])

        return None
    def variation_count(self) -> int:
        return len(size_list)
    
    # def cleanup(self) -> None:
    #     for obj, pose in zip(self.object_list, self.obj_init_pose):
    #         if obj.still_exists():
    #             obj.set_pose(pose)
    #     return super().cleanup()