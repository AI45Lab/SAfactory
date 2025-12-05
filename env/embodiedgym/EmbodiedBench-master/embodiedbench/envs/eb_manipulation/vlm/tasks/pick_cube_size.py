import itertools
from typing import List
import numpy as np
from amsolver.backend.unit_tasks import VLM_Object, TargetSpace
from pyrep.objects.shape import Shape
from amsolver.const import colors
from amsolver.backend.utils import scale_object, select_color
from vlm.tasks.pick_cube import PickCube

size_permutations = list(itertools.product(["small", "large"], repeat=2))

class PickCubeSize(PickCube):
    def init_task(self) -> None:
        # self.model_path = "cube/cube_basic/cube_basic.ttm"
        super().init_task()

    def modified_init_episode(self, index: int) -> List[str]:
        obj_size, target_size = size_permutations[index]
        # samll_scale_factor = np.random.uniform(0.6, 0.9, 2)
        # large_scale_factor = np.random.uniform(1.1, 1.5, 2)
        if obj_size == "small":
            small_obj = self.object_list[0]
            large_obj = self.object_list[1]
        else:
            small_obj = self.object_list[1]
            large_obj = self.object_list[0]
        small_obj.manipulated_part.descriptions = "the {} object".format("smaller")
        large_obj.manipulated_part.descriptions = "the {} object".format("larger")
        for obj, scale_factor in zip([small_obj, large_obj],[np.random.uniform(0.6, 0.9), np.random.uniform(1.1, 1.2)]):
            relative_factor = scale_object(obj, scale_factor)
            if abs(relative_factor-1)>1e-2:
                local_grasp_pose = obj.manipulated_part.local_grasp
                local_grasp_pose[:, :3, 3] *= relative_factor
                obj.manipulated_part.local_grasp = local_grasp_pose

        if target_size == "small":
            small_target = self.target_spaces[0]
            large_target = self.target_spaces[1]
        else:
            small_target = self.target_spaces[1]
            large_target = self.target_spaces[0]
        small_target.target_space_descriptions = "the smaller container"
        large_target.target_space_descriptions = "the larger container"
        for target, scale_factor in zip([small_target, large_target],[np.random.uniform(0.8, 0.9), np.random.uniform(1.1, 1.25)]):
            relative_factor = scale_object(Shape(target.focus_obj_id), scale_factor)

        color_index = np.random.choice(len(colors), len(self.object_list), replace=True)
        for i, obj in enumerate(self.object_list):
            Shape(obj.manipulated_part.visual).set_color(colors[color_index[i]][1])

        target_space_colors = np.random.choice(len(colors), len(self.target_spaces), replace=True)
        for i, target_space in enumerate(self.target_spaces):
            Shape(target_space.focus_obj_id).set_color(colors[target_space_colors[i]][1])
        self.pyrep.step()
        return None
    def variation_count(self) -> int:
        return len(size_permutations)
