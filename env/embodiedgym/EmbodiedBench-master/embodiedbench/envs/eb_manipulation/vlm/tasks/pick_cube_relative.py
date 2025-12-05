import itertools
from typing import List
import numpy as np
from amsolver.backend.unit_tasks import VLM_Object, TargetSpace
from pyrep.objects.shape import Shape
from amsolver.const import colors
from amsolver.backend.utils import get_relative_position_xy, scale_object, select_color
from vlm.tasks.pick_cube import PickCube

relative_pos_list = list(itertools.product(["left", "right", "front", "rear"], repeat=2))
class PickCubeRelative(PickCube):
    def init_task(self) -> None:
        super().init_task()
        self.get_relative_pos=True

    def modified_init_episode(self, index: int) -> List[str]:
        color_index = np.random.choice(len(colors), len(self.object_list), replace=True)
        for i, obj in enumerate(self.object_list):
            Shape(obj.manipulated_part.visual).set_color(colors[color_index[i]][1])

        target_space_colors = np.random.choice(len(colors), len(self.target_spaces), replace=True)
        for i, target_space in enumerate(self.target_spaces):
            Shape(target_space.focus_obj_id).set_color(colors[target_space_colors[i]][1])

        self.object_target_relative, self.destination_target_relative = relative_pos_list[index]

        return None
    def variation_count(self) -> int:
        return len(relative_pos_list)

    def sample_method(self):
        satisfied = False
        while not satisfied:
            self.spawn_space.clear()
            for space in self.target_spaces:
                self.spawn_space.sample(Shape(space.focus_obj_id), min_distance=0.25)

            if hasattr(self, "get_relative_pos"):
                comparted_target = self.target_spaces[1]
                target_relative_pose = get_relative_position_xy(Shape(comparted_target.focus_obj_id), Shape(self.target_space.focus_obj_id), self.robot.arm)
                if target_relative_pose != self.destination_target_relative:
                    continue
                self.target_space.target_space_descriptions = f"the {target_relative_pose} container"

            for obj in self.object_list:
                self.spawn_space.sample(obj, min_distance=0.1)
            if hasattr(self, "get_relative_pos"):
                comparted_obj = self.object_list[1]
                object_relative_pose = get_relative_position_xy(comparted_obj.manipulated_part, self.manipulated_obj, self.robot.arm)
                if object_relative_pose != self.object_target_relative:
                    continue
                self.manipulated_obj.property["relative_pos"] = object_relative_pose
                self.manipulated_obj.descriptions = "the {} object".format(self.manipulated_obj.property["relative_pos"])
            satisfied = True
            self.pyrep.step()