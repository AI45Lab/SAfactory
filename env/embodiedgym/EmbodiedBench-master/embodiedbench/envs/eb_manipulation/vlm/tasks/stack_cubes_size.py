from typing import List
import numpy as np
import itertools
from amsolver.backend.task import Task
from pyrep.objects.shape import Shape
from pyrep.objects.proximity_sensor import ProximitySensor
from amsolver.backend.unit_tasks import VLM_Object
from amsolver.const import colors
from amsolver.backend.conditions import DetectedCondition
from amsolver.backend.spawn_boundary import SpawnBoundary
from amsolver.backend.task import Task
from amsolver.backend.utils import scale_object, select_color
from vlm.tasks.stack_cubes import StackCubes
from pyrep.const import ObjectType, PrimitiveShape

sequence = list(itertools.permutations(list(range(3)), 3))
class StackCubesSize(StackCubes):

    def init_task(self) -> None:
        self.model_num = 3
        super().init_task()

    def modified_init_episode(self, index: int) -> List[str]:
        assert self.model_num == 3
        for i in range(self.model_num):
            cube = self.cube_list[i]
            if i==0:
                cube.manipulated_part.descriptions = "the large {}".format(cube.manipulated_part.property["shape"])
                scale_factor = np.random.uniform(1.2, 1.4)
            elif i==1:
                cube.manipulated_part.descriptions = "the medium {}".format(cube.manipulated_part.property["shape"])
                scale_factor = np.random.uniform(0.9, 1.1)
            elif i==2:
                cube.manipulated_part.descriptions = "the small {}".format(cube.manipulated_part.property["shape"])
                scale_factor = np.random.uniform(0.7, 0.9)
            relative_factor = scale_object(cube, scale_factor)
            scale_object(cube.target, scale_factor)
            if abs(relative_factor-1)>1e-2:
                local_grasp_pose = cube.manipulated_part.local_grasp
                local_grasp_pose[:, :3, 3] *= relative_factor
                cube.manipulated_part.local_grasp = local_grasp_pose

        color_index = np.random.choice(len(colors), self.model_num, replace=True)
        for cube, i in zip(self.cube_list, color_index):
            Shape(cube.manipulated_part.visual).set_color(colors[i][1])
        self.cube_list = [self.cube_list[i] for i in sequence[index]]
        return None

    def variation_count(self) -> int:
        # TODO: The number of variations for this task.
        return len(sequence)
    
    # def import_objects(self, number=4):
    #     large_model_path = self.model_dir+"cube/cube_large/cube_large.ttm"
    #     normal_model_path = self.model_dir+"cube/cube_normal/cube_normal.ttm"
    #     small_model_path = self.model_dir+"cube/cube_small/cube_small.ttm"
    #     self.large_cube = VLM_Object(self.pyrep, large_model_path, 0)
    #     self.normal_cube = VLM_Object(self.pyrep, normal_model_path, 0)
    #     self.small_cube = VLM_Object(self.pyrep, small_model_path, 0)
    #     self.large_cube.manipulated_part.descriptions = "the large cube"
    #     self.normal_cube.manipulated_part.descriptions = "the medium cube"
    #     self.small_cube.manipulated_part.descriptions = "the small cube"
    #     self.cube_list = [self.large_cube, self.normal_cube, self.small_cube]
    #     self.cube_num = len(self.cube_list)
    #     for cube in self.cube_list:
    #         cube.set_model(False)
    #         cube.set_parent(self.get_base())
    #         cube_bbox = cube.get_bounding_box()
    #         x, y = cube_bbox[1]-cube_bbox[0]+0.02,  cube_bbox[3]-cube_bbox[2]+0.02
    #         cube_target = Shape.create(PrimitiveShape.CUBOID, [x,y,0], respondable=False, static=True, renderable=False)
    #         cube_target.set_parent(cube)
    #         cube.target = cube_target
    #     self.register_graspable_objects(self.cube_list)
