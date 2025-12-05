from typing import List
import numpy as np
from amsolver.backend.unit_tasks import VLM_Object, TargetSpace
from pyrep.objects.shape import Shape
from amsolver.const import colors, object_shapes
from amsolver.backend.utils import select_color
from vlm.tasks.pick_cube import PickCube

class PickCubeShape(PickCube):

    def modified_init_episode(self, index: int) -> List[str]:
        self.object_list = [self.shape_lib[index]]
        other_obj_index = list(range(len(self.shape_lib)))
        other_obj_index.remove(index)
        distractor_number = np.random.randint(1,len(self.shape_lib))
        distractor_index = np.random.choice(other_obj_index, distractor_number, replace=False)
        for i in distractor_index:
            self.object_list.append(self.shape_lib[i])
        color_index = np.random.choice(len(colors), len(self.object_list), replace=True)
        for i, obj in enumerate(self.object_list):
            Shape(obj.manipulated_part.visual).set_color(colors[color_index[i]][1])
            # obj.manipulated_part.descriptions = "the {} {}".format(colors[color_index[i]][0], obj.manipulated_part.property["shape"])
            obj.manipulated_part.descriptions = "the {}".format(obj.manipulated_part.property["shape"])

        target_space_colors = np.random.choice(len(colors), len(self.target_spaces), replace=False)
        for i, target_space in enumerate(self.target_spaces):
            target_space.target_space_descriptions = "the {} container".format(colors[target_space_colors[i]][0])
            Shape(target_space.focus_obj_id).set_color(colors[target_space_colors[i]][1])

        return None
    
    def variation_count(self) -> int:
        return len(object_shapes)

    def import_objects(self, num):
        object_numbers = [1]*len(object_shapes)
        self.shape_lib = []
        for obj, num in zip(object_shapes, object_numbers):
            for i in range(num):
                model = VLM_Object(self.pyrep, self.model_dir+object_shapes[obj]["path"], i)
                model.set_parent(self.taks_base)
                model.set_position([0,0,0])
                self.shape_lib.append(model)
        self.register_graspable_objects(self.shape_lib)
        self._need_remove_objects+=self.shape_lib
                