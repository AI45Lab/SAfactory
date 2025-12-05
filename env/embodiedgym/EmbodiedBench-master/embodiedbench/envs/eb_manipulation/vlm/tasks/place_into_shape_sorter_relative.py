import itertools
from typing import List
import numpy as np
from pyrep.objects.shape import Shape
from amsolver.const import colors
from amsolver.backend.task import Task
from amsolver.backend.utils import get_relative_position_xy, select_color
from vlm.tasks.place_into_shape_sorter import PlaceIntoShapeSorter

relative_pos_list = ["left", "right", "front", "rear"]
class PlaceIntoShapeSorterRelative(PlaceIntoShapeSorter):

    def init_task(self) -> None:
        self.object_numbers = [2 for _ in self.object_dict]
        return super().init_task()
        
    def init_episode(self, index: int) -> List[str]:
        self.object_target_relative = relative_pos_list[index]
        for obj_name in self.objects:
            obj_lens = len(self.objects[obj_name])
            color_index = np.random.choice(len(colors), obj_lens, replace=True)
            for i, obj in enumerate(self.objects[obj_name]):
                Shape(obj.manipulated_part.visual).set_color(colors[color_index[i]][1])
        return super().init_episode(index)

    def sample_method(self):
        self.pyrep.set_configuration_tree(self._task_init_states)
        distractors = set(self.objects.keys())-set(self.pick_objs)
        if len(distractors)>0 and self.need_distractors:
            distractor_number = np.random.randint(1,len(distractors))
            select_distractors = np.random.choice(list(distractors), distractor_number, replace=False).tolist()
        else:
            select_distractors = []
        while True:
            satisfied = True
            self.spawn_space.clear()
            self.spawn_space.sample(self.sorter)
            for obj in (self.pick_objs+select_distractors):
                for instance in self.objects[obj]:
                    self.spawn_space.sample(instance)
                if obj in self.pick_objs:
                    object_relative_pose = get_relative_position_xy(self.objects[obj][1], self.objects[obj][0], self.robot.arm)
                    if object_relative_pose!= self.object_target_relative:
                        satisfied = False
                        break
                    obj = self.objects[obj][0].manipulated_part
                    obj.descriptions = "the {} {}".format(object_relative_pose, obj.property["shape"]) 
            if satisfied:
                break

    def variation_count(self) -> int:
        # TODO: The number of variations for this task.
        return len(relative_pos_list)
