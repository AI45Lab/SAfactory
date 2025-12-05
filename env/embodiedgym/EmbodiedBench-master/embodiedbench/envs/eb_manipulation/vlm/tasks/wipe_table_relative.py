from typing import List
import numpy as np
from amsolver.backend.unit_tasks import VLM_Object, TargetSpace
from pyrep.objects.shape import Shape
from amsolver.const import colors
from amsolver.backend.utils import get_relative_position_xy, select_color
from vlm.tasks.wipe_table import WipeTable

relative_pos_list = ["left", "right", "front", "rear"]
class WipeTableRelative(WipeTable):
    def init_task(self) -> None:
        return super().init_task()
        
    def modified_init_episode(self, index: int):
        self.target_relative = relative_pos_list[index]
        target_space_colors = np.random.choice(len(colors), len(self.target_list), replace=True)
        for i, target in enumerate(self.target_list):
            target.set_color(colors[target_space_colors[i]][1])

        return None
    
    def variation_count(self) -> int:
        return len(relative_pos_list)
    
    def sample_method(self):
        
        while True:
            self.spawn_space.clear()
            
            for t in self.target_list:
                self.spawn_space.sample(t, min_distance=0.1)
            target = self.target_list[0]
            distractor = self.target_list[1]        
            object_relative_pose = get_relative_position_xy(distractor, target, self.robot.arm)
            if object_relative_pose!= self.target_relative:
                continue
            target.target_space_descriptions = f"the {object_relative_pose} area"
            for obj in self.object_list:
                self.spawn_space.sample(obj, min_distance=0.1)
            break