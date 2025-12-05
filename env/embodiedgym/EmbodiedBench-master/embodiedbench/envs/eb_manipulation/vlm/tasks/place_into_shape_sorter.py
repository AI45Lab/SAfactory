from typing import List
import numpy as np
import os
from amsolver.backend.task import Task
from pyrep.objects.shape import Shape
from pyrep.objects.proximity_sensor import ProximitySensor
from pyrep.objects.dummy import Dummy
from amsolver.backend.unit_tasks import T0_ObtainControl, T1_MoveObjectGoal, TargetSpace, VLM_Object
from amsolver.backend.utils import scale_object
from amsolver.const import colors, sorter_objects
from amsolver.backend.conditions import DetectedCondition
from amsolver.backend.spawn_boundary import SpawnBoundary
from amsolver.backend.task import Task

class PlaceIntoShapeSorter(Task):
    def __init__(self, pyrep, robot):
        super().__init__(pyrep, robot)
        self.object_dict = sorter_objects
        
    def init_task(self) -> None:
        self.model_dir = os.path.dirname(os.path.realpath(__file__)).replace("tasks","object_models/")
        self.spawn_space = SpawnBoundary([Shape('workspace')])
        self.success_sensor = ProximitySensor('success')
        self.sorter = Shape("shape_sorter")
        self.sorter_visual = Shape("shape_sorter_visual")
        self.sorter_visual.set_color([1, 0, 0])
        self.objects = {}
        self.temporary_waypoints = []
        self.taks_base = self.get_base()
        if not hasattr(self, "object_numbers"):
            self.object_numbers = [1 for _ in self.object_dict]
        self.import_objects()
        self._task_init_states = self.get_state()[0]
        self.need_shape_resample = True
        self.pick_objs = []
        if not hasattr(self, "need_distractors"):
            self.need_distractors = False

    def init_episode(self, index: int) -> List[str]:
        self.variation_index = index
        pick_number = 1
        try_times = 0
        while len(self.temporary_waypoints)==0:
            if try_times%10 == 0 and self.need_shape_resample:
                # pick_number = np.random.randint(1,min(len(self.objects)-1,2))
                self.pick_objs = np.random.choice(list(self.objects.keys()), pick_number, replace=False).tolist()
            self.sample_method()
            self.pyrep.step()
            init_states = self.get_state()[0]
            task_sequence, manipulated_objs, conditions = [], [], []
            for obj_name in self.pick_objs:
                obj = self.objects[obj_name][0]
                conditions.append(DetectedCondition(obj.manipulated_part, self.success_sensor))
                target = obj.target
                target_space = TargetSpace(obj.target, self.success_sensor, None, None, target.descriptions, self.sorter_visual.get_handle())
                target_space.set_target(obj.manipulated_part, linear=False, ignore_collisions= False, release=True)
                MoveTask = T1_MoveObjectGoal(self.robot, self.pyrep, target_space, self.taks_base, fail_times=10)
                GraspTask = T0_ObtainControl(self.robot, self.pyrep, obj.manipulated_part, self.taks_base, try_times=20)
                task_sequence.append(GraspTask)
                task_sequence.append(MoveTask)
                manipulated_objs.append(obj)
            for i in range(-1, -len(task_sequence), -1):
                task_sequence[i-1].next_task_fuc = task_sequence[i].get_path
            if try_times<100:
                waypoints = task_sequence[0].get_path(try_ik_sampling=False)
            else:
                waypoints = task_sequence[0].get_path(try_ik_sampling=True)
            if waypoints is not None:
                self.temporary_waypoints += waypoints
            self.reset_robot()
            self.pyrep.set_configuration_tree(init_states)
            try_times += 1
        self.register_success_conditions(conditions)
        for i,waypoint in enumerate(self.temporary_waypoints):
            waypoint.set_name('waypoint{}'.format(i))
        descriptions = f"Put "
        for i, obj in enumerate(manipulated_objs):
            descriptions += f"{obj.manipulated_part.descriptions}"
            if i == len(manipulated_objs)-2:
                descriptions += " and "
            elif i < len(manipulated_objs)-2:
                descriptions += ", "
        descriptions += " into the shape sorter."
        return [descriptions]

    def variation_count(self) -> int:
        # TODO: The number of variations for this task.
        return 1

    def import_objects(self):
        for obj, num in zip(self.object_dict, self.object_numbers):
            for i in range(num):
                model_path = self.model_dir+self.object_dict[obj]["path"]
                model = VLM_Object(self.pyrep, model_path, i)
                model.set_parent(self.taks_base)
                model.set_position([0, 0, 0])
                target = Dummy(f"{model.obj_class}_target")
                target.descriptions = f"the hole of {model.obj_class} shape"
                target.set_position(target.get_position()+[0,0,0.01])
                model.target = target
                if obj in self.objects:
                    self.objects[obj].append(model)
                else:
                    self.objects[obj]=[model]
        self.register_graspable_objects([obj for i in self.objects for obj in self.objects[i]])

    def sample_method(self):
        self.pyrep.set_configuration_tree(self._task_init_states)
        distractors = set(self.objects.keys())-set(self.pick_objs)
        if len(distractors)>0 and self.need_distractors:
            distractor_number = np.random.randint(1,len(distractors))
            select_distractors = np.random.choice(list(distractors), distractor_number, replace=False).tolist()
        else:
            select_distractors = []
        self.spawn_space.clear()
        self.spawn_space.sample(self.sorter)
        for obj in (self.pick_objs+select_distractors):
            for instance in self.objects[obj]:
                self.spawn_space.sample(instance)
    
    def is_static_workspace(self) -> bool:
        return True

    def load(self, ttms_folder=None):
        if Shape.exists('place_into_shape_sorter'):
            return Dummy('place_into_shape_sorter')
        ttm_file = os.path.join(ttms_folder, 'place_into_shape_sorter.ttm')
        self._base_object = self.pyrep.import_model(ttm_file)
        return self._base_object
    
    def get_base(self) -> Dummy:
        self._base_object = Dummy('place_into_shape_sorter')
        return self._base_object