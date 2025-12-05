from typing import List
import numpy as np
import os
import random
from amsolver.backend.task import Task
from pyrep.objects.dummy import Dummy
from pyrep.objects.shape import Shape
from pyrep.objects.proximity_sensor import ProximitySensor
from pyrep.const import ObjectType, PrimitiveShape
from amsolver.backend.unit_tasks import T0_ObtainControl, T1_MoveObjectGoal, TargetSpace, VLM_Object
from amsolver.backend.utils import scale_object
from amsolver.const import colors, object_shapes
from amsolver.backend.conditions import DetectedCondition
from amsolver.backend.spawn_boundary import SpawnBoundary
from amsolver.backend.task import Task

class StackCubes(Task):

    def init_task(self) -> None:
        self.success_sensor = ProximitySensor('success')
        self.success_sensor.set_collidable(False)
        self.spawn_space = SpawnBoundary([Shape('workspace')])
        self.model_dir = os.path.dirname(os.path.realpath(__file__)).replace("tasks","object_models/")
        self.temporary_waypoints = []
        self.taks_base = self.get_base()
        if not hasattr(self, "model_num"):
            self.model_num = 4
        if not hasattr(self, "class_num"):
            self.class_num = 1

    def init_episode(self, index: int) -> List[str]:
        self.variation_index = index
        self.import_objects()
        self.init_pose = []
        for cube in self.cube_list:
            self.init_pose.append(cube.get_pose())
        self.modified_init_episode(index)
        try_times = 200
        # pick_cube_number = np.random.randint(1,self.cube_num)
        pick_cube_number = 1
        while len(self.temporary_waypoints)==0 and try_times>0:
            for cube, pose in zip(self.cube_list, self.init_pose):
                cube.set_pose(pose)
            self.sample_method()
            self.success_sensor.set_position(self.cube_list[0].get_position())
            self.pyrep.step()
            init_states = self.get_state()[0]
            task_sequence = []
            conditions = [DetectedCondition(self.cube_list[0].manipulated_part, self.success_sensor)]
            for i in range(pick_cube_number):
                below_cube = self.cube_list[i]
                above_cube = self.cube_list[i+1]
                conditions.append(DetectedCondition(above_cube.manipulated_part, self.success_sensor))
                target_space_mesh = above_cube.target
                below_obj_zmax = below_cube.get_bounding_box()[-1]
                target_space_mesh.set_position([0,0,below_obj_zmax+0.005], relative_to = below_cube)
                # target_space_mesh.set_parent(below_cube)
                target_space = TargetSpace(SpawnBoundary([target_space_mesh]), self.success_sensor,
                                        (0,0,-3.14), (0,0,3.14), below_cube.manipulated_part.descriptions, below_cube.manipulated_part.visual)
                target_space.set_target(above_cube.manipulated_part, try_ik_sampling=True, linear=False, release=True)
                MoveTask = T1_MoveObjectGoal(self.robot, self.pyrep, target_space, self.taks_base, fail_times=2)
                GraspTask = T0_ObtainControl(self.robot, self.pyrep, above_cube.manipulated_part, self.taks_base, try_times=100)
                task_sequence.append(GraspTask)
                task_sequence.append(MoveTask)
            for i in range(-1, -len(task_sequence), -1):
                task_sequence[i-1].next_task_fuc = task_sequence[i].get_path
            waypoints = task_sequence[0].get_path(try_ik_sampling=False, ignore_collisions=False)
            if waypoints is not None:
                self.temporary_waypoints += waypoints
            self.reset_robot()
            self.pyrep.set_configuration_tree(init_states)
            try_times -= 1
        self.register_success_conditions(conditions)
        for i,waypoint in enumerate(self.temporary_waypoints):
            waypoint.set_name('waypoint{}'.format(i))
        descriptions = f"Stack {self.cube_list[0].manipulated_part.descriptions}"
        for i in range(pick_cube_number):
            if i == pick_cube_number-1:
                descriptions += " and "
            elif i < pick_cube_number-1:
                descriptions += ", "
            descriptions += f"{self.cube_list[i+1].manipulated_part.descriptions}"
        descriptions += " in sequence."
        return [descriptions]

    def import_objects(self):
        self.cube_list = []
        self.shape_lib = {}
        selected_objs = random.sample(list(object_shapes.keys()), self.class_num)
        for selected_obj in selected_objs:
            model_path = self.model_dir+object_shapes[selected_obj]['path']
            self.shape_lib[selected_obj] = []
            for i in range(self.model_num):
                cube = VLM_Object(self.pyrep, model_path, i)
                # scale_factor = np.random.uniform(1.0, 1.25)
                scale_factor = 1.5
                relative_factor = scale_object(cube, scale_factor)
                if abs(relative_factor-1)>1e-2:
                    local_grasp_pose = cube.manipulated_part.local_grasp
                    local_grasp_pose[:, :3, 3] *= relative_factor
                    cube.manipulated_part.local_grasp = local_grasp_pose
                cube.scale_factor = scale_factor
                # cube.set_model(False)
                cube.set_parent(self.taks_base)
                cube_bbox = cube.get_bounding_box()
                x, y = cube_bbox[1]-cube_bbox[0],  cube_bbox[3]-cube_bbox[2]
                x, y = x*1.2, y*1.2
                cube_target = Shape.create(PrimitiveShape.CUBOID, [x,y,0], respondable=False, static=True, renderable=False)
                # cube_target.set_parent(self.taks_base)
                cube_target._is_plane = True
                cube_target.set_transparency(0)
                cube.target = cube_target
                self.cube_list.append(cube)
                self.shape_lib[selected_obj].append(cube)
                self._need_remove_objects.append(cube_target)
        self.register_graspable_objects(self.cube_list)
        self._need_remove_objects+=self.cube_list

    def sample_method(self):
        self.spawn_space.clear()
        for cube in self.cube_list:
            self.spawn_space.sample(cube, min_distance=0.1)
    
    def is_static_workspace(self) -> bool:
        return True

    def load(self, ttms_folder=None):
        if Shape.exists('stack_cubes'):
            return Dummy('stack_cubes')
        ttm_file = os.path.join(ttms_folder, 'stack_cubes.ttm')
        self._base_object = self.pyrep.import_model(ttm_file)
        return self._base_object
    
    def get_base(self) -> Dummy:
        self._base_object = Dummy('stack_cubes')
        return self._base_object
