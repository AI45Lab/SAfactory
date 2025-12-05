from typing import List
import numpy as np
import os
from amsolver.backend.task import Task
from pyrep.objects.dummy import Dummy
from pyrep.objects.shape import Shape
from pyrep.objects.proximity_sensor import ProximitySensor
from pyrep.const import ObjectType, PrimitiveShape
from pyrep.backend._sim_cffi import ffi, lib
from amsolver.backend.unit_tasks import T0_ObtainControl, T1_MoveObjectGoal, TargetSpace, VLM_Object
from amsolver.const import colors
from amsolver.backend.conditions import DetectedCondition
from amsolver.backend.spawn_boundary import SpawnBoundary
from amsolver.backend.task import Task

class DropPen(Task):

    def init_task(self) -> None:
        self.spawn_space = SpawnBoundary([Shape('workspace')])
        self.model_dir = os.path.dirname(os.path.realpath(__file__)).replace("tasks","object_models/")
        self.object_list = []
        self.target_list = []
        self.temporary_waypoints = []
        self.taks_base = self.get_base()
        if not hasattr(self, "model_num"):
            self.model_num = 2
        self.import_objects(self.model_num)
        self._task_init_states = self.get_state()[0]

    def init_episode(self, index: int) -> List[str]:
        self.variation_index = index
        self.pyrep.set_configuration_tree(self._task_init_states)
        try_times = 200
        # pick_number = np.random.randint(1,self.model_num)
        pick_obj =  self.object_list[0]
        target_obj = self.target_list[0]
        while len(self.temporary_waypoints)==0 and try_times>0:
            self.sample_method()
            self.pyrep.step()
            init_states = self.get_state()[0]
            obj_space_args = {"obj":pick_obj.manipulated_part, "container":target_obj}
            target_space_descriptions = f"along the opening of {target_obj.target_space_descriptions}"
            target_space = TargetSpace(self.drop_pose, space_args=obj_space_args,
                    target_space_descriptions = target_space_descriptions, focus_obj_id= target_obj.visual.get_handle())
            target_space.set_target(pick_obj.manipulated_part, try_ik_sampling=False, linear=False, release=True)
            MoveTask = T1_MoveObjectGoal(self.robot, self.pyrep, target_space, self.taks_base, fail_times=2)
            GraspTask = T0_ObtainControl(self.robot, self.pyrep, pick_obj.manipulated_part, self.taks_base, try_times=500,
                                        next_task_fuc=MoveTask.get_path)
            waypoints = GraspTask.get_path(try_ik_sampling=False, ignore_collisions=True)
            if waypoints is not None:
                self.temporary_waypoints += waypoints
            self.reset_robot()
            self.pyrep.set_configuration_tree(init_states)
            try_times -= 1
        conditions = [DetectedCondition(pick_obj.manipulated_part, target_obj.successor)]
        self.register_success_conditions(conditions)
        for i,waypoint in enumerate(self.temporary_waypoints):
            waypoint.set_name('waypoint{}'.format(i))
        descriptions = "Drop {} into {}.".format(pick_obj.manipulated_part.descriptions, target_obj.target_space_descriptions)
        return [descriptions]

    @staticmethod
    def drop_pose(obj:Shape, container:Shape, n_sample=36):
        container_pos = container.get_position()
        container_size = container.get_bounding_box()
        container_x, container_y = container_size[1] - container_size[0], container_size[3] - container_size[2]
        container_r = (container_x**2+container_y**2)**0.5/2

        obj_size = obj.get_bounding_box()
        obj_x, obj_y, obj_h = obj_size[1] - obj_size[0], obj_size[3] - obj_size[2], obj_size[5] - obj_size[4]
        h_offset = max(obj_h, obj_x, obj_y)/1.5
        h = h_offset + container_pos[2] + container_size[5]
        random_r = np.random.uniform(0, container_r*0.5, n_sample)
        random_angle = np.random.uniform(0, np.pi, n_sample)
        obj_goal_pose_set = np.zeros((n_sample, 4, 4))
        # obj_goal_pose_set[:, :3, 3] = np.array([container_pos[0], container_pos[1], h])
        obj_goal_pose_set[:, 3, 3] = 1
        axis_y = np.array([0, 0, -1])
        obj_goal_pose_set[:, :3, 1] = axis_y
        tmp = np.array((0, 1, 0))
        theta = 2*np.pi/ n_sample
        for i in range(n_sample):
            x = container_pos[0]+ random_r[i]*np.cos(random_angle[i])
            y = container_pos[1]+ random_r[i]*np.sin(random_angle[i])
            sample_h = h + np.random.normal(scale=0.01)
            obj_goal_pose_set[i, :3, 3] = np.array([x, y, sample_h])
            axis_x = np.cos(i*theta) * tmp + np.sin(i*theta) * np.cross(axis_y, tmp)
            axis_x = axis_x/ np.linalg.norm(axis_x)
            obj_goal_pose_set[i, :3, 0] = axis_x
            obj_goal_pose_set[i, :3, 2] = np.cross(axis_x, axis_y)
        rand_idx = np.random.permutation(len(obj_goal_pose_set))
        obj_goal_pose_set = obj_goal_pose_set[rand_idx]
        return obj_goal_pose_set

    def import_objects(self, num=2):
        if not hasattr(self, "model_path"):
            model_path = self.model_dir+"pencil/pencil1/pencil1.ttm"
        else:
            model_path = self.model_dir+self.model_path
        for i in range(num):
            obj = VLM_Object(self.pyrep, model_path, i)
            obj.set_parent(self.taks_base)
            self.object_list.append(obj)
        self.register_graspable_objects(self.object_list)
        target_path = self.model_dir+"container/basket1.ttm"
        for i in range(2):
            target = self.pyrep.import_model(target_path)
            target.set_model_dynamic(False)
            target.scale_factor = lib.simGetObjectSizeFactor(ffi.cast('int',target._handle))
            target.set_parent(self.taks_base)
            for children in target.get_objects_in_tree(exclude_base=True):
                if children.get_type() == ObjectType.PROXIMITY_SENSOR:
                    target.successor = children
                if "visual" in children.get_name():
                    target.visual = children
            self.target_list.append(target)
        return

    def is_static_workspace(self) -> bool:
        return True
        
    def sample_method(self):
        self.spawn_space.clear()
        for tg in self.target_list:
            self.spawn_space.sample(tg, min_distance=0.1)
        for obj in self.object_list:
            self.spawn_space.sample(obj, min_distance=0.2)
        for _ in range(5):
            self.pyrep.step()
    
    def load(self, ttms_folder=None):
        if Shape.exists('drop_pen'):
            return Dummy('drop_pen')
        ttm_file = os.path.join(ttms_folder, 'drop_pen.ttm')
        self._base_object = self.pyrep.import_model(ttm_file)
        return self._base_object
    
    def get_base(self) -> Dummy:
        self._base_object = Dummy('drop_pen')
        return self._base_object

