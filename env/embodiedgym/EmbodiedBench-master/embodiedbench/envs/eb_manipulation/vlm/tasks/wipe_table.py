import random
from typing import List
import numpy as np
import os
from scipy.spatial.transform import Rotation as R
from pyrep.objects.dummy import Dummy
from pyrep.const import ObjectType, PrimitiveShape
from amsolver.backend.task import Task
from pyrep.objects.shape import Shape
from pyrep.objects.proximity_sensor import ProximitySensor
from amsolver.backend.unit_tasks import T0_ObtainControl, T1_MoveObjectGoal, T2_MoveObjectConstraints, TargetSpace, VLM_Object
from amsolver.backend.utils import scale_object
from amsolver.const import colors, planes
from amsolver.backend.conditions import DetectedCondition
from amsolver.backend.spawn_boundary import SpawnBoundary
from amsolver.backend.task import Task
from amsolver.backend.conditions import Condition
from pyrep.backend._sim_cffi import ffi, lib

DIRT_POINTS = 50
class WipeTable(Task):

    def init_task(self) -> None:
        self.spawn_space = SpawnBoundary([Shape('workspace')])
        self.model_dir = os.path.dirname(os.path.realpath(__file__)).replace("tasks","object_models/")
        self.temporary_waypoints = []
        self.object_list = []
        self.target_list = []
        self.dirt_spots = []
        self.taks_base = self.get_base()
        if not hasattr(self, "model_num"):
            self.model_num = 1
        if not hasattr(self, "area_num"):
            self.area_num = 2
        if not hasattr(self, "area_class_num"):
            self.area_class_num = 1
        self.import_objects(self.model_num)
    
    def init_episode(self, index: int) -> List[str]:
        try_init_episode_times = 200
        self.variation_index = index
        self.create_area()
        self.modified_init_episode(index)
        self.manipulated_obj = self.object_list[0].manipulated_part
        self.sensor = self.object_list[0].sensor
        self.sensor_name = self.sensor.get_name()
        target = self.target_list[0]
        self.manipulated_obj.descriptions = "the {}".format(self.manipulated_obj.property["shape"])
        while len(self.temporary_waypoints)==0 and try_init_episode_times>0:
            self.sample_method()
            # self.register_success_conditions([DetectedCondition(self.manipulated_obj, self.target_space.successor)])
            init_states = self.get_state()[0]
            end_args = {"target":target,"step":"end"}
            if target.directional:
                target_space_descriptions1 = f"Move the object along the main direction of {target.target_space_descriptions}."
            else:
                target_space_descriptions1 = f"Move the object cross the center of {target.target_space_descriptions}."
            target_space1 = TargetSpace(self.move_to_point, space_args=end_args,
                    target_space_descriptions = target_space_descriptions1, focus_obj_id= target.get_handle())
            target_space1.set_target(self.sensor, try_ik_sampling=False, linear=True, ignore_collisions=True, release=False)
            MoveTask1 = T2_MoveObjectConstraints(self.robot, self.pyrep, target_space1, self.taks_base, fail_times=2)
            
            start_args = {"target":target,"step":"start"}
            target_space_descriptions0 = f"to the side of {target.target_space_descriptions}"
            target_space0 = TargetSpace(self.move_to_point, space_args=start_args,
                    target_space_descriptions = target_space_descriptions0, focus_obj_id= target.get_handle())
            target_space0.set_target(self.sensor, try_ik_sampling=False, linear=False, ignore_collisions=True, release=False)
            MoveTask0 = T1_MoveObjectGoal(self.robot, self.pyrep, target_space0, self.taks_base, fail_times=2, next_task_fuc=MoveTask1.get_path_with_constraints)
            # self.target_space.set_target(self.manipulated_obj, try_ik_sampling=False, release=True)
            GraspTask = T0_ObtainControl(self.robot, self.pyrep, self.manipulated_obj,self.taks_base, try_times=200,
                                     next_task_fuc=MoveTask0.get_path)
            if try_init_episode_times>100:
                waypoints = GraspTask.get_path(try_ik_sampling=False)
            else:
                waypoints = GraspTask.get_path(try_ik_sampling=True)
            if waypoints is not None:
                self.temporary_waypoints += waypoints
            self.reset_robot()
            self.pyrep.set_configuration_tree(init_states)
            try_init_episode_times -= 1
        self._place_dirt(target)
        self.register_success_conditions([LengthCondition(self.dirt_spots, DIRT_POINTS*0.5)])
        for i,waypoint in enumerate(self.temporary_waypoints):
            waypoint.set_name('waypoint{}'.format(i))
        description = f"Wipe {target.target_space_descriptions}."
        return [description]
    
    def is_static_workspace(self) -> bool:
        return True

    def load(self, ttms_folder=None):
        if Shape.exists('wipe_table'):
            return Dummy('wipe_table')
        ttm_file = os.path.join(ttms_folder, 'wipe_table.ttm')
        self._base_object = self.pyrep.import_model(ttm_file)
        return self._base_object
    
    def get_base(self) -> Dummy:
        self._base_object = Dummy('wipe_table')
        return self._base_object

    def import_objects(self, num=1):
        if not hasattr(self, "model_path"):
            model_path = self.model_dir+"wiper/sponge/sponge.ttm"
        else:
            model_path = self.model_dir+self.model_path
        for i in range(num):
            obj = VLM_Object(self.pyrep, model_path, i)
            for children in obj.get_objects_in_tree(exclude_base=True):
                if children.get_type() == ObjectType.PROXIMITY_SENSOR:
                    obj.sensor = children
            obj.set_parent(self.taks_base)
            self.object_list.append(obj)
        self.register_graspable_objects(self.object_list)

    def create_area(self):
        self.target_list = []
        self.shape_lib = {}
        selected_objs = random.sample(list(planes.keys()), self.area_class_num)
        for selected_obj in selected_objs:
            model_path = self.model_dir+planes[selected_obj]['path']
            self.shape_lib[selected_obj] = []
            for i in range(self.area_num):
                dirt_area = self.pyrep.import_model(model_path)
                dirt_area.scale_factor = lib.simGetObjectSizeFactor(ffi.cast('int',dirt_area._handle))
                scale_object(dirt_area, np.random.uniform(0.9, 1.1))
                dirt_area.directional = planes[selected_obj]['directional']
                dirt_area.set_parent(self.taks_base)
                dirt_area.target_space_descriptions = f"the {selected_obj} area"
                self.shape_lib[selected_obj].append(dirt_area)
                self.target_list.append(dirt_area)
                self._need_remove_objects.append(dirt_area)
    
    def sample_method(self):
        self.spawn_space.clear()
        
        for target in self.target_list:
            self.spawn_space.sample(target, min_distance=0.1)

        for obj in self.object_list:
            self.spawn_space.sample(obj, min_distance=0.1)
    
    @staticmethod
    def move_to_point(target: Shape, step, n_sample=36):
        target_pose = target.get_matrix()
        target_size = target.get_bounding_box()
        if step == "start":
            offset = target_size[0]
        else:
            offset = target_size[1]
        related_pos = target_pose.dot(np.array([offset, 0, 0, 1]))
        obj_poses = np.zeros((n_sample, 4, 4))
        obj_poses[:, :, 3] = related_pos
        theta = 2*np.pi/ n_sample
        for i in range(n_sample):
            rot_m = R.from_euler("xyz", [0, 0, i*theta]).as_matrix()
            obj_poses[i, :3, :3] = target_pose[:3, :3].dot(rot_m)

        return obj_poses
    
    def _place_dirt(self, space):
        target_pose = space.get_matrix()
        target_size = space.get_bounding_box()
        if space.directional:
            step = (target_size[1]-target_size[0])/DIRT_POINTS
            for i in range(DIRT_POINTS):
                spot = Shape.create(type=PrimitiveShape.CUBOID,
                                    size=[.005, .005, .001],
                                    mass=0, static=True, respondable=False,
                                    renderable=False,
                                    color=[0.58, 0.29, 0.0])
                spot.set_parent(space)
                delta_y = np.random.normal(scale=0.005)
                related_pos = target_pose.dot(np.array([target_size[0]+step*i, delta_y, 0.001, 1]))
                spot.set_position(related_pos[:3])
                self.dirt_spots.append(spot.get_name())
        else:
            max_r = min(target_size[1], target_size[3])/2
            for i in range(DIRT_POINTS):
                spot = Shape.create(type=PrimitiveShape.CUBOID,
                                    size=[.005, .005, .001],
                                    mass=0, static=True, respondable=False,
                                    renderable=False,
                                    color=[0.58, 0.29, 0.0])
                spot.set_parent(space)
                r = np.random.uniform(0, max_r)
                theta = np.random.uniform(-np.pi, np.pi)
                x, y = r*np.cos(theta), r*np.sin(theta)
                related_pos = target_pose.dot(np.array([x, y, 0.001, 1]))
                spot.set_position(related_pos[:3])
                self.dirt_spots.append(spot.get_name())
    
    def step(self) -> None:
        sensor = ProximitySensor(self.sensor_name)
        for d in self.dirt_spots:
            if sensor.is_detected(Shape(d)):
                self.dirt_spots.remove(d)
                Shape(d).remove()

    def cleanup(self) -> None:
        for d in self.dirt_spots:
            Shape(d).remove()
        self.dirt_spots = []
        super().cleanup()

class LengthCondition(Condition):

    def __init__(self, container: list, num_bound):
        self._container = container
        self.num_bound = num_bound

    def condition_met(self):
        count = 0
        for obj in self._container:
            if Shape.exists(obj):
                count+=1
        met = count <= self.num_bound
        return met, False