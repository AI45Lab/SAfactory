import random
from turtle import shape
from typing import List
import numpy as np
import os
from pyrep.objects.shape import Shape
from pyrep.objects.dummy import Dummy
from pyrep.const import PYREP_SCRIPT_TYPE, PrimitiveShape
from pyrep.objects.proximity_sensor import ProximitySensor
from amsolver.backend.task import Task
from amsolver.backend.unit_tasks import T0_ObtainControl, T1_MoveObjectGoal, T2_MoveObjectConstraints, TargetSpace, VLM_Object
from amsolver.backend.utils import get_relative_position_xy, scale_object
from amsolver.const import mug_list
from amsolver.backend.conditions import ConditionSet, DetectedCondition, Condition
from amsolver.backend.spawn_boundary import SpawnBoundary
from scipy.spatial.transform import Rotation as R

class PourDemo(Task):

    def init_task(self) -> None:
        self.spawn_space = SpawnBoundary([Shape('workspace')])
        self.model_dir = os.path.dirname(os.path.realpath(__file__)).replace("tasks","object_models/")
        self.temporary_waypoints = []
        self.taks_base = self.get_base()
        if not hasattr(self, "model_num"):
            self.model_num = 2
        
        self.ignore_collisions = False

    def init_episode(self, index: int) -> List[str]:
        # self.pyrep.set_configuration_tree(self.task_init_states)
        self.import_objects(self.model_num)
        self.modified_init_episode(index)
        self.variation_index = index
        try_times = 200
        pour_obj = self.object_list[0]
        recv_obj = self.object_list[1]
        self.manipulated_obj = pour_obj.manipulated_part
        # recv = self.object_list[1].manipulated_part
        # successor0 = self.object_list[0].succssor
        # successor1 = self.object_list[1].succssor
        while len(self.temporary_waypoints)==0 and try_times > 0:
            self.sample_method()
            init_states = self.get_state()[0]
            # pre_pour_pose_space = self.calculate_pre_pour_pose(self.manipulated_obj, recv)
            pour_descriptions = f"Rotate {self.manipulated_obj.descriptions} toward {recv_obj.manipulated_part.descriptions}."
            pour_space_args = {"container_pour": pour_obj, "container_recv":recv_obj}
            pour_space = TargetSpace(self.create_pour_goal_pose,space_args=pour_space_args,
                                    target_space_descriptions=pour_descriptions, focus_obj_id=self.manipulated_obj.visual)
            pour_space.set_target(pour_obj.buttom_point, try_ik_sampling=False, linear=True, ignore_collisions=self.ignore_collisions)
            MoveTask1 = T2_MoveObjectConstraints(self.robot, self.pyrep, pour_space, self.taks_base,fail_times=36)

            move_descriptions = f"Move the object to the top of {recv_obj.manipulated_part.descriptions} with the opening upwards."
            target_space_args = {"container_pour": pour_obj, "container_recv":recv_obj, "tip":self.robot.arm._ik_tip,"angle_threshold":np.pi/8}
            target_space = TargetSpace(self.calculate_pre_pour_pose,space_args=target_space_args,
                                target_space_descriptions=move_descriptions, focus_obj_id=recv_obj.manipulated_part.visual)
            target_space.set_target(pour_obj.buttom_point, try_ik_sampling=False, linear=True)
            # target_space = TargetSpace(pre_pour_pose_space.tolist(), None, None, None, "the receiver mug", recv.visual)
            # target_space.set_target(self.manipulated_obj, try_ik_sampling=False, release=False)
            MoveTask0 = T2_MoveObjectConstraints(self.robot, self.pyrep, target_space, self.taks_base,fail_times=10,
                                                next_task_fuc=MoveTask1.get_path_with_constraints)
            # MoveTask0 = T1_MoveObjectGoal(self.robot, self.pyrep, self.target_space, self.taks_base, fail_times=10)
            GraspTask0 = T0_ObtainControl(self.robot, self.pyrep, self.manipulated_obj,self.taks_base,\
                        grasp_sort_key="horizontal",next_task_fuc=MoveTask0.get_path_with_constraints, try_times=200)
            if try_times<100:
                waypoints = GraspTask0.get_path(try_ik_sampling=True)
            else:
                waypoints = GraspTask0.get_path(try_ik_sampling=False)
            if waypoints is not None:
                self.temporary_waypoints += waypoints
            self.reset_robot()
            self.pyrep.set_configuration_tree(init_states)
            self.pyrep.step()
            try_times-=1
        for i,waypoint in enumerate(self.temporary_waypoints):
            waypoint.set_name('waypoint{}'.format(i))
        self.drops = []
        conditions = []
        LIQUID_BALLS = 20
        success_rate = 0.5
        for i in range(LIQUID_BALLS):
            drop = Shape.create(PrimitiveShape.SPHERE, mass=0.0001, size=[0.005, 0.005, 0.005])
            drop.set_parent(self.taks_base)
            drop.set_color([0.1, 0.1, 0.9])
            drop.set_position(list(np.random.normal(0, 0.0005, size=(3,))), relative_to=self.manipulated_obj)
            self.drops.append(drop)
            self.pyrep.step()
            conditions.append(DetectedCondition(drop, self.object_list[1].succssor))
            # if np.random.rand() < success_rate:
            #     conditions.append(DetectedCondition(drop, self.object_list[1].succssor))
        self.register_success_conditions([NumberCondition(conditions, LIQUID_BALLS*success_rate)])
        description = f"Pour water from {self.manipulated_obj.descriptions} to {recv_obj.manipulated_part.descriptions}."
        return [description]

    def variation_count(self) -> int:
        # TODO: The number of variations for this task.
        return 1

    def cleanup(self) -> None:
        for d in self.drops:
            if d.still_exists():
                d.remove()
        self.drops.clear()
        super().cleanup()

    def is_static_workspace(self) -> bool:
        return True

    def load(self, ttms_folder=None):
        if Shape.exists('pour_demo'):
            return Dummy('pour_demo')
        ttm_file = os.path.join(ttms_folder, 'pour_demo.ttm')
        self._base_object = self.pyrep.import_model(ttm_file)
        return self._base_object
    
    def get_base(self) -> Dummy:
        self._base_object = Dummy('pour_demo')
        return self._base_object
    
    def import_objects(self, num=2):
        self.object_list = []
        if not hasattr(self, "model_path"):
            selected_obj = random.choice(mug_list)
            model_path = self.model_dir+selected_obj['path']
        else:
            model_path = self.model_dir+self.model_path
        for i in range(num):
            obj = VLM_Object(self.pyrep, model_path, i)
            succssor = ProximitySensor(f"success{i}")
            buttom_point = Dummy.create()
            buttom_point.set_name(f"buttom_point{i}")
            buttom_point.set_parent(obj)
            buttom_point.set_position(succssor.get_position())
            buttom_point.set_orientation(obj.get_orientation())
            obj.succssor = succssor
            obj.buttom_point = buttom_point
            # relative_factor = scale_object(obj, 1.5)
            # if abs(relative_factor-1)>1e-2:
            #     local_grasp_pose = obj.manipulated_part.local_grasp
            #     local_grasp_pose[:, :3, 3] *= relative_factor
            #     obj.manipulated_part.local_grasp = local_grasp_pose
            obj.set_parent(self.taks_base)
            self.object_list.append(obj)
            self._need_remove_objects.append(obj)
        self.register_graspable_objects(self.object_list)
        self.pyrep.step()
    
    def sample_method(self):
        self.spawn_space.clear()
        for obj in self.object_list:
            self.spawn_space.sample(obj, min_distance=0.1)
        for _ in range(5):
            self.pyrep.step()

    @staticmethod
    def calculate_pre_pour_pose(container_pour, container_recv, tip, n_pos_sample=20, n_rot_sample=5, angle_threshold=np.pi/8):
        # randomize the container to pour with its center on a horizontal circle whose center is right above the container to receive
        # radius = radius of container to receive + half height of container to pour
        # container to pour needs to have z axis upwards, no other rotational constraints
        recv_pos = container_recv.buttom_point.get_position()
        pour_size = container_pour.get_bounding_box() # min_x, max_x, min_y, max_y, min_z, max_z
        recv_size = container_recv.get_bounding_box()
        pour_x, pour_y, pour_h = pour_size[1] - pour_size[0], pour_size[3] - pour_size[2], pour_size[5] - pour_size[4]
        recv_x, recv_y, recv_h = recv_size[1] - recv_size[0], recv_size[3] - recv_size[2], recv_size[5] - recv_size[4]
        
        succssor_size = container_recv.succssor.get_bounding_box()
        succ_x, succ_y = succssor_size[1] - succssor_size[0], succssor_size[3] - succssor_size[2]
        # r = (succ_x**2 + succ_y**2)**0.5/2
        r = pour_h
        h_offset = max(pour_h, pour_x, pour_y)
        h = h_offset + recv_pos[2] + recv_h / 2

        h += np.random.normal(scale=0.005)
        # make sure <gripper->container_pour, container_pour->container_recv> < angle_threshold
        # pos_vec, rot_vec = [], []
        pose_vec = []
        pour2gripper_pos = container_pour.buttom_point.get_position(tip) # container_pour.get_parent().get_position(container_pour)
        pour2gripper_theta = np.arctan2(pour2gripper_pos[1], pour2gripper_pos[0])
        for i in range(n_pos_sample):
            angle = i * 2*np.pi / n_pos_sample
            x, y = recv_pos[0] + r * np.cos(angle), recv_pos[1] + r * np.sin(angle)
            for j in range(n_rot_sample):
                rz = angle - angle_threshold + (j+0.5) * 2*angle_threshold / n_rot_sample - pour2gripper_theta
                # rz += np.pi
                # pos_vec.append((x, y, h))
                # rot_vec.append((0, 0, rz))
                pose_vec.append([x, y, h, 0, 0, rz])
                pose_vec.append([x, y, h, 0, 0, rz+np.pi])
                # dm = Dummy.create()
                # dm.set_position([x, y, h])
                # dm.set_orientation([0, 0, rz])
        # rand_idx = np.random.permutation(len(pos_vec))
        rand_idx = np.random.permutation(len(pose_vec))
        pose_vec = np.array(pose_vec)[rand_idx]
        pose_matrx = np.zeros((pose_vec.shape[0], 4, 4))
        for i, vec in enumerate(pose_vec):
            matrix = np.eye(4)
            matrix[:3, 3] = vec[:3]
            matrix[:3, :3] = R.from_euler("xyz", vec[3:]).as_matrix()
            pose_matrx[i] = matrix
        return pose_matrx
    
    @staticmethod
    def create_pour_goal_pose(container_pour, container_recv, n_sample=36):
        obj_goal_pose_set = np.zeros((n_sample, 4, 4))
        obj_goal_pose_set[:, 3, 3] = 1
        obj_goal_pose_set[:, :3, 3] = container_pour.buttom_point.get_position()
        axis_z = container_recv.buttom_point.get_position() - container_pour.buttom_point.get_position()
        pour_angle = np.random.uniform(20, 40)
        axis_z[2] = -np.linalg.norm(axis_z[:2])*np.tan(np.deg2rad(pour_angle))
        axis_z = axis_z / np.linalg.norm(axis_z)
        obj_goal_pose_set[:, :3, 2] = axis_z
        tmp = np.cross([1, 0, 1], axis_z)
        tmp = tmp / np.linalg.norm(tmp)
        theta = 2*np.pi / n_sample
        for i in range(n_sample):
            axis_x = np.cos(i*theta) * tmp + np.sin(i*theta) * np.cross(tmp, axis_z)
            axis_x = axis_x / np.linalg.norm(axis_x)
            obj_goal_pose_set[i, :3, 0] = axis_x
            obj_goal_pose_set[i, :3, 1] = np.cross(axis_z, axis_x)
            # dm = Dummy.create(size=0.005)
            # dm.set_matrix(obj_goal_pose_set[i])
        rand_idx = np.random.permutation(len(obj_goal_pose_set))
        obj_goal_pose_set = obj_goal_pose_set[rand_idx]
        return obj_goal_pose_set

class NumberCondition(Condition):

    def __init__(self,  conditions: List[Condition], num_bound):
        self._conditions = conditions
        self.num_bound = num_bound

    def condition_met(self):
        count = 0
        for cond in self._conditions:
            ismet, term = cond.condition_met()
            if ismet:
                count+= 1
        met = count >= self.num_bound
        return met, False