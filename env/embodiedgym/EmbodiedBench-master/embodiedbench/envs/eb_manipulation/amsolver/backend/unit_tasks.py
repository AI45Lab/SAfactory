#Modified From the rlbench: https://github.com/stepjam/RLBench
from copy import deepcopy
import json
import pickle
from platform import release
import numpy as np
import os
from typing import Any, Dict, List, Tuple, Union
from scipy.spatial.transform import Rotation as R
from pyrep.pyrep import PyRep
from pyrep.objects.shape import Shape
from pyrep.objects.joint import Joint
from pyrep.objects.dummy import Dummy
from pyrep.objects.proximity_sensor import ProximitySensor
from pyrep.objects.cartesian_path import CartesianPath
from pyrep.const import PYREP_SCRIPT_TYPE, JointType
from pyrep.backend._sim_cffi import ffi, lib
from amsolver.backend.conditions import DetectedCondition
from amsolver.backend.robot import Robot
from amsolver.backend.spawn_boundary import BoundingBox, SpawnBoundary
from amsolver.backend.task import Task
from amsolver.backend.utils import WriteCustomDataBlock, execute_grasp, execute_path, get_relative_position_xy, get_sorted_grasp_pose, test_reachability

gripper_step = 0.5
def fast_path_test(point, robot):
    final_config = point._path_points[-len(robot.arm.joints):]
    robot.arm.set_joint_positions(final_config, True)
    
class VLM_Object(Shape):
    def __init__(self, pr: PyRep, model_path: str, instance_id: int):
        with open(model_path.replace("ttm", "json"), 'r') as f:
            self.config = json.load(f)
        self.obj_class = self.config["class"]
        self.highest_part_name = self.config["parts"][self.config["highest_part"]]["name"]
        if self.exists(self.highest_part_name):
            super().__init__(self.highest_part_name)
        else:
            model = pr.import_model(model_path)
            super().__init__(model.get_handle())
        self.scale_factor = lib.simGetObjectSizeFactor(ffi.cast('int',self._handle))
        self.instance_id = instance_id
        for m in self.get_objects_in_tree(exclude_base=False):
            m.set_name(m.get_name()+str(instance_id))
        self.under_control = False
        self.parts = []
        for p in self.config["parts"]:
            part = Shape(p["name"]+str(instance_id))
            part.graspable = p["graspable"]
            part.local_grasp = None
            if "local_grasp_pose_path" in p:
                grasp_pose_path = os.path.join(os.path.dirname(model_path),p["local_grasp_pose_path"])
                with open(grasp_pose_path, 'rb') as f:
                    part.local_grasp = pickle.load(f)
            part.property = p["property"]
            if self.exists(p["name"]+"_visual"+str(instance_id)):
                part.set_transparency(0)
                part.visual = Shape(p["name"]+"_visual"+str(instance_id)).get_handle()
            else:
                part.set_transparency(1)
                part.visual = part.get_handle()
            part.descriptions = None
            self.parts.append(part)

        self.manipulated_parts = []
        for i in self.config["manipulated_part"]:
            self.manipulated_parts.append(self.parts[i])
        self.manipulated_part = self.manipulated_parts[0]
        
        self.articulated = self.config["articulated"]
        if self.articulated:
            self.constraints = []
            for c_name in self.config["constraints"]:
                name = c_name+str(instance_id)
                if Shape.exists(name):
                    if Shape.get_object_type(name).name == "JOINT":
                        constraint = Joint(name)
                        self.constraints.append(constraint)
        else:
            self.constraints = None

class TargetSpace(object):
    def __init__(self, space: SpawnBoundary or Joint or set, successor=None, min_range=None, max_range=None, 
                target_space_descriptions=None, focus_obj_id=None, space_args=None) -> None:
        super().__init__()
        self.space = space
        self.min_range = min_range
        self.max_range = max_range
        self.target_space_descriptions = target_space_descriptions
        self.focus_obj_id = focus_obj_id
        self.successor = successor
        self.space_args = space_args

    def set_target(self, target_object:Shape, try_ik_sampling=True, linear=False, ignore_collisions=False, release=False):
        self.target_object = target_object
        self.try_ik_sampling = try_ik_sampling
        self.linear = linear
        self.ignore_collisions = ignore_collisions
        self.release = release

class T0_ObtainControl(object):
    def __init__(self, robot: Robot, pyrep: PyRep, target_obj:Shape, task_base:Dummy, try_times = 200,
                        table_height=0.752,need_post_grasp=True, 
                        table_offset_dist=0.01, 
                        pregrasp_dist=0.08, 
                        postgrasp_height=0.1, grasp_sort_key="vertical",*, next_task_fuc=None,next_task_args=None) -> None:
        super().__init__()
        self.robot = robot
        self.pyrep = pyrep
        self.task_base = task_base
        self.target_obj = target_obj
        
        self.try_times = try_times
        self.table_height = table_height
        self.table_offset_dist = table_offset_dist
        self.need_post_grasp = need_post_grasp
        self.pregrasp_dist = pregrasp_dist
        self.postgrasp_height = postgrasp_height
        self.grasp_sort_key = grasp_sort_key
        self.next_task_fuc = next_task_fuc
        self.next_task_args = next_task_args

    @staticmethod
    def sample_postgrasp_pose(grasp_pose, offset_height):
        # sample 3d pos, remain same orientation in a cuboid area above grasp pose
        def sample_from_center(center, range, sample):
            pos_list = np.linspace(center, center + range, sample // 2).tolist()
            neg_list = np.linspace(center, center - range, sample // 2).tolist()
            result = [None] * (len(pos_list) + len(neg_list) - 1)
            result[::2] = pos_list
            result[1::2] = neg_list[1:]
            return result
        pose_list = []
        xy_range = 0.03
        sample_n = 10
        for h in sample_from_center(offset_height, offset_height / 2, sample_n):
            for x in sample_from_center(0, xy_range, sample_n):
                for y in sample_from_center(0, xy_range, sample_n):
                    pose = deepcopy(grasp_pose)
                    pose[:3, 3] -= pose[:3,:3].dot([x, y, h])
                    pose_list.append(pose)
        return np.array(pose_list)
    
    @staticmethod
    def check_path_length(path, pose1, pose2, arm, threshold=3.0):
        # print('checking path length')
        linear_dist = np.linalg.norm(pose1[:3, 3] - pose2[:3, 3])
        path_dist = 0
        tip = arm.get_tip()
        init_angles = arm.get_joint_positions()
        arm.set_joint_positions(path._path_points[0: len(arm.joints)])
        prev_point = tip.get_position()
        for i in range(len(arm.joints), len(path._path_points),
                        len(arm.joints)):
            points = path._path_points[i:i + len(arm.joints)]
            arm.set_joint_positions(points)
            p = tip.get_position()
            path_dist += np.linalg.norm(p - prev_point)
            prev_point = p
        arm.set_joint_positions(init_angles)
        return path_dist / linear_dist < threshold

    def test_reachability(self, arm, pose, try_ik_sampling=False, linear=False, ignore_collisions=False):
        new_target = Dummy.create()
        new_target.set_matrix(pose)
        pos, ori = new_target.get_position(), new_target.get_orientation()
        res, path = False, None
        success = False
        try:
            _ = arm.solve_ik_via_jacobian(pos, ori)
            success = True
        except:
            if try_ik_sampling:
                try:
                    _ = arm.solve_ik_via_sampling(pos, ori) # much slower than jacobian
                    success = True
                except:
                    pass
            else:
                pass
        if success:
            try:
                path = arm.get_linear_path(pos, ori, ignore_collisions=ignore_collisions) if linear else arm.get_path(pos, ori, ignore_collisions=ignore_collisions)
                res = True
            except:
                pass
        new_target.remove()
        return res, path

    def get_path(self, try_ik_sampling=False, linear=False, ignore_collisions=False):
        self.object_pose = self.target_obj.get_matrix()
        sorted_grasp_pose = get_sorted_grasp_pose(self.object_pose, self.target_obj.local_grasp, self.grasp_sort_key)
        sorted_grasp_pose[:,:3, 3] -= sorted_grasp_pose[:,:3, 2] * np.random.normal(scale=0.005, size=(sorted_grasp_pose.shape[0],1))
        objs_init_states = self.task_base.get_configuration_tree()
        saved_states = self.robot.save_state()
        success_grasps = None
        trial = 0
        grasp_pose_idx = 0
        arm = self.robot.arm
        gripper = self.robot.gripper
        waypoint_dumpy = Dummy.create(size=0.03)
        init_pose_dummy = Dummy.create()
        init_pose_dummy.set_matrix(self.object_pose)
        find_path = False
        if self.try_times == -1:
            self.try_times = len(sorted_grasp_pose)
        grasp_index_step = len(sorted_grasp_pose)//self.try_times // 2
        grasp_index_step = max(1, grasp_index_step)
        while trial < self.try_times and grasp_pose_idx < len(sorted_grasp_pose):
            # print('trial:', trial, 'grasp idx:', grasp_pose_idx)
            success = False
            grasp_pose = sorted_grasp_pose[grasp_pose_idx]
            pre_grasp_pose = deepcopy(grasp_pose)
            pre_grasp_pose[:3, 3] -= pre_grasp_pose[:3, 2] * self.pregrasp_dist
            grasp_pose_idx += grasp_index_step
            if pre_grasp_pose[2, 3] - self.table_height < self.table_offset_dist:
                # print('pre-grasp pose too near to table, ignore')
                continue
            trial += 1
            post_grasp_pose = deepcopy(grasp_pose)
            post_grasp_pose[2, 3] += self.postgrasp_height
            # grasp_pose[:3, 3] += grasp_pose[:3, 2] * 0.01
            waypoint_dumpy.set_matrix(grasp_pose)
            success, path0 = test_reachability(arm, pre_grasp_pose, try_ik_sampling=try_ik_sampling, linear = linear, ignore_collisions=ignore_collisions)
            moved = False
            if success:
                # move arm to pregrasp pose, then test grasp, this time only check cartesian move (linear path)
                # execute_path(path0, self.pyrep)
                fast_path_test(path0, self.robot)
                # path0.set_to_end()
                moved = True
                # print('pregrasp path:', success)
                success, path1 = test_reachability(arm, grasp_pose, False, linear=True,  ignore_collisions=ignore_collisions)
                if success:
                    # test grasp here
                    # execute_path(path1, self.pyrep)
                    fast_path_test(path1, self.robot)
                    # path1.set_to_end()
                    # print('grasp path:', success)
                    # print('moving ')
                    success = execute_grasp(gripper, self.target_obj, self.pyrep)
                    grasp_arm_config = arm.get_joint_positions()
                    grasp_gripper_config = gripper.get_joint_positions()
                    # obj.set_dynamic(True)
                    # print('gripper grasp:', success)
                    if success:
                    # move arm to grasp pose, grasp, test whether in hand, then test lift, this time only check cartesian move (linear path)
                    # execute_path(path1, pyrep)
                        if self.need_post_grasp:
                            post_grasp_pose_set = self.sample_postgrasp_pose(grasp_pose, self.postgrasp_height)
                            for post_grasp_pose in post_grasp_pose_set:
                                success, path2 = self.test_reachability(arm, post_grasp_pose, ignore_collisions=True)
                                if success and self.check_path_length(path2, grasp_pose, post_grasp_pose, arm):
                                    break
                            # print('postgrasp path:', success)
                            if success:
                                # execute_path(path2, self.pyrep)
                                # path2.set_to_end()
                                fast_path_test(path2, self.robot)
                                find_path = True
                                # print('Find a grasp!')
                        else:
                            post_grasp_pose = None
                            path2 = None
                            find_path=True
                            # print('Find a grasp!')
            if find_path:
                grasp_point = {}
                grasp_point['waypoints'] = [pre_grasp_pose, grasp_pose, post_grasp_pose]
                grasp_point['path'] = [path0, path1, path2]
                grasp_point['grasp_arm_config'] = grasp_arm_config
                grasp_point['grasp_gripper_config'] = grasp_gripper_config
                waypoint_dumpy.set_matrix(pre_grasp_pose)
                relative_position_description = get_relative_position_xy(init_pose_dummy, waypoint_dumpy, arm)
                grasp_point['relative'] = relative_position_description
                # success_grasps.append(grasp_point)
                find_path = False
                waypoints = []
                for i,pose in enumerate([pre_grasp_pose, grasp_pose, post_grasp_pose]):
                    if pose is not None:
                        waypoint = Dummy.create()
                        waypoint.set_matrix(pose)
                        if i==0:
                            low_level_descriptions = "Move to the {} of {}.".format(relative_position_description,self.target_obj.descriptions)
                            WriteCustomDataBlock(waypoint.get_handle(),"waypoint_type","pre_grasp")
                        elif i==1:
                            low_level_descriptions = "Grasp {}.".format(self.target_obj.descriptions)
                            WriteCustomDataBlock(waypoint.get_handle(),"gripper","['close',0]")
                            WriteCustomDataBlock(waypoint.get_handle(),"waypoint_type","grasp")
                        elif i==2:
                            low_level_descriptions = "Move upward."
                            WriteCustomDataBlock(waypoint.get_handle(),"waypoint_type","post_grasp")
                            # WriteCustomDataBlock(waypoint.get_handle(),"ignore_collisions","True")
                        WriteCustomDataBlock(waypoint.get_handle(),"low_level_descriptions",low_level_descriptions)
                        WriteCustomDataBlock(waypoint.get_handle(),"focus_obj_id",str(self.target_obj.visual))
                        WriteCustomDataBlock(waypoint.get_handle(),"focus_obj_name",Shape.get_object_name(self.target_obj.visual))
                        if ignore_collisions or i==2:
                            WriteCustomDataBlock(waypoint.get_handle(),"ignore_collisions","True")
                        waypoints.append(waypoint)
                if self.next_task_fuc is not None:
                    next_path = self.next_task_fuc() if self.next_task_args is None else self.next_task_fuc(**self.next_task_args)
                    if next_path is None:
                        for w in waypoints:
                            w.remove()
                        self.robot.recover_state(saved_states, release=True)
                        self.pyrep.set_configuration_tree(objs_init_states)
                        self.pyrep.step()
                        continue
                    else:
                        success_grasps = waypoints + next_path
                        break
                else:
                    success_grasps = waypoints
                    break
            if moved:
                self.robot.recover_state(saved_states, release=True)
                self.pyrep.set_configuration_tree(objs_init_states)
                self.pyrep.step()
        waypoint_dumpy.remove()
        init_pose_dummy.remove()
        return success_grasps

class T1_MoveObjectGoal(object):
    def __init__(self, robot: Robot, pyrep: PyRep, target_space:TargetSpace, task_base,
                fail_times = 10, *, next_task_fuc=None,next_task_args=None) -> None:
        super().__init__()
        self.robot = robot
        self.pyrep = pyrep
        self.target_space = target_space
        self.arm = robot.arm
        self.gripper = robot.gripper
        self.fail_times = fail_times
        self.task_base = task_base

        self.next_task_fuc = next_task_fuc
        self.next_task_args = next_task_args
    
    def get_path(self):
        spaces = self.target_space.space
        manipulated_obj = self.target_space.target_object
        min_range = self.target_space.min_range
        max_range = self.target_space.max_range
        target_space_descriptions = self.target_space.target_space_descriptions
        focus_obj_id = self.target_space.focus_obj_id
        try_ik_sampling = self.target_space.try_ik_sampling
        linear = self.target_space.linear
        ignore_collisions = self.target_space.ignore_collisions
        release = self.target_space.release
        path = None
        if type(spaces) == SpawnBoundary:
            path = self.get_path_from_space(spaces, manipulated_obj, min_range, max_range,
                                            try_ik_sampling, linear, ignore_collisions, release,
                                            target_space_descriptions, focus_obj_id)
        elif type(spaces) == Dummy:
            path = self.get_path_from_point(spaces, manipulated_obj, min_range, max_range,
                                            try_ik_sampling, linear, ignore_collisions, release,
                                            target_space_descriptions, focus_obj_id)
        
        elif callable(spaces):
            path = self.get_path_from_func(spaces, manipulated_obj, self.target_space.space_args, 
                                            try_ik_sampling, linear, ignore_collisions, release,
                                            target_space_descriptions, focus_obj_id)
        return path

    def get_path_from_space(self, target_spaces:SpawnBoundary, obj:VLM_Object, 
                            min_rotation, max_rotation, try_ik_sampling, linear, ignore_collisions, release, 
                            target_space_descriptions, focus_obj_id):
        assert len(min_rotation) == 3
        objs_init_states = self.task_base.get_configuration_tree()
        saved_states =self.robot.save_state()
        target_space = np.random.choice(target_spaces._boundaries,p=target_spaces._probabilities)
        if obj.is_model():
            bb = obj.get_model_bounding_box()
        else:
            bb = obj.get_bounding_box()
        obj_bbox = BoundingBox(*bb)
        point2obj_pose = self.arm._ik_tip.get_matrix(relative_to=obj)
        target_space_pose = target_space._boundary.get_matrix()
        new_path = None
        for i in range(self.fail_times):
            find_path = False
            new_position = target_space._get_position_within_boundary(obj, obj_bbox)
            rotations = np.random.uniform(list(min_rotation), list(max_rotation))
            trans_matrix = np.eye(4)
            trans_matrix[:3, 3] = new_position
            trans_matrix[:3,:3] = R.from_euler('xyz', rotations).as_matrix()
            target_obj_pose = target_space_pose.dot(trans_matrix)
            target_gripper_pose = target_obj_pose.dot(point2obj_pose)
            find_path, path = test_reachability(self.arm, target_gripper_pose, try_ik_sampling=try_ik_sampling, 
                        linear=linear, ignore_collisions=ignore_collisions)
            if find_path:
                # execute_path(path, self.pyrep)
                # path.set_to_end()
                fast_path_test(path, self.robot)
                if release:
                    self.robot.gripper.release()
                    done = False
                    while not done:
                        done = self.robot.gripper.actuate(1,gripper_step)
                        self.pyrep.step()
                    for _ in range(10):
                        self.pyrep.step()
                target = Dummy.create()
                target.set_matrix(target_gripper_pose)
                if target_space._is_plane:
                    low_level_descriptions = "Move the object on {}.".format(target_space_descriptions)
                else:
                    low_level_descriptions = "Move the object above {}.".format(target_space_descriptions)
                if release:
                    low_level_descriptions += " Release the gripper."
                    WriteCustomDataBlock(target.get_handle(),"gripper","['open',1]")
                WriteCustomDataBlock(target.get_handle(),"low_level_descriptions",low_level_descriptions)
                WriteCustomDataBlock(target.get_handle(),"focus_obj_id",str(focus_obj_id))
                WriteCustomDataBlock(target.get_handle(),"focus_obj_name",Shape.get_object_name(focus_obj_id))
                WriteCustomDataBlock(target.get_handle(),"waypoint_type","goal_move")
                if self.next_task_fuc is not None:
                    next_path = self.next_task_fuc() if self.next_task_args is None else self.next_task_fuc(**self.next_task_args)
                    if next_path is None:
                        target.remove()
                        # self.robot.recover_state(saved_states)
                        # self.pyrep.set_configuration_tree(objs_init_states)
                        # self.pyrep.step()
                        # continue
                        break
                    else:
                        new_path = [target] + next_path
                else:
                    new_path = [target]
                break
        return new_path
    
    def get_path_from_point(self, target_spaces:Dummy, obj:VLM_Object, 
                            min_rotation, max_rotation, try_ik_sampling, linear, ignore_collisions, release, 
                            target_space_descriptions, focus_obj_id):
        objs_init_states = self.task_base.get_configuration_tree()
        saved_states =self.robot.save_state()
        new_path = None
        point2obj_pose = self.arm._ik_tip.get_matrix(relative_to=obj)
        target_pose = target_spaces.get_matrix()
        target_gripper_pose = target_pose.dot(point2obj_pose)
        pre_target_gripper_pose = deepcopy(target_gripper_pose)
        pre_target_gripper_pose[:3, 3] -= pre_target_gripper_pose[:3, 2] * 0.08
        # target_gripper_pose = target_pose.dot(point2obj_pose)
        find_path, path = test_reachability(self.arm, pre_target_gripper_pose, try_ik_sampling=try_ik_sampling, 
                            linear=linear, ignore_collisions=ignore_collisions)
        if find_path:
            # execute_path(path, self.pyrep)
            fast_path_test(path, self.robot)
            find_path, path = test_reachability(self.arm, target_gripper_pose, try_ik_sampling=try_ik_sampling, 
                        linear=linear, ignore_collisions=True)
            if not find_path:
                self.robot.recover_state(saved_states)
                self.pyrep.set_configuration_tree(objs_init_states)
                self.pyrep.step()
        if find_path:
            # execute_path(path, self.pyrep)
            fast_path_test(path, self.robot)
            if release:
                self.robot.gripper.release()
                done = False
                while not done:
                    done = self.robot.gripper.actuate(1,gripper_step)
                    self.pyrep.step()
                for _ in range(10):
                    self.pyrep.step()
            pre_target = Dummy.create()
            pre_target.set_matrix(pre_target_gripper_pose)

            target = Dummy.create()
            target.set_matrix(target_gripper_pose)
            low_level_descriptions = "Move the object to {}.".format(target_space_descriptions)
            if release:
                low_level_descriptions += " Release the gripper."
                WriteCustomDataBlock(target.get_handle(),"gripper","['open',1]")
            WriteCustomDataBlock(target.get_handle(),"low_level_descriptions",low_level_descriptions)
            WriteCustomDataBlock(target.get_handle(),"focus_obj_id",str(focus_obj_id))
            WriteCustomDataBlock(target.get_handle(),"focus_obj_name",Shape.get_object_name(focus_obj_id))
            WriteCustomDataBlock(target.get_handle(),"ignore_collisions","True")
            WriteCustomDataBlock(target.get_handle(),"waypoint_type","goal_move")

            WriteCustomDataBlock(pre_target.get_handle(),"low_level_descriptions","Move the object to the {} of {}".format(get_relative_position_xy(target, pre_target, self.arm), target_space_descriptions))
            WriteCustomDataBlock(pre_target.get_handle(),"focus_obj_id",str(focus_obj_id))
            WriteCustomDataBlock(pre_target.get_handle(),"focus_obj_name",Shape.get_object_name(focus_obj_id))
            WriteCustomDataBlock(pre_target.get_handle(),"waypoint_type","pre_goal_move")
            if self.next_task_fuc is not None:
                next_path = self.next_task_fuc() if self.next_task_args is None else self.next_task_fuc(**self.next_task_args)
                if next_path is None:
                    target.remove()
                    pre_target.remove()
                else:
                    new_path = [pre_target, target] + next_path
            else:
                new_path = [pre_target, target]
        
        return new_path
    
    def get_path_from_func(self, target_spaces, obj:VLM_Object, target_args,
                            try_ik_sampling, linear, ignore_collisions, release, 
                            target_space_descriptions, focus_obj_id):
        all_g_poses = target_spaces(**target_args)
        objs_init_states = self.task_base.get_configuration_tree()
        saved_states =self.robot.save_state()
        new_path = None
        # obj_target = Dummy.create()
        gripper_target = Dummy.create()
        gripper2obj = self.arm._ik_tip.get_matrix(obj)
        for g_pose in all_g_poses:
            find_path = False
            # obj_target.set_position(g_pose[:3])
            # obj_target.set_orientation(g_pose[3:])
            # gripper_matrix = obj_target.get_matrix().dot(gripper2obj)
            gripper_matrix = g_pose.dot(gripper2obj)
            gripper_target.set_matrix(gripper_matrix)
            find_path, path = test_reachability(self.arm, gripper_matrix, try_ik_sampling=try_ik_sampling, 
                            linear=linear, ignore_collisions=ignore_collisions)
            if find_path:
                fast_path_test(path, self.robot)
                if release:
                    self.robot.gripper.release()
                    done = False
                    while not done:
                        done = self.robot.gripper.actuate(1,gripper_step)
                        self.pyrep.step()
                    for _ in range(10):
                        self.pyrep.step()
                    WriteCustomDataBlock(gripper_target.get_handle(),"gripper","['open',1]")
                low_level_descriptions = "Move the object {}.".format(target_space_descriptions)
                if release:
                    low_level_descriptions += " Release the gripper."
                WriteCustomDataBlock(gripper_target.get_handle(),"low_level_descriptions",low_level_descriptions)
                WriteCustomDataBlock(gripper_target.get_handle(),"focus_obj_id",str(focus_obj_id))
                WriteCustomDataBlock(gripper_target.get_handle(),"focus_obj_name",Shape.get_object_name(focus_obj_id))
                WriteCustomDataBlock(gripper_target.get_handle(),"waypoint_type","goal_move")
                WriteCustomDataBlock(gripper_target.get_handle(), "ignore_collisions",str(ignore_collisions))
                WriteCustomDataBlock(gripper_target.get_handle(), "linear",str(linear))
                if self.next_task_fuc is not None:
                    next_path = self.next_task_fuc() if self.next_task_args is None else self.next_task_fuc(**self.next_task_args)
                    if next_path is None:
                        self.robot.recover_state(saved_states)
                        self.pyrep.set_configuration_tree(objs_init_states)
                        continue
                    else:
                        new_path = [gripper_target] + next_path
                        break
                else:
                    new_path = [gripper_target]
                    break
        if not find_path:
            gripper_target.remove()
        # obj_target.remove()
        return new_path

class T2_MoveObjectConstraints(T1_MoveObjectGoal):
    def __init__(self, robot: Robot, pyrep: PyRep, target_space: TargetSpace, init_states, 
                fail_times=10, *, next_task_fuc=None,next_task_args=None) -> None:
        super().__init__(robot, pyrep, target_space, init_states, fail_times, 
                    next_task_fuc=next_task_fuc,next_task_args=next_task_args)
    
    def get_path_with_constraints(self):
        spaces = self.target_space.space
        manipulated_obj = self.target_space.target_object
        min_range = self.target_space.min_range
        max_range = self.target_space.max_range
        target_space_descriptions = self.target_space.target_space_descriptions
        focus_obj_id = self.target_space.focus_obj_id
        try_ik_sampling = self.target_space.try_ik_sampling
        linear = self.target_space.linear
        ignore_collisions = self.target_space.ignore_collisions
        release = self.target_space.release
        path = None
        if type(spaces) == Joint:
            path = self.get_path_from_joint(spaces, manipulated_obj, min_range, max_range, 
                                            try_ik_sampling, linear, ignore_collisions, release,
                                            target_space_descriptions, focus_obj_id)
        # elif type(spaces) == list:
        #     path = self.get_path_from_list(spaces, manipulated_obj, min_range, max_range, 
        #                                     try_ik_sampling, linear, ignore_collisions, release,
        #                                     target_space_descriptions, focus_obj_id)
        elif callable(spaces):
            path = self.get_path_from_func(spaces, manipulated_obj, self.target_space.space_args, 
                                            try_ik_sampling, linear, ignore_collisions, release,
                                            target_space_descriptions, focus_obj_id)
        return path
        
    def get_path_from_joint(self, target_spaces:Joint, obj:VLM_Object, 
                            min_rotation, max_rotation, try_ik_sampling, linear, ignore_collisions, release, 
                            target_space_descriptions, focus_obj_id):
        intermediate_target = Dummy.create()
        intermediate_target.set_pose(self.arm._ik_tip.get_pose())
        objs_init_states = self.task_base.get_configuration_tree()
        saved_states =self.robot.save_state()
        current_angle = target_spaces.get_joint_position()
        point2joint_pose = intermediate_target.get_matrix(relative_to=target_spaces)
        joint2world_pose = target_spaces.get_matrix()
        for _ in range(self.fail_times):
            find_path = False
            end_angle = np.random.uniform(min_rotation, max_rotation)
            intermediate_angles = np.linspace(current_angle, end_angle, num=10)
            new_path = CartesianPath.create(automatic_orientation=False)
            control_points = []
            can_reach = True
            start_joint = self.arm.get_joint_positions()
            for angle in intermediate_angles:
                transform_matrix = np.eye(4)
                if target_spaces.get_joint_type() == JointType.REVOLUTE:
                    rotation = R.from_euler('z', angle).as_matrix()
                    transform_matrix[:3,:3] = rotation
                elif target_spaces.get_joint_type() == JointType.PRISMATIC:
                    trans = np.array([0,0,angle])
                    transform_matrix[:3,3] = trans
                intermediate_target.set_matrix(joint2world_pose.dot(transform_matrix).dot(point2joint_pose))
                can_reach,sub_path = test_reachability(self.arm, intermediate_target.get_matrix(), linear=True, ignore_collisions=True)
                if not can_reach:
                    break
                pos = intermediate_target.get_position().tolist()
                ori = intermediate_target.get_orientation().tolist()
                # execute_path(sub_path, self.pyrep)
                sub_path.set_to_end()
                control_points.append(pos+ori)
            self.arm.set_joint_positions(start_joint)
            if not can_reach:
                new_path.remove()
                self.robot.recover_state(saved_states)
                self.pyrep.set_configuration_tree(objs_init_states)
                self.pyrep.step()
                continue
            new_path.insert_control_points(control_points)
            try:
                try:
                    path = self.arm.get_path_from_cartesian_path(new_path)
                except:
                    path = self.arm.get_linear_path(pos, ori, ignore_collisions=True)
                    new_path.remove()
                    new_path = Dummy.create()
                    new_path.set_position(pos)
                    new_path.set_orientation(ori)
                    WriteCustomDataBlock(new_path.get_handle(), "linear","True")
                execute_path(path, self.pyrep)
                if target_spaces.get_joint_type() == JointType.PRISMATIC:
                    tolerance = 0.05
                elif target_spaces.get_joint_type() == JointType.REVOLUTE:
                    tolerance = np.deg2rad(5)
                if abs(target_spaces.get_joint_position()-end_angle)>tolerance:
                    new_path.remove()
                    self.robot.recover_state(saved_states)
                    self.pyrep.set_configuration_tree(objs_init_states)
                    self.pyrep.step()
                    continue
                WriteCustomDataBlock(new_path.get_handle(), "ignore_collisions","True")
                if target_spaces.get_joint_type() == JointType.PRISMATIC:
                    low_level_descriptions = "Move along the axis of {}.".format(target_space_descriptions)
                elif target_spaces.get_joint_type() == JointType.REVOLUTE:
                    low_level_descriptions = "Rotate around the axis of {}.".format(target_space_descriptions)
                WriteCustomDataBlock(new_path.get_handle(),"low_level_descriptions",low_level_descriptions)
                WriteCustomDataBlock(new_path.get_handle(),"focus_obj_id",str(focus_obj_id))
                WriteCustomDataBlock(new_path.get_handle(),"focus_obj_name",Shape.get_object_name(focus_obj_id))
                WriteCustomDataBlock(new_path.get_handle(),"waypoint_type","path_move")
                if self.next_task_fuc is not None:
                    next_path = self.next_task_fuc() if self.next_task_args is None else self.next_task_fuc(**self.next_task_args)
                    if next_path is None:
                        new_path.remove()
                        # reverse_path = deepcopy(path)
                        # original_path_points = path._path_points.reshape(-1,7)
                        # reverse_path._path_points = original_path_points[::-1].reshape(-1,)
                        # execute_path(path, self.pyrep)
                        self.robot.recover_state(saved_states)
                        self.pyrep.set_configuration_tree(objs_init_states)
                        self.pyrep.step()
                        continue
                    else:
                        new_path = [new_path] + next_path
                else:
                    new_path = [new_path]
                find_path=True
                break
            except:
                new_path.remove()
        intermediate_target.remove()  
        return new_path if find_path else None
    
    def get_path_from_func(self, target_spaces, obj:VLM_Object, target_args,
                            try_ik_sampling, linear, ignore_collisions, release, 
                            target_space_descriptions, focus_obj_id):
        all_g_poses = target_spaces(**target_args)
        objs_init_states = self.task_base.get_configuration_tree()
        saved_states =self.robot.save_state()
        new_path = None
        # obj_target = Dummy.create()
        # gripper_target = Dummy.create()
        gripper2obj = self.arm._ik_tip.get_matrix(obj)
        for g_pose in all_g_poses:
            find_path = False
            # obj_target.set_position(g_pose[:3])
            # obj_target.set_orientation(g_pose[3:])
            # gripper_matrix = obj_target.get_matrix().dot(gripper2obj)
            gripper_matrix = g_pose.dot(gripper2obj)
            # gripper_target.set_matrix(gripper_matrix)
            find_path, path = test_reachability(self.arm, gripper_matrix, try_ik_sampling=try_ik_sampling, 
                            linear=linear, ignore_collisions=ignore_collisions)
            if find_path:
                new_path = CartesianPath.create(automatic_orientation=False)
                control_points = []
                all_path_configs =  path._path_points.reshape(-1, len(self.arm.joints))
                for config in all_path_configs:
                    self.arm.set_joint_positions(config)
                    pos = self.arm._ik_tip.get_position().tolist()
                    ori = self.arm._ik_tip.get_orientation().tolist()
                    control_points.append(pos+ori)
                new_path.insert_control_points(control_points)
                fast_path_test(path, self.robot)
                WriteCustomDataBlock(new_path.get_handle(),"low_level_descriptions", target_space_descriptions)
                WriteCustomDataBlock(new_path.get_handle(),"focus_obj_id",str(focus_obj_id))
                WriteCustomDataBlock(new_path.get_handle(),"focus_obj_name",Shape.get_object_name(focus_obj_id))
                WriteCustomDataBlock(new_path.get_handle(),"waypoint_type","path_move")
                WriteCustomDataBlock(new_path.get_handle(), "ignore_collisions",str(ignore_collisions))
                WriteCustomDataBlock(new_path.get_handle(), "linear",str(linear))
                if self.next_task_fuc is not None:
                    next_path = self.next_task_fuc() if self.next_task_args is None else self.next_task_fuc(**self.next_task_args)
                    if next_path is None:
                        self.robot.recover_state(saved_states)
                        self.pyrep.set_configuration_tree(objs_init_states)
                        new_path.remove()
                        new_path = None
                        continue
                    else:
                        new_path = [new_path] + next_path
                        break
                else:
                    new_path = [new_path]
                    break
        # if not find_path:
        #     new_path.remove()
        # obj_target.remove()
        return new_path

if __name__=="__main__":
    pr = PyRep()
    pr.launch('', responsive_ui=True, headless=False)
    model_path = "../vlm/object_models/cube/cube_normal/cube_normal.ttm"
    model = VLM_Object(pr, model_path, 0)
    model_2 = VLM_Object(pr, model_path, 1)
    pr.shutdown()