from pyrep.objects.object import Object
from pyrep.robots.configuration_paths.arm_configuration_path import (
    ArmConfigurationPath)
from amsolver.backend.robot import Robot
from amsolver.backend.utils import ReadCustomDataBlock
from scipy.spatial.transform import Rotation as R
import numpy as np


class Waypoint(object):

    def __init__(self, waypoint: Object, robot: Robot, ignore_collisions=False,
                 start_of_path_func=None, end_of_path_func=None):
        self.name = waypoint.get_name()
        self._waypoint = waypoint
        self._robot = robot
        self._ext = waypoint.get_extension_string()
        self._ignore_collisions = ignore_collisions
        self._linear_only = False
        self._start_of_path_func = start_of_path_func
        self._end_of_path_func = end_of_path_func
        self.gripper_control = None
        if len(self._ext) > 0:
            self._ignore_collisions = 'ignore_collision' in self._ext
            self._linear_only = 'linear' in self._ext

            contains_param = False
            start_of_bracket = -1
            if 'open_gripper(' in self._ext:
                start_of_bracket = self._ext.index('open_gripper(') + 13
                contains_param = self._ext[start_of_bracket] != ')'
                if not contains_param:
                    self.gripper_control = ['open', 1]
                else:
                    rest = self._ext[start_of_bracket:]
                    num = float(rest[:rest.index(')')])
                    self.gripper_control = ['open', num]
            elif 'close_gripper(' in self._ext:
                start_of_bracket = self._ext.index('close_gripper(') + 14
                contains_param = self._ext[start_of_bracket] != ')'
                if not contains_param:
                    self.gripper_control = ['close', 0]
                else:
                    rest = self._ext[start_of_bracket:]
                    num = float(rest[:rest.index(')')])
                    self.gripper_control = ['close', num]

        if ReadCustomDataBlock(waypoint._handle, "ignore_collisions") != None:
            self._ignore_collisions = eval(ReadCustomDataBlock(waypoint._handle, "ignore_collisions"))
        if ReadCustomDataBlock(waypoint._handle, "linear") != None:
            self._linear_only = eval(ReadCustomDataBlock(waypoint._handle, "linear"))
        if ReadCustomDataBlock(waypoint._handle, "gripper")!=None:
            self.gripper_control = eval(ReadCustomDataBlock(waypoint._handle, "gripper"))
            assert self.gripper_control[0] in ["open", "close"], "gripper only can open or close"
            assert self.gripper_control[1]<=1 and self.gripper_control[1]>=0, "gripper distance should between 0 and 1"
        self.low_level_descriptions = None
        if ReadCustomDataBlock(waypoint._handle, "low_level_descriptions")!=None:
            self.low_level_descriptions = ReadCustomDataBlock(waypoint._handle, "low_level_descriptions")
        self.focus_obj_id = None
        if ReadCustomDataBlock(waypoint._handle, "focus_obj_id")!=None:
            self.focus_obj_id = eval(ReadCustomDataBlock(waypoint._handle, "focus_obj_id"))
        self.focus_obj_name = None
        if ReadCustomDataBlock(waypoint._handle,"focus_obj_name")!=None:
            self.focus_obj_name = ReadCustomDataBlock(waypoint._handle, "focus_obj_name")
        self.waypoint_type = None
        if ReadCustomDataBlock(waypoint._handle, "waypoint_type")!=None:
            self.waypoint_type = ReadCustomDataBlock(waypoint._handle, "waypoint_type")
            
    def get_path(self, ignore_collisions=False) -> ArmConfigurationPath:
        raise NotImplementedError()

    def get_ext(self) -> str:
        return self._ext

    def get_waypoint_object(self) -> Object:
        return self._waypoint

    def remove(self) -> None:
        self._waypoint.remove()

    def start_of_path(self) -> None:
        if self._start_of_path_func is not None:
            self._start_of_path_func(self)

    def end_of_path(self) -> None:
        if self._end_of_path_func is not None:
            self._end_of_path_func(self)


class Point(Waypoint):
    def __init__(self, waypoint: Object, robot: Robot, ignore_collisions=False, start_of_path_func=None, end_of_path_func=None):
        super().__init__(waypoint, robot, ignore_collisions, start_of_path_func, end_of_path_func)
        self.pose = waypoint.get_pose()

    def get_path(self, ignore_collisions=False) -> ArmConfigurationPath:
        arm = self._robot.arm
        if self._linear_only:
            path = arm.get_linear_path(self._waypoint.get_position(),
                                euler=self._waypoint.get_orientation(),
                                ignore_collisions=(self._ignore_collisions or
                                                   ignore_collisions))
        else:
            path = arm.get_path(self._waypoint.get_position(),
                                euler=self._waypoint.get_orientation(),
                                ignore_collisions=(self._ignore_collisions or
                                                   ignore_collisions))
        return path


class PredefinedPath(Waypoint):
    def __init__(self, waypoint: Object, robot: Robot, ignore_collisions=False, start_of_path_func=None, end_of_path_func=None):
        super().__init__(waypoint, robot, ignore_collisions, start_of_path_func, end_of_path_func)
        start_pos, start_ori = waypoint.get_pose_on_path(0)
        self.start_pose = np.array(start_pos + R.from_euler('zyx', start_ori[::-1]).as_quat().tolist())
        end_pos, end_ori = waypoint.get_pose_on_path(1)
        self.end_pose = np.array(end_pos + R.from_euler('zyx', end_ori[::-1]).as_quat().tolist())
        
    def get_path(self, ignore_collisions=False) -> ArmConfigurationPath:
        arm = self._robot.arm
        path = arm.get_path_from_cartesian_path(self._waypoint)
        return path
