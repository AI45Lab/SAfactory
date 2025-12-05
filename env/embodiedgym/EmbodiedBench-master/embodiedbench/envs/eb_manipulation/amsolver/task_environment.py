#Modified From the rlbench: https://github.com/stepjam/RLBench
import logging
import pickle
from sys import api_version
from typing import List, Callable, Tuple

import numpy as np
from pyquaternion import Quaternion
from pyrep import PyRep
from pyrep.const import ObjectType, ConfigurationPathAlgorithms
from pyrep.errors import IKError
from pyrep.objects import Dummy, Object

from amsolver import utils
from amsolver.action_modes import ArmActionMode, ActionMode
from amsolver.backend.exceptions import BoundaryError, WaypointError
from amsolver.backend.observation import Observation
from amsolver.backend.robot import Robot
from amsolver.backend.scene import Scene
from amsolver.backend.task import Task
from amsolver.backend.utils import execute_path
from amsolver.demo import Demo
from amsolver.observation_config import ObservationConfig
from scipy.spatial.transform import Rotation as R

_TORQUE_MAX_VEL = 9999
_DT = 0.05
_MAX_RESET_ATTEMPTS = 40
_MAX_DEMO_ATTEMPTS = 10
TTMS_FOLDER = 'embodiedbench/envs/eb_manipulation/'

class InvalidActionError(Exception):
    pass


class TaskEnvironmentError(Exception):
    pass

class TaskConfigs(object):
    def __init__(self) -> None:
        pass

class TaskEnvironment(object):

    def __init__(self, pyrep: PyRep, robot: Robot, scene: Scene, task: Task,
                 action_mode: ActionMode, dataset_root: str,
                 obs_config: ObservationConfig,
                 static_positions: bool = False,
                 attach_grasped_objects: bool = True):
        self._pyrep = pyrep
        self._robot = robot
        self._scene = scene
        self._task = task
        self._variation_number = 0
        self._action_mode = action_mode
        self._dataset_root = dataset_root
        self._obs_config = obs_config
        self._static_positions = static_positions
        self._attach_grasped_objects = attach_grasped_objects
        self._reset_called = False
        self._prev_ee_velocity = None
        self._enable_path_observations = False
        tasks_folder = self._task.__module__.split('.')[0]
        ttms_folder = TTMS_FOLDER + tasks_folder+'/task_ttms' # change to absolute path to vlmbench folder
        # ttms_folder = './'+tasks_folder+'/task_ttms'
        self._scene.load(self._task, ttms_folder=ttms_folder)
        self._pyrep.start()
        self._robot_shapes = self._robot.arm.get_objects_in_tree(
            object_type=ObjectType.SHAPE)

    def get_name(self) -> str:
        return self._task.get_name()

    def sample_variation(self) -> int:
        self._variation_number = np.random.randint(
            0, self._task.variation_count())
        return self._variation_number

    def set_variation(self, v: int) -> None:
        if v >= self.variation_count():
            raise TaskEnvironmentError(
                'Requested variation %d, but there are only %d variations.' % (
                    v, self.variation_count()))
        self._variation_number = v

    def variation_count(self) -> int:
        return self._task.variation_count()

    def reset(self) -> Tuple[List[str], Observation]:
        self._scene.reset()
        try:
            ctr_loop = self._robot.arm.joints[0].is_control_loop_enabled()
            self._robot.arm.set_control_loop_enabled(True)
            desc = self._scene.init_episode(
                self._variation_number, max_attempts=_MAX_RESET_ATTEMPTS,
                randomly_place=not self._static_positions)
            self._robot.arm.set_control_loop_enabled(ctr_loop)
        except (BoundaryError, WaypointError) as e:
            raise TaskEnvironmentError(
                'Could not place the task %s in the scene. This should not '
                'happen, please raise an issues on this task.'
                % self._task.get_name()) from e

        self._reset_called = True
        # Returns a list of descriptions and the first observation
        return desc, self._scene.get_observation()

    def get_observation(self) -> Observation:
        return self._scene.get_observation()

    def _assert_action_space(self, action, expected_shape):
        if np.shape(action) != expected_shape:
            raise RuntimeError(
                'Expected the action shape to be: %s, but was shape: %s' % (
                    str(expected_shape), str(np.shape(action))))

    def _assert_unit_quaternion(self, quat):
        if not np.isclose(np.linalg.norm(quat), 1.0):
            raise RuntimeError('Action contained non unit quaternion!')

    def _torque_action(self, action):
        self._robot.arm.set_joint_target_velocities(
            [(_TORQUE_MAX_VEL if t < 0 else -_TORQUE_MAX_VEL)
             for t in action])
        self._robot.arm.set_joint_forces(np.abs(action))

    def _ee_action(self, action, relative_to=None):
        self._assert_unit_quaternion(action[3:])
        try:
            joint_positions = self._robot.arm.solve_ik_via_jacobian(
                action[:3], quaternion=action[3:], relative_to=relative_to)
            self._robot.arm.set_joint_target_positions(joint_positions)
        except IKError as e:
            raise InvalidActionError(
                'Could not perform IK via Jacobian. This is because the current'
                ' end-effector pose is too far from the given target pose. '
                'Try limiting your action space, or sapping to an alternative '
                'action mode, e.g. ABS_EE_POSE_PLAN_WORLD_FRAME') from e
        done = False
        prev_values = None
        # Move until reached target joint positions or until we stop moving
        # (e.g. when we collide wth something)
        while not done:
            self._scene.step()
            cur_positions = self._robot.arm.get_joint_positions()
            reached = np.allclose(cur_positions, joint_positions, atol=0.01)
            not_moving = False
            if prev_values is not None:
                not_moving = np.allclose(
                    cur_positions, prev_values, atol=0.001)
            prev_values = cur_positions
            done = reached or not_moving

    def _path_action_get_path(self, action, collision_checking, relative_to):
        try:
            path = self._robot.arm.get_path(
                action[:3], quaternion=action[3:],
                ignore_collisions=not collision_checking,
                relative_to=relative_to,
              )
            return path
        except IKError as e:
            raise InvalidActionError('Could not find a path.') from e

    def _path_action(self, action, collision_checking=False, relative_to=None, recorder=None):
        self._assert_unit_quaternion(action[3:])
        # Check if the target is in the workspace; if not, then quick reject
        # Only checks position, not rotation
        pos_to_check = action[:3]
        if relative_to is not None:
            self._scene.target_workspace_check.set_position(
                pos_to_check, relative_to)
            pos_to_check = self._scene.target_workspace_check.get_position()
        valid = self._scene.check_target_in_workspace(pos_to_check)
        if not valid:
            raise InvalidActionError('Target is outside of workspace.')

        observations = []
        done = False
        success_in_path = []
        if collision_checking:
            # First check if we are colliding with anything
            colliding = self._robot.arm.check_arm_collision()
            if colliding:
                # Disable collisions with the objects that we are colliding with
                grasped_objects = self._robot.gripper.get_grasped_objects()
                colliding_shapes = [s for s in self._pyrep.get_objects_in_tree(
                    object_type=ObjectType.SHAPE) if (
                        s.is_collidable() and
                        s not in self._robot_shapes and
                        s not in grasped_objects and
                        self._robot.arm.check_arm_collision(s))]
                [s.set_collidable(False) for s in colliding_shapes]
                path = self._path_action_get_path(
                    action, collision_checking, relative_to)
                [s.set_collidable(True) for s in colliding_shapes]
                # Only run this path until we are no longer colliding
                while not done:
                    done = path.step()
                    self._scene.step()
                    if self._enable_path_observations:
                        observations.append(self._scene.get_observation())
                    if recorder is not None:
                        recorder.take_snap()
                    colliding = self._robot.arm.check_arm_collision()
                    if not colliding:
                        break
                    success, terminate = self._task.success()
                    # If the task succeeds while traversing path, then break early
                    if success:
                        done = True
                        break
        if not done:
            path = self._path_action_get_path(
                action, collision_checking, relative_to)
            small_step = 0
            while not done:
                done = path.step()
                self._scene.step()
                if self._enable_path_observations:
                    observations.append(self._scene.get_observation())
                if recorder is not None:
                    recorder.take_snap()
                success, terminate = self._task.success()
                # If the task succeeds while traversing path, then break early
                # if success:
                #     break
                if success:
                    success_in_path.append(small_step)
                small_step += 1

        return observations, success_in_path

    def step(self, action, collision_checking=None, use_auto_move=True, recorder = None, need_grasp_obj = None) -> Tuple[Observation, int, bool]:
        # returns observation, reward, done, info
        if not self._reset_called:
            raise RuntimeError(
                "Call 'reset' before calling 'step' on a task.")

        # action should contain 1 extra value for gripper open close state
        arm_action = np.array(action[:-1])
        ee_action = action[-1]

        if 0.0 > ee_action > 1.0:
            raise ValueError('Gripper action expected to be within 0 and 1.')

        # Discretize the gripper action
        open_condition = all(x > 0.9 for x in self._robot.gripper.get_open_amount())
        open_condition &= (len(self._robot.gripper.get_grasped_objects())==0)
        current_ee = 1.0 if open_condition else 0.0

        if ee_action > 0.5:
            ee_action = 1.0
        elif ee_action < 0.5:
            ee_action = 0.0

        success_in_path= []
        if self._action_mode.arm == ArmActionMode.ABS_JOINT_VELOCITY:

            self._assert_action_space(arm_action,
                                      (len(self._robot.arm.joints),))
            self._robot.arm.set_joint_target_velocities(arm_action)
            self._scene.step()
            self._robot.arm.set_joint_target_velocities(
                np.zeros_like(arm_action))

        elif self._action_mode.arm == ArmActionMode.DELTA_JOINT_VELOCITY:

            self._assert_action_space(arm_action,
                                      (len(self._robot.arm.joints),))
            cur = np.array(self._robot.arm.get_joint_velocities())
            self._robot.arm.set_joint_target_velocities(cur + arm_action)
            self._scene.step()
            self._robot.arm.set_joint_target_velocities(
                np.zeros_like(arm_action))

        elif self._action_mode.arm == ArmActionMode.ABS_JOINT_POSITION:

            self._assert_action_space(arm_action,
                                      (len(self._robot.arm.joints),))
            self._robot.arm.set_joint_target_positions(arm_action)
            self._scene.step()
            self._robot.arm.set_joint_target_positions(
                self._robot.arm.get_joint_positions())

        elif self._action_mode.arm == ArmActionMode.DELTA_JOINT_POSITION:

            self._assert_action_space(arm_action,
                                      (len(self._robot.arm.joints),))
            cur = np.array(self._robot.arm.get_joint_positions())
            self._robot.arm.set_joint_target_positions(cur + arm_action)
            self._scene.step()
            self._robot.arm.set_joint_target_positions(
                self._robot.arm.get_joint_positions())

        elif self._action_mode.arm == ArmActionMode.ABS_JOINT_TORQUE:

            self._assert_action_space(
                arm_action, (len(self._robot.arm.joints),))
            self._torque_action(arm_action)
            self._scene.step()
            self._torque_action(self._robot.arm.get_joint_forces())
            self._robot.arm.set_joint_target_velocities(
                np.zeros_like(arm_action))

        elif self._action_mode.arm == ArmActionMode.DELTA_JOINT_TORQUE:

            cur = np.array(self._robot.arm.get_joint_forces())
            new_action = cur + arm_action
            self._torque_action(new_action)
            self._scene.step()
            self._torque_action(self._robot.arm.get_joint_forces())
            self._robot.arm.set_joint_target_velocities(
                np.zeros_like(arm_action))

        elif self._action_mode.arm == ArmActionMode.ABS_EE_POSE_WORLD_FRAME:

            self._assert_action_space(arm_action, (7,))
            self._ee_action(list(arm_action))

        elif self._action_mode.arm == ArmActionMode.ABS_EE_POSE_PLAN_WORLD_FRAME:

            self._assert_action_space(arm_action, (7,))
            pass_this_step = False
            # if current_ee == 1.0 and ee_action == 0.0:
            if current_ee != ee_action and use_auto_move:
                obs = self._scene.get_observation()
                _, new_arm_action = self.auto_grasp(obs, arm_action, ee_action)
                if new_arm_action is not None:
                    arm_action = new_arm_action
            if collision_checking is None:
                collision_checking = False
            if not pass_this_step:
                self._path_observations = []
                self._path_observations, success_in_path = self._path_action(
                    list(arm_action), collision_checking=collision_checking, recorder=recorder)

        elif self._action_mode.arm == ArmActionMode.ABS_EE_POSE_PLAN_WORLD_FRAME_WITH_COLLISION_CHECK:

            self._assert_action_space(arm_action, (7,))
            pass_this_step = False
            # if current_ee == 1.0 and ee_action == 0.0:
            if current_ee != ee_action and use_auto_move:
                obs = self._scene.get_observation()
                _, new_arm_action = self.auto_grasp(obs, arm_action, ee_action)
                if new_arm_action is not None:
                    arm_action = new_arm_action
            if collision_checking is None:
                collision_checking = True
            if not pass_this_step:
                self._path_observations = []
                self._path_observations, success_in_path = self._path_action(
                    list(arm_action), collision_checking=collision_checking, recorder = recorder)

        elif self._action_mode.arm == ArmActionMode.DELTA_EE_POSE_PLAN_WORLD_FRAME:

            self._assert_action_space(arm_action, (7,))
            a_x, a_y, a_z, a_qx, a_qy, a_qz, a_qw = arm_action
            x, y, z, qx, qy, qz, qw = self._robot.arm.get_tip().get_pose()
            new_rot = Quaternion(a_qw, a_qx, a_qy, a_qz) * Quaternion(qw, qx,
                                                                      qy, qz)
            qw, qx, qy, qz = list(new_rot)
            new_pose = [a_x + x, a_y + y, a_z + z] + [qx, qy, qz, qw]
            self._path_observations = []
            self._path_observations, success_in_path = self._path_action(list(new_pose))

        elif self._action_mode.arm == ArmActionMode.DELTA_EE_POSE_WORLD_FRAME:

            self._assert_action_space(arm_action, (7,))
            a_x, a_y, a_z, a_qx, a_qy, a_qz, a_qw = arm_action
            x, y, z, qx, qy, qz, qw = self._robot.arm.get_tip().get_pose()
            new_rot = Quaternion(a_qw, a_qx, a_qy, a_qz) * Quaternion(
                qw, qx, qy, qz)
            qw, qx, qy, qz = list(new_rot)
            new_pose = [a_x + x, a_y + y, a_z + z] + [qx, qy, qz, qw]
            # point = Dummy('Dummy')
            # point.set_pose(list(new_pose))
            self._ee_action(list(new_pose))

        elif self._action_mode.arm == ArmActionMode.EE_POSE_EE_FRAME:

            self._assert_action_space(arm_action, (7,))
            self._ee_action(
                list(arm_action), relative_to=self._robot.arm.get_tip())

        elif self._action_mode.arm == ArmActionMode.EE_POSE_PLAN_EE_FRAME:

            self._assert_action_space(arm_action, (7,))
            self._path_observations = []
            self._path_observations, success_in_path = self._path_action(
                list(arm_action), relative_to=self._robot.arm.get_tip())

        else:
            raise RuntimeError('Unrecognised action mode.')

        obs = self._scene.get_observation()
        grasp_sucess = False
        if current_ee != ee_action:
            done = False
            if ee_action == 0.0 and self._attach_grasped_objects:
                # If gripper close action, the check for grasp.
                for g_obj in self._task.get_graspable_objects():
                    succ = self._robot.gripper.grasp(g_obj)
                    if need_grasp_obj is not None:
                        if g_obj.get_name() == need_grasp_obj and succ:
                            grasp_sucess = True
                if need_grasp_obj is not None:
                    if self._robot.gripper._proximity_sensor.is_detected(Object.get_object(need_grasp_obj)):
                        grasp_sucess = True
            else:
                # If gripper open action, the check for ungrasp.
                self._robot.gripper.release()
            while not done:
                done = self._robot.gripper.actuate(ee_action, velocity=0.2)
                self._pyrep.step()
                self._task.step()
            if ee_action == 1.0:
                # Step a few more times to allow objects to drop
                for _ in range(10):
                    self._pyrep.step()
                    self._task.step()

        success, terminate = self._task.success()
        # task_reward = self._task.reward(steps)
        # reward = float(success) if task_reward is None else task_reward
        if len(success_in_path) > 0:
            success = 1.0
        elif grasp_sucess:
            success = 0.5
        reward = float(success)
        return obs, reward, terminate

    def auto_grasp(self, obs, goal_tip_pose, ee_action):
        def angle_distance(q1, q2):
            # same as rad2deg(2arccos(theta)), here qw = cos(theta/2). need to select min of (x, 2pi - x)
            # reference: https://math.stackexchange.com/questions/90081/quaternion-distance
            q1_matrix = R.from_quat(q1).as_matrix()
            q2_matrix = R.from_quat(q2).as_matrix()
            # v = 2 * np.arccos(np.clip(2 * np.dot(q1, q2) ** 2 - 1, -1, 1))
            # return min(v, 2*np.pi - v)
            return np.dot(q1_matrix[:3, 2], q2_matrix[:3, 2])
        def get_errors(target_pose, current_ee_pose):
            translate_error = ((target_pose[:3]-current_ee_pose[:3])**2).sum()**(1/2)
            angle_error = angle_distance(target_pose[3:], current_ee_pose[3:])
            return translate_error, angle_error
        def execute_waypoint(point):
            path = point.get_path()
            done = False
            while not done:
                done = path.step()
                self._pyrep.step()
                self._task.step()
            if point.gripper_control is not None:
                gripper = self._robot.gripper
                if point.gripper_control[0]=='open':
                    gripper.release()
                done = False
                while not done:
                    done = gripper.actuate(point.gripper_control[1], 0.04)
                    self._pyrep.step()
                    self._task.step()
                if point.gripper_control[0]=='close':
                    for g_obj in self._task.get_graspable_objects():
                        gripper.grasp(g_obj)

        info = obs.object_informations
        waypoints = self._task.get_waypoints()
        success = False
        new_action = None
        # for w in waypoints:
        #     name = w.name
        #     gripper_control = w.gripper_control
        #     if gripper_control is not None:
        #         if gripper_control[1]!=ee_action:
        #             continue
        #         target_pose = info[name]['pose'][0]
        #         t_error, r_error = get_errors(target_pose, goal_tip_pose)
        #         print("Waypoint: ", name, "Translate Error: ", t_error, "Rotation Error: ", r_error)
                # if t_error<0.05 and r_error>0.9:
                #     print("Using automatic grasp")
                #     # execute_waypoint(w)
                #     new_action = goal_tip_pose.copy()
                #     wpoint_pose = w.pose if hasattr(w, "pose") else w.end_pose
                #     new_action[:3] = wpoint_pose[:3]
                #     success = True
                #     break
        return success, new_action
            
    def enable_path_observations(self, value: bool) -> None:
        if (self._action_mode.arm != ArmActionMode.DELTA_EE_POSE_PLAN_WORLD_FRAME and
                self._action_mode.arm != ArmActionMode.ABS_EE_POSE_PLAN_WORLD_FRAME and
                self._action_mode.arm != ArmActionMode.ABS_EE_POSE_PLAN_WORLD_FRAME_WITH_COLLISION_CHECK and
                self._action_mode.arm != ArmActionMode.EE_POSE_PLAN_EE_FRAME):
            raise RuntimeError('Only available in DELTA_EE_POSE_PLAN or '
                               'ABS_EE_POSE_PLAN action mode.')
        self._enable_path_observations = value

    def get_path_observations(self):
        if (self._action_mode.arm != ArmActionMode.DELTA_EE_POSE_PLAN_WORLD_FRAME and
                self._action_mode.arm != ArmActionMode.ABS_EE_POSE_PLAN_WORLD_FRAME and
                self._action_mode.arm != ArmActionMode.ABS_EE_POSE_PLAN_WORLD_FRAME_WITH_COLLISION_CHECK and
                self._action_mode.arm != ArmActionMode.EE_POSE_PLAN_EE_FRAME):
            raise RuntimeError('Only available in DELTA_EE_POSE_PLAN or '
                               'ABS_EE_POSE_PLAN action mode.')
        return self._path_observations

    def get_demos(self, amount: int, live_demos: bool = False,
                  image_paths: bool = False,
                  callable_each_step: Callable[[Observation], None] = None,
                  max_attempts: int = _MAX_DEMO_ATTEMPTS,
                  random_selection: bool = True,
                  from_episode_number: int = 0
                  ) -> List[Demo]:
        """Negative means all demos"""

        if not live_demos and (self._dataset_root is None
                       or len(self._dataset_root) == 0):
            raise RuntimeError(
                "Can't ask for a stored demo when no dataset root provided.")

        if not live_demos:
            if self._dataset_root is None or len(self._dataset_root) == 0:
                raise RuntimeError(
                    "Can't ask for stored demo when no dataset root provided.")
            demos = utils.get_stored_demos(
                amount, image_paths, self._dataset_root, self._variation_number,
                self._task.get_name(), self._obs_config,
                random_selection, from_episode_number)
            return (demos)
        else:
            ctr_loop = self._robot.arm.joints[0].is_control_loop_enabled()
            self._robot.arm.set_control_loop_enabled(True)
            demos, success_all = self._get_live_demos(
                amount, callable_each_step, max_attempts)
            self._robot.arm.set_control_loop_enabled(ctr_loop)
            return (demos, success_all)

    def _get_live_demos(self, amount: int,
                        callable_each_step: Callable[
                            [Observation], None] = None,
                        max_attempts: int = _MAX_DEMO_ATTEMPTS, record=True) -> List[Demo]:
        demos = []
        success_all = []
        for i in range(amount):
            attempts = max_attempts
            while attempts > 0:
                random_seed = np.random.get_state()
                self.reset()
                try:
                    demo, success = self._scene.get_demo(
                        record = record, callable_each_step=callable_each_step)
                    demo.random_seed = random_seed
                    demos.append(demo)
                    success_all.append(success)
                    break
                except Exception as e:
                    attempts -= 1
                    logging.info('Bad demo. ' + str(e))
            if attempts <= 0:
                raise RuntimeError(
                    'Could not collect demos. Maybe a problem with the task?')
        return demos, success_all

    def reset_to_demo(self, demo: Demo) -> Tuple[List[str], Observation]:
        demo.restore_state()
        return self.reset()
    
    def save_config(self, max_attempts: int = _MAX_DEMO_ATTEMPTS):
        random_seed = np.random.get_state()
        desc,_ = self.reset()
        task_base, waypoint_sets, config = self.read_config(desc)
        config.random_seed = random_seed

        """
        attempts = max_attempts
        while attempts > 0:
            random_seed = np.random.get_state()
            desc,_ = self.reset()
            task_base, waypoint_sets, config = self.read_config(desc)
            config.random_seed = random_seed
            try:
                demo, success = self._scene.get_demo(record = False)
                if not success:
                    attempts -= 1
                    continue
                break
            except Exception as e:
                attempts -= 1
                logging.info('Bad demo. ' + str(e))
        """
            # ctr_loop = self._robot.arm.joints[0].is_control_loop_enabled()
            # self._robot.arm.set_control_loop_enabled(True)
            # demos, success_all = self._get_live_demos(amount=1,record=False)
            # desc = demos[0].high_level_instructions
            # desc = self._scene.init_episode(
            #     self._variation_number, max_attempts=_MAX_RESET_ATTEMPTS,
            #     randomly_place=not self._static_positions)
            # self._robot.arm.set_control_loop_enabled(ctr_loop)
        # except (BoundaryError, WaypointError) as e:
        #     raise TaskEnvironmentError(
        #         'Could not place the task %s in the scene. This should not '
        #         'happen, please raise an issues on this task.'
        #         % self._task.get_name()) from e

        # self._reset_called = True
        
        return task_base, waypoint_sets, config
    
    def read_config(self, desc):
        task_base = self._task.get_base()
        if Dummy.exists("waypoint_sets"):
            waypoint_sets = Dummy("waypoint_sets")
        else:
            waypoint_sets = Dummy.create()
            waypoint_sets.set_name("waypoint_sets")
            waypoint_sets.set_model(True)
        for waypoint in self._task.temporary_waypoints:
            waypoint.set_parent(waypoint_sets)
        # config = self._scene.get_observation()
        # for key, val in config.__dict__.items():
        #     if "rgb" in key or "depth" in key or "point" in key or "gripper" in key:
        #         config.__setattr__(key, None)
        config = TaskConfigs()
        config.high_level_descriptions = desc
        config.success_conditions = self._task._success_conditions
        graspable_objects = []
        for obj in self._task._graspable_objects:
            graspable_objects.append(obj.get_name())
        config.graspable_objects = graspable_objects
        task_attributes = {}
        for key, val in self._task.__dict__.items():
            if key[0]!="_" and key!="pyrep" and key!="robot":
                task_attributes[key] = val
        config.task_attributes = task_attributes
        return task_base, waypoint_sets, config
    
    def load_config(self, task_base, waypoint_sets, config_path):
        ctr_loop = self._robot.arm.joints[0].is_control_loop_enabled()
        self._scene._has_init_task = True
        self._robot.gripper.release()

        arm, gripper = self._scene._initial_robot_state
        self._pyrep.set_configuration_tree(arm)
        self._pyrep.set_configuration_tree(gripper)
        self._robot.arm.set_joint_positions(self._scene._start_arm_joint_pos, disable_dynamics=True)
        self._robot.arm.set_joint_target_velocities(
            [0] * len(self._robot.arm.joints))
        self._robot.gripper.set_joint_positions(
            self._scene._starting_gripper_joint_pos, disable_dynamics=True)
        self._robot.gripper.set_joint_target_velocities(
            [0] * len(self._robot.gripper.joints))
        self._robot.arm.set_control_loop_enabled(ctr_loop)

        self._task.unload()
        if Dummy.exists("waypoint_sets"):
            Dummy("waypoint_sets").remove()
        new_base = self._pyrep.import_model(task_base)
        waypoints = self._pyrep.import_model(waypoint_sets)
        with open(config_path, 'rb') as f:
            config = pickle.load(f)
        self._task._success_conditions = config.success_conditions
        graspable_objects = []
        for obj_name in config.graspable_objects:
            graspable_objects.append(Object.get_object(obj_name))
        self._task._graspable_objects = graspable_objects
        self._task.set_initial_objects_in_scene()
        if not hasattr(self, "attr_retrivel"):
            self.attr_retrivel = []
        for key, val in config.task_attributes.items():
            if not hasattr(self._task, key):
                self.attr_retrivel.append(key)
        for key in self.attr_retrivel:
            self._task.__setattr__(key, config.task_attributes[key])
        self._reset_called=True

        return config.high_level_descriptions, self._scene.get_observation()