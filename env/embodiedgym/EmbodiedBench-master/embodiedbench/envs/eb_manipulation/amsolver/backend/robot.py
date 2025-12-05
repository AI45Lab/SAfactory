#Modified From the rlbench: https://github.com/stepjam/RLBench
from pyrep.robots.arms.arm import Arm
from pyrep.robots.end_effectors.gripper import Gripper


class Robot(object):
    """Simple container for the robot components.
    """

    def __init__(self, arm: Arm, gripper: Gripper):
        self.arm = arm
        self.gripper = gripper
        self._start_arm_joint_pos = arm.get_joint_positions()
        self._starting_gripper_joint_pos = gripper.get_joint_positions()
        self._initial_robot_state = (arm.get_configuration_tree(),
                                     gripper.get_configuration_tree())
    
    def reset(self):
        self.gripper.release()
        self.arm.set_joint_positions(self._start_arm_joint_pos, disable_dynamics=True)
        self.arm.set_joint_target_velocities([0] * len(self.arm.joints))
        self.gripper.set_joint_positions(self._starting_gripper_joint_pos, disable_dynamics=True)
        self.gripper.set_joint_target_velocities([0] * len(self.gripper.joints))
    
    def save_state(self):
        saved_arm_joint_pos = self.arm.get_joint_positions()
        saved_gripper_joint_pos = self.gripper.get_joint_positions()
        return [saved_arm_joint_pos, saved_gripper_joint_pos]
    
    def recover_state(self, saved_states, release=False):
        if release:
            self.gripper.release()
        saved_arm_joint_pos, saved_gripper_joint_pos = saved_states
        self.arm.set_joint_positions(saved_arm_joint_pos, disable_dynamics=True)
        self.arm.set_joint_target_velocities([0] * len(self.arm.joints))
        self.gripper.set_joint_positions(saved_gripper_joint_pos, disable_dynamics=True)
        self.gripper.set_joint_target_velocities([0] * len(self.gripper.joints))