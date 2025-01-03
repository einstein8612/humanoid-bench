import os

import mujoco._structs
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium.spaces import Box
from dm_control.utils import rewards

from humanoid_bench.tasks import Task


# Height of head above which stand reward is 1.
_STAND_HEIGHT = 1.65


class Powerlift(Task):
    qpos0_robot = {
        "h1hand": """
            0 0 0.98 1 0 0 0 0 0 -0.4 0.8 -0.4 0 0 -0.4 0.8 -0.4 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0.5 0 0.2 0 0 0 0
        """,
        "h1touch": """
            0 0 0.98 1 0 0 0 0 0 -0.4 0.8 -0.4 0 0 -0.4 0.8 -0.4 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            0.5 0 0.2 0 0 0 0
        """,
        "g1": """
            0 0 0.75
            1 0 0 0
            0 0 0 0 0 0
            0 0 0 0 0 0
            0
            0 0 0 0 -1.57
            0 0 0 0 0 0 0
            0 0 0 0 1.57
            0 0 0 0 0 0 0
            0.5 0 0.2 0 0 0 0
        """
    }

    dof = 7

    success_bar = 800

    def __init__(self, robot=None, env=None, **kwargs):
        super().__init__(robot, env, **kwargs)
        if robot.__class__.__name__ == "G1":
            global _STAND_HEIGHT
            _STAND_HEIGHT = 1.28
            
        self.prev_torso_rotation = np.array([1, 0, 0, 0]) # Default pose
        self.terminate_on_collision = ["torso", "pelvis", "hip", "elbow", "knee"] # Torso and head are fused together
        # self.qpos0 = robot._env.model.key_qpos[0].copy() # The actual default pose of the robot, there's always only one key_qpos
        self.qpos0 = np.fromstring(self.qpos0_robot["h1hand"], dtype=float, sep=" ")

    @property
    def observation_space(self):
        return Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.robot.dof * 2 - 1 + self.dof * 2 - 1,),
            dtype=np.float64,
        )
        
    def get_pelvis_height(self):
        return self._env.named.data.xpos["pelvis", "z"]

    def get_balance_reward(self):
        # Calculate the robot's center of mass in the horizontal plane (x, y)
        com_xy = self.robot.center_of_mass_position()[:2]

        # Calculate the midpoint between the robot's feet (support polygon center)
        feet_midpoint = (
            self._env.named.data.xpos["left_ankle_link", :2]
            + self._env.named.data.xpos["right_ankle_link", :2]
        ) / 2

        # Compute the horizontal distance between CoM and the support polygon center
        com_to_support_dist = np.linalg.norm(com_xy - feet_midpoint)

        # Reward the robot for keeping its CoM close to the center of the support polygon
        balance = rewards.tolerance(
            com_to_support_dist,
            bounds=(0, 0.05),  # Target: CoM stays within 5 cm of the support polygon center
            margin=0.2,        # Allow up to 20 cm for a smoother gradient
            value_at_margin=0, # Reward drops to 0 outside the margin
            sigmoid="quadratic"
        )
        
        return balance

    def get_reward(self):
        # Reward standing
        standing = rewards.tolerance(
            self.robot.head_height(),
            bounds=(_STAND_HEIGHT, _STAND_HEIGHT),
            sigmoid="linear",
            margin=_STAND_HEIGHT / 4,
            value_at_margin=0
        )
        upright = rewards.tolerance(
            self.robot.torso_upright(),
            bounds=(0.8, float("inf")),
            sigmoid="linear",
            margin=1.9,
            value_at_margin=0,
        )
        stand_reward = standing * upright

        # Reward small control for smooth motions
        small_control = rewards.tolerance(
            self.robot.actuator_forces(),
            margin=10,
            value_at_margin=0,
            sigmoid="quadratic",
        ).mean()
        # small_control = (4 + small_control) / 5
        
        # Reward for feet staying near the ground
        foot = rewards.tolerance(
            np.array([self._env.named.data.sensordata["left_foot_sensor"][0], self._env.named.data.sensordata["right_foot_sensor"][0]]),
            bounds=(40, 250),
            margin=100,
            value_at_margin=0,
            sigmoid="quadratic",
        ).mean()
        
        # Reward for feet staying at certain distance
        foot_space = rewards.tolerance(
            np.linalg.norm(self._env.named.data.xpos["left_ankle_link", :2] - self._env.named.data.xpos["right_ankle_link", :2]),
            bounds=(0.3, 0.7), # 50cm between feet
            margin=0.1,
            value_at_margin=0,
            sigmoid="quadratic",
        )
        
        # Reward for feet staying at certain height
        foot_height = rewards.tolerance(
            np.array([self._env.named.data.xpos["left_ankle_link", 2], self._env.named.data.xpos["right_ankle_link", 2]]),
            bounds=(0.0, 0.1), # 50cm between feet
            margin=0.05,
            value_at_margin=0,
            sigmoid="quadratic",
        ).min()

        # Reward for slow joints
        slow_joints = rewards.tolerance(
            self.robot.joint_velocities(),
            margin=1,
            bounds=(-1, 1),
            value_at_margin=0,
            sigmoid="quadratic",
        ).mean()
        
        balance = self.get_balance_reward()
        
        ref_reward = rewards.tolerance(
            np.linalg.norm(self.qpos0 - self._env.data.qpos),
            bounds=(0, 0.5),
            margin=5,
            value_at_margin=0,
            sigmoid="quadratic",
        )

        reward = slow_joints * (0.2 * foot_height + 0.15 * foot + 0.2 * stand_reward + 0.2 * ref_reward + 0.05 * foot_space + 0.1 * balance + 0.1 * small_control)

        return reward, {
            "stand_reward": stand_reward,
            "small_control": small_control,
            "foot_space": foot_space,
            "foot_height": foot_height,
            "foot": foot,
            "slow_joints": slow_joints,
            "balance": balance,
            "ref": ref_reward,
            # "reward_dumbbell_lifted": reward_hasdumbbell_lifted,
            "standing": standing,
            "upright": upright,
        }
        
    def check_blacklisted_geoms(self):
        geoms_on_floor = []
        for contact in self._env.data.contact:
            # Get the names of the bodies involved in the contact
            geom1 = mujoco.mj_id2name(self._env.model, 5, contact.geom1)
            geom2 = mujoco.mj_id2name(self._env.model, 5, contact.geom2)
            
            # Check if the contact is with the ground
            if geom1 == "floor" and geom2 is not None:
                geoms_on_floor.append(geom2)
        
        for geom in self.terminate_on_collision:
            if any(geom in floor_geom for floor_geom in geoms_on_floor):
                return True
            
        return False

    # If pelvis too low then abort
    def get_terminated(self):
        return self._env.data.qpos[2] < 0.2 or self.get_pelvis_height() < 0.2 or self.check_blacklisted_geoms(), {}
