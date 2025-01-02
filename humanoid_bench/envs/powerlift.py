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

    def get_rotation_reward(self):
        cur_torso_rotation = self._env.named.data.xquat["torso_link"]
        r = rewards.tolerance(
            np.dot(cur_torso_rotation, self.prev_torso_rotation),
            bounds=(0.95, 1),
            sigmoid="linear",
            margin=0.5,
            value_at_margin=0,
        )
        self.prev_torso_rotation = cur_torso_rotation
        return r

    def get_reward(self):
        # Reward standing
        standing = rewards.tolerance(
            self.robot.head_height(),
            bounds=(_STAND_HEIGHT, float("inf")),
            margin=_STAND_HEIGHT / 3,
        )
        upright = rewards.tolerance(
            self.robot.torso_upright(),
            bounds=(0.8, float("inf")),
            sigmoid="linear",
            margin=1.9,
            value_at_margin=0,
        )
        stand_reward = standing * upright

        # Reward not rotating
        no_rotation = self.get_rotation_reward()

        # Reward small control for smooth motions
        small_control = rewards.tolerance(
            self.robot.actuator_forces(),
            margin=10,
            value_at_margin=0,
            sigmoid="quadratic",
        ).mean()
        small_control = (4 + small_control) / 5
        
        reward = 0.6 * stand_reward + 0.2 * no_rotation + 0.2 * small_control

        # reward = 0.2 * (small_control * stand_reward) + 0.8 * reward_dumbbell_lifted
        return reward, {
            "stand_reward": stand_reward,
            "small_control": small_control,
            "no_rotation": no_rotation,
            # "reward_dumbbell_lifted": reward_dumbbell_lifted,
            "standing": standing,
            "upright": upright,
        }

    # If pelvis too low then abort
    def get_terminated(self):
        return self.get_pelvis_height() < 0.1, {}
