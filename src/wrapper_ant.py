import gym
import QDgym
import numpy as np
import time


LEG_IDX = {
    "front_left": (0, 1),
    "front_right": (2, 3),
    "back_left": (4, 5),
    "back_right": (6, 7),
}


class DamagedAnt(gym.Wrapper):
    """
    The wrapper is used to damage the ant robot by removing one of its legs setting the
    corresponding joint angles to zero.
    """

    def __init__(
        self, env_name="QDAntBulletEnv-v0", leg_name="front_left", render=False
    ):
        super().__init__(gym.make(env_name, render=render))
        self.sleep = render
        self.leg_name = leg_name
        if isinstance(self.leg_name, list):
            self.leg_idx = [LEG_IDX[leg] for leg in self.leg_name if leg in LEG_IDX]
        elif self.leg_name in LEG_IDX:
            self.leg_idx = [LEG_IDX[self.leg_name]]
        else:
            self.leg_idx = []

    def step(self, action):
        action = np.float32(action).copy()
        if self.leg_idx:
            if isinstance(self.leg_idx[0], tuple):
                for idx in np.ravel(self.leg_idx):
                    action[idx] = 0.0

        obs, rew, done, info = super().step(action)
        if self.sleep:
            time.sleep(1 / 60)

        return obs, rew, done, info
