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

        self.contacts = []

    def reset(self, **kwargs):
        self.contacts.clear()
        self.tot_reward = 0.0
        self.desc = np.zeros((4,), dtype=np.float32)
        return super().reset(**kwargs)

    def step(self, action):
        action = np.float32(action).copy()
        if self.leg_idx:
            if isinstance(self.leg_idx[0], tuple):
                for idx in np.ravel(self.leg_idx):
                    action[idx] = 0.0

        obs, rew, done, info = super().step(action)
        self.contacts.append(np.asarray(info["bc"], dtype=bool))
        self.tot_reward += rew

        if done:
            contacts_arr = np.vstack(self.contacts)
            self.desc = contacts_arr.mean(axis=0).astype(np.float32)

        if self.sleep:
            time.sleep(1 / 60)

        return obs, rew, done, info

    def duty_factor(self, contacts):
        """
        Returns the duty factor of the ant robot.
        The duty factor is the ratio of the time a leg is in contact with the ground to the total cycle time.
        """
        return contacts.mean(axis=0).astype(
            np.float32
        )  # Average over all legs shape (4,)
