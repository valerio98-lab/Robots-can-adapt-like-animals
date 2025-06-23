import numpy as np, gym
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from controller import CPG
from wrapper_ant import DamagedAnt


class MBOA:
    """
    Model-Based Optimization Algorithm for the QDAntBulletEnv environment.
    This script uses a Gaussian Process to model the fitness landscape and adaptively
    samples candidates to optimize the performance of a damaged ant robot.
    """

    def __init__(
        self,
        env_name="QDAntBulletEnv-v0",
        map_file="map_ant.npz",
        damaged_leg="back_left",
        max_trials=20,
        kappa=0.5,
        alpha_gp=1e-4,
        render=False,
    ):
        self.env_name = env_name
        self.damaged_leg = damaged_leg
        self.max_trials = max_trials
        self.kappa = kappa
        self.alpha_gp = alpha_gp
        self.render = render
        self.env = DamagedAnt(env_name, damaged_leg, render=render)

        self.data = np.load(map_file, allow_pickle=True)
        self.theta_map, self.fit_map, self.bd_map = (
            self.data["theta"],
            self.data["fitness"],
            self.data["bd"],
        )
        self.best_sim = self.fit_map.max()

        self.kernel = 1.0 * Matern(nu=2.5, length_scale_bounds=(1e-6, 1e2))
        self.gp = GaussianProcessRegressor(
            self.kernel, alpha=self.alpha_gp, normalize_y=True
        )
        self.gp.fit(self.bd_map, np.zeros(len(self.bd_map)))

        self.target = 0.9 * self.best_sim
        print(f"Target: {self.target:.2f}  (90 % of best simulation fitness)")

    def run(self):
        for t in range(1, self.max_trials + 1):
            mu, sigma = self.gp.predict(self.bd_map, return_std=True)
            score = (self.fit_map + mu) + self.kappa * sigma
            idx = np.argmax(score)
            theta = self.theta_map[idx]

            ctrl = CPG(theta)
            s = self.env.reset()
            done = False
            while not done:
                a = ctrl.select_action(s)
                s, r, done, _ = self.env.step(a)
            f_real = self.env.tot_reward
            bd_real = self.env.desc

            print(f"[{t:02d}/{self.max_trials}] reward = {f_real:.2f}")

            if f_real >= self.target:
                print("Target reached!")
                break

            self.gp.fit(
                np.vstack([self.gp.X_train_, bd_real.reshape(1, -1)]),
                np.hstack([self.gp.y_train_, [f_real - self.fit_map[idx]]]),
            )

        self.env.close()
