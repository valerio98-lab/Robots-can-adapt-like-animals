import numpy as np
import random
import QDgym
import torch

from wrapper_ant import DamagedAnt
from tqdm import tqdm
from joblib import Parallel, delayed

from controller import CPG

from autoencoder import AutoEncoder

ENC_PATH = "ae_final.pt"
LATENT_DIM = 2

encoder = AutoEncoder(latent_dim=LATENT_DIM)
encoder.load_state_dict(torch.load(ENC_PATH, map_location="cpu"))
encoder.eval()


class MapElites:
    """
    Map Elites algorithm for QDAntBulletEnv.
    This script generates a map of behaviors and fitnesses for the QDAntBulletEnv environment.
    It uses a naive approach to sample candidates and evaluate them.
    """

    def __init__(
        self,
        env_name="QDAntBulletEnv-v0",
        n_buckets=5,
        theta_dim=16,
        sigma=0.2,
        n_evals=5000,
        batch_size=64,
        n_jobs=12,
        render=False,
    ):
        self.env_name = env_name
        self.n_buckets = n_buckets
        self.theta_dim = theta_dim
        self.sigma = sigma
        self.n_evals = n_evals
        self.batch_size = batch_size
        self.render = render
        self.n_jobs = n_jobs if render == False else 1

        self.env = DamagedAnt(
            env_name=self.env_name,
            leg_name=None,
            render=self.render,
        )
        self.archive = {}

    def bd_to_cell(self, bd):
        z = (
            encoder.encoder(torch.from_numpy(bd * 2.0 - 1.0).float().to("cpu"))
            .cpu()
            .numpy()
        )
        z_norm = (z + 1.0) / 2.0
        idx = np.floor(z_norm * self.n_buckets).astype(int)
        idx = np.clip(idx, 0, self.n_buckets - 1)
        return tuple(idx), z

    def evaluate(self, theta):
        ctrl = CPG(theta)
        s = self.env.reset()
        done = False
        while not done:
            a = ctrl.select_action(s)
            s, _, done, _ = self.env.step(a)
        return self.env.tot_reward, self.env.desc

    def sample_candidate(self):
        if self.archive:
            theta = random.choice(list(self.archive.values()))[1]
            new_theta = theta + np.random.randn(self.theta_dim) * self.sigma
        else:
            new_theta = np.random.uniform(-1, 1, self.theta_dim)
        return np.clip(new_theta, -1, 1)

    def run(self):
        evals = 0

        pbar = tqdm(total=self.n_evals, desc="Evals", unit="evals")

        while evals < self.n_evals:
            thetas = [self.sample_candidate() for _ in range(self.batch_size)]

            results = Parallel(n_jobs=self.n_jobs, prefer="processes")(
                delayed(self.evaluate)(theta) for theta in thetas
            )
            for theta, (fitness, bd) in zip(thetas, results):
                cell, z = self.bd_to_cell(bd)
                best = self.archive.get(cell, (-np.inf, None, None))
                if fitness > best[0]:
                    self.archive[cell] = (fitness, theta, z)
                evals += 1
                if evals % 10000 == 0:
                    print(f"{evals:,} evaluations - filled {len(self.archive)} cells")
            pbar.update(self.batch_size)

        cells, fits, params, descs, latents = [], [], [], [], []
        for cell, (fit, theta, z) in self.archive.items():
            cells.append(cell)
            fits.append(fit)
            params.append(theta)
            latents.append(z)
            descs.append(np.array(bd))

        np.savez(
            "map_ant.npz",
            theta=np.vstack(params),
            fitness=np.array(fits),
            latent=np.vstack(latents),
            cells=np.array(cells),
        )
        self.env.close()


# if __name__ == "__main__":
#     map_elites = MapElites(
#         env_name="QDAntBulletEnv-v0",
#         n_buckets=5,
#         theta_dim=16,
#         sigma=0.2,
#         n_evals=5000,
#         batch_size=64,
#         n_jobs=12,
#         render=False,
#     )
#     map_elites.run()
