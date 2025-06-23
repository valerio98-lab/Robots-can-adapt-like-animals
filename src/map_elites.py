import numpy as np
import random
import gym
import QDgym
from tqdm import tqdm
from joblib import Parallel, delayed

from controller import CPG

ENV_NAME = "QDAntBulletEnv-v0"
N_BD_DIMS = 4
N_BUCKETS = 5  # 0-0.2, 0.2-0.4, ...
THETA_DIM = 16
SIGMA = 0.2  # std dev mutazione
N_EVALS = 1000
BATCH_SIZE = 64

env = gym.make(ENV_NAME, render=False)
archive = {}


def bd_to_cell(bd):
    idx = np.floor(bd * N_BUCKETS).astype(int)
    idx = np.clip(idx, 0, N_BUCKETS - 1)
    return tuple(idx)


def evaluate(theta):
    ctrl = CPG(theta)
    s = env.reset()
    done = False
    while not done:
        a = ctrl.select_action(s)
        s, _, done, _ = env.step(a)
    return env.tot_reward, env.desc


def sample_candidate():
    if archive:
        theta = random.choice(list(archive.values()))[1]
        new_theta = theta + np.random.randn(THETA_DIM) * SIGMA
    else:
        new_theta = np.random.uniform(-1, 1, THETA_DIM)
    return np.clip(new_theta, -1, 1)


evals = 0

pbar = tqdm(total=N_EVALS, desc="Evals", unit="evals")

while evals < N_EVALS:
    thetas = [sample_candidate() for _ in range(BATCH_SIZE)]

    results = Parallel(n_jobs=8, prefer="processes")(
        delayed(evaluate)(theta) for theta in thetas
    )
    for theta, (fitness, bd) in zip(thetas, results):
        cell = bd_to_cell(bd)
        best = archive.get(cell, (-np.inf, None))
        if fitness > best[0]:
            archive[cell] = (fitness, theta)
        evals += 1
        if evals % 50_000 == 0:
            print(f"{evals:,} evaluations - filled {len(archive)} cells")
    pbar.update(BATCH_SIZE)

cells, fits, params, descs = [], [], [], []
for cell, (fit, theta) in archive.items():
    cells.append(cell)
    fits.append(fit)
    params.append(theta)
    descs.append(np.array(cell) / (N_BUCKETS - 1))

np.savez(
    "map_ant_naive.npz",
    theta=np.vstack(params),
    fitness=np.array(fits),
    bd=np.vstack(descs),
    cells=np.array(cells),
)
env.close()
print("Mappa salvata in map_ant_naive.npz")
