# plot_map_vs_latent.py
"""Visual comparison between the original 4‑D MAP‑Elites grid and the 2‑D latent
embedding learned by the auto‑encoder.

Generates **three** side‑by‑side panels:
  1. Heat‑map occupancy of the original 4‑D grid (projected on two BD axes).
  2. Scatter of all solutions in the latent space (continuous).
  3. Heat‑map occupancy of the latent space after discretisation into an
     `n_latent_buckets × n_latent_buckets` regular grid.

Example:
    python plot_map_vs_latent.py \
        --npz map_ant.npz \
        --latent latent_embeddings.npy \
        --grid-buckets 5 \
        --latent-buckets 50 \
        --dims 0 1

Produces `compare_map_vs_latent.png`.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def build_heatmap(cells: np.ndarray, n_buckets: int, dims: tuple[int, int]):
    """Count how many solutions land in each 2D bucket (dims specify which axes)."""
    hm = np.zeros((n_buckets, n_buckets), dtype=int)
    for c in cells:
        hm[c[dims[0]], c[dims[1]]] += 1
    return np.flipud(hm)  # flip for y‑axis origin bottom


def latent_histogram(z: np.ndarray, n_bins: int):
    """2D histogram (occupancy) of latent samples in [min,max] per axis."""
    # determine bounds with a small margin
    margin = 1e-6
    x_min, x_max = z[:, 0].min() - margin, z[:, 0].max() + margin
    y_min, y_max = z[:, 1].min() - margin, z[:, 1].max() + margin
    hist, xedges, yedges = np.histogram2d(
        z[:, 0], z[:, 1], bins=n_bins, range=[[x_min, x_max], [y_min, y_max]]
    )
    return hist.T, xedges, yedges  # orient like imshow


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--npz", default="../map_ant.npz", help="map file with 'cells'")
    p.add_argument("--latent", default="latent_embeddings.npy")
    p.add_argument(
        "--grid_buckets",
        type=int,
        default=5,
        help="# buckets per BD axis (original grid)",
    )
    p.add_argument(
        "--latent_buckets",
        type=int,
        default=50,
        help="# buckets per latent axis for histogram",
    )
    p.add_argument(
        "--dims", nargs=2, type=int, default=(0, 1), help="BD axes to project (0‑3)"
    )
    p.add_argument("--out", default="compare_map_vs_latent.png")
    args = p.parse_args()

    # load data
    data = np.load(args.npz)
    if "cells" not in data:
        raise KeyError("'cells' array missing in npz file")
    cells = data["cells"].astype(int)  # (N,4)
    z = np.load(args.latent)  # (N,2)
    if z.shape[0] != cells.shape[0]:
        raise ValueError("latent and cells row mismatch")

    # original grid heatmap (projected)
    hm_grid = build_heatmap(cells, args.grid_buckets, tuple(args.dims))

    # latent histogram
    hm_lat, xedges, yedges = latent_histogram(z, args.latent_buckets)

    # ── plotting ────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.5))

    im0 = ax[0].imshow(hm_grid, origin="lower", cmap="viridis")
    ax[0].set_title("Original grid\nproj dims %d & %d" % tuple(args.dims))
    ax[0].set_xlabel(f"cell dim {args.dims[0]}")
    ax[0].set_ylabel(f"cell dim {args.dims[1]}")
    fig.colorbar(im0, ax=ax[0], shrink=0.7, label="solutions")

    ax[1].scatter(z[:, 0], z[:, 1], s=2, alpha=0.4)
    ax[1].set_title("Latent scatter 2D")
    ax[1].set_xlabel("z₁")
    ax[1].set_ylabel("z₂")
    ax[1].set_aspect("equal", adjustable="box")

    im2 = ax[2].imshow(
        hm_lat,
        origin="lower",
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        cmap="viridis",
    )
    ax[2].set_title(f"Latent heatmap ({args.latent_buckets}×{args.latent_buckets})")
    ax[2].set_xlabel("z₁")
    ax[2].set_ylabel("z₂")
    fig.colorbar(im2, ax=ax[2], shrink=0.7, label="solutions")

    plt.tight_layout()
    fig.savefig(args.out, dpi=300)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
