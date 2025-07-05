import argparse
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class AutoEncoder(nn.Module):
    def __init__(self, latent_dim: int = 2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 4),
            nn.Sigmoid(),
        )

    def forward(self, x):
        latent = self.encoder(x)
        recon = self.decoder(latent)
        return recon, latent


def train(model, dl, epochs=50, lr=1e-3, device="cpu"):
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    model.to(device)
    model.train()

    for ep in tqdm(range(1, epochs + 1), desc="Training AutoEncoder", unit="epoch"):
        running = 0.0
        for (xb,) in dl:
            xb = xb.to(device)
            recon, _ = model(xb)
            loss = criterion(recon, xb)
            optim.zero_grad()
            loss.backward()
            optim.step()
            running += loss.item() * xb.size(0)
        print(f"epoch {ep:3d} | loss {running/len(dl.dataset):.6f}")
        if ep % 10 == 0:
            ckpt_dir = Path("checkpoints")
            ckpt_dir.mkdir(exist_ok=True)
            torch.save(model.state_dict(), ckpt_dir / f"ae_epoch{ep:02d}.pt")


def plot_latent(z: np.ndarray, out: str = "latent_scatter.png"):
    if plt is None:
        print("matplotlib not available skipping plot.")
        return
    plt.figure(figsize=(6, 5))
    plt.scatter(z[:, 0], z[:, 1], s=2, alpha=0.4)
    plt.xlabel("z₁")
    plt.ylabel("z₂")
    plt.title("Latent space of dutyfactor descriptors")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    print(f"saved scatter to {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--map_file", type=str, default="map_ant.npz")
    parser.add_argument("--latent_dim", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--save_path", type=str, default="latent_embeddings.npy", nargs="?"
    )
    parser.add_argument(
        "--plot", action="store_true", help="plot latent scatter at the end"
    )
    args = parser.parse_args()

    data = np.load(args.map_file)
    bd = data["bd"].astype(np.float32)  # shape (N,4), already in [0,1]

    # Convert to tensor in [-1,1] for smoother gradients
    bd_tensor = torch.from_numpy((bd * 2.0) - 1.0)
    ds = TensorDataset(bd_tensor)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True)

    model = AutoEncoder(latent_dim=args.latent_dim)
    train(model, dl, epochs=args.epochs, device=args.device)

    torch.save(model.state_dict(), "ae_final.pt")

    with torch.no_grad():
        z_all = model.encoder(bd_tensor.to(args.device)).cpu().numpy()
    np.save(args.save_path, z_all)
    print(f"Training done. Saved ae_final.pt and {args.save_path}")

    if args.plot:
        plot_latent(z_all)


if __name__ == "__main__":
    main()
