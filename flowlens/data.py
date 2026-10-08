"""Dataset generation and loading.

On-disk format (a torch .pt file), shared by the synthetic generator and any real dataset:

    {
        "responses": float16/float32 tensor (B, C, N, N)  # e.g. intensity, cos(phase), sin(phase)
        "designs":   float32 tensor (B, 1, N, N)          # per-cell meta-atom dimension in um
        "d_min", "d_max": float                           # design range in um
        "config": dict                                     # optional, LensConfig used for synthesis
        "meta": dict of (B,) tensors                       # optional per-sample parameters
    }

To use FDTD data, save it in this format and pass `--data path.pt` to the training scripts.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .physics import ForwardModel, LensConfig, sample_designs


@torch.no_grad()
def generate(n: int, cfg: LensConfig, seed: int = 0, batch: int = 2500, device: str | None = None) -> dict:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    gen = torch.Generator(device=device).manual_seed(seed)
    fwd = ForwardModel(cfg)
    responses, designs, metas = [], [], []
    for start in range(0, n, batch):
        b = min(batch, n - start)
        d, wg, params = sample_designs(cfg, fwd.lib, b, device, generator=gen)
        r = fwd(d, wg, noise=True, generator=gen)
        responses.append(r.half().cpu())
        designs.append(d[:, None].float().cpu())
        metas.append({k: v.cpu() for k, v in params.items()})
    meta = {k: torch.cat([m[k] for m in metas]) for k in metas[0]}
    return {
        "responses": torch.cat(responses),
        "designs": torch.cat(designs),
        "d_min": cfg.d_min,
        "d_max": cfg.d_max,
        "config": cfg.to_dict(),
        "meta": meta,
    }


class LensDataset(Dataset):
    """Normalised (response, design) pairs.

    Responses are standardised per channel using training-set statistics.
    Designs are mapped from [d_min, d_max] um to [-1, 1]; use `to_um` to map back.
    """

    def __init__(self, blob: dict, indices: torch.Tensor, stats: dict | None = None):
        self.indices = indices
        self.responses = blob["responses"][indices].float()
        self.designs = blob["designs"][indices].float()
        self.d_min, self.d_max = float(blob["d_min"]), float(blob["d_max"])
        if stats is None:
            stats = {
                "mean": self.responses.mean(dim=(0, 2, 3), keepdim=True)[0],
                "std": self.responses.std(dim=(0, 2, 3), keepdim=True)[0].clamp_min(1e-6),
            }
        self.stats = stats
        self.responses = (self.responses - stats["mean"]) / stats["std"]
        self.designs_norm = self.to_norm(self.designs)

    def to_norm(self, d_um: torch.Tensor) -> torch.Tensor:
        return 2 * (d_um - self.d_min) / (self.d_max - self.d_min) - 1

    def to_um(self, d_norm: torch.Tensor) -> torch.Tensor:
        return self.d_min + (d_norm + 1) * (self.d_max - self.d_min) / 2

    def __len__(self):
        return len(self.designs)

    def __getitem__(self, i):
        return self.responses[i], self.designs_norm[i]


def load_splits(path: str, val_frac: float = 0.2, train_frac: float = 1.0, seed: int = 0):
    """80/20 train/val split (paper, Sec. 3). `train_frac` subsamples the training split (Fig. 2b).

    The validation set is the same for every `train_frac`.
    """
    blob = torch.load(path, weights_only=False)
    n = len(blob["designs"])
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    n_val = int(round(val_frac * n))
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    train_idx = train_idx[: max(1, int(round(train_frac * len(train_idx))))]
    train = LensDataset(blob, train_idx)
    val = LensDataset(blob, val_idx, stats=train.stats)
    return train, val, blob


def main():
    p = argparse.ArgumentParser(description="Generate the synthetic FlowLens dataset.")
    p.add_argument("--n", type=int, default=50_000, help="number of response-design pairs (paper: 50k)")
    p.add_argument("--n-cells", type=int, default=48)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="data/synthetic_50k.pt")
    args = p.parse_args()

    cfg = LensConfig(n_cells=args.n_cells)
    t0 = time.time()
    blob = generate(args.n, cfg, seed=args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(blob, args.out)
    r, d = blob["responses"], blob["designs"]
    print(f"wrote {args.out}: responses {tuple(r.shape)}, designs {tuple(d.shape)} in {time.time() - t0:.1f}s")
    print(f"design range [{d.min():.3f}, {d.max():.3f}] um, period {cfg.period:.4f} um")


if __name__ == "__main__":
    main()
