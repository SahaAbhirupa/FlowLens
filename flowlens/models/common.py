"""Building blocks shared by FlowLens and the baselines."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def norm(ch: int) -> nn.GroupNorm:
    return nn.GroupNorm(min(8, ch), ch)


class ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.body = nn.Sequential(
            norm(ch), nn.SiLU(), nn.Conv2d(ch, ch, 3, padding=1),
            norm(ch), nn.SiLU(), nn.Conv2d(ch, ch, 3, padding=1),
        )

    def forward(self, x):
        return x + self.body(x)


class CondEncoder(nn.Module):
    """Response map (B, C, N, N) -> features at resolutions N, N/2, N/4.

    Mirrors the conditioning path of Fig. 1c: strided convolutions ("Conv 1",
    "Conv 2") give the local multi-scale features, and two linear layers
    ("Linear 1", "Linear 2") on the pooled coarsest features give a global
    context vector, which is broadcast-added at every scale. The global vector
    carries quantities that are not local, such as the feed type and the
    focal length.
    """

    def __init__(self, in_ch: int = 3, chs=(32, 64, 96), global_dim: int = 128):
        super().__init__()
        c0, c1, c2 = chs
        self.chs = chs
        self.stem = nn.Sequential(nn.Conv2d(in_ch, c0, 3, padding=1), ResBlock(c0))
        self.conv1 = nn.Sequential(nn.Conv2d(c0, c1, 3, stride=2, padding=1), ResBlock(c1))
        self.conv2 = nn.Sequential(nn.Conv2d(c1, c2, 3, stride=2, padding=1), ResBlock(c2))
        self.linear1 = nn.Linear(c2, global_dim)
        self.linear2 = nn.Linear(global_dim, global_dim)
        self.to_scale = nn.ModuleList([nn.Linear(global_dim, c) for c in chs])

    def forward(self, c: torch.Tensor) -> list[torch.Tensor]:
        f0 = self.stem(c)
        f1 = self.conv1(f0)
        f2 = self.conv2(f1)
        g = self.linear2(F.silu(self.linear1(f2.mean(dim=(2, 3)))))
        g = F.silu(g)
        return [f + proj(g)[:, :, None, None] for f, proj in zip((f0, f1, f2), self.to_scale)]


class CondDecoder(nn.Module):
    """Latent vector + conditioning features -> design map (B, 1, N, N).

    Shared by the cVAE decoder and the cGAN generator.
    """

    def __init__(self, n_cells: int, latent_dim: int, cond_chs=(32, 64, 96), chs=(32, 64, 96)):
        super().__init__()
        c0, c1, c2 = chs
        self.base = n_cells // 4
        self.fc = nn.Linear(latent_dim, c2 * self.base * self.base)
        self.c2 = c2
        self.b2 = nn.Sequential(nn.Conv2d(c2 + cond_chs[2], c2, 3, padding=1), ResBlock(c2))
        self.up1 = nn.ConvTranspose2d(c2, c1, 4, stride=2, padding=1)  # "DeConv 2"
        self.b1 = nn.Sequential(nn.Conv2d(c1 + cond_chs[1], c1, 3, padding=1), ResBlock(c1))
        self.up0 = nn.ConvTranspose2d(c1, c0, 4, stride=2, padding=1)  # "DeConv 1"
        self.b0 = nn.Sequential(nn.Conv2d(c0 + cond_chs[0], c0, 3, padding=1), ResBlock(c0))
        self.out = nn.Sequential(norm(c0), nn.SiLU(), nn.Conv2d(c0, 1, 3, padding=1))

    def forward(self, z: torch.Tensor, feats: list[torch.Tensor]) -> torch.Tensor:
        f0, f1, f2 = feats
        h = self.fc(z).view(-1, self.c2, self.base, self.base)
        h = self.b2(torch.cat([h, f2], 1))
        h = self.b1(torch.cat([self.up1(h), f1], 1))
        h = self.b0(torch.cat([self.up0(h), f0], 1))
        return self.out(h)


class InverseModel(nn.Module):
    """Common interface used by the training harness.

    Designs are in normalised units, [-1, 1].
    """

    name = "base"

    def configure(self, lr: float):
        self.opt = torch.optim.Adam(self.parameters(), lr=lr)

    def train_step(self, c: torch.Tensor, d: torch.Tensor) -> dict:
        raise NotImplementedError

    @torch.no_grad()
    def predict(self, c: torch.Tensor, n_samples: int = 1, temperature: float = 0.0) -> torch.Tensor:
        """Returns (B, S, 1, N, N) design samples. temperature=0 gives the point estimate."""
        raise NotImplementedError

    def _step(self, loss: torch.Tensor, opt=None, params=None, clip: float = 10.0):
        opt = opt or self.opt
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params if params is not None else self.parameters(), clip)
        opt.step()
