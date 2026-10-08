"""FlowLens: conditional RealNVP over metasurface designs (paper Sec. 2, Fig. 1c).

The flow F_theta maps a design map d (B, 1, N, N) to a latent z of the same
dimension. Every affine coupling layer is conditioned on the target phase and
intensity response through `CondEncoder`. Training minimises the exact
negative log-likelihood

    L(theta) = -sum_i [ log p_Z(F_theta(d_i; x_i)) + log |det dF_theta/dd (d_i; x_i)| ],

where x_i is the response. Paper Eq. (2) prints "+" before the log-det term;
the NLL needs it subtracted. Inverse design samples z ~ N(0, T^2 I) and
evaluates d = F_theta^{-1}(z; x).

Multi-scale layout (RealNVP, Dinh et al. 2016):
    N x N x 1      : checkerboard couplings
    squeeze -> N/2 : channel-wise couplings + checkerboard couplings
    squeeze -> N/4 : channel-wise couplings + checkerboard couplings
Couplings are interleaved with ActNorm, a data-initialised replacement for
RealNVP's batch norm that behaves the same in train and eval mode.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .common import CondEncoder, InverseModel, ResBlock


def squeeze(x: torch.Tensor) -> torch.Tensor:
    b, c, h, w = x.shape
    x = x.view(b, c, h // 2, 2, w // 2, 2).permute(0, 1, 3, 5, 2, 4)
    return x.reshape(b, c * 4, h // 2, w // 2)


def unsqueeze(x: torch.Tensor) -> torch.Tensor:
    b, c, h, w = x.shape
    x = x.view(b, c // 4, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3)
    return x.reshape(b, c // 4, h * 2, w * 2)


def checkerboard(h: int, w: int, parity: int, device) -> torch.Tensor:
    ii, jj = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
    return (((ii + jj) % 2) == parity).float()[None, None]


class ActNorm(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.loc = nn.Parameter(torch.zeros(1, ch, 1, 1))
        self.log_scale = nn.Parameter(torch.zeros(1, ch, 1, 1))
        self.register_buffer("initialized", torch.tensor(False))

    @torch.no_grad()
    def _init(self, x):
        self.loc.data = -x.mean(dim=(0, 2, 3), keepdim=True)
        self.log_scale.data = -torch.log(x.std(dim=(0, 2, 3), keepdim=True) + 1e-6)
        self.initialized.fill_(True)

    def forward(self, x):
        if not self.initialized and self.training:
            self._init(x)
        y = (x + self.loc) * torch.exp(self.log_scale)
        logdet = self.log_scale.sum() * x.shape[2] * x.shape[3]
        return y, logdet.expand(x.shape[0])

    def inverse(self, y):
        return y * torch.exp(-self.log_scale) - self.loc


class AffineCoupling(nn.Module):
    """y_b = x_b * exp(s(x_a, h)) + t(x_a, h), where h are the conditioning features."""

    def __init__(self, ch: int, cond_ch: int, hidden: int, mask: str, parity: int, n_res: int = 2):
        super().__init__()
        assert mask in ("checker", "channel")
        self.mask, self.parity, self.ch = mask, parity, ch
        in_ch = ch if mask == "checker" else ch // 2
        out_ch = ch if mask == "checker" else ch - ch // 2
        self.net = nn.Sequential(
            nn.Conv2d(in_ch + cond_ch + (1 if mask == "checker" else 0), hidden, 3, padding=1),
            *[ResBlock(hidden) for _ in range(n_res)],
            nn.GroupNorm(min(8, hidden), hidden), nn.SiLU(),
            nn.Conv2d(hidden, 2 * out_ch, 3, padding=1),
        )
        nn.init.zeros_(self.net[-1].weight)  # start as the identity map
        nn.init.zeros_(self.net[-1].bias)
        self.s_scale = nn.Parameter(torch.ones(1, out_ch, 1, 1))

    def _st(self, x_a, cond, extra=None):
        inp = [x_a, cond] if extra is None else [x_a, cond, extra.expand(x_a.shape[0], -1, -1, -1)]
        s, t = self.net(torch.cat(inp, 1)).chunk(2, dim=1)
        return self.s_scale * torch.tanh(s), t

    def _split(self, x):
        k = self.ch // 2
        a, b = x[:, :k], x[:, k:]
        return (a, b) if self.parity == 0 else (b, a)

    def _merge(self, a, b):
        return torch.cat([a, b], 1) if self.parity == 0 else torch.cat([b, a], 1)

    def forward(self, x, cond):
        if self.mask == "checker":
            m = checkerboard(x.shape[2], x.shape[3], self.parity, x.device)
            s, t = self._st(x * m, cond, m)
            s, t = s * (1 - m), t * (1 - m)
            return x * torch.exp(s) + t, s.sum(dim=(1, 2, 3))
        a, b = self._split(x)
        s, t = self._st(a, cond)
        return self._merge(a, b * torch.exp(s) + t), s.sum(dim=(1, 2, 3))

    def inverse(self, y, cond):
        if self.mask == "checker":
            m = checkerboard(y.shape[2], y.shape[3], self.parity, y.device)
            s, t = self._st(y * m, cond, m)
            s, t = s * (1 - m), t * (1 - m)
            return (y - t) * torch.exp(-s)
        a, b = self._split(y)
        s, t = self._st(a, cond)
        return self._merge(a, (b - t) * torch.exp(-s))


class FlowLens(InverseModel):
    name = "flow"

    # Default size (~2.7M params) is matched to the baselines (1.5-2.4M) and fits an 8 GB GPU at batch 128.
    def __init__(self, n_cells: int, in_ch: int = 3, cond_chs=(32, 64, 96), hidden=(32, 64, 128),
                 layout=(("checker",) * 2, ("channel", "checker") * 2, ("channel", "checker") * 2), n_res: int = 1):
        super().__init__()
        assert n_cells % 4 == 0, "n_cells must be divisible by 4 (two squeeze steps)"
        self.n_cells = n_cells
        self.cond = CondEncoder(in_ch, cond_chs)
        self.levels = nn.ModuleList()
        for lvl, kinds in enumerate(layout):
            ch = 4**lvl
            layers = nn.ModuleList()
            for i, kind in enumerate(kinds):
                parity = i % 2 if kind == "checker" else (i // 2) % 2
                layers.append(ActNorm(ch))
                layers.append(AffineCoupling(ch, cond_chs[lvl], hidden[lvl], kind, parity, n_res))
            self.levels.append(layers)
        self.dim = n_cells * n_cells

    def _forward(self, d, feats):
        x, logdet = d, torch.zeros(d.shape[0], device=d.device)
        for lvl, layers in enumerate(self.levels):
            if lvl > 0:
                x = squeeze(x)
            for layer in layers:
                x, ld = layer(x) if isinstance(layer, ActNorm) else layer(x, feats[lvl])
                logdet = logdet + ld
        return x, logdet

    def _inverse(self, z, feats):
        x = z
        for lvl in reversed(range(len(self.levels))):
            for layer in reversed(self.levels[lvl]):
                x = layer.inverse(x) if isinstance(layer, ActNorm) else layer.inverse(x, feats[lvl])
            if lvl > 0:
                x = unsqueeze(x)
        return x

    def log_prob(self, d, c):
        z, logdet = self._forward(d, self.cond(c))
        log_pz = -0.5 * (z**2 + math.log(2 * math.pi)).flatten(1).sum(1)
        return log_pz + logdet

    def train_step(self, c, d):
        nll = -self.log_prob(d, c).mean() / self.dim  # nats per design cell
        self._step(nll)
        return {"loss": nll.item()}

    @torch.no_grad()
    def predict(self, c, n_samples=1, temperature=0.0):
        feats = self.cond(c)
        b = c.shape[0]
        lvls = len(self.levels) - 1
        zshape = (4**lvls, self.n_cells // 2**lvls, self.n_cells // 2**lvls)
        feats_rep = [f.repeat_interleave(n_samples, 0) for f in feats]
        z = temperature * torch.randn(b * n_samples, *zshape, device=c.device)
        d = self._inverse(z, feats_rep)
        return d.view(b, n_samples, 1, self.n_cells, self.n_cells)
