"""Baselines from the paper: segmentation CNN [1], cGAN [4] and cVAE [5]."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import CondDecoder, CondEncoder, InverseModel


# --------------------------------------------------------------------------
# Segmentation CNN (SegNet-style encoder-decoder with pooling indices)
# --------------------------------------------------------------------------


def _cbr(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class SegmentationCNN(InverseModel):
    """Per-cell classification of the design into `n_bins` discrete diameters.

    As in SegNet, the decoder upsamples with the encoder's max-pooling indices.
    The prediction is the centre of the argmax bin. With the default 31 bins
    over 0.08-0.38 um, the bin width is 0.01 um.
    """

    name = "segmentation"

    def __init__(self, n_cells: int, in_ch: int = 3, chs=(48, 96, 160), n_bins: int = 31):
        super().__init__()
        assert n_cells % 8 == 0, "SegNet path needs n_cells divisible by 8"
        self.n_bins = n_bins
        c0, c1, c2 = chs
        self.enc = nn.ModuleList([
            nn.Sequential(_cbr(in_ch, c0), _cbr(c0, c0)),
            nn.Sequential(_cbr(c0, c1), _cbr(c1, c1)),
            nn.Sequential(_cbr(c1, c2), _cbr(c2, c2), _cbr(c2, c2)),
        ])
        self.dec = nn.ModuleList([
            nn.Sequential(_cbr(c2, c2), _cbr(c2, c2), _cbr(c2, c1)),
            nn.Sequential(_cbr(c1, c1), _cbr(c1, c0)),
            nn.Sequential(_cbr(c0, c0)),
        ])
        self.head = nn.Conv2d(c0, n_bins, 3, padding=1)
        self.register_buffer("centers", torch.linspace(-1, 1, n_bins))

    def logits(self, c):
        h, idx, sizes = c, [], []
        for block in self.enc:
            h = block(h)
            sizes.append(h.shape)
            h, i = F.max_pool2d(h, 2, return_indices=True)
            idx.append(i)
        for block, i, s in zip(self.dec, reversed(idx), reversed(sizes)):
            h = block(F.max_unpool2d(h, i, 2, output_size=s[-2:]))
        return self.head(h)

    def train_step(self, c, d):
        target = ((d[:, 0] + 1) / 2 * (self.n_bins - 1)).round().long().clamp(0, self.n_bins - 1)
        loss = F.cross_entropy(self.logits(c), target)
        self._step(loss)
        return {"loss": loss.item()}

    @torch.no_grad()
    def predict(self, c, n_samples=1, temperature=0.0):
        d = self.centers[self.logits(c).argmax(1)][:, None]
        return d[:, None].expand(-1, n_samples, -1, -1, -1)


# --------------------------------------------------------------------------
# Conditional VAE
# --------------------------------------------------------------------------


class ConditionalVAE(InverseModel):
    """q(z | d, x) and p(d | z, x) with a Gaussian likelihood of fixed std `recon_sigma`."""

    name = "vae"

    def __init__(self, n_cells: int, in_ch: int = 3, latent_dim: int = 64, cond_chs=(32, 64, 96),
                 recon_sigma: float = 0.1, beta: float = 1.0):
        super().__init__()
        self.latent_dim, self.recon_sigma, self.beta = latent_dim, recon_sigma, beta
        self.cond = CondEncoder(in_ch, cond_chs)
        c2 = cond_chs[2]
        self.post = nn.Sequential(
            nn.Conv2d(1 + in_ch, 32, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(64, c2, 3, padding=1), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(c2, 2 * latent_dim),
        )
        self.dec = CondDecoder(n_cells, latent_dim, cond_chs)

    def train_step(self, c, d):
        mu, logvar = self.post(torch.cat([d, c], 1)).chunk(2, dim=1)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        recon = self.dec(z, self.cond(c))
        rec = ((recon - d) ** 2).flatten(1).sum(1) / (2 * self.recon_sigma**2)
        kl = 0.5 * (mu**2 + logvar.exp() - 1 - logvar).sum(1)
        loss = (rec + self.beta * kl).mean() / d[0].numel()
        self._step(loss)
        return {"loss": loss.item(), "kl": kl.mean().item()}

    @torch.no_grad()
    def predict(self, c, n_samples=1, temperature=0.0):
        feats = [f.repeat_interleave(n_samples, 0) for f in self.cond(c)]
        z = temperature * torch.randn(c.shape[0] * n_samples, self.latent_dim, device=c.device)
        d = self.dec(z, feats)
        return d.view(c.shape[0], n_samples, *d.shape[1:])


# --------------------------------------------------------------------------
# Conditional GAN (pix2pix-style: adversarial + L1)
# --------------------------------------------------------------------------


class PatchDiscriminator(nn.Module):
    def __init__(self, in_ch: int, ch: int = 64):
        super().__init__()
        sn = nn.utils.spectral_norm
        self.net = nn.Sequential(
            sn(nn.Conv2d(in_ch + 1, ch, 4, stride=2, padding=1)), nn.LeakyReLU(0.2),
            sn(nn.Conv2d(ch, 2 * ch, 4, stride=2, padding=1)), nn.LeakyReLU(0.2),
            sn(nn.Conv2d(2 * ch, 4 * ch, 3, padding=1)), nn.LeakyReLU(0.2),
            sn(nn.Conv2d(4 * ch, 1, 3, padding=1)),
        )

    def forward(self, c, d):
        return self.net(torch.cat([c, d], 1))


class _Generator(nn.Module):
    def __init__(self, n_cells, in_ch, latent_dim, cond_chs):
        super().__init__()
        self.cond = CondEncoder(in_ch, cond_chs)
        self.dec = CondDecoder(n_cells, latent_dim, cond_chs)

    def forward(self, z, c, n_samples=1):
        feats = [f.repeat_interleave(n_samples, 0) for f in self.cond(c)]
        return torch.tanh(self.dec(z, feats))


class ConditionalGAN(InverseModel):
    name = "gan"

    def __init__(self, n_cells: int, in_ch: int = 3, latent_dim: int = 64, cond_chs=(32, 64, 96),
                 l1_weight: float = 100.0):
        super().__init__()
        self.latent_dim, self.l1_weight = latent_dim, l1_weight
        self.G = _Generator(n_cells, in_ch, latent_dim, cond_chs)
        self.D = PatchDiscriminator(in_ch)

    def configure(self, lr: float):
        self.opt_g = torch.optim.Adam(self.G.parameters(), lr=lr, betas=(0.5, 0.999))
        self.opt_d = torch.optim.Adam(self.D.parameters(), lr=lr, betas=(0.5, 0.999))

    def train_step(self, c, d):
        z = torch.randn(c.shape[0], self.latent_dim, device=c.device)
        fake = self.G(z, c)

        real_logits, fake_logits = self.D(c, d), self.D(c, fake.detach())
        loss_d = (F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits))
                  + F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits)))
        self._step(loss_d, self.opt_d, self.D.parameters())

        adv_logits = self.D(c, fake)
        adv = F.binary_cross_entropy_with_logits(adv_logits, torch.ones_like(adv_logits))
        l1 = (fake - d).abs().mean()
        loss_g = adv + self.l1_weight * l1
        self._step(loss_g, self.opt_g, self.G.parameters())
        return {"loss": loss_g.item(), "loss_d": loss_d.item(), "l1": l1.item()}

    @torch.no_grad()
    def predict(self, c, n_samples=1, temperature=0.0):
        z = temperature * torch.randn(c.shape[0] * n_samples, self.latent_dim, device=c.device)
        d = self.G(z, c, n_samples)
        return d.view(c.shape[0], n_samples, *d.shape[1:])
