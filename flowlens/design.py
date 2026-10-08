"""Inverse design with a trained model: lens specification -> metasurface design.

    python -m flowlens.design --ckpt runs/paper/flow_frac1_seed0.pt --f 20 --feed waveguide --out design.npz

The target is the ideal focusing response: the hyperbolic phase of Eq. (1),
carried by the feed's amplitude profile, propagated to the observation plane.
K designs are sampled from the flow, and each is checked with the forward
model. The design whose simulated response matches the target best is kept.
"""

from __future__ import annotations

import argparse
import math

import numpy as np
import torch

from .experiments import load_model
from .physics import ForwardModel, LensConfig, illumination, lens_phase


@torch.no_grad()
def ideal_response(cfg: LensConfig, f: float, x0: float, y0: float, waveguide: bool, device) -> torch.Tensor:
    fwd = ForwardModel(cfg)
    t = lambda v: torch.tensor([v], device=device, dtype=torch.float64)
    wg = torch.tensor([waveguide], device=device)
    # real pillars transmit |t(d)| < 1; use the library's mean so the target stays in-distribution
    amp = illumination(cfg, wg).abs() * fwd.lib.amplitude(fwd.lib.d_table).mean().item()
    field = amp * torch.exp(1j * lens_phase(cfg, t(f), t(x0), t(y0)))
    e = fwd.propagate(field)
    ang = torch.angle(e)
    return torch.stack([e.abs() ** 2, torch.cos(ang), torch.sin(ang)], 1).float()


@torch.no_grad()
def design(ckpt: str, data_cfg: dict, f: float, x0: float = 0.0, y0: float = 0.0, waveguide: bool = False,
           k: int = 64, temperature: float = 0.7, device: str | None = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(ckpt, weights_only=False, map_location="cpu")
    model = load_model(ckpt, device)
    cfg = LensConfig(**data_cfg)
    fwd = ForwardModel(cfg)
    target = ideal_response(cfg, f, x0, y0, waveguide, device)
    c = (target - ck["stats"]["mean"].to(device)) / ck["stats"]["std"].to(device)
    d_min, d_max = ck["d_range"]
    point = model.predict(c, n_samples=1, temperature=0.0)
    d = torch.cat([point, model.predict(c, n_samples=k, temperature=temperature)], 1)
    d = d[0, :, 0].clamp(-1, 1)  # candidate 0 is the T=0 point estimate
    d_um = d_min + (d + 1) * (d_max - d_min) / 2
    sim = fwd(d_um.double(), torch.full((len(d_um),), waveguide, device=device), noise=False)
    err = ((sim - target) ** 2).flatten(1).mean(1)
    best = int(err.argmin())
    return d_um[best].cpu().numpy(), sim[best].cpu().numpy(), target[0].cpu().numpy(), err.cpu().numpy()


@torch.no_grad()
def lookup_design(data_cfg: dict, f: float, x0: float = 0.0, y0: float = 0.0, waveguide: bool = False,
                  device: str | None = None):
    """Unit-cell baseline: invert the periodic library phase per cell, ignoring coupling (Sec. 1).

    Both diameter branches are tried; the one whose simulated response matches the target best is kept.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    cfg = LensConfig(**data_cfg)
    fwd = ForwardModel(cfg)
    target = ideal_response(cfg, f, x0, y0, waveguide, device)
    t = lambda v: torch.tensor([v], device=device, dtype=torch.float64)
    wg = torch.tensor([waveguide], device=device)
    phase = lens_phase(cfg, t(f), t(x0), t(y0)) - torch.angle(illumination(cfg, wg))
    d = torch.cat([fwd.lib.invert(phase, torch.tensor(b, device=device)) for b in (False, True)])
    sim = fwd(d, wg.repeat(2), noise=False)
    err = ((sim - target) ** 2).flatten(1).mean(1)
    best = int(err.argmin())
    return d[best].cpu().numpy(), sim[best].cpu().numpy(), err[best].item()


def phase_rms(sim, target) -> float:
    """RMS phase error (in units of pi) between two (3, N, N) responses."""
    dphi = np.angle(np.exp(1j * (np.arctan2(sim[2], sim[1]) - np.arctan2(target[2], target[1]))))
    return float(np.sqrt((dphi**2).mean()) / math.pi)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data", default="data/synthetic_50k.pt", help="dataset whose LensConfig the model was trained on")
    p.add_argument("--f", type=float, required=True, help="focal length (um)")
    p.add_argument("--x0", type=float, default=0.0)
    p.add_argument("--y0", type=float, default=0.0)
    p.add_argument("--feed", choices=["parallel", "waveguide"], default="parallel")
    p.add_argument("--k", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--out", default="design.npz")
    a = p.parse_args()
    cfg = torch.load(a.data, weights_only=False)["config"]
    d, sim, target, err = design(a.ckpt, cfg, a.f, a.x0, a.y0, a.feed == "waveguide", a.k, a.temperature)
    d_lu, sim_lu, err_lu = lookup_design(cfg, a.f, a.x0, a.y0, a.feed == "waveguide")
    np.savez(a.out, diameters_um=d, simulated_response=sim, target_response=target, lookup_diameters_um=d_lu)
    print(f"wrote {a.out}: {d.shape} pillar diameters in [{d.min():.3f}, {d.max():.3f}] um")
    print(f"FlowLens point estimate: response MSE {err[0]:.4f}")
    print(f"FlowLens best of {a.k}+1: response MSE {err.min():.4f}, RMS phase error {phase_rms(sim, target):.3f} pi")
    print(f"unit-cell lookup:        response MSE {err_lu:.4f}, RMS phase error {phase_rms(sim_lu, target):.3f} pi")


if __name__ == "__main__":
    main()
