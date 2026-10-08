"""Synthetic metalens surrogate: meta-atom library, Eq. (1) lens phase, coupled forward model.

This is a stand-in for the full-wave (FDTD) dataset used in the paper. It is
designed to keep the properties that make the inverse problem interesting:

* Meta-atoms are nanopillars on a square lattice whose design parameter is the
  pillar diameter d (um). The phase library spans more than 2*pi, so some
  target phases have two valid diameters (a "low" and a "high" branch). The
  inverse map from response to design is therefore multi-valued, which is the
  motivation for a generative model.
* Neighbouring pillars couple through their gap, so a cell's response depends
  on its neighbours and not only on its own unit-cell (periodic) response.
* Two feed configurations (Fig. 1a/1b): a parallel-fed plane wave, and a
  waveguide-fed guided mode with propagation phase n_wg*k0*y and an amplitude
  that decays along the propagation direction.
* The "response" is the complex field at a plane z_obs above the metasurface,
  obtained by angular-spectrum propagation and corrupted by measurement noise.
  It is returned as (intensity, cos(phase), sin(phase)).

All functions are batched torch operations so the dataset can be generated
on the GPU.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import torch


@dataclass
class LensConfig:
    wavelength: float = 0.633  # um (paper: 633 nm)
    size: float = 20.0  # um, lens aperture side (paper: 20 um x 20 um)
    n_cells: int = 48  # cells per side -> period = size / n_cells
    d_min: float = 0.08  # um, smallest pillar diameter
    d_max: float = 0.38  # um, largest pillar diameter
    n_pillar: float = 2.4  # TiO2-like pillar index at 633 nm
    phase_span: float = 2.4 * math.pi  # total phase coverage of the library
    # neighbour coupling: kappa = coupling_strength * exp(-gap / coupling_length)
    coupling_strength: float = 0.25
    coupling_length: float = 0.06  # um
    coupling_phase: float = math.pi / 4
    # waveguide feed (Fig. 1a)
    n_wg: float = 1.85  # effective index of the feeding slab mode
    wg_loss: float = 0.04  # 1/um, amplitude decay from out-coupling
    # observation
    z_obs: float = 0.5  # um above the metasurface
    noise_std: float = 0.02  # complex Gaussian noise on the observed field
    # dataset sampling
    f_range: tuple = (8.0, 40.0)  # um focal length
    offset_range: float = 3.0  # um, |x0|, |y0| focal-spot offset
    aberration_std: float = 0.5  # rad, low-order random aberrations
    jitter_std: float = 0.008  # um, per-pillar fabrication jitter

    @property
    def period(self) -> float:
        return self.size / self.n_cells

    @property
    def k0(self) -> float:
        return 2 * math.pi / self.wavelength

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------
# Meta-atom library
# --------------------------------------------------------------------------


class MetaAtomLibrary:
    """Pillar diameter -> (unwrapped phase, transmission amplitude)."""

    def __init__(self, cfg: LensConfig, n_table: int = 4001):
        self.cfg = cfg
        d = torch.linspace(cfg.d_min, cfg.d_max, n_table, dtype=torch.float64)
        n_lo, n_hi = self._n_eff(torch.tensor([cfg.d_min, cfg.d_max], dtype=torch.float64))
        # pillar height chosen so the library spans exactly `phase_span`
        self.height = cfg.phase_span / (cfg.k0 * (n_hi - n_lo).item())
        self.d_table = d
        self.phase_table = self.phase(d)  # monotonically increasing
        self.phi_min = self.phase_table[0].item()
        self.phi_max = self.phase_table[-1].item()

    def _n_eff(self, d: torch.Tensor) -> torch.Tensor:
        fill = math.pi * (d / 2) ** 2 / self.cfg.period**2
        return torch.sqrt(1 + (self.cfg.n_pillar**2 - 1) * fill**0.7)

    def phase(self, d: torch.Tensor) -> torch.Tensor:
        return self.cfg.k0 * self._n_eff(d) * self.height

    def amplitude(self, d: torch.Tensor) -> torch.Tensor:
        # slow roll-off, a Mie-like resonance dip and a Fabry-Perot ripple
        span = self.cfg.d_max - self.cfg.d_min
        ripple = 0.04 * torch.cos(2 * self.phase(d))
        dip = 0.25 * torch.exp(-(((d - 0.30) / 0.025) ** 2))
        return 0.95 - 0.10 * (d - self.cfg.d_min) / span - dip + ripple

    def transmission(self, d: torch.Tensor) -> torch.Tensor:
        return torch.polar(self.amplitude(d), self.phase(d))

    def invert(self, target_phase: torch.Tensor, branch: torch.Tensor) -> torch.Tensor:
        """Diameter realising `target_phase` (mod 2*pi).

        Where two diameters realise the phase, `branch` (bool, broadcastable)
        selects the larger one (True) or the smaller one (False).
        """
        two_pi = 2 * math.pi
        wrapped = self.phi_min + torch.remainder(target_phase - self.phi_min, two_pi)
        upper = wrapped + two_pi
        has_upper = upper <= self.phi_max
        phi = torch.where(branch & has_upper, upper, wrapped)
        return _interp(phi, self.phase_table.to(phi), self.d_table.to(phi))


def _interp(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
    """1-D linear interpolation (torch has no np.interp); xp must be increasing."""
    idx = torch.searchsorted(xp, x.contiguous()).clamp(1, len(xp) - 1)
    x0, x1 = xp[idx - 1], xp[idx]
    y0, y1 = fp[idx - 1], fp[idx]
    w = ((x - x0) / (x1 - x0)).clamp(0, 1)
    return y0 + w * (y1 - y0)


# --------------------------------------------------------------------------
# Geometry, illumination and the lens phase of Eq. (1)
# --------------------------------------------------------------------------


def cell_coords(cfg: LensConfig, device=None, dtype=torch.float32):
    c = (torch.arange(cfg.n_cells, device=device, dtype=dtype) - (cfg.n_cells - 1) / 2) * cfg.period
    y, x = torch.meshgrid(c, c, indexing="ij")  # rows = y (waveguide propagation axis)
    return x, y


def illumination(cfg: LensConfig, waveguide: torch.Tensor) -> torch.Tensor:
    """Incident complex field per cell, shape (B, N, N).

    waveguide: bool tensor (B,). Parallel-fed cells see a unit plane wave.
    Waveguide-fed cells see exp(-wg_loss*s) * exp(i*n_wg*k0*s), where s is the
    distance travelled along +y from the input edge.
    """
    x, y = cell_coords(cfg, device=waveguide.device)
    s = (y - y.min())[None]
    amp = torch.where(waveguide[:, None, None], torch.exp(-cfg.wg_loss * s), torch.ones_like(s))
    ph = torch.where(waveguide[:, None, None], cfg.n_wg * cfg.k0 * s, torch.zeros_like(s))
    return torch.polar(amp, ph)


def lens_phase(cfg: LensConfig, f, x0, y0) -> torch.Tensor:
    """Hyperbolic focusing phase -k0 * (sqrt((x-x0)^2 + (y-y0)^2 + f^2) - f).

    This is the first term of Eq. (1). Its second term, the -n_eff*k0 feed
    phase, is applied via `illumination` and compensated in `sample_designs`.
    """
    x, y = cell_coords(cfg, device=f.device)
    r2 = (x[None] - x0[:, None, None]) ** 2 + (y[None] - y0[:, None, None]) ** 2
    fb = f[:, None, None]
    return -cfg.k0 * (torch.sqrt(r2 + fb**2) - fb)


def aberration_phase(cfg: LensConfig, coeffs: torch.Tensor) -> torch.Tensor:
    """Astigmatism (x2) + spherical terms; coeffs (B, 3) in radians."""
    x, y = cell_coords(cfg, device=coeffs.device)
    R = cfg.size / 2
    u, v = x / R, y / R
    basis = torch.stack([u**2 - v**2, 2 * u * v, (u**2 + v**2) ** 2 - (u**2 + v**2)])
    return torch.einsum("bk,kij->bij", coeffs, basis)


# --------------------------------------------------------------------------
# Forward model: design -> response
# --------------------------------------------------------------------------


class ForwardModel:
    """Maps pillar diameters (B, N, N) in um to responses (B, 3, N, N)."""

    def __init__(self, cfg: LensConfig):
        self.cfg = cfg
        self.lib = MetaAtomLibrary(cfg)
        self._kernel = {}

    def coupled_transmission(self, d: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        tau = self.lib.transmission(d)
        out = tau.clone()
        rot = complex(math.cos(cfg.coupling_phase), math.sin(cfg.coupling_phase))
        for dim in (-1, -2):
            for shift in (1, -1):
                # non-periodic neighbour: pad by replicating the edge (zero coupling term)
                d_n = _shift(d, shift, dim)
                tau_n = _shift(tau, shift, dim)
                gap = cfg.period - (d + d_n) / 2
                kappa = cfg.coupling_strength * torch.exp(-gap / cfg.coupling_length)
                out = out + rot * kappa * (tau_n - tau)
        return out

    def near_field(self, d: torch.Tensor, waveguide: torch.Tensor) -> torch.Tensor:
        return illumination(self.cfg, waveguide) * self.coupled_transmission(d)

    def propagate(self, field: torch.Tensor) -> torch.Tensor:
        """Angular-spectrum propagation by z_obs with 2x zero padding."""
        cfg = self.cfg
        n = field.shape[-1]
        m = 2 * n
        key = (field.device, field.dtype)
        if key not in self._kernel:
            k = 2 * math.pi * torch.fft.fftfreq(m, d=cfg.period, device=field.device, dtype=torch.float64)
            ky, kx = torch.meshgrid(k, k, indexing="ij")
            kz = torch.sqrt((cfg.k0**2 - kx**2 - ky**2).to(torch.complex128))
            self._kernel[key] = torch.exp(1j * kz * cfg.z_obs).to(field.dtype)
        pad = (n // 2, m - n - n // 2)
        big = torch.nn.functional.pad(field, pad + pad)
        out = torch.fft.ifft2(torch.fft.fft2(big) * self._kernel[key])
        return out[..., pad[0] : pad[0] + n, pad[0] : pad[0] + n]

    def __call__(self, d: torch.Tensor, waveguide: torch.Tensor, noise: bool = True, generator=None) -> torch.Tensor:
        e = self.propagate(self.near_field(d, waveguide))
        if noise and self.cfg.noise_std > 0:
            nr = torch.randn(e.shape, device=e.device, generator=generator)
            ni = torch.randn(e.shape, device=e.device, generator=generator)
            e = e + self.cfg.noise_std * torch.complex(nr, ni)
        ang = torch.angle(e)
        return torch.stack([e.abs() ** 2, torch.cos(ang), torch.sin(ang)], dim=1).float()


def _shift(t: torch.Tensor, shift: int, dim: int) -> torch.Tensor:
    """Neighbour along `dim`, replicating the edge so boundary cells see no coupling."""
    n = t.shape[dim]
    if shift == 1:
        idx = torch.cat([torch.arange(1, n), torch.tensor([n - 1])])
    else:
        idx = torch.cat([torch.tensor([0]), torch.arange(0, n - 1)])
    return t.index_select(dim, idx.to(t.device))


# --------------------------------------------------------------------------
# Design sampling
# --------------------------------------------------------------------------


def sample_designs(cfg: LensConfig, lib: MetaAtomLibrary, batch: int, device, generator=None):
    """Random lens designs (B, N, N) in um, plus the waveguide flags and parameters."""

    def u(lo, hi, *shape):
        return lo + (hi - lo) * torch.rand(*(shape or (batch,)), device=device, generator=generator)

    f = u(*cfg.f_range)
    x0 = u(-cfg.offset_range, cfg.offset_range)
    y0 = u(-cfg.offset_range, cfg.offset_range)
    phi0 = u(0.0, 2 * math.pi)
    waveguide = torch.rand(batch, device=device, generator=generator) < 0.5
    branch = torch.rand(batch, device=device, generator=generator) < 0.5  # per-design branch policy
    aberr = cfg.aberration_std * torch.randn(batch, 3, device=device, generator=generator)

    # target field phase = lens (Eq. 1) + offset + aberrations; the feed phase is compensated
    feed = torch.angle(illumination(cfg, waveguide))
    target = lens_phase(cfg, f, x0, y0) + aberration_phase(cfg, aberr) + phi0[:, None, None] - feed
    d = lib.invert(target, branch[:, None, None])

    if cfg.jitter_std > 0:
        d = d + cfg.jitter_std * torch.randn(d.shape, device=device, generator=generator)
        d = _reflect(d, cfg.d_min, cfg.d_max)

    params = dict(f=f, x0=x0, y0=y0, phi0=phi0, waveguide=waveguide, branch=branch, aberration=aberr)
    return d, waveguide, params


def _reflect(d: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    d = torch.where(d < lo, 2 * lo - d, d)
    return torch.where(d > hi, 2 * hi - d, d)
