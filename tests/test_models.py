import math

import pytest
import torch

from flowlens.models import MODELS, build_model
from flowlens.models.flow import FlowLens, squeeze, unsqueeze
from flowlens.physics import ForwardModel, LensConfig, MetaAtomLibrary, sample_designs

torch.manual_seed(0)


def _warm(flow, d, c):
    """Initialise ActNorm and perturb the zero-initialised coupling outputs so the flow is non-trivial."""
    flow.train()
    flow.log_prob(d, c)
    with torch.no_grad():
        for m in flow.modules():
            if hasattr(m, "net"):
                m.net[-1].weight.normal_(0, 0.05)
                m.net[-1].bias.normal_(0, 0.05)
    flow.eval()


def test_squeeze_roundtrip():
    x = torch.randn(2, 4, 8, 8)
    assert torch.equal(unsqueeze(squeeze(x)), x)


def test_flow_inverse_and_logdet():
    n = 8  # 64-dim design -> brute-force Jacobian is cheap
    flow = FlowLens(n_cells=n, cond_chs=(8, 8, 8), hidden=(8, 8, 8), n_res=1).double()
    c = torch.randn(3, 3, n, n, dtype=torch.float64)
    d = torch.randn(3, 1, n, n, dtype=torch.float64)
    _warm(flow, d, c)

    feats = flow.cond(c)
    z, logdet = flow._forward(d, feats)
    assert torch.allclose(flow._inverse(z, feats), d, atol=1e-10)

    for i in range(len(d)):
        f = lambda x: flow._forward(x.view(1, 1, n, n), [ft[i : i + 1] for ft in feats])[0].flatten()
        J = torch.autograd.functional.jacobian(f, d[i].flatten())
        assert torch.allclose(torch.linalg.slogdet(J)[1], logdet[i], atol=1e-8)


@pytest.mark.parametrize("name", list(MODELS))
def test_model_step_and_predict(name):
    n = 16
    model = build_model(name, n_cells=n)
    model.configure(1e-3)
    c, d = torch.randn(8, 3, n, n), torch.rand(8, 1, n, n) * 2 - 1
    logs = model.train_step(c, d)
    assert math.isfinite(logs["loss"])
    model.eval()
    out = model.predict(c, n_samples=3, temperature=1.0)
    assert out.shape == (8, 3, 1, n, n)
    assert torch.isfinite(out).all()


def test_library_inversion_roundtrip():
    cfg = LensConfig()
    lib = MetaAtomLibrary(cfg)
    psi = torch.rand(1000, dtype=torch.float64) * 2 * math.pi
    for branch in (False, True):
        d = lib.invert(psi, torch.tensor(branch))
        err = torch.remainder(lib.phase(d) - psi + math.pi, 2 * math.pi) - math.pi
        assert err.abs().max() < 1e-6
        assert d.min() >= cfg.d_min and d.max() <= cfg.d_max


def test_ideal_lens_has_focusing_phase():
    # without coupling/jitter/aberration the near-field phase must equal the target lens phase (mod 2pi)
    cfg = LensConfig(coupling_strength=0.0, jitter_std=0.0, aberration_std=0.0, noise_std=0.0)
    fwd = ForwardModel(cfg)
    d, wg, p = sample_designs(cfg, fwd.lib, 6, "cpu", generator=torch.Generator().manual_seed(0))
    from flowlens.physics import lens_phase

    target = lens_phase(cfg, p["f"], p["x0"], p["y0"]) + p["phi0"][:, None, None]
    got = torch.angle(fwd.near_field(d.double(), wg))
    err = torch.remainder(got - target + math.pi, 2 * math.pi) - math.pi
    assert err.abs().max() < 1e-4
