import time

import pytest
import torch

from aeroshape.diffusion.denoiser import ShapeDenoiser
from aeroshape.diffusion.guided_shape_diffusion import GuidedSampler, diffusion_loss
from aeroshape.diffusion.schedule import CosineSchedule
from aeroshape.geometry.reference_shapes import is_valid, lhs_shapes, shape_distance
from aeroshape.pinn.transformer_pinn import GeometryCrossAttentionPINN
from tests.pinn import make_nurbs

CPU = torch.device("cpu")


def build(nurbs, **kw):
    xc = nurbs.x_ctrl[nurbs.free_idx % nurbs.k]
    return ShapeDenoiser(nurbs.n_free, xc, (nurbs.free_idx // nurbs.k).long(), **kw)


class Oracle(torch.nn.Module):
    """Perfect eps for a known z0 (ignores condition)."""

    def __init__(self, z0, sched):
        super().__init__()
        self.z0, self.g = z0, sched.gamma_bar

    def forward(self, z_t, t, cond, null_mask=None):
        gb = self.g[t.cpu()].to(z_t)[:, None]
        z0 = self.z0.to(z_t).repeat(z_t.shape[0] // self.z0.shape[0], 1)
        return (z_t - gb.sqrt() * z0) / (1 - gb).sqrt()


def test_ddim_oracle_recovers_z0():
    n, s = make_nurbs(), CosineSchedule()
    z0 = torch.randn(4, n.n_free).clamp(-2, 2)
    mean, std = torch.zeros(n.n_free), torch.ones(n.n_free)
    smp = GuidedSampler(Oracle(z0, s), s, n, None, (mean, std), physics_fn=lambda a, c: (a.sum(1), a.sum(1)))
    r = smp.sample(torch.zeros(4, 4), torch.full((n.k,), -10.0), guide_scale=0.0, generator=torch.Generator().manual_seed(0), record=True)
    assert torch.allclose(r.z_trajectory[-1], z0, atol=1e-3)
    assert torch.allclose(r.alpha, n.free_to_full(z0), atol=1e-3)


@pytest.fixture(scope="module")
def trained():
    torch.manual_seed(0)
    n, s = make_nurbs(), CosineSchedule()
    alpha = lhs_shapes(make_nurbs(dtype=torch.float64), 16, seed=0).float()
    free = n.full_to_free(alpha)
    mean, std = free.mean(0), free.std(0).clamp_min(1e-3)
    z0 = (free - mean) / std
    cond = torch.randn(16, 4)  # arbitrary distinct label per shape
    m = build(n)
    opt = torch.optim.AdamW(m.parameters(), 1e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 2000)
    g = torch.Generator().manual_seed(0)
    t0 = time.time()
    for _ in range(2000):
        loss = diffusion_loss(m, s, z0.repeat(4, 1), cond.repeat(4, 1), 0.1, g)
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    print(f"\ntrain {time.time() - t0:.1f}s final loss {loss.item():.4f}")
    return n, s, m.eval().requires_grad_(False), alpha, cond, (mean, std)


def _sample(trained, B, seed=0, c=None, **kw):
    n, s, m, alpha, cond, norm = trained
    smp = GuidedSampler(m, s, n, None, norm, physics_fn=kw.pop("physics_fn", None))
    c = cond.repeat(B // 16 + 1, 1)[:B] if c is None else c
    return smp, c, smp.sample(c, torch.zeros(n.k), generator=torch.Generator().manual_seed(seed), **kw)


@pytest.mark.slow
def test_overfit_and_validity(trained):
    n, _, _, alpha, _, _ = trained
    _, _, r = _sample(trained, 64, cfg_scale=1.0, guide_scale=0.0)
    d = torch.stack([shape_distance(n, r.alpha[i:i + 1].expand(16, -1), alpha) for i in range(64)]).min(1).values
    valid = is_valid(n, r.alpha).float().mean().item()
    print(f"nearest-training-shape RMS dist (chord): max {d.max():.4f} mean {d.mean():.4f}; validity {valid:.3f}")
    assert d.max() < 0.01 and valid >= 0.95


@pytest.mark.slow
def test_guidance_helps(trained):
    n, _, _, alpha, _, _ = trained
    a_star = alpha[3:4]  # a training shape; the condition below points at a different one (7)
    phys = lambda a, c: ((((a - a_star) / 0.03) ** 2).sum(1), torch.zeros(a.shape[0]))
    c = trained[4][7:8].repeat(16, 1)
    c[:, :2] = 0.0  # CL_ref = CD_ref = 0 (stand-in CL is 0, so J = CD^2 / 2)
    smp, c, r0 = _sample(trained, 16, c=c, cfg_scale=1.0, guide_scale=0.0, physics_fn=phys)
    _, _, r1 = _sample(trained, 16, c=c, cfg_scale=1.0, guide_scale=0.3, physics_fn=phys)
    t_min = torch.zeros(n.k)
    j0, j1 = smp.objective(r0.alpha, c, t_min).mean(), smp.objective(r1.alpha, c, t_min).mean()
    print(f"J unguided {j0:.3e} guided {j1:.3e}")
    assert j1 < 0.5 * j0


def test_mps_matches_cpu_and_real_pinn():
    if not torch.backends.mps.is_available():
        pytest.skip("no mps")
    torch.manual_seed(0)
    n, s = make_nurbs(), CosineSchedule()
    m = build(n, d=32, layers=2).eval().requires_grad_(False)
    norm = (torch.zeros(n.n_free), torch.ones(n.n_free))
    phys = lambda a, c: ((a ** 2).sum(1), a.sum(1))
    cond = torch.randn(8, 4)
    t_min = torch.zeros(n.k)
    out = {}
    for dev in ("cpu", "mps"):
        d = torch.device(dev)
        sm = GuidedSampler(build(n, d=32, layers=2).load_state_dict(m.state_dict()) and m.to(d), s, make_nurbs(dev), None,
                           tuple(x.to(d) for x in norm), physics_fn=phys)
        out[dev] = sm.sample(cond.to(d), t_min.to(d), n_steps=10, guide_scale=0.1, generator=torch.Generator().manual_seed(1)).alpha.cpu()
    assert torch.allclose(out["cpu"], out["mps"], atol=1e-3)

    # real (tiny, untrained) transformer PINN on MPS through surface_forces
    d = torch.device("mps")
    nm = make_nurbs("mps")
    pinn = GeometryCrossAttentionPINN(nm, d_model=32).to(d).requires_grad_(False)
    sm = GuidedSampler(m.to(d), s, nm, pinn, tuple(x.to(d) for x in norm))
    c = torch.stack([torch.zeros(16), torch.full((16,), 0.1), torch.full((16,), 4.6), torch.full((16,), 0.05)], 1).to(d)
    sm.sample(c, torch.zeros(n.k, device=d), n_steps=3)  # warm-up
    torch.mps.synchronize(); t0 = time.time()
    r = sm.sample(c, torch.zeros(n.k, device=d), n_steps=50, generator=torch.Generator().manual_seed(0))
    torch.mps.synchronize()
    print(f"\n50-step guided B=16 real PINN on MPS: {time.time() - t0:.2f}s")
    assert torch.isfinite(r.J_history[10:]).all() and torch.isfinite(r.alpha).all()
