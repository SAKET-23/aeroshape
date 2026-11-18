import time

import pytest
import torch

from aeroshape.pinn.embeddings import predict_flow, wall_distance
from aeroshape.pinn.mlp_pinn import MLPPINN
from aeroshape.pinn.navier_stokes_loss import hybrid_loss, ns_residuals
from aeroshape.pinn.sampling import sample_batch
from tests.pinn import kovasznay, make_cfg, make_nurbs, synthetic_anchors, thin_alpha

DEVICES = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])


def _inputs(B, N, D=8, dtype=torch.float32, device="cpu"):
    g = torch.Generator().manual_seed(0)
    xy = (torch.rand(B, N, 2, generator=g) * 3 - 1.5).to(device, dtype)
    return xy, torch.randn(B, D, generator=g).to(device, dtype), torch.randn(B, 3, generator=g).to(device, dtype), \
        torch.rand(B, N, generator=g).to(device, dtype)


@pytest.mark.parametrize("B,N", [(1, 1), (1, 100), (4, 1), (4, 100)])
@pytest.mark.parametrize("v2", [False, True])
def test_shapes(B, N, v2):
    out = MLPPINN(8, v2=v2)(*_inputs(B, N))
    assert out.psi.shape == out.p.shape == (B, N)
    assert (out.fx is not None) == v2 and (not v2 or out.fy.shape == (B, N))
    assert out.attn is None


def test_param_count_and_cambered():
    print("params", sum(p.numel() for p in MLPPINN(8).parameters()))
    assert MLPPINN(16)(*_inputs(2, 5, D=16)).psi.shape == (2, 5)


def test_divergence_free():
    torch.manual_seed(0)
    m = MLPPINN(8)
    xy, a, c, _ = _inputs(2, 200)
    b = make_nurbs()(thin_alpha(2))
    f = predict_flow(m, xy, a, c, b, derivs=1)
    assert (f.derivs["u_x"] + f.derivs["v_y"]).abs().max() < 1e-5


def test_point_independence():
    torch.manual_seed(0)
    m = MLPPINN(8)
    xy, a, c, d = _inputs(2, 8)
    perm = torch.randperm(8)
    o, op = m(xy, a, c, d), m(xy[:, perm], a, c, d[:, perm])
    assert torch.allclose(o.psi[:, perm], op.psi, atol=1e-5) and torch.allclose(o.p[:, perm], op.p, atol=1e-5)
    # sum-trick gradient equals per-point gradient
    x1 = xy.clone().requires_grad_(True)
    g = torch.autograd.grad(m(x1, a, c, d).psi.sum(), x1)[0]
    for i in range(8):
        xi = xy[:, i:i + 1].clone().requires_grad_(True)
        gi = torch.autograd.grad(m(xi, a, c, d[:, i:i + 1]).psi.sum(), xi)[0]
        assert torch.allclose(g[:, i:i + 1], gi, atol=1e-5)


@pytest.mark.parametrize("device", DEVICES)
def test_third_derivatives_finite(device):
    m = MLPPINN(8).to(device)
    xy, a, c, _ = _inputs(2, 50, device=device)
    b = make_nurbs(device)(thin_alpha(2).to(device))
    f = predict_flow(m, xy, a, c, b, derivs=2)
    for t in (f.u, f.v, f.p, *f.derivs.values()):
        assert torch.isfinite(t).all()


@pytest.mark.parametrize("device", DEVICES)
def test_hybrid_loss_backward(device):
    cfg = make_cfg()
    nurbs = make_nurbs(device)
    batch = sample_batch(synthetic_anchors(), nurbs, cfg, torch.Generator().manual_seed(0), device=device)
    m = MLPPINN(8).to(device)
    ts = []
    for _ in range(2):  # second pass is the warm timing
        t0 = time.time()
        loss, _ = hybrid_loss(m, nurbs, batch, cfg, step=10_000)
        m.zero_grad()
        loss.backward()
        loss.item()
        ts.append(time.time() - t0)
    print(device, "hybrid_loss fwd+bwd s", ts)
    assert torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())


@pytest.mark.slow
def test_fit_kovasznay():
    re, dt = 20.0, torch.float64
    torch.manual_seed(0)
    m = MLPPINN(8, scale=0.25, psi_scale=1.0).to(dt)
    b = make_nurbs(dtype=dt)(thin_alpha(1).to(dt))  # only feeds d_wall
    a, c = torch.zeros(1, 8, dtype=dt), torch.zeros(1, 3, dtype=dt)
    lo, hi = torch.tensor([-0.5, -0.5], dtype=dt), torch.tensor([1.0, 1.5], dtype=dt)
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 2000, 1e-5)
    nu = torch.tensor([[1 / re]], dtype=dt)
    t0 = time.time()
    for step in range(2000):
        xd = (lo + (hi - lo) * torch.rand(1, 256, 2, dtype=dt))
        fd = predict_flow(m, xd, a, c, b)
        _, u, v, p = kovasznay(xd[..., 0], xd[..., 1], re)
        xc = (lo + (hi - lo) * torch.rand(1, 256, 2, dtype=dt))
        rx, ry = ns_residuals(predict_flow(m, xc, a, c, b, derivs=2), nu)
        loss = 100 * ((fd.u - u) ** 2 + (fd.v - v) ** 2 + (fd.p - p) ** 2).mean() + (rx ** 2 + ry ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    xt = lo + (hi - lo) * torch.rand(1, 2000, 2, dtype=dt)
    f = predict_flow(m, xt, a, c, b)
    _, u, v, p = kovasznay(xt[..., 0], xt[..., 1], re)
    err = [((x - y).norm() / y.norm()).item() for x, y in ((f.u, u), (f.v, v), (f.p - f.p.mean(), p - p.mean()))]
    print("kovasznay rel L2 (u,v,p)", err, "time", time.time() - t0)
    assert max(err) < 1e-2
