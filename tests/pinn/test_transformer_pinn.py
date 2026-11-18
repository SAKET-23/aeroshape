import time

import pytest
import torch

from aeroshape.geometry.nurbs import NURBSAirfoil
from aeroshape.pinn.embeddings import predict_flow
from aeroshape.pinn.navier_stokes_loss import hybrid_loss
from aeroshape.pinn.sampling import sample_batch
from aeroshape.pinn.transformer_pinn import GeometryCrossAttentionPINN as GCA
from tests.pinn import kovasznay, make_cfg, make_nurbs, synthetic_anchors, thin_alpha

DEVICES = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
MODES = {8: "symmetric", 16: "cambered"}


def _nurbs(D, device="cpu"):
    return NURBSAirfoil(k=8, M=40, mode=MODES[D], device=device)


def _inputs(B, N, D=8, device="cpu"):
    g = torch.Generator().manual_seed(0)
    return ((torch.rand(B, N, 2, generator=g) * 3 - 1.5).to(device), (0.05 * torch.randn(B, D, generator=g)).to(device),
            torch.randn(B, 3, generator=g).to(device), torch.rand(B, N, generator=g).to(device))


@pytest.mark.parametrize("D", [8, 16])
@pytest.mark.parametrize("B,N", [(1, 1), (1, 100), (4, 1), (4, 100)])
def test_shapes_and_attn(D, B, N):
    m = GCA(_nurbs(D), d_model=32)
    xy, a, c, d = _inputs(B, N, D)
    out = m(xy, a, c, d)
    assert out.psi.shape == out.p.shape == (B, N) and out.attn is None and out.fx is None
    out = m(xy, a, c, d, return_attn=True)
    assert out.attn.shape == (B, 4, N, D + 1)
    assert torch.allclose(out.attn.sum(-1), torch.ones(B, 4, N), atol=1e-5)
    assert GCA(_nurbs(D), d_model=32, v2=True)(xy, a, c, d).fy.shape == (B, N)


def test_tokens_match_nurbs():
    for D in (8, 16):
        n, m = _nurbs(D), GCA(_nurbs(D), d_model=32)
        a = _inputs(2, 1, D)[1]
        ref = n.control_tokens(a)[:, :D]  # symmetric: upper only; cambered: all 2k
        got = m.tokens(a)
        assert torch.allclose(got[..., 0], ref[..., 0]) and torch.allclose(got[..., 2], ref[..., 2])
        assert torch.allclose(got[..., 1], ref[..., 1])  # InputNorm is identity until fit


def test_param_counts():
    for d, ns, nc in ((64, 1, 1), (128, 2, 2)):
        for D in (8, 16):
            kw = dict(d_model=d, n_self=ns, n_cross=nc, ffn_mult=2 if d == 64 else 4)
            print("params d", d, "D", D, sum(p.numel() for p in GCA(_nurbs(D), **kw).parameters()))


def test_divergence_free():
    torch.manual_seed(0)
    m = GCA(_nurbs(8), d_model=32)
    xy, a, c, _ = _inputs(2, 200)
    f = predict_flow(m, xy, a, c, make_nurbs()(thin_alpha(2)), derivs=1)
    assert (f.derivs["u_x"] + f.derivs["v_y"]).abs().max() < 1e-5


def test_point_independence():
    torch.manual_seed(0)
    m = GCA(_nurbs(8), d_model=32)
    xy, a, c, d = _inputs(2, 8)
    perm = torch.randperm(8)
    o, op = m(xy, a, c, d), m(xy[:, perm], a, c, d[:, perm])
    assert torch.allclose(o.psi[:, perm], op.psi, atol=1e-5) and torch.allclose(o.p[:, perm], op.p, atol=1e-5)
    x1 = xy.clone().requires_grad_(True)
    g = torch.autograd.grad(m(x1, a, c, d).psi.sum(), x1)[0]
    for i in range(8):
        xi = xy[:, i:i + 1].clone().requires_grad_(True)
        gi = torch.autograd.grad(m(xi, a, c, d[:, i:i + 1]).psi.sum(), xi)[0]
        assert torch.allclose(g[:, i:i + 1], gi, atol=1e-5)


def test_geometry_caching():
    m = GCA(_nurbs(8), d_model=32)
    xy, a, c, d = _inputs(2, 20)
    G = m.encode_geometry(a, c)
    assert G.shape == (2, 9, 32)
    assert torch.equal(m(xy, a, c, d).psi, m(xy, a, c, d, G=G).psi)


def test_condition_sensitivity():
    torch.manual_seed(0)
    m = GCA(_nurbs(8), d_model=32)
    for p in m.head[-1].parameters():  # un-shrink the head so the effect is well above float noise
        p.data.normal_(0, 0.5)
    xy, a, c, d = _inputs(2, 20)
    assert (m(xy, a, c, d).psi - m(xy, a, c + 1.0, d).psi).abs().max() > 1e-4


def _third(m, xy, a, c, d, device):
    x = xy.clone().requires_grad_(True)
    g = torch.autograd.grad(m(x, a, c, d).psi.sum(), x, create_graph=True)[0]
    g2 = torch.autograd.grad(g[..., 0].sum(), x, create_graph=True)[0]
    return torch.autograd.grad(g2[..., 1].sum(), x)[0].cpu()


@pytest.mark.skipif("mps" not in DEVICES, reason="no MPS")
@pytest.mark.parametrize("impl", ["sdpa", "explicit"])
def test_mps_third_derivatives(impl):
    torch.manual_seed(0)
    cpu = GCA(_nurbs(8), d_model=32, scale=1.0)  # CPU reference is always explicit (CPU SDPA has no double-backward)
    gpu = GCA(_nurbs(8, "mps"), d_model=32, scale=1.0, attn_impl=impl).to("mps")
    gpu.load_state_dict(cpu.state_dict())
    xy, a, c, d = _inputs(2, 30)
    ref = _third(cpu, xy, a, c, d, "cpu")
    got = _third(gpu, *(t.to("mps") for t in (xy, a, c, d)), "mps")
    assert (got - ref).norm() / ref.norm() < 1e-3


@pytest.mark.parametrize("device", DEVICES)
def test_hybrid_loss_backward(device):
    cfg = make_cfg()
    nurbs = make_nurbs(device)
    batch = sample_batch(synthetic_anchors(), nurbs, cfg, torch.Generator().manual_seed(0), device=device)
    m = GCA(_nurbs(8), d_model=64, n_self=1, n_cross=1, ffn_mult=2).to(device)
    ts = []
    for _ in range(2):
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
@pytest.mark.skipif("mps" not in DEVICES, reason="no MPS")
def test_overfit_one_shape():
    """Anchor field: Kovasznay flow (Re=20, exact NS) on the window [-0.5,1]x[-0.5,1.5], data loss only, one shape."""
    dev, re = "mps", 20.0
    torch.manual_seed(0)
    m = GCA(_nurbs(8, dev), scale=0.25, psi_scale=1.0).to(dev)
    b = make_nurbs(dev)(thin_alpha(1).to(dev))
    a, c = thin_alpha(1).to(dev), torch.zeros(1, 3, device=dev)
    lo, hi = torch.tensor([-0.5, -0.5], device=dev), torch.tensor([1.0, 1.5], device=dev)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 2000, 1e-5)
    t0 = time.time()
    for _ in range(2000):
        xd = lo + (hi - lo) * torch.rand(1, 512, 2, device=dev)
        f = predict_flow(m, xd, a, c, b)
        _, u, v, p = kovasznay(xd[..., 0], xd[..., 1], re)
        loss = ((f.u - u) ** 2 + (f.v - v) ** 2 + (f.p - p) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    xt = lo + (hi - lo) * torch.rand(1, 4000, 2, device=dev)
    f = predict_flow(m, xt, a, c, b)
    _, u, v, _ = kovasznay(xt[..., 0], xt[..., 1], re)
    err = (((f.u - u) ** 2 + (f.v - v) ** 2).sum() / (u ** 2 + v ** 2).sum()).sqrt().item()
    print("overfit velocity rel L2", err, "time", time.time() - t0)
    assert err < 0.02
