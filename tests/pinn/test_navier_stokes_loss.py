import time

import pytest
import torch

from aeroshape.pinn.embeddings import predict_flow
from aeroshape.pinn.navier_stokes_loss import hybrid_loss, ns_residuals, pde_weight
from aeroshape.pinn.sampling import sample_batch
from tests.pinn import RE, KovasznayModel, TinyMLP, make_cfg, make_nurbs, synthetic_anchors, thin_alpha

KEYS = {"total", "pde", "wall", "edge", "pin", "data", "force", "force_data", "grad_data", "w_pde"}


def _batch(device="cpu", seed=0, anchors=None, cfg=None):
    cfg = cfg or make_cfg()
    nurbs = make_nurbs(device)
    g = torch.Generator().manual_seed(seed)
    return cfg, nurbs, sample_batch(anchors or synthetic_anchors(), nurbs, cfg, g, device=device)


def test_kovasznay_residual():
    torch.manual_seed(0)
    xy = torch.rand(1, 500, 2, dtype=torch.float64) * torch.tensor([5.5, 3.0], dtype=torch.float64) \
        + torch.tensor([-1.5, -1.5], dtype=torch.float64)
    b = make_nurbs(dtype=torch.float64)(thin_alpha(1).double())
    f = predict_flow(KovasznayModel(), xy, torch.zeros(1, 8, dtype=torch.float64), torch.zeros(1, 3, dtype=torch.float64), b, 2)
    rx, ry = ns_residuals(f, torch.tensor([[1 / RE]], dtype=torch.float64))
    m = max(rx.abs().max().item(), ry.abs().max().item())
    print("kovasznay max residual", m)
    assert m < 1e-4


def test_pde_weight_curriculum():
    cfg = make_cfg("train.warmup_data_only=10", "train.pde_ramp=20", "loss.w_pde=2.0")
    assert pde_weight(0, cfg) == 0 and pde_weight(10, cfg) == 0
    assert pde_weight(20, cfg) == pytest.approx(1.0) and pde_weight(30, cfg) == 2.0 and pde_weight(99, cfg) == 2.0


def test_warmup_loss_has_no_pde_and_keys():
    cfg, nurbs, batch = _batch()
    m = TinyMLP(8)
    _, d = hybrid_loss(m, nurbs, batch, cfg, step=0, log=True)
    assert set(d) == KEYS and all(isinstance(v, float) for v in d.values())
    assert d["w_pde"] == 0 and d["pde"] == 0
    _, d2 = hybrid_loss(m, nurbs, batch, cfg, step=10_000, log=True)
    assert d2["pde"] > 0 and d2["w_pde"] == cfg.loss.w_pde
    assert hybrid_loss(m, nurbs, batch, cfg, step=0)[1] == {}


def test_v2_force_term():
    cfg = make_cfg("pinn.version=v2")
    _, nurbs, batch = _batch(cfg=cfg)
    _, d = hybrid_loss(TinyMLP(8, v2=True), nurbs, batch, cfg, step=10_000, log=True)
    assert set(d) == KEYS and d["force"] > 0


def _train(device, steps, step0, lr=1e-2):
    cfg = make_cfg()
    nurbs, anchors = make_nurbs(device), synthetic_anchors()
    torch.manual_seed(0)
    m = TinyMLP(8).to(device)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    g = torch.Generator().manual_seed(1)
    losses = []
    for i in range(steps):
        batch = sample_batch(anchors, nurbs, cfg, g, device=device)
        loss, _ = hybrid_loss(m, nurbs, batch, cfg, step=step0 + i)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.detach())
    return torch.stack(losses).cpu()


def test_loss_decreases():
    l = _train("cpu", 50, 0)
    assert l[-5:].mean() < 0.5 * l[:5].mean()


def test_no_nan_on_mps_with_pde():
    if not torch.backends.mps.is_available():
        pytest.skip("no MPS")
    l = _train("mps", 100, 2000)
    assert torch.isfinite(l).all()


def test_mps_step_time():
    if not torch.backends.mps.is_available():
        pytest.skip("no MPS")
    cfg, nurbs, batch = _batch("mps")
    m = TinyMLP(8).to("mps")
    for i in range(6):
        if i == 1:
            torch.mps.synchronize()
            t = time.perf_counter()
        hybrid_loss(m, nurbs, batch, cfg, step=10_000)[0].backward()
    torch.mps.synchronize()
    print("mps hybrid_loss fwd+bwd s/step", (time.perf_counter() - t) / 5)
