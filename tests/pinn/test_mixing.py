"""v2 data mixing: layout conversion, per-kind loss weights, symmetric-gradient projection."""
import pytest
import torch

from aeroshape.core.io import check_anchors
from aeroshape.geometry.nurbs import NURBSAirfoil
from aeroshape.pinn.mixing import mix_anchors, sym_grad, to_cambered
from aeroshape.pinn.navier_stokes_loss import hybrid_loss
from aeroshape.pinn.sampling import sample_batch
from tests.pinn import TinyMLP, make_cfg, make_nurbs, synthetic_anchors, thin_alpha


def test_to_cambered_gives_same_surface():
    sym, cam = make_nurbs(dtype=torch.float64), NURBSAirfoil(k=8, M=40, mode="cambered", dtype=torch.float64)
    a = thin_alpha(3).double()
    for u, v in zip(sym.surface_y(a), cam.surface_y(to_cambered(a))):
        assert torch.allclose(u, v, atol=1e-12)
    bs, bc = sym(a), cam(to_cambered(a))
    for f in ("x", "y", "nx", "ny", "ds"):
        assert torch.allclose(getattr(bs, f), getattr(bc, f), atol=1e-12)


def test_sym_grad_is_chain_rule():
    w = torch.randn(16, dtype=torch.float64)
    f = lambda a: (torch.sin(a) * w).sum()  # any function of the cambered alpha
    a = thin_alpha(1).double()
    s = a.clone().requires_grad_(True)
    g_ref = torch.autograd.grad(f(to_cambered(s)), s)[0]
    c = to_cambered(a).requires_grad_(True)
    assert torch.allclose(sym_grad(torch.autograd.grad(f(c), c)[0]), g_ref)


def _tavg(S=2):
    A = synthetic_anchors(S=S)
    A.update(mode="cambered", alpha=to_cambered(A["alpha"]), re=torch.full((S,), 1000.0), meta={"runs": [{"id": f"t{i}"} for i in range(S)]})
    A["split"] = torch.tensor([0, 1], dtype=torch.int8)[:S]
    return A


def test_mix_anchors_contract_kinds_and_holdout():
    st = synthetic_anchors(S=3)
    st["split"] = torch.tensor([0, 1, 0], dtype=torch.int8)
    st["meta"] = {"runs": [{"id": f"s{i}"} for i in range(3)]}
    m = mix_anchors(st, _tavg())
    assert not check_anchors(m) and m["alpha"].shape == (4, 16)
    assert m["kind"].tolist() == [0, 0, 1, 1] and m["split"].tolist() == [0, 0, 0, 1]  # steady holdout dropped
    assert [r["id"] for r in m["meta"]["runs"]] == ["s0", "s2", "t0", "t1"]


@pytest.mark.parametrize("kind,pde_x,data_x", [(1, 0.1, 3.0), (0, 1.0, 1.0)])
def test_per_kind_weights(kind, pde_x, data_x):
    cfg = make_cfg("pinn.version=v2", "train.shapes_per_step=2")
    nurbs = make_nurbs()
    torch.manual_seed(0)
    m = TinyMLP(8, v2=True)
    out = {}
    for k in (0, kind):
        A = synthetic_anchors(S=2)
        A["kind"] = torch.full((2,), k, dtype=torch.int8)
        b = sample_batch(A, nurbs, cfg, torch.Generator().manual_seed(1))
        out[k] = hybrid_loss(m, nurbs, b, cfg, step=10_000, log=True)[1]
    assert out[kind]["pde"] == pytest.approx(pde_x * out[0]["pde"], rel=1e-5)
    assert out[kind]["data"] == pytest.approx(data_x * out[0]["data"], rel=1e-5)
