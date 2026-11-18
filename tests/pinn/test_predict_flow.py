import torch

from aeroshape.pinn.embeddings import predict_flow, wall_distance
from tests.pinn import KovasznayModel, TinyMLP, make_nurbs, thin_alpha


def _setup(model, dtype=torch.float64):
    torch.manual_seed(0)
    b = make_nurbs(dtype=dtype)(thin_alpha(1).to(dtype))
    xy = torch.rand(1, 8, 2, dtype=dtype) * 2 - 0.5 + torch.tensor([0.0, 0.3], dtype=dtype)
    return b, xy, torch.zeros(1, 8, dtype=dtype), torch.zeros(1, 3, dtype=dtype)


def test_divergence_free():
    b, xy, a, c = _setup(None)
    m = TinyMLP(8).double()
    f = predict_flow(m, xy, a, c, b, derivs=1)
    assert (f.derivs["u_x"] + f.derivs["v_y"]).abs().max() < 1e-5


def test_sum_trick_equals_per_point_loop():
    b, xy, a, c = _setup(None)
    m = TinyMLP(8).double()
    f = predict_flow(m, xy, a, c, b, derivs=2)
    for i in range(8):
        fi = predict_flow(m, xy[:, i:i + 1].clone(), a, c, b, derivs=2)
        for k in ("u_x", "u_y", "u_xx", "u_yy", "v_x", "v_y", "v_xx", "v_yy", "p_x", "p_y"):
            torch.testing.assert_close(f.derivs[k][:, i], fi.derivs[k][:, 0], rtol=1e-9, atol=1e-9)


def test_kovasznay_velocity_and_wall_distance():
    b, xy, a, c = _setup(None)
    f = predict_flow(KovasznayModel(), xy, a, c, b, derivs=0)
    from tests.pinn import kovasznay
    _, u, v, p = kovasznay(xy[..., 0], xy[..., 1])
    torch.testing.assert_close(f.u, u); torch.testing.assert_close(f.v, v); torch.testing.assert_close(f.p, p)
    # a point at the leading edge region, and one far away
    d = wall_distance(torch.tensor([[[0.5, 0.0], [0.5, 1.0]]], dtype=torch.float64), b)
    assert d[0, 0] < 0.07 and abs(d[0, 1] - 0.94) < 0.07 and not d.requires_grad
