import pytest
import torch

from aeroshape.geometry import reference_shapes as rs
from aeroshape.geometry.nurbs import NURBSAirfoil

K_CAMBERED = 16  # G2 default for cambered


def make(k, mode):
    return NURBSAirfoil(k=k, mode=mode, dtype=torch.float64)


def max_err(n, alpha, name):
    c = rs.reference_coords(name)
    x, yu, yl = n.surface_y(alpha)
    import numpy as np
    eu = np.abs(np.interp(x.numpy(), c["x_up"], c["y_up"]) - yu[0].numpy()).max()
    el = np.abs(np.interp(x.numpy(), c["x_lo"], c["y_lo"]) - yl[0].numpy()).max()
    return max(eu, el)


def test_naca0012_fit():
    n = make(8, "symmetric")
    assert max_err(n, rs.naca4_alpha(n, "0012"), "naca0012") < 2e-3


@pytest.mark.parametrize("k", [8, K_CAMBERED])  # 8 = smallest passing k (fit report); 16 = G2 cambered default
def test_rae2822_fit(k):
    n = make(k, "cambered")
    assert max_err(n, rs.rae2822_alpha(n), "rae2822") < 2e-3


def test_naca_formula_thickness():
    c = rs.reference_coords("naca0012")
    import numpy as np
    t = np.interp(0.30, c["x_up"], c["y_up"]) - np.interp(0.30, c["x_lo"], c["y_lo"])
    assert abs(t - 0.12) < 1e-3


def test_invalid_rejected():
    n = make(8, "cambered")
    a = rs.naca4_alpha(n, "0012")
    flipped = torch.cat([a[:, n.k:], a[:, :n.k]], 1)  # lower above upper
    assert rs.is_valid(n, a).all() and not rs.is_valid(n, flipped).any()


@pytest.mark.parametrize("family", ["dataset", "diffusion"])
@pytest.mark.parametrize("mode", ["symmetric", "cambered"])
def test_lhs_reproducible_and_valid(family, mode):
    n = make(8 if mode == "symmetric" else K_CAMBERED, mode)
    a = rs.lhs_shapes(n, 12, seed=3, family=family)
    assert a.shape == (12, n.D) and a.dtype == torch.float64
    assert torch.equal(a, rs.lhs_shapes(n, 12, seed=3, family=family))
    assert not torch.equal(a, rs.lhs_shapes(n, 12, seed=4, family=family))
    assert rs.is_valid(n, a).all()


def test_shape_distance_and_perturb():
    n = make(8, "symmetric")
    a = rs.naca4_alpha(n, "0012")
    assert rs.shape_distance(n, a, a).item() == 0
    p = rs.perturb(a, 5, 0.01, 0.3, 0, n)
    assert p.shape == (5, n.D) and (p[:, 0] == 0).all() and (p[:, -1] == 0).all()
    assert 0 < rs.shape_distance(n, a, p).max() < 0.05
