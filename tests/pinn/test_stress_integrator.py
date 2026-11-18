"""WP14: differentiable surface forces (Plan/04_models/stress_integrator.md)."""
import math
import time

import pytest
import torch
from torch import nn

from aeroshape.core.types import Boundary, Condition, PINNOutput
from aeroshape.geometry.nurbs import NURBSAirfoil
from aeroshape.pinn.mlp_pinn import MLPPINN
from aeroshape.pinn.stress_integrator import pinn_value_and_grad, surface_cp, surface_forces
from aeroshape.pinn.transformer_pinn import GeometryCrossAttentionPINN
from tests.pinn import thin_alpha

F64 = torch.float64


class Circle(nn.Module):
    """Stand-in 'nurbs': circle of radius r about (cx, cy), CCW, outward normals, closed trapezoid ds."""

    def __init__(self, r=0.5, c=(0.25, 0.0), n=401):
        super().__init__()
        t = torch.linspace(0, 2 * math.pi, n, dtype=F64)
        ds = torch.full((n,), 2 * math.pi * r / (n - 1), dtype=F64)
        ds[0] = ds[-1] = ds[0] / 2
        self.b = Boundary(x=(c[0] + r * t.cos())[None], y=(c[1] + r * t.sin())[None], nx=t.cos()[None], ny=t.sin()[None],
                          ds=ds[None])

    def forward(self, alpha):
        return self.b


class Plate(nn.Module):
    """Open flat plate y = 0 traversed x = 1 -> 0, so the edge normal (dy, -dx) is +y."""

    def forward(self, alpha):
        x = torch.linspace(1, 0, 101, dtype=F64)[None]
        ds = torch.full((1, 101), 0.01, dtype=F64)
        ds[:, [0, -1]] = 0.005
        return Boundary(x=x, y=0 * x, nx=0 * x, ny=1 + 0 * x, ds=ds)


class Analytic(nn.Module):
    """psi, p given as functions of (x, y); psi gets a 0*(x^2+y^2) term so autograd paths stay connected."""

    def __init__(self, psi, p):
        super().__init__()
        self.psi, self.p = psi, p

    def forward(self, xy, alpha, cond, d_wall):
        x, y = xy[..., 0], xy[..., 1]
        return PINNOutput(psi=self.psi(x, y) + 0 * (x * x + y * y), p=self.p(x, y) + 0 * (x + y))


def _cond(re=100.0, aoa=0.0, B=1):
    return Condition(re, aoa).as_tensor(dtype=F64)[None].repeat(B, 1)


A1 = torch.zeros(1, 8, dtype=F64)


def test_constant_pressure_zero_force():
    m = Analytic(lambda x, y: 0 * x, lambda x, y: 3.7 + 0 * x)
    nurbs = NURBSAirfoil(k=8, M=40, dtype=F64)
    cd, cl = surface_forces(m, Circle(), A1, _cond(), 0.0)  # exactly closed loop
    assert cd.abs().max() < 1e-6 and cl.abs().max() < 1e-6
    cd, cl = surface_forces(m, nurbs, thin_alpha(2).double(), _cond(B=2), 5.0)  # edge-based: closes to machine precision
    assert cd.abs().max() < 1e-6 and cl.abs().max() < 1e-6


def test_circle_pressure_analytic():
    cx, r = 0.25, 0.5
    m = Analytic(lambda x, y: 0 * x, lambda x, y: -(x - cx) / r)  # p = -cos(theta) on the circle
    cd, cl = surface_forces(m, Circle(r), A1, _cond(), 0.0)
    # Fx = oint cos^2 r dtheta = pi/2 ; CD = 2 Fx
    assert abs(cd.item() - math.pi) < 2e-4 and abs(cl.item()) < 2e-4


def test_pure_shear_plate():
    g, re = 0.7, 50.0
    m = Analytic(lambda x, y: g * y * y / 2, lambda x, y: 0 * x)
    cd, cl = surface_forces(m, Plate(), A1, _cond(re=re), 0.0)
    assert abs(cd.item() - 2 * g / re) < 1e-9 and abs(cl.item()) < 1e-9


def _mlp(D=8):
    torch.manual_seed(0)
    m = MLPPINN(D, hidden=32, layers=3, fourier=16).double()
    with torch.no_grad():
        m.out.weight.normal_(0, 0.3)
    return m.requires_grad_(False)


def test_gradient_vs_central_difference():
    m, nurbs = _mlp(), NURBSAirfoil(k=8, M=40, dtype=F64)
    a = thin_alpha(1).double()
    cond = _cond(aoa=3.0)
    r = pinn_value_and_grad(m, nurbs, a, cond)
    assert r.dcd_dalpha.shape == (8,) and r.dcd_dalpha.abs().max() > 1e-6
    for fn, g in ((lambda x: surface_forces(m, nurbs, x, cond, 3.0)[0], r.dcd_dalpha),
                  (lambda x: surface_forces(m, nurbs, x, cond, 3.0)[1], r.dcl_dalpha)):
        d = torch.randn(1, 8, dtype=F64) * nurbs.mask
        e = 1e-6
        fd = (fn(a + e * d) - fn(a - e * d)) / (2 * e)
        an = (g * d[0]).sum()
        assert abs(fd.item() - an.item()) / abs(an.item()) < 1e-5


def test_batch_equals_separate_calls():
    m, nurbs = _mlp(), NURBSAirfoil(k=8, M=40, dtype=F64)
    a = thin_alpha(3).double()
    cond = torch.cat([_cond(80 + 40 * i, 2.0 * i) for i in range(3)])
    cd, cl = surface_forces(m, nurbs, a, cond, None)  # AoA from cond
    for i in range(3):
        c1, l1 = surface_forces(m, nurbs, a[i:i + 1], cond[i:i + 1], 2.0 * i)
        torch.testing.assert_close(cd[i], c1[0], rtol=1e-10, atol=1e-12)
        torch.testing.assert_close(cl[i], l1[0], rtol=1e-10, atol=1e-12)
    assert surface_cp(m, nurbs, a, cond).shape == (3, 79)


def test_rotation_antisymmetry():
    # mirror-symmetric fields: psi odd in y, p even in y; body rotated by +-theta is a y-mirror image
    m = Analytic(lambda x, y: y * (1 + 0.3 * torch.cos(x) + 0.2 * y * y), lambda x, y: torch.cos(2 * x) + 0.5 * y * y)
    nurbs = NURBSAirfoil(k=8, M=40, dtype=F64)
    a = thin_alpha(1).double()
    cd1, cl1 = surface_forces(m, nurbs, a, _cond(aoa=7.0), None)
    cd2, cl2 = surface_forces(m, nurbs, a, _cond(aoa=-7.0), None)
    assert abs(cl1.item()) > 1e-3
    assert abs(cl1.item() + cl2.item()) < 1e-9 and abs(cd1.item() - cd2.item()) < 1e-9


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="no MPS")
@pytest.mark.parametrize("d_model,M", [(64, 40), (128, 160)])
def test_mps_transformer_value_and_grad(d_model, M, capsys):
    dev = "mps"
    nurbs = NURBSAirfoil(k=8, M=M, device=dev)
    m = GeometryCrossAttentionPINN(nurbs, d_model=d_model).to(dev).requires_grad_(False)
    a = thin_alpha(1).to(dev)
    cond = Condition(100.0, 2.0).as_tensor(device=dev)[None]
    pinn_value_and_grad(m, nurbs, a, cond)  # warm-up
    t = [pinn_value_and_grad(m, nurbs, a, cond) for _ in range(5)]
    r = t[-1]
    for v in (r.cd, r.cl, r.dcd_dalpha, r.dcl_dalpha):
        assert torch.isfinite(v).all()
    with capsys.disabled():
        print(f"\nMPS Transformer d_model={d_model} M={M}: pinn_value_and_grad wall_s = "
              f"{[round(x.wall_s * 1e3, 2) for x in t]} ms")
