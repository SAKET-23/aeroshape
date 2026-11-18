import math
import time
from dataclasses import replace
from functools import lru_cache

import pytest
import torch

from aeroshape.core.config import load_config
from aeroshape.core.types import Boundary, Condition, ForceHistory, LinkSet
from aeroshape.geometry.ibb_grid_mapper import map_to_grid
from aeroshape.lbm import d2q9_solver as S
from aeroshape.lbm.aero_forces import (coefficients, force_fn, momentum_exchange, period_average,
                                       surface_pressure_coefficient)
from aeroshape.lbm.lattice import make_params
from tests.lbm.test_d2q9_solver import empty_geom

CFG = load_config(None, "dev", [])
F64 = torch.float64


def circle(r=0.5, n=4096):
    th = (torch.arange(n + 1, dtype=F64) + 0.5) * (2 * math.pi / n)
    x, y = (r * torch.cos(th))[None], (r * torch.sin(th))[None]
    z = torch.zeros_like(x)
    return Boundary(x, y, z, z, z)


def setup(N):
    p = make_params(CFG, Condition(20, 0), N, dtype=F64)
    return p, map_to_grid(circle(), p)


def test_no_body_zero_force():
    p = make_params(CFG, Condition(20, 0), 10, dtype=F64)
    g = empty_geom(p.Nx, p.Ny)
    f = S.init_state(p, g, F64)
    fn, fc = S.step(f, g, p)
    assert momentum_exchange(fc, fn, g.links, p.Nx).abs().max() == 0
    assert force_fn(fc, fn, g, p)[0] == 0


def test_uniform_flow_symmetric_one_step():
    p, g = setup(10)
    fn, fc = S.step(S.init_state(p, g, F64), g, p)
    F = momentum_exchange(fc, fn, g.links, p.Nx)
    assert F.shape == (2,) and F[1].abs() < 1e-10 and F[0] > 0


def test_coefficients_and_force_fn_signature():
    p, g = setup(10)
    fn, fc = S.step(S.init_state(p, g, F64), g, p)
    F = momentum_exchange(fc, fn, g.links, p.Nx)
    cd, cl = force_fn(fc, fn, g, p)
    assert cd.ndim == 0 and cl.ndim == 0
    assert torch.allclose(cd, 2 * F[0] / (p.u_lb**2 * p.N)) and torch.allclose(cl, 2 * F[1] / (p.u_lb**2 * p.N))
    assert torch.equal(torch.stack(coefficients(F, p)), torch.stack([cd, cl]))


def test_period_average():
    step = torch.arange(0, 1000, 10)
    t = step.double()
    cl = torch.sin(2 * math.pi * t / 100)
    cd, clm = period_average(ForceHistory(step, 2 + 0.1 * cl, cl), 100)
    assert abs(cd - 2) < 1e-9 and abs(clm) < 1e-9


def test_gradient_wrt_q_matches_central_difference():
    p, g = setup(10)
    f = S.init_state(p, g, F64)
    for _ in range(20):
        f, _ = S.step(f, g, p)
    f = f + 1e-3 * torch.rand_like(f)
    v = torch.randn_like(g.links.q)

    def cd_of(q):
        gq = replace(g, links=replace(g.links, q=q))
        fn, fc = S.step(f, gq, p)
        return force_fn(fc, fn, gq, p)[0]

    q = g.links.q.clone().requires_grad_(True)
    (gr,) = torch.autograd.grad(cd_of(q), q)
    eps = 1e-6
    fd = (cd_of(g.links.q + eps * v) - cd_of(g.links.q - eps * v)) / (2 * eps)
    assert abs((gr * v).sum() - fd) < 1e-6 * max(1.0, abs(fd))


def test_surface_cp_uniform_density():
    p, g = setup(10)
    f = S.init_state(p, g, F64)
    cp = surface_pressure_coefficient(S._state(f, 0), g, p, circle())
    assert cp.shape == (4097,) and cp.abs().max() < 1e-9


@lru_cache(maxsize=None)
def steady_cd(N):
    p, g = setup(N)
    t0 = time.perf_counter()
    r = S.run(S.init_state(p, g, F64), g, p, force_fn=force_fn, record_every=20)
    print(f"\nCircle Re20 N={N}: CD={r.cd.item():.4f} CL={r.cl.item():.2e} converged={r.converged} "
          f"wall={time.perf_counter() - t0:.1f}s device=cpu")
    return r


@pytest.mark.slow
def test_circle_re20_smoke_cd_range():
    r = steady_cd(10)
    assert math.isfinite(r.cd.item()) and 1.5 < r.cd.item() < 3.5


@pytest.mark.slow
def test_circle_symmetric_cl_zero():
    assert abs(steady_cd(20).cl.item()) < 1e-3


@pytest.mark.slow
def test_circle_cd_convergence_trend():
    a, b = steady_cd(10).cd.item(), steady_cd(20).cd.item()
    assert abs(a - b) / b < 0.05
