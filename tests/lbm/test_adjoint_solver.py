"""Validation ladder for the discrete adjoint (Plan/03_lbm/adjoint_solver.md); CPU float64, 100 x 50 grid, N = 10, Re 20."""
import time
from dataclasses import replace
from functools import lru_cache

import pytest
import torch

from aeroshape.core.config import load_config
from aeroshape.core.types import Condition
from aeroshape.geometry.ibb_grid_mapper import map_to_grid
from aeroshape.geometry.nurbs import NURBSAirfoil
from aeroshape.geometry.reference_shapes import circle_alpha
from aeroshape.lbm import d2q9_solver as S
from aeroshape.lbm import adjoint_solver as AS
from aeroshape.lbm.adjoint_solver import (adjoint_step_hand, adjoint_step_vjp, fd_gradient, lbm_value, lbm_value_and_grad,
                                          solve_adjoint)
from aeroshape.lbm.aero_forces import force_fn
from aeroshape.lbm.lattice import make_params

F64 = torch.float64
OV = ["lbm.domain_d=[10,5]", "lbm.steady_tol=1.0e-13", "lbm.max_convective_times=300"]
CFG = load_config(None, "dev", OV)
COND = Condition(20, 0)
N = 10


def cd_fn(fc, fn, g, p):
    return force_fn(fc, fn, g, p)[0]


@lru_cache(maxsize=None)
def base():
    nurbs = NURBSAirfoil(k=8, dtype=F64)
    alpha = circle_alpha(nurbs)
    p = make_params(CFG, COND, N, dtype=F64)
    g = map_to_grid(nurbs(alpha), p)
    f = S.init_state(p, g, F64)
    for _ in range(300):
        f, _ = S.step(f, g, p)
    return nurbs, alpha, p, g, f


def steady(g, p, f0):
    r = S.run(f0, g, p, force_fn=force_fn, record_every=20, steady_tol=1.0e-13, max_convective_times=300)
    assert r.converged
    return r.state.f


def cd_at(f, g, p):
    fn, fc = S.step(f, g, p)
    return cd_fn(fc, fn, g, p)


@lru_cache(maxsize=None)
def converged():
    _, _, p, g, f = base()
    return steady(g, p, f)


def rand_like(f, seed):
    return torch.randn(f.shape, generator=torch.Generator().manual_seed(seed), dtype=F64)


def test_level1_hand_equals_vjp():
    _, _, p, g, f = base()
    Fs = rand_like(f, 1)
    d = (adjoint_step_hand(Fs, f, g, p) - adjoint_step_vjp(Fs, f, g, p)).abs().max()
    print(f"\nlevel 1 max abs diff {d:.2e}")
    assert d < 1e-10, d


def test_level2_dot_product():
    _, _, p, g, f = base()
    v, w = rand_like(f, 2), rand_like(f, 3)
    _, Jv = torch.func.jvp(lambda x: S.step(x, g, p)[0], (f,), (v,))
    lhs, rhs = (w * Jv).sum(), (adjoint_step_hand(w, f, g, p) * v).sum()
    print(f"\nlevel 2 rel {abs(lhs - rhs) / abs(lhs):.2e}")
    assert abs(lhs - rhs) / abs(lhs) < 1e-10


def _count_matvecs(monkeypatch):
    n = [0]
    orig = AS.adjoint_step_hand

    def counted(*a, **k):
        n[0] += 1
        return orig(*a, **k)

    monkeypatch.setattr(AS, "adjoint_step_hand", counted)
    return n


def test_krylov_equals_fixed_point_and_warm_start(monkeypatch):
    _, _, p, g, _ = base()
    f, n = converged(), _count_matvecs(monkeypatch)
    fp = solve_adjoint(f, g, p, cd_fn, tol=1e-12, horizon_steps=200000, solver="fixed_point")
    n[0] = 0
    kr = solve_adjoint(f, g, p, cd_fn, tol=1e-12, horizon_steps=200000, solver="krylov")
    cold = n[0]
    rel = (kr.F_star - fp.F_star).norm() / fp.F_star.norm()
    print(f"\nkrylov vs fixed point rel {rel:.2e}; krylov matvecs {cold}")
    assert rel < 1e-8
    n[0] = 0
    solve_adjoint(f, g, p, cd_fn, tol=1e-8, solver="krylov", F0=0.99 * kr.F_star)
    print(f"warm matvecs {n[0]} vs cold {cold}")
    assert n[0] < cold


def test_lbm_value_hooks():
    nurbs, alpha, *_ = base()
    cd, cl, f = lbm_value(alpha, nurbs, COND, CFG, N=N)
    r = lbm_value_and_grad(alpha, nurbs, COND, CFG, N=N, f0=f)
    assert r.extras["f"].shape == f.shape == r.extras["F_star"].shape
    assert abs(float(r.cd - cd)) < 1e-8


@pytest.mark.slow
def test_level3_dI_dq_vs_fd():
    _, _, p, g, _ = base()
    f = converged()
    res = solve_adjoint(f, g, p, cd_fn, tol=1e-12, horizon_steps=200000)
    # 10 random links among the 40 most sensitive ones (FD noise ~1e-8 would swamp links with |dI/dq| ~ 1e-5)
    top = res.dI_dq.abs().topk(40).indices
    links = top[torch.randperm(40, generator=torch.Generator().manual_seed(0))[:10]]
    eps, fd = 3e-5, []
    for l in links.tolist():
        v = []
        for s in (1, -1):
            q = g.links.q.clone()
            q[l] += s * eps
            gq = replace(g, links=replace(g.links, q=q))
            v.append(cd_at(steady(gq, p, f), gq, p))
        fd.append((v[0] - v[1]) / (2 * eps))
    fd, ad = torch.stack(fd), res.dI_dq[links]
    rel = (ad - fd).norm() / fd.norm()
    print(f"\nlevel 3 rel err {rel:.3e}")
    assert rel < 1e-4


def _topology(alpha, nurbs):
    g = map_to_grid(nurbs(alpha), make_params(CFG, COND, N, dtype=F64))
    return g.solid, g.links.node, g.links.dir


@pytest.mark.slow
def test_level4_dI_dalpha_vs_fd():
    nurbs, alpha, *_ = base()
    delta = 1e-4
    for k in range(8):
        for s in (1, -1):
            a = alpha.clone()
            a[0, k] += s * delta
            assert all(torch.equal(x, y) for x, y in zip(_topology(a, nurbs), _topology(alpha, nurbs))), (k, s)
    t0 = time.perf_counter()
    r = lbm_value_and_grad(alpha, nurbs, COND, CFG, N=N)
    t_all = time.perf_counter() - t0
    t0 = time.perf_counter()
    fd = fd_gradient(alpha, nurbs, COND, CFG, delta=delta, N=N)
    t_fd = time.perf_counter() - t0
    rel = (r.dcd_dalpha - fd).norm() / fd.norm()
    print(f"\nlevel 4 rel L2 {rel:.3e}; forward {r.extras['t_forward']:.1f}s adjoint {r.extras['t_adjoint']:.1f}s "
          f"total {t_all:.1f}s; fd {t_fd:.1f}s")
    assert rel < 1e-3


@pytest.mark.slow
@pytest.mark.gpu
def test_level6_mps_float32_vs_cpu():
    nurbs, alpha, *_ = base()
    ref = lbm_value_and_grad(alpha, nurbs, COND, CFG, N=N).dcd_dalpha
    cfg32 = load_config(None, "dev", ["lbm.domain_d=[10,5]", "lbm.steady_tol=1.0e-6", "lbm.max_convective_times=300"])
    n32 = NURBSAirfoil(k=8, device="mps", dtype=torch.float32)
    a32 = alpha.to("mps", torch.float32)
    r = lbm_value_and_grad(a32, n32, COND, cfg32, N=N).dcd_dalpha.cpu().double()
    cos = torch.nn.functional.cosine_similarity(r, ref, dim=0)
    print(f"\nlevel 6 cosine {cos:.6f}")
    assert cos > 0.999


def _stub_run(monkeypatch, f_mean, seen):
    from types import SimpleNamespace

    def run(f0, g, p, **kw):
        seen.append(f0)
        return SimpleNamespace(state=SimpleNamespace(f=f0 + 0.0), f_mean=f_mean, cd=torch.tensor(1.23, dtype=F64),
                               cl=torch.tensor(-0.4, dtype=F64), converged=True)
    monkeypatch.setattr(AS.S, "run", run)


def test_unsteady_forward_returns_period_mean_force(monkeypatch):  # M4
    _, _, p, g, f = base()
    f_mean = 0.999 * f
    _stub_run(monkeypatch, f_mean, [])
    _, f_lin, cd, cl, f_warm, conv = AS._forward(g, p, CFG, "time_averaged")
    assert float(cd) == 1.23 and float(cl) == -0.4 and conv
    assert f_lin is f_mean and f_warm is not f_mean  # linearize at f_mean, warm-start from the instantaneous state


def test_unsteady_warm_start_is_re_kicked(monkeypatch):  # M5
    _, _, p, g, f = base()
    seen = []
    _stub_run(monkeypatch, None, seen)
    cfg = load_config(None, "dev", OV + ["lbm.perturb=0.1"])
    AS._forward(g, p, cfg, "time_averaged", f)
    kick = seen[0] - f
    assert kick.abs().max() > 1e-6 * p.u_lb
    AS._forward(g, p, CFG, "steady", f)
    assert (seen[1] - f).abs().max() == 0  # steady warm start untouched


def test_gmres_zero_rhs():  # L4
    x, hist, n = AS._gmres(lambda v: v, torch.zeros(5, dtype=F64), torch.zeros(5, dtype=F64), 1e-8, 10)
    assert x.abs().max() == 0 and n == 0
