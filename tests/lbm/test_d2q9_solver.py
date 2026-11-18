import math
from dataclasses import replace

import numpy as np
import pytest
import torch

from aeroshape.core.types import GridGeometry, LBMParams, LinkSet
from aeroshape.lbm import d2q9_solver as S
from aeroshape.lbm.lattice import C, W

F64 = torch.float64
CPU = torch.device("cpu")


def mkparams(Nx, Ny, tau=0.8, collision="bgk", magic=3 / 16, u_lb=0.05, N=10):
    tm = 0.5 + magic / (tau - 0.5)
    return LBMParams(tau=tau, omega=1 / tau, omega_minus=(1 / tm if collision == "trt" else 1 / tau),
                     u_lb=u_lb, N=N, Nx=Nx, Ny=Ny, collision=collision)


def empty_geom(Nx, Ny, device=CPU):
    z = torch.zeros(0, dtype=torch.long, device=device)
    links = LinkSet(z, z, torch.zeros(0, device=device), torch.zeros(0, dtype=torch.bool, device=device), z)
    return GridGeometry(torch.zeros(Ny, Nx, dtype=torch.bool, device=device), links, 10, (0, 0))


def channel_geom(Nx, Nf, qb, qt, device=CPU, dtype=F64):
    """Fluid rows 1..Nf, solid rows 0 and Nf+1; wall is a fraction q from the nearest fluid row."""
    Ny = Nf + 2
    solid = torch.zeros(Ny, Nx, dtype=torch.bool, device=device)
    solid[0] = solid[-1] = True
    cols = torch.arange(Nx, device=device)
    node, dirs, q = [], [], []
    for row, ds, qq in ((1, (4, 7, 8), qb), (Nf, (2, 5, 6), qt)):
        for d in ds:
            node.append(row * Nx + cols)
            dirs.append(torch.full((Nx,), d, device=device))
            q.append(qq * torch.ones(Nx, device=device, dtype=dtype))
    node, dirs, q = torch.cat(node), torch.cat(dirs), torch.cat(q)
    links = LinkSet(node, dirs, q, torch.ones_like(node, dtype=torch.bool), torch.zeros_like(node))
    return GridGeometry(solid, links, Nf, (0, 0))


def channel_u(Nf, qb, qt, g, nu, device=CPU):
    y0, y1 = 1 - qb, Nf + qt
    y = torch.arange(Nf + 2, dtype=F64, device=device)
    return (g / (2 * nu) * (y - y0) * (y1 - y)).clamp(min=0)


def poiseuille_error(Nf, q, tau=0.8, umax=0.02, collision="trt"):
    Nx, p = 4, mkparams(4, Nf + 2, tau, collision)
    nu, H = (tau - 0.5) / 3, Nf - 1 + 2 * q
    g = 8 * nu * umax / H**2
    geom = channel_geom(Nx, Nf, q, q)
    ue = channel_u(Nf, q, q, g, nu)
    f = S.equilibrium(torch.ones(Nf + 2, Nx, dtype=F64), ue[:, None].expand(-1, Nx), torch.zeros(Nf + 2, Nx, dtype=F64))
    f = torch.where(geom.solid, W[:, None, None], f)
    for _ in range(int(6 * H**2 / (math.pi**2 * nu))):
        f, _ = S.step(f, geom, p, force=g, open_x=False)
    _, ux, _ = S.moments(f)
    u = (ux[:, 0] + g / 2)[1:-1]  # first-order forcing: physical velocity is u + g/2
    return ((u - ue[1:-1]).norm() / ue[1:-1].norm()).item()


def test_equilibrium_moments():
    rho, ux, uy = torch.rand(5, 6, dtype=F64) + 0.5, torch.rand(5, 6, dtype=F64) * 0.1, torch.rand(5, 6, dtype=F64) * 0.1
    feq = S.equilibrium(rho, ux, uy)
    r, vx, vy = S.moments(feq)
    assert (r - rho).abs().max() < 1e-12 and (vx - ux).abs().max() < 1e-12 and (vy - uy).abs().max() < 1e-12


@pytest.mark.parametrize("coll", ["bgk", "trt"])
def test_collision_conserves(coll):
    f = torch.rand(9, 5, 6, dtype=F64) * 0.1 + 0.05
    p = mkparams(6, 5, 0.7, coll)
    fc = S.collide(f, p, torch.zeros(5, 6, dtype=torch.bool))
    for a, b in zip(S.moments(f), S.moments(fc)):
        assert (a - b).abs().max() < 1e-12


def test_trt_equals_bgk_when_omegas_equal():
    f = torch.rand(9, 5, 6, dtype=F64) * 0.1 + 0.05
    solid = torch.zeros(5, 6, dtype=torch.bool)
    pb = mkparams(6, 5, 0.7, "bgk")
    pt = replace(pb, collision="trt", omega_minus=pb.omega)
    assert (S.collide(f, pb, solid) - S.collide(f, pt, solid)).abs().max() < 1e-14


def test_roll_sign_convention():
    for i in range(9):
        f = torch.zeros(9, 6, 7)
        f[i, 2, 3] = 1.0
        out = S.stream(f)
        assert out.sum() == 1 and out[i, 2 + int(C[i, 1]), 3 + int(C[i, 0])] == 1


def test_periodic_box_mass():
    Nx, Ny = 16, 12
    p = mkparams(Nx, Ny, 0.7)
    rho = 1 + 0.01 * torch.rand(Ny, Nx, dtype=F64)
    f = S.equilibrium(rho, 0.02 * torch.rand(Ny, Nx, dtype=F64), torch.zeros(Ny, Nx, dtype=F64))
    m0 = f.sum()
    for _ in range(1000):
        f, _ = S.step(f, empty_geom(Nx, Ny), p, open_x=False)
    assert abs(f.sum() - m0) / m0 < 1e-6


@pytest.mark.parametrize("outlet", ["fixed", "extrapolate"])
def test_uniform_flow_stays_uniform(outlet):
    p = mkparams(20, 10, 0.7)
    geom = empty_geom(20, 10)
    f0 = S.init_state(p, geom, dtype=F64)
    f = f0
    for _ in range(50):
        f, _ = S.step(f, geom, p, outlet=outlet)
    assert (f - f0).abs().max() < 1e-6


@pytest.mark.parametrize("q", [0.5, 0.3])
def test_poiseuille_1pct(q):
    err = poiseuille_error(20, q)
    print(f"Poiseuille N=20 q={q}: rel L2 error {err:.2e}")
    assert err < 0.01


def test_wall_position_convergence():
    ns = [10, 20, 40]
    for q in (0.25, 0.75):
        e = [poiseuille_error(n, q) for n in ns]
        slope = -float(np.polyfit(np.log(ns), np.log(e), 1)[0])
        print(f"q={q}: errors {e}, slope {slope:.2f}")
        assert slope >= 1.8


def test_differentiable_in_q():
    Nf, Nx = 8, 4
    p = mkparams(Nx, Nf + 2, 0.8, "trt")
    q = torch.tensor([0.3, 0.7], dtype=F64, requires_grad=True)
    g = channel_geom(Nx, Nf, 0.5, 0.5)
    qq = torch.cat([q[0].expand(3 * Nx), q[1].expand(3 * Nx)])
    geom = replace(g, links=replace(g.links, q=qq))
    f = S.init_state(p, geom, dtype=F64)
    for _ in range(20):
        f, _ = S.step(f, geom, p, force=1e-5, open_x=False)
    (grad,) = torch.autograd.grad(S.moments(f)[1].sum(), q)
    assert torch.isfinite(grad).all() and (grad.abs() > 0).all()


@pytest.mark.gpu
def test_cpu_mps_parity():
    if not torch.backends.mps.is_available():
        pytest.skip("no mps")
    Nf, Nx = 8, 16
    p = mkparams(Nx, Nf + 2, 0.7, "trt")
    g = channel_geom(Nx, Nf, 0.3, 0.6, dtype=torch.float32)
    f = S.init_state(p, g, dtype=torch.float32) * (1 + 0.01 * torch.rand(9, Nf + 2, Nx))
    gm = GridGeometry(g.solid.to("mps"), LinkSet(*(t.to("mps") for t in (g.links.node, g.links.dir, g.links.q, g.links.far_ok, g.links.seg))), g.N, g.origin)
    a, _ = S.step(f, g, p)
    b, _ = S.step(f.to("mps"), gm, p)
    assert (a - b.cpu()).abs().max() < 1e-6


def test_run_modes():
    p = mkparams(20, 10, 0.7, N=2)  # K = 35 steps per convective time
    geom = empty_geom(20, 10)
    f0 = S.init_state(p, geom, dtype=F64)
    r = S.run(f0, geom, p, n_steps=30)
    assert r.state.step == 30 and r.history.cd.numel() == 0
    r = S.run(f0, geom, p, record_every=5, force_fn=lambda fc, fn, g, pp: (fn[1].sum(), fn[1].sum() * 0))
    assert r.converged and r.state.step < 60 * 35

    cnt = [0]

    def fn(fc, f_next, g, pp):
        cnt[0] += 5
        return torch.zeros((), dtype=F64), torch.sin(torch.tensor(2 * math.pi * cnt[0] / 100 + 0.3, dtype=F64))

    r = S.run(f0, geom, p, mode="unsteady", record_every=5, force_fn=fn, transient=2, average_periods=2)
    assert r.period_steps == 100 and r.f_mean.shape == f0.shape


def _sig_fn(sig):
    cnt = [0]

    def fn(fc, f_next, g, pp):
        cnt[0] += 5
        return torch.zeros((), dtype=F64), torch.tensor(sig(cnt[0]), dtype=F64)
    return fn


def test_unsteady_period_ignores_startup_wiggle():
    p = mkparams(20, 10, 0.7, N=2)  # K = 35; transient=4 -> 140 steps
    geom = empty_geom(20, 10)
    f0 = S.init_state(p, geom, dtype=F64)
    sig = lambda n: math.sin(2 * math.pi * n / 20) if n <= 140 else math.sin(2 * math.pi * (n - 140) / 100 + 0.3)
    r = S.run(f0, geom, p, mode="unsteady", record_every=5, force_fn=_sig_fn(sig), transient=4, average_periods=1)
    assert r.period_steps == 100


def test_unsteady_period_needs_three_consistent_periods():
    p = mkparams(20, 10, 0.7, N=2)
    geom = empty_geom(20, 10)
    f0 = S.init_state(p, geom, dtype=F64)
    # periods 100, 60, 100, 100, 100 steps after the transient: only the last three agree
    edges = [0, 100, 160, 260, 360, 460]

    def sig(n):
        t = n - 70
        if t < 0:
            return math.sin(0.5 * n)
        k = max(i for i, e in enumerate(edges) if e <= t) if t < edges[-1] else 4
        return math.sin(2 * math.pi * (t - edges[k]) / (edges[k + 1] - edges[k]))
    r = S.run(f0, geom, p, mode="unsteady", record_every=5, force_fn=_sig_fn(sig), transient=2, average_periods=1)
    assert r.period_steps == 100


def test_init_state_perturb_breaks_symmetry():
    p, geom = mkparams(40, 20, N=10), empty_geom(40, 20)
    f_sym, f_p = S.init_state(p, geom, dtype=F64), S.init_state(p, geom, dtype=F64, perturb=0.1)
    assert (S.moments(f_sym)[2]).abs().max() == 0
    uy = S.moments(f_p)[2]
    assert uy.abs().max() > 0.05 * p.u_lb and uy.abs().max() <= 0.1 * p.u_lb * 1.001
    assert (f_p - f_p[[0, 1, 4, 3, 2, 8, 7, 6, 5]].flip(1)).abs().max() > 1e-6


def test_unsteady_no_period_warns_or_raises():
    p, geom = mkparams(20, 10, 0.7, N=2), empty_geom(20, 10)
    f0 = S.init_state(p, geom, dtype=F64)
    kw = dict(mode="unsteady", record_every=5, force_fn=_sig_fn(lambda n: 1.0), transient=1, max_convective_times=4)
    with pytest.warns(RuntimeWarning, match="no period"):
        r = S.run(f0, geom, p, **kw)
    assert r.f_mean is None and not r.converged
    with pytest.raises(RuntimeError):
        S.run(f0, geom, p, require_period=True, **kw)


@pytest.mark.slow
def test_re1000_smoke_stays_finite():
    """F1: Re 1000 TRT circle-like start at the smoke-tier N (tau ~ 0.504) must not blow up (Lambda 3/16 does at ~1.4 T)."""
    from aeroshape.core.config import load_config
    from aeroshape.core.types import Condition
    from aeroshape.geometry.ibb_grid_mapper import map_to_grid
    from aeroshape.geometry.nurbs import NURBSAirfoil
    from aeroshape.geometry.reference_shapes import circle_alpha
    from aeroshape.lbm.lattice import make_params
    cfg = load_config("case2_circle_to_rae_re1000.yaml", "smoke", [])
    nb = NURBSAirfoil(k=8, mode="cambered", dtype=F64)
    p = make_params(cfg, Condition(1000, 0), cfg.lbm.cells_per_d_high_re, dtype=torch.float32)
    b = nb(circle_alpha(nb))
    b = type(b)(*[t.float() for t in (b.x, b.y, b.nx, b.ny, b.ds)])
    g = map_to_grid(b, p, body_offset_d=cfg.lbm.body_offset_d)
    f = S.init_state(p, g, torch.float32, perturb=cfg.lbm.perturb)
    f = S.run(f, g, p, n_steps=round(4 * p.N / p.u_lb), outlet=cfg.lbm.outlet).state.f  # past the 1.4 T failure
    assert torch.isfinite(f).all() and S.moments(f)[1].abs().max() < 0.5


def _records(fn, n=60):
    return torch.tensor([fn(i) for i in range(n)], dtype=F64)


def test_detect_period_offset_sinusoid():  # M6: one-signed CL (cambered shape) still has a period
    cl = _records(lambda i: 1.0 + 0.2 * math.sin(2 * math.pi * i / 10))
    assert S.detect_period(cl, 5) == 50


def test_detect_period_rejects_decaying_oscillation():  # M6
    cl = _records(lambda i: math.exp(-i / 12) * math.sin(2 * math.pi * i / 10))
    assert S.detect_period(cl, 5) is None


def test_steady_convergence_checks_cl_too():  # L5: CD flat but CL still drifting must not count as converged
    p, geom = mkparams(20, 10, 0.7, N=2), empty_geom(20, 10)
    f0 = S.init_state(p, geom, dtype=F64)
    cnt = [0]

    def fn(fc, f_next, g, pp):
        cnt[0] += 5
        return torch.ones((), dtype=F64), torch.tensor(1e-3 * cnt[0], dtype=F64)
    r = S.run(f0, geom, p, record_every=5, force_fn=fn, steady_tol=1e-6, max_convective_times=5)
    assert not r.converged
