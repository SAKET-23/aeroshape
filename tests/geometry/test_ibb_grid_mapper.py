import math
import time

import pytest
import torch

from aeroshape.core.config import load_config
from aeroshape.core.types import Boundary, Condition
from aeroshape.geometry.ibb_grid_mapper import (find_links, link_q, map_to_grid, points_in_polygon, solid_mask,
                                                to_lattice)
from aeroshape.geometry.nurbs import NURBSAirfoil, rotate
from aeroshape.lbm import d2q9_solver as S
from aeroshape.lbm.lattice import C, make_params

CFG = load_config(None, "dev", [])
F64 = torch.float64
MIRROR = [0, 1, 4, 3, 2, 8, 7, 6, 5]  # direction with c_y flipped


def params(N, dtype=torch.float32):
    return make_params(CFG, Condition(20, 0), N, dtype=dtype)


def circle_boundary(r, cx=0.0, cy=0.0, n=4096, dtype=F64):
    th = (torch.arange(n + 1, dtype=F64) + 0.5) * (2 * math.pi / n)
    x, y = (cx + r * torch.cos(th))[None].to(dtype), (cy + r * torch.sin(th))[None].to(dtype)
    z = torch.zeros_like(x)
    return Boundary(x=x, y=y, nx=z, ny=z, ds=z)


def naca_alpha(m, cambered):
    """Fit NURBS control y to NACA thickness (+ parabolic camber) by lstsq through the R matrix."""
    x = m.R @ m.x_ctrl
    yt = 0.6 * 0.12 / 0.2 * (0.2969 * x.sqrt() - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4)
    yc = 0.06 * x * (1 - x) if cambered else torch.zeros_like(x)
    A = m.R[:, 1:m.k - 1]
    fit = lambda y: torch.linalg.lstsq(A, y[:, None]).solution[:, 0]
    a = torch.zeros(2 if cambered else 1, m.k, dtype=F64)
    a[0, 1:m.k - 1] = fit(yc + yt)
    if cambered:
        a[1, 1:m.k - 1] = fit(yc - yt)
        return a.reshape(1, -1)
    return a


def shape(kind, aoa=0.0):
    m = NURBSAirfoil(mode="cambered" if kind == "cambered" else "symmetric", dtype=F64)
    b = m(naca_alpha(m, kind == "cambered"))
    return rotate(b, aoa) if aoa else b


def mapped(b, N, dtype=torch.float32):
    p = params(N, dtype)
    return p, map_to_grid(b, p)


def test_points_in_polygon():
    b = circle_boundary(1.0, n=64)
    g = torch.tensor([[0.0, 0.5, 0.99, 1.01, -2.0, 0.0]], dtype=F64)
    h = torch.tensor([[0.0, 0.5, 0.0, 0.0, 0.0, 1.2]], dtype=F64)
    assert points_in_polygon(g, h, b.x, b.y).tolist() == [[True, True, True, False, False, False]]


def test_circle_mask_area_and_symmetry():
    N = 40
    p, geom = mapped(circle_boundary(0.5), N, F64)
    area = geom.solid.sum().item()
    assert abs(area / (math.pi * (0.5 * N) ** 2) - 1) < 0.01
    j0, s = geom.origin[1], geom.solid
    k = min(j0, p.Ny - 1 - j0)
    assert torch.equal(s[j0 + 1:j0 + k], s[j0 - k + 1:j0].flip(0))


def test_circle_q_analytic():
    N, (cx, cy) = 10, (0.013, -0.011)
    b = circle_boundary(0.5, cx, cy)
    xl, yl = to_lattice(b, N, (100, 50))
    with torch.no_grad():
        solid = solid_mask(xl, yl, 100, 200)
    node, d, far_ok = find_links(solid)
    q, seg = link_q(node, d, xl, yl, 200)
    assert (seg >= 0).all()
    A = torch.stack([node % 200, node // 200], 1).to(F64) - torch.tensor([100 + N * cx, 50 + N * cy], dtype=F64)
    c = C[d].to(F64)
    R = 0.5 * N
    bq, cc = (A * c).sum(1), (c * c).sum(1)
    t = (-bq - torch.sqrt(bq**2 - cc * ((A * A).sum(1) - R**2))) / cc
    assert (q - t.clamp(0.01, 1)).abs().max() < 1e-5


@pytest.mark.parametrize("kind", ["circle", "sym", "cambered"])
def test_link_invariants_and_pinholes(kind):
    b = circle_boundary(0.5, dtype=torch.float32) if kind == "circle" else shape("symmetric" if kind == "sym" else kind, 3.0)
    p, g = mapped(b, 20)
    L = g.links
    s = g.solid.reshape(-1)
    Nx = p.Nx
    nb = (L.node % Nx + C[L.dir, 0]) + ((L.node // Nx + C[L.dir, 1]) * Nx)
    assert not s[L.node].any() and s[nb].all()
    assert (L.q >= 0.01).all() and (L.q <= 1).all()
    assert (L.seg >= 0).all() and L.node.numel() > 0
    sh = lambda dy, dx: torch.roll(g.solid, (dy, dx), (0, 1))
    assert not (~g.solid & sh(1, 0) & sh(-1, 0) & sh(0, 1) & sh(0, -1)).any()


def test_link_symmetry():
    N = 20
    p, g = mapped(shape("symmetric"), N, F64)
    L, (Ny, Nx) = g.links, g.solid.shape
    Q = torch.full((9, Ny, Nx), float("nan"), dtype=F64)
    Q[L.dir, L.node // Nx, L.node % Nx] = L.q
    j0 = g.origin[1]
    j = torch.arange(Ny)
    for d in range(1, 9):
        m = Q[d].isnan() == Q[MIRROR[d]][(2 * j0 - j).clamp(0, Ny - 1)].isnan()
        assert m.all()
        a, bq = Q[d], Q[MIRROR[d]][(2 * j0 - j).clamp(0, Ny - 1)]
        ok = ~a.isnan()
        assert (a[ok] - bq[ok]).abs().max() < 1e-10


def test_dq_dalpha():
    m = NURBSAirfoil(mode="cambered", dtype=F64)
    a0 = naca_alpha(m, True)
    v = torch.rand(1, 16, generator=torch.Generator().manual_seed(1), dtype=F64) - 0.5
    w = None

    def build(a):
        return to_lattice(m(a), 40, (100.37, 100.21))

    xl, yl = build(a0)
    solid = solid_mask(xl, yl, 200, 200)
    node, d, _ = find_links(solid)
    w = torch.rand(node.numel(), generator=torch.Generator().manual_seed(2), dtype=F64)

    def f(a):
        return (link_q(node, d, *build(a), 200)[0] * w).sum()

    a = a0.clone().requires_grad_()
    f(a).backward()
    ad = (a.grad * v).sum()
    h = 1e-6
    fd = (f(a0 + h * v) - f(a0 - h * v)) / (2 * h)
    assert abs(ad - fd) / abs(fd) < 1e-6
    for s in (1, -1):  # topology identical at +-h
        assert torch.equal(find_links(solid_mask(*build(a0 + s * h * v), 200, 200))[0], node)


@pytest.mark.parametrize("N", [10, 20, 40])
@pytest.mark.parametrize("kind", ["circle", "sym", "cambered"])
def test_no_fallbacks(kind, N, capsys):
    b = circle_boundary(0.5, dtype=torch.float32) if kind == "circle" else shape("symmetric" if kind == "sym" else kind)
    _, g = mapped(b, N)
    with capsys.disabled():
        print(f"\n[{kind} N={N}] links={g.links.node.numel()} fallbacks={(g.links.seg < 0).sum().item()}")
    assert (g.links.seg >= 0).all()


def test_device_parity(device):
    b = shape("cambered", 4.0)
    b = Boundary(*[t.float() for t in (b.x, b.y, b.nx, b.ny, b.ds)])
    p = params(20)
    ref = map_to_grid(b, p)
    got = map_to_grid(Boundary(*[t.to(device) for t in (b.x, b.y, b.nx, b.ny, b.ds)]), p)
    assert torch.equal(got.solid.cpu(), ref.solid)
    assert torch.equal(got.links.node.cpu(), ref.links.node) and torch.equal(got.links.dir.cpu(), ref.links.dir)
    assert (got.links.q.cpu() - ref.links.q).abs().max() < 1e-6


def test_end_to_end_step_gradient():
    N = 10
    p = params(N)
    alpha = torch.tensor(0.0, requires_grad=True)
    b = circle_boundary(0.5 + alpha * 0.02, dtype=torch.float32)
    g = map_to_grid(b, p)
    f = S.init_state(p, g)
    for _ in range(200):
        f, _ = S.step(f, g, p)
    assert torch.isfinite(f).all()
    f[:, :, p.Nx // 2:].pow(2).sum().backward()
    assert torch.isfinite(alpha.grad) and alpha.grad.abs() > 0


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="no MPS")
def test_map_to_grid_timing(capsys):
    b = shape("cambered", 2.0)
    dev = torch.device("mps")
    b = Boundary(*[t.float().to(dev) for t in (b.x, b.y, b.nx, b.ny, b.ds)])
    p = params(100)
    map_to_grid(b, p)
    torch.mps.synchronize()
    t0 = time.perf_counter()
    g = map_to_grid(b, p)
    torch.mps.synchronize()
    dt = time.perf_counter() - t0
    with capsys.disabled():
        print(f"\nmap_to_grid N=100 MPS: {dt:.3f}s, links={g.links.node.numel()}")
    assert dt < 0.5

