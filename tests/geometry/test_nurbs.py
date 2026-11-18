import pytest
import torch

from aeroshape.geometry.nurbs import NURBSAirfoil, basis_matrix, rotate
from tests.conftest import assert_close


def rand_alpha(B, D, seed=0, scale=0.08):
    g = torch.Generator().manual_seed(seed)
    a = torch.rand(B, D, generator=g, dtype=torch.float64) * scale
    if D == 16:  # cambered: lower half negative so the surfaces do not cross
        a[:, 8:] *= -1
    return a


def make(device, dtype=torch.float32, **kw):
    return NURBSAirfoil(device=device, dtype=dtype, **kw)


def ellipse_alpha(m, t=0.12):
    # alpha = control y of a thin ellipse-like shape (positive bump), pins left to mask
    a = t * torch.sqrt(m.x_ctrl * (1 - m.x_ctrl)) * 2
    a[1] = 0.8 * a[2]  # index 1 sits at x=0: free y that sets the nose radius (0 would degenerate the tangent)
    return a[None]


def test_partition_of_unity():
    s = torch.linspace(0, 1, 101, dtype=torch.float64)
    for w in (None, torch.rand(8, dtype=torch.float64) + 0.5):
        R, dR = basis_matrix(8, 3, s, torch.ones(8, dtype=torch.float64) if w is None else w)
        assert_close(R.sum(1), torch.ones(101, dtype=torch.float64), 0, 1e-12)
        assert_close(dR.sum(1), torch.zeros(101, dtype=torch.float64), 0, 1e-10)


@pytest.mark.parametrize("mode", ["symmetric", "cambered"])
def test_pinning_and_loop(device, mode):
    m = make(device, mode=mode)
    D = 8 if mode == "symmetric" else 16
    b = m(rand_alpha(4, D).float().to(device))
    M = 160
    assert b.x.shape == (4, 2 * M - 1)
    # trailing edge (both ends) and leading edge on chord line, exactly
    assert (b.y[:, 0] == 0).all() and (b.y[:, -1] == 0).all() and (b.y[:, M - 1] == 0).all()
    assert_close(b.x[:, 0], b.x[:, -1], 0, 0)
    assert_close(b.y[:, 0], b.y[:, -1], 0, 0)
    assert (m.area(rand_alpha(4, D).float().to(device)) > 0).all()


def test_normals_outward(device):
    m = make(device)
    b = m(ellipse_alpha(m).to(device))
    assert_close(torch.hypot(b.nx, b.ny), torch.ones_like(b.nx), 0, 1e-6)
    cx, cy = b.x.mean(1, keepdim=True), b.y.mean(1, keepdim=True)
    assert ((b.nx * (b.x - cx) + b.ny * (b.y - cy)) > 0).all()
    assert (b.ds > 0).all()
    # ds sums to the perimeter of a closed loop
    per = torch.hypot(b.x[:, 1:] - b.x[:, :-1], b.y[:, 1:] - b.y[:, :-1]).sum(1)
    assert_close(b.ds.sum(1), per, 1e-5, 1e-6)


def test_symmetric_mirror(device):
    m = make(device)
    _, yu, yl = m.surface_y(rand_alpha(3, 8).float().to(device))
    assert_close(yl, -yu, 0, 0)


@pytest.mark.parametrize("mode", ["symmetric", "cambered"])
def test_linearity(mode):
    m = make("cpu", torch.float64, mode=mode)
    D = 8 if mode == "symmetric" else 16
    a, c = rand_alpha(2, D, 1), rand_alpha(2, D, 2)
    ba, bc, bs = m(a), m(c), m(a + c)
    assert_close(bs.y, ba.y + bc.y, 0, 1e-12)
    assert_close(bs.x, ba.x, 0, 0)


@pytest.mark.parametrize("mode", ["symmetric", "cambered"])
def test_gradcheck(mode):
    m = make("cpu", torch.float64, mode=mode, M=24)
    D = 8 if mode == "symmetric" else 16
    a = rand_alpha(2, D, 3).requires_grad_()
    assert torch.autograd.gradcheck(lambda t: m(t).y, (a,))
    assert torch.autograd.gradcheck(lambda t: m(t).nx, (a,))
    assert torch.autograd.gradcheck(m.thickness_at_ctrl, (a,))


def test_thickness_at_ctrl():
    m = make("cpu", torch.float64)
    a = rand_alpha(1, 8, 4)
    th = m.thickness_at_ctrl(a)
    assert th.shape == (1, 8) and th[0, 0] == 0 and th[0, -1] == 0
    assert th[0, 1] == 0 and (th[0, 2:-1] > 0).all()  # x_ctrl[1] = 0 -> s = 0


def test_free_full_roundtrip():
    m = make("cpu", torch.float64, mode="cambered")
    z = torch.randn(3, m.n_free, dtype=torch.float64)
    assert m.n_free == 12
    assert_close(m.full_to_free(m.free_to_full(z)), z, 0, 0)
    assert m.control_tokens(m.free_to_full(z)).shape == (3, 16, 3)


def test_rotate():
    m = make("cpu", torch.float64)
    b = m(ellipse_alpha(m).double())
    r = rotate(b, 10.0)
    assert_close(torch.hypot(r.nx, r.ny), torch.ones_like(r.nx), 0, 1e-10)
    # positive AoA: trailing edge (x=1) drops
    assert r.y[0, 0] < 0
    assert_close(r.ds, b.ds, 0, 0)
    # pivot fixed
    p = rotate(b, 10.0, pivot=(0.25, 0.0))
    assert_close(torch.hypot(p.x[:, 0] - 0.25, p.y[:, 0]), torch.hypot(b.x[:, 0] - 0.25, b.y[:, 0]), 0, 1e-12)


# TE has finite slope by design (G21), so the circle's sqrt(1-x) tip is excluded: assert on x <= 0.95,
# print the full-chord error without asserting.
@pytest.mark.parametrize("k,tol", [(8, 2.5e-2), (41, 3e-3)])
def test_circle_fit(k, tol):
    m = make("cpu", torch.float64, k=k)
    x = m.R @ m.x_ctrl
    target = torch.sqrt(x * (1 - x)).clamp_min(0)
    free = m.free_idx
    sol = torch.linalg.lstsq(m.R[:, free], target[:, None]).solution[:, 0]
    e = (m.R[:, free] @ sol - target).abs()
    print(f"circle fit k={k} max err x<=0.95: {e[x <= 0.95].max().item():.3e}, full chord: {e.max().item():.3e}")
    assert e[x <= 0.95].max().item() < tol


def test_device_parity(device):
    a = rand_alpha(3, 8).float()
    ref = make("cpu")(a)
    out = make(device)(a.to(device))
    for f in ("x", "y", "nx", "ny", "ds"):
        assert_close(getattr(out, f).cpu(), getattr(ref, f), 0, 1e-6)
