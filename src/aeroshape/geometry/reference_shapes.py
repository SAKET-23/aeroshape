"""Reference shapes, least-squares fitting and shape sampling (Plan/02_geometry/reference_shapes.md). CPU float64."""
from pathlib import Path

import numpy as np
import torch

from aeroshape.geometry.nurbs import NURBSAirfoil

Tensor = torch.Tensor
_CSV = Path(__file__).parent / "data" / "rae2822.csv"


def _cpu(t):
    return t.detach().to(device="cpu", dtype=torch.float64)


def _naca_coords(m, p, t, n=400):
    x = (1 - np.cos(np.linspace(0, np.pi, n))) / 2
    yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1036 * x**4)  # closed TE
    if m == 0:
        return dict(x_up=x, y_up=yt, x_lo=x, y_lo=-yt)
    yc = np.where(x < p, m / p**2 * (2 * p * x - x**2), m / (1 - p) ** 2 * (1 - 2 * p + 2 * p * x - x**2))
    th = np.arctan(np.where(x < p, 2 * m / p**2 * (p - x), 2 * m / (1 - p) ** 2 * (p - x)))
    xu, yu = x - yt * np.sin(th), yc + yt * np.cos(th)
    i = xu.argmin()  # camber-normal offset pushes the nose to x < 0: start the upper curve at its true leading edge
    return dict(x_up=np.maximum.accumulate(np.maximum(xu[i:], 0)), y_up=yu[i:], x_lo=x + yt * np.sin(th), y_lo=yc - yt * np.cos(th))


def reference_coords(name: str) -> dict:
    """Raw coordinates {x_up, y_up, x_lo, y_lo} (numpy) for 'circle', 'naca<4 digits>' or 'rae2822'."""
    name = name.lower()
    if name == "circle":
        x = (1 - np.cos(np.linspace(0, np.pi, 400))) / 2
        return dict(x_up=x, y_up=np.sqrt(x * (1 - x)), x_lo=x, y_lo=-np.sqrt(x * (1 - x)))
    if name.startswith("naca"):
        c = name[4:]
        return _naca_coords(int(c[0]) / 100, int(c[1]) / 10, int(c[2:]) / 100)
    if name == "rae2822":
        # published TE is already closed (y = 0 at x = 1), so no blend is needed
        d = {s: np.array([[float(v) for v in r.split(",")[1:]] for r in _CSV.read_text().splitlines()
                          if r.startswith(s)]) for s in ("upper", "lower")}
        return dict(x_up=d["upper"][:, 0], y_up=d["upper"][:, 1], x_lo=d["lower"][:, 0], y_lo=d["lower"][:, 1])
    raise ValueError(name)


def fit_alpha(nurbs: NURBSAirfoil, x_up, y_up, x_lo=None, y_lo=None) -> Tensor:
    """Least squares on CPU float64 with targets interpolated at the NURBS sample x. [1, D].
    Symmetric mode fits the half thickness (y_up - y_lo)/2 (lower = -upper when not given)."""
    R, xs = _cpu(nurbs.R), _cpu(nurbs.R @ nurbs.x_ctrl).numpy()
    yu = np.interp(xs, x_up, y_up)
    yl = -yu if x_lo is None else np.interp(xs, x_lo, y_lo)
    A = R[:, 1:-1]
    sol = lambda y: torch.linalg.lstsq(A, torch.from_numpy(y)[:, None]).solution[:, 0]
    z = torch.zeros(1, nurbs.k, dtype=torch.float64)
    if nurbs.mode == "symmetric":
        z[0, 1:-1] = sol((yu - yl) / 2)
        return z
    zl = z.clone()
    z[0, 1:-1], zl[0, 1:-1] = sol(yu), sol(yl)
    return torch.cat([z, zl], 1)


def _fit(nurbs, name):
    return fit_alpha(nurbs, *(reference_coords(name)[k] for k in ("x_up", "y_up", "x_lo", "y_lo")))


def circle_alpha(nurbs):
    return _fit(nurbs, "circle")


def naca4_alpha(nurbs, code: str):
    return _fit(nurbs, "naca" + code)


def rae2822_alpha(nurbs):
    return _fit(nurbs, "rae2822")


def shape_distance(nurbs, alpha_a, alpha_b) -> Tensor:
    """RMS y-difference over both surfaces (chord units), [B]."""
    _, ua, la = nurbs.surface_y(alpha_a)
    _, ub, lb = nurbs.surface_y(alpha_b)
    return _cpu((((ua - ub) ** 2 + (la - lb) ** 2) / 2).mean(1).sqrt())


def is_valid(nurbs, alpha, t_floor=0.02, x_hi=0.85) -> Tensor:
    """[B] bool: upper >= lower everywhere and thickness >= t_floor on x in (0.05, x_hi).
    x_hi = 0.85, not the spec's 0.95: NACA0012 and RAE2822 are only 0.014 / 0.009 thick at x = 0.95."""
    x, yu, yl = (_cpu(v) for v in nurbs.surface_y(alpha))
    t = yu - yl
    mid = (x > 0.05) & (x < x_hi)
    return (t >= -1e-9).all(1) & (t[:, mid] >= t_floor).all(1)


def _lhs(n, d, g):
    return (torch.stack([torch.randperm(n, generator=g) for _ in range(d)], 1) + torch.rand(n, d, generator=g, dtype=torch.float64)) / n


def perturb(alpha, n, scale, smooth_len, seed, nurbs) -> Tensor:
    """[n, D]: alpha ([1, D] or [n, D]) + zero-mean GP noise (squared exponential in control x,
    length smooth_len in chord units, std scale) on free entries."""
    g = torch.Generator().manual_seed(seed)
    xc = _cpu(nurbs.x_ctrl)
    K = scale**2 * torch.exp(-((xc[:, None] - xc[None]) ** 2) / (2 * smooth_len**2)) + 1e-10 * torch.eye(nurbs.k, dtype=torch.float64)
    L = torch.linalg.cholesky(K)
    S = nurbs.D // nurbs.k
    noise = torch.randn(n, S, nurbs.k, generator=g, dtype=torch.float64) @ L.T
    return _cpu(alpha) + noise.reshape(n, nurbs.D) * _cpu(nurbs.mask)


def lhs_shapes(nurbs, n, seed, family="dataset") -> Tensor:
    """[n, D] valid shapes: first n valid candidates of a seeded Latin hypercube (candidates doubled until n are valid)."""
    base = {k: _fit(nurbs, k) for k in ("circle", "naca0012", "rae2822")}
    rae = base["rae2822"]
    m = 3 * n
    while True:
        g = torch.Generator().manual_seed(seed)
        u = _lhs(m, 6, g)
        if family == "dataset":
            w = u[:, :3] / u[:, :3].sum(1, keepdim=True)
            a = (w @ torch.stack([base[k][0] for k in base])) * (0.6 + 0.8 * u[:, 3:4])
            a = perturb(a, m, 0.01, 0.3, seed + 1, nurbs)
        else:
            nac = torch.cat([fit_alpha(nurbs, *(_naca_coords(0.06 * r[0], 0.2 + 0.4 * r[1], 0.08 + 0.16 * r[2])[c]
                             for c in ("x_up", "y_up", "x_lo", "y_lo"))) for r in u.tolist()])
            kind = torch.arange(m)[:, None] % 10
            blend = u[:, 3:4] * rae + (1 - u[:, 3:4]) * nac
            a = torch.where(kind < 4, nac, blend)
            pert = perturb(torch.where(u[:, 4:5] < 0.5, nac, rae.expand_as(nac)), m, 0.01, 0.3, seed + 1, nurbs)
            a = torch.where(kind >= 7, pert, a)
        ok = is_valid(nurbs, a)
        if int(ok.sum()) >= n:
            return a[ok][:n]
        m *= 2
