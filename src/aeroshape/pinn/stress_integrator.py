"""Differentiable surface forces from the PINN (Plan/04_models/stress_integrator.md, derivations 5.5)."""
import time

import torch

from aeroshape.core.timing import sync
from aeroshape.core.types import Boundary, EvalResult
from aeroshape.geometry.nurbs import NURBSAirfoil
from aeroshape.pinn.embeddings import predict_flow

Tensor = torch.Tensor


def _rotated(nurbs, alpha, cond, aoa_deg, n_pts):
    """Rotated surface Boundary; AoA per shape (float, [B] tensor, or from cond sin/cos)."""
    if n_pts is not None:  # shortcut: unit NURBS weights/degree 3, rebuilt at the requested resolution
        nurbs = NURBSAirfoil(k=nurbs.k, M=(n_pts + 1) // 2, mode=nurbs.mode, device=alpha.device, dtype=alpha.dtype)
    b = nurbs(alpha)
    if aoa_deg is None:
        th = torch.atan2(cond[:, 1], cond[:, 2])[:, None]
    else:
        th = torch.as_tensor(aoa_deg, dtype=alpha.dtype, device=alpha.device).deg2rad().reshape(-1, 1)
    c, s = th.cos(), -th.sin()  # same convention as geometry.nurbs.rotate (body rotated by -AoA about the pivot)
    px = 0.25
    return Boundary(x=px + c * (b.x - px) - s * b.y, y=s * (b.x - px) + c * b.y,
                    nx=c * b.nx - s * b.ny, ny=s * b.nx + c * b.ny, ds=b.ds)


def surface_forces(model, nurbs, alpha: Tensor, cond: Tensor, aoa_deg=None, n_pts: int | None = None):
    """-> CD [B], CL [B]; differentiable in alpha through the geometry and the model's alpha input.
    Midpoint rule over polygon edges: sum(n_e |e|) = 0 exactly, so a constant pressure gives zero force."""
    b = _rotated(nurbs, alpha, cond, aoa_deg, n_pts)
    ex, ey = b.x[:, 1:] - b.x[:, :-1], b.y[:, 1:] - b.y[:, :-1]
    L = torch.sqrt(ex * ex + ey * ey) + 1e-12
    nx, ny = ey / L, -ex / L  # outward for the CCW loop
    mid = torch.stack([b.x[:, 1:] + b.x[:, :-1], b.y[:, 1:] + b.y[:, :-1]], -1) / 2
    f = predict_flow(model, mid, alpha, cond, b, derivs=1)
    d = f.derivs
    nu = torch.exp(-cond[:, :1])  # cond[:, 0] = log Re
    sxy = d["u_y"] + d["v_x"]
    tx = -f.p * nx + nu * (2 * d["u_x"] * nx + sxy * ny)
    ty = -f.p * ny + nu * (sxy * nx + 2 * d["v_y"] * ny)
    return 2 * (tx * L).sum(1), 2 * (ty * L).sum(1)


def surface_cp(model, nurbs, alpha: Tensor, cond: Tensor, aoa_deg=None) -> Tensor:
    """[B, 2M-1] surface pressure coefficient at the loop nodes (rho = U = 1, so Cp = 2p)."""
    b = _rotated(nurbs, alpha, cond, aoa_deg, None)
    return 2 * predict_flow(model, torch.stack([b.x, b.y], -1), alpha, cond, b).p


def pinn_value_and_grad(model, nurbs, alpha: Tensor, cond: Tensor) -> EvalResult:
    """One shape: alpha [D] or [1, D], cond [3] or [1, 3] -> cd, cl (0-dim) and d/dalpha [D]."""
    t0 = time.perf_counter()
    a = alpha.detach().reshape(1, -1).requires_grad_(True)
    cd, cl = surface_forces(model, nurbs, a, cond.reshape(1, 3))
    gd, = torch.autograd.grad(cd.sum(), a, retain_graph=True)
    gl, = torch.autograd.grad(cl.sum(), a)
    sync(a.device)
    return EvalResult(cd[0].detach(), cl[0].detach(), gd[0], gl[0], wall_s=time.perf_counter() - t0)
