"""Drag and lift by momentum exchange (derivations 3.1-3.3); Cp for plots (Plan/03_lbm/forces.md)."""
import torch

from aeroshape.core.types import Boundary, ForceHistory, GridGeometry, LBMParams, LinkSet
from aeroshape.geometry.ibb_grid_mapper import to_lattice
from aeroshape.lbm.d2q9_solver import moments
from aeroshape.lbm.lattice import lattice_constants


def momentum_exchange(f_col, f_next, links: LinkSet, Nx) -> torch.Tensor:
    """F = sum_l c_i (f_col_i(x_A) + f_next_opp_i(x_A)) in lattice units -> [2]. f_next holds the IBB value at x_A."""
    Cl, _, opp = lattice_constants(f_col.device, f_col.dtype)
    S = f_col.shape[1] * f_col.shape[2]
    d, n = links.dir, links.node
    s = f_col.reshape(-1)[d * S + n] + f_next.reshape(-1)[opp[d] * S + n]
    return (Cl[d].to(s.dtype) * s[:, None]).sum(0)


def coefficients(force, params: LBMParams):
    cd, cl = 2 * force / (params.u_lb**2 * params.N)  # rho_inf = 1
    return cd, cl


def force_fn(f_col, f_next, geom: GridGeometry, params: LBMParams):
    """Signature expected by d2q9_solver.run."""
    return coefficients(momentum_exchange(f_col, f_next, geom.links, params.Nx), params)


def period_average(history: ForceHistory, period_steps) -> tuple[float, float]:
    """Mean over the largest whole number of periods ending at the last record (all records if < 1 period)."""
    s = history.step
    n = max(int((s[-1] - s[0]) // period_steps), 1)
    m = s > s[-1] - n * period_steps if (s[-1] - s[0]) >= period_steps else torch.ones_like(s, dtype=torch.bool)
    return history.cd[m].mean().item(), history.cl[m].mean().item()


@torch.no_grad()
def surface_pressure_coefficient(state, geom: GridGeometry, params: LBMParams, boundary: Boundary) -> torch.Tensor:
    """Cp = (rho - 1)/3 / (u_lb^2/2), bilinear at the boundary points. shortcut: stencil may include solid nodes (rho=1 there)."""
    rho = state.rho if hasattr(state, "rho") else moments(state)[0]
    xl, yl = to_lattice(boundary, geom.N, geom.origin)
    i0, j0 = xl.floor().long().clamp(0, params.Nx - 2), yl.floor().long().clamp(0, params.Ny - 2)
    tx, ty = (xl - i0).to(rho.dtype), (yl - j0).to(rho.dtype)
    r = (rho[j0, i0] * (1 - tx) + rho[j0, i0 + 1] * tx) * (1 - ty) + (rho[j0 + 1, i0] * (1 - tx) + rho[j0 + 1, i0 + 1] * tx) * ty
    return (r - 1) / 3 / (0.5 * params.u_lb**2)
