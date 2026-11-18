"""Body onto the lattice with differentiable wall fractions q (Plan/02_geometry/grid_mapping_ibb.md).
Lattice node (j, i) = row y, column x; node coordinates are (x, y) = (i, j)."""
import math

import torch

from aeroshape.core.types import Boundary, GridGeometry, LBMParams, LinkSet
from aeroshape.lbm.lattice import C

Tensor = torch.Tensor
_PAIRS = 4_000_000  # max points x edges per chunk


@torch.no_grad()
def points_in_polygon(px: Tensor, py: Tensor, bx: Tensor, by: Tensor, chunk: int | None = None) -> Tensor:
    """px, py [B, N]; closed polygon bx, by [B, E+1] -> [B, N] bool (even-odd rule, no gradient)."""
    B, N = px.shape
    x1, y1, x2, y2 = bx[:, None, :-1], by[:, None, :-1], bx[:, None, 1:], by[:, None, 1:]
    dy = y2 - y1
    dy = torch.where(dy == 0, torch.ones_like(dy), dy)
    chunk = chunk or max(1, _PAIRS // (B * x1.shape[-1]))
    out = []
    for s in range(0, N, chunk):
        x, y = px[:, s:s + chunk, None], py[:, s:s + chunk, None]
        cross = ((y1 > y) != (y2 > y)) & (x < x1 + (y - y1) * (x2 - x1) / dy)
        out.append(cross.sum(-1) % 2 == 1)
    return torch.cat(out, 1)


def to_lattice(boundary: Boundary, N: int, origin: tuple[float, float]) -> tuple[Tensor, Tensor]:
    assert boundary.x.shape[0] == 1, "to_lattice needs B == 1"
    return origin[0] + N * boundary.x[0], origin[1] + N * boundary.y[0]


def _nb(s: Tensor, cy: int, cx: int) -> Tensor:
    """out[j, i] = s[j + cy, i + cx], False outside the grid."""
    Ny, Nx = s.shape
    out = torch.zeros_like(s)
    out[max(0, -cy):Ny - max(0, cy), max(0, -cx):Nx - max(0, cx)] = s[max(0, cy):Ny + min(0, cy), max(0, cx):Nx + min(0, cx)]
    return out


@torch.no_grad()
def solid_mask(xl: Tensor, yl: Tensor, Ny: int, Nx: int, chunk_rows: int = 64) -> Tensor:
    xl, yl = xl.detach().reshape(-1), yl.detach().reshape(-1)
    i0, i1, j0, j1 = (int(v) for v in (xl.min().floor(), xl.max().ceil(), yl.min().floor(), yl.max().ceil()))
    assert i0 >= 2 and j0 >= 2 and i1 <= Nx - 3 and j1 <= Ny - 3, "body closer than 2 cells to the grid edge"
    jj, ii = torch.meshgrid(torch.arange(j0, j1 + 1, device=xl.device), torch.arange(i0, i1 + 1, device=xl.device), indexing="ij")
    inside = points_in_polygon(ii.reshape(1, -1).to(xl.dtype), jj.reshape(1, -1).to(xl.dtype), xl[None], yl[None],
                               chunk=chunk_rows * ii.shape[1])
    solid = torch.zeros(Ny, Nx, dtype=torch.bool, device=xl.device)
    solid[j0:j1 + 1, i0:i1 + 1] = inside.view_as(ii)
    pin = ~solid & _nb(solid, 1, 0) & _nb(solid, -1, 0) & _nb(solid, 0, 1) & _nb(solid, 0, -1)
    return solid | pin


@torch.no_grad()
def find_links(solid: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Fluid nodes whose neighbour along c_i (i = 1..8) is solid -> node [L] (flat), dir [L], far_ok [L]."""
    Ny, Nx = solid.shape
    cs = C.tolist()[1:]
    link = torch.stack([~solid & _nb(solid, cy, cx) for cx, cy in cs])
    far = torch.stack([~_nb(solid, -cy, -cx) for cx, cy in cs])
    k, j, i = link.nonzero(as_tuple=True)
    return j * Nx + i, k + 1, far[k, j, i]


def link_q(node: Tensor, dir: Tensor, xl: Tensor, yl: Tensor, Nx: int, q_min: float = 0.01) -> tuple[Tensor, Tensor]:
    """q = smallest t in [0, 1] where the ray x_A + t c_i meets a surface segment (differentiable).
    seg = hit segment index, -1 where no hit (q falls back to 0.5)."""
    tol = 1e-6
    ax, ay = xl[:-1], yl[:-1]
    ex, ey = xl[1:] - ax, yl[1:] - ay
    xA, yA = (node % Nx).to(xl.dtype), (node // Nx).to(xl.dtype)
    c = C.to(node.device)[dir].to(xl.dtype)
    step = max(1, _PAIRS // ax.numel())
    ts, segs = [], []
    for s in range(0, node.numel(), step):
        cx, cy = c[s:s + step, :1], c[s:s + step, 1:]
        rx, ry = ax - xA[s:s + step, None], ay - yA[s:s + step, None]
        den = cx * ey - cy * ex
        ok = den.abs() > 1e-12
        den = torch.where(ok, den, torch.ones_like(den))
        t, u = (rx * ey - ry * ex) / den, (rx * cy - ry * cx) / den
        valid = ok & (u >= -tol) & (u <= 1 + tol) & (t >= -tol) & (t <= 1 + tol)
        tm, sg = torch.where(valid, t, torch.full_like(t, math.inf)).min(1)
        ts.append(tm)
        segs.append(torch.where(torch.isfinite(tm), sg, torch.full_like(sg, -1)))
    t, seg = torch.cat(ts), torch.cat(segs)
    q = torch.where(seg >= 0, t, torch.full_like(t, 0.5)).clamp(q_min, 1.0)
    return q, seg


def map_to_grid(boundary: Boundary, params: LBMParams, origin: tuple[int, int] | None = None,
                body_offset_d: float = 5) -> GridGeometry:
    """origin (i0, j0) defaults to (body_offset_d * N, Ny // 2), the config placement of the body reference."""
    N = params.N
    origin = origin or (round(body_offset_d * N), params.Ny // 2)
    xl, yl = to_lattice(boundary, N, origin)
    solid = solid_mask(xl, yl, params.Ny, params.Nx)
    node, d, far_ok = find_links(solid)
    q, seg = link_q(node, d, xl, yl, params.Nx)
    return GridGeometry(solid, LinkSet(node, d, q, far_ok, seg), N, origin)
