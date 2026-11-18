"""Differentiable NURBS airfoil (Plan/02_geometry/nurbs.md)."""
import math
from typing import Literal

import torch
from torch import nn

from aeroshape.core.types import Boundary

Tensor = torch.Tensor
_TABLE = 4096  # fixed fine s-table for thickness_at_ctrl


def _div(num: Tensor, den: Tensor) -> Tensor:
    """num/den with 0/0 := 0."""
    return torch.where(den == 0, torch.zeros_like(num), num / torch.where(den == 0, torch.ones_like(den), den))


def basis_matrix(k: int, degree: int, s: Tensor, weights: Tensor) -> tuple[Tensor, Tensor]:
    """Rational B-spline basis R [M, k] and dR/ds [M, k]; clamped uniform knots (derivations 1.1-1.3)."""
    p = degree
    s = s.to(torch.float64).cpu()
    w = weights.to(torch.float64).cpu()
    t = torch.cat([torch.zeros(p), torch.linspace(0, 1, k - p + 1, dtype=torch.float64), torch.ones(p)]).to(torch.float64)
    sc = s[:, None]
    N = ((t[:-1] <= sc) & (sc < t[1:])).to(torch.float64)  # [M, k+p]
    N[s >= 1.0, k - 1] = 1.0  # close the last span on the right
    prev = N
    for q in range(1, p + 1):
        n = k + p - q
        prev = N
        N = _div(sc - t[:n], t[q:q + n] - t[:n]) * N[:, :n] + _div(t[q + 1:q + n + 1] - sc, t[q + 1:q + n + 1] - t[1:n + 1]) * N[:, 1:n + 1]
    dN = p * (_div(prev[:, :k], t[p:p + k] - t[:k]) - _div(prev[:, 1:k + 1], t[p + 1:p + k + 1] - t[1:k + 1]))
    S, dS = (N * w).sum(1, keepdim=True), (dN * w).sum(1, keepdim=True)
    return N * w / S, (dN * w * S - N * w * dS) / S**2


class NURBSAirfoil(nn.Module):
    def __init__(self, k: int = 8, degree: int = 3, M: int = 160,
                 mode: Literal["symmetric", "cambered"] = "symmetric",
                 weights: Tensor | None = None, device=None, dtype=None):
        super().__init__()
        dtype = dtype or torch.get_default_dtype()
        self.k, self.M, self.mode = k, M, mode
        w = torch.ones(k, dtype=torch.float64) if weights is None else weights
        xc = torch.zeros(k, dtype=torch.float64)  # [0, 0, cosine interior..., 1]
        j = torch.arange(1, k - 2, dtype=torch.float64) / (k - 2)
        xc[2:k - 1] = (1 - torch.cos(math.pi * j)) / 2
        xc[-1] = 1.0
        s = (1 - torch.cos(math.pi * torch.arange(M, dtype=torch.float64) / (M - 1))) / 2
        R, dR = basis_matrix(k, degree, s, w)
        # thickness table: s -> x(s) is monotone; fold interpolation at x_ctrl into a [k, k] matrix
        Rf, _ = basis_matrix(k, degree, torch.linspace(0, 1, _TABLE, dtype=torch.float64), w)
        xt = Rf @ xc
        i = (torch.searchsorted(xt, xc) - 1).clamp(0, _TABLE - 2)
        f = ((xc - xt[i]) / (xt[i + 1] - xt[i])).clamp(0, 1)[:, None]
        Tc = (1 - f) * Rf[i] + f * Rf[i + 1]
        m = torch.ones(k, dtype=torch.float64)
        m[0] = m[-1] = 0
        D = k if mode == "symmetric" else 2 * k
        mask = m if mode == "symmetric" else torch.cat([m, m])
        self.n_free = int(mask.sum())
        for name, v in dict(x_ctrl=xc, R=R, dR=dR, Tc=Tc, mask=mask).items():
            self.register_buffer(name, v.to(device=device, dtype=dtype))
        self.register_buffer("free_idx", mask.nonzero()[:, 0].to(device))
        self.D = D

    def _split(self, alpha: Tensor) -> tuple[Tensor, Tensor]:
        a = alpha * self.mask
        return (a, -a) if self.mode == "symmetric" else (a[:, :self.k], a[:, self.k:])

    def surface_y(self, alpha: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        au, al = self._split(alpha)
        return self.R @ self.x_ctrl, au @ self.R.T, al @ self.R.T

    def forward(self, alpha: Tensor) -> Boundary:
        M = self.M
        au, al = self._split(alpha)
        B = au.shape[0]
        x = (self.R @ self.x_ctrl).expand(B, M)
        dx = (self.dR @ self.x_ctrl).expand(B, M)
        yu, yl = au @ self.R.T, al @ self.R.T
        dyu, dyl = au @ self.dR.T, al @ self.dR.T
        # CCW loop: TE -> upper surface (s descending) -> LE -> lower surface -> TE.
        # (The spec says lower first, but that traversal is clockwise for y up.)
        # Tangent of the upper part is reversed; unit outward normals n = (Ty, -Tx)/|T|.
        def unit(tx, ty):
            n = torch.sqrt(tx * tx + ty * ty) + 1e-12
            return ty / n, -tx / n
        nux, nuy = unit(-dx, -dyu)
        nlx, nly = unit(dx, dyl)
        ax, ay = nux[:, :1] + nlx[:, :1], nuy[:, :1] + nly[:, :1]  # LE: average the two one-sided normals
        an = torch.sqrt(ax * ax + ay * ay) + 1e-12
        nx = torch.cat([nux.flip(1)[:, :M - 1], ax / an, nlx[:, 1:]], 1)
        ny = torch.cat([nuy.flip(1)[:, :M - 1], ay / an, nly[:, 1:]], 1)
        X = torch.cat([x.flip(1), x[:, 1:]], 1)
        Y = torch.cat([yu.flip(1), yl[:, 1:]], 1)
        seg = torch.sqrt((X[:, 1:] - X[:, :-1]) ** 2 + (Y[:, 1:] - Y[:, :-1]) ** 2)
        z = seg.new_zeros(B, 1)
        ds = 0.5 * (torch.cat([z, seg], 1) + torch.cat([seg, z], 1))
        return Boundary(x=X, y=Y, nx=nx, ny=ny, ds=ds)

    def thickness_at_ctrl(self, alpha: Tensor) -> Tensor:
        au, al = self._split(alpha)
        return (au - al) @ self.Tc.T

    def area(self, alpha: Tensor) -> Tensor:
        b = self(alpha)
        return 0.5 * (b.x[:, :-1] * b.y[:, 1:] - b.x[:, 1:] * b.y[:, :-1]).sum(1)

    def control_tokens(self, alpha: Tensor) -> Tensor:
        """[B, 2k, 3] = (x_c, y_c, surface_id) with upper (0) then lower (1); symmetric lower = -upper."""
        au, al = self._split(alpha)
        B, k = au.shape
        x = self.x_ctrl.expand(B, k)
        sid = torch.cat([torch.zeros_like(x), torch.ones_like(x)], 1)
        return torch.stack([torch.cat([x, x], 1), torch.cat([au, al], 1), sid], -1)

    def free_to_full(self, z: Tensor) -> Tensor:
        return z.new_zeros(z.shape[0], self.D).index_copy(1, self.free_idx, z)

    def full_to_free(self, alpha: Tensor) -> Tensor:
        return alpha[:, self.free_idx]


def rotate(boundary: Boundary, aoa_deg: float, pivot=(0.25, 0.0)) -> Boundary:
    """Rotate the body by -AoA about pivot (inflow stays +x); normals rotate, ds unchanged."""
    c, s = math.cos(math.radians(aoa_deg)), -math.sin(math.radians(aoa_deg))
    px, py = pivot
    b = boundary
    return Boundary(x=px + c * (b.x - px) - s * (b.y - py), y=py + s * (b.x - px) + c * (b.y - py),
                    nx=c * b.nx - s * b.ny, ny=s * b.nx + c * b.ny, ds=b.ds)
