"""PINN input/derivative helpers (Plan/04_models/mlp_pinn.md): wall_distance, predict_flow, shared layers."""
import math

import torch
from torch import nn

from aeroshape.core.types import Boundary, FlowFields

Tensor = torch.Tensor
_PAIRS = 4_000_000  # max points x segments per chunk


@torch.no_grad()
def wall_distance(xy: Tensor, boundary: Boundary) -> Tensor:
    """xy [B, N, 2] -> [B, N] min distance to the boundary polyline (point-to-segment), detached."""
    ax, ay = boundary.x[:, None, :-1], boundary.y[:, None, :-1]
    ex, ey = boundary.x[:, None, 1:] - ax, boundary.y[:, None, 1:] - ay
    L2 = (ex * ex + ey * ey).clamp_min(1e-12)
    B, N, _ = xy.shape
    chunk = max(1, _PAIRS // (B * ax.shape[-1]))
    out = []
    for s in range(0, N, chunk):
        px, py = xy[:, s:s + chunk, 0, None] - ax, xy[:, s:s + chunk, 1, None] - ay
        t = ((px * ex + py * ey) / L2).clamp(0, 1)
        out.append(((px - t * ex) ** 2 + (py - t * ey) ** 2).amin(-1).sqrt())
    return torch.cat(out, 1)


def _grad(f: Tensor, xy: Tensor) -> Tensor:
    # "sum then grad": valid because every output point depends only on its own xy
    return torch.autograd.grad(f.sum(), xy, create_graph=True)[0]


def predict_flow(model, xy: Tensor, alpha: Tensor, cond: Tensor, boundary: Boundary, derivs: int = 0) -> FlowFields:
    """psi -> (u, v) = (psi_y, -psi_x). derivs>=1 adds first derivs of u, v, p (+ fx, fy if the model has them);
    derivs>=2 adds u_xx, u_yy, v_xx, v_yy."""
    if not xy.requires_grad:
        xy = xy.detach().requires_grad_(True)
    out = model(xy, alpha, cond, wall_distance(xy.detach(), boundary))
    g = _grad(out.psi, xy)
    u, v = g[..., 1], -g[..., 0]
    d = None
    if derivs >= 1:
        gu, gv, gp = _grad(u, xy), _grad(v, xy), _grad(out.p, xy)
        d = dict(u_x=gu[..., 0], u_y=gu[..., 1], v_x=gv[..., 0], v_y=gv[..., 1], p_x=gp[..., 0], p_y=gp[..., 1])
        if out.fx is not None:
            d.update(fx=out.fx, fy=out.fy)
        if derivs >= 2:
            d.update(u_xx=_grad(d["u_x"], xy)[..., 0], u_yy=_grad(d["u_y"], xy)[..., 1],
                     v_xx=_grad(d["v_x"], xy)[..., 0], v_yy=_grad(d["v_y"], xy)[..., 1])
    return FlowFields(u=u, v=v, p=out.p, derivs=d)


class FourierFeatures(nn.Module):
    """[..., 2] -> [..., 2F] = [sin(2 pi x B), cos(2 pi x B)], fixed B ~ N(0, scale^2) from a seeded CPU generator."""

    def __init__(self, features: int = 64, scale: float = 2.0, seed: int = 0):
        super().__init__()
        self.register_buffer("B", scale * torch.randn(2, features, generator=torch.Generator().manual_seed(seed)))

    def forward(self, xy: Tensor) -> Tensor:
        z = 2 * math.pi * xy @ self.B
        return torch.cat([z.sin(), z.cos()], -1)


class Wavelet(nn.Module):
    """w1 sin(z) + w2 cos(z) with learnable per-channel w1, w2 (init 1)."""

    def __init__(self, channels: int):
        super().__init__()
        self.w1, self.w2 = nn.Parameter(torch.ones(channels)), nn.Parameter(torch.ones(channels))

    def forward(self, z: Tensor) -> Tensor:
        return self.w1 * z.sin() + self.w2 * z.cos()


class InputNorm(nn.Module):
    """Standardizes alpha [B, D] and cond [B, 3]; stats are buffers (identity until fit), saved in the checkpoint."""

    def __init__(self, D: int, C: int = 3):
        super().__init__()
        for n, k in (("a", D), ("c", C)):
            self.register_buffer(f"{n}_mean", torch.zeros(k))
            self.register_buffer(f"{n}_std", torch.ones(k))

    @torch.no_grad()
    def fit(self, alpha: Tensor, cond: Tensor) -> None:
        for n, t in (("a", alpha), ("c", cond)):
            getattr(self, f"{n}_mean").copy_(t.mean(0))
            getattr(self, f"{n}_std").copy_(t.std(0).clamp_min(1e-2))  # constant training dims (e.g. AoA=0) must not blow up other values at inference

    def forward(self, alpha: Tensor, cond: Tensor) -> tuple[Tensor, Tensor]:
        return (alpha - self.a_mean) / self.a_std, (cond - self.c_mean) / self.c_std
