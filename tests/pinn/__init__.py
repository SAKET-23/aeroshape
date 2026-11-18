"""Shared stand-ins for the PINN tests: analytic Kovasznay model, tiny MLP, synthetic anchors."""
import math

import torch
from torch import nn

from aeroshape.core.config import load_config
from aeroshape.core.types import PINNOutput
from aeroshape.geometry.ibb_grid_mapper import points_in_polygon
from aeroshape.geometry.nurbs import NURBSAirfoil

RE = 40.0


def kovasznay(x, y, re=RE):
    lam = re / 2 - math.sqrt(re ** 2 / 4 + 4 * math.pi ** 2)
    e = torch.exp(lam * x)
    psi = y - e * torch.sin(2 * math.pi * y) / (2 * math.pi)
    u = 1 - e * torch.cos(2 * math.pi * y)
    v = lam / (2 * math.pi) * e * torch.sin(2 * math.pi * y)
    return psi, u, v, (1 - e ** 2) / 2


class KovasznayModel(nn.Module):
    def forward(self, xy, alpha, cond, d_wall):
        psi, _, _, p = kovasznay(xy[..., 0], xy[..., 1])
        return PINNOutput(psi=psi, p=p)


class TinyMLP(nn.Module):
    def __init__(self, D, hidden=64, v2=False):
        super().__init__()
        self.v2 = v2
        self.net = nn.Sequential(nn.Linear(6 + D, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(),
                                 nn.Linear(hidden, 4 if v2 else 2))

    def forward(self, xy, alpha, cond, d_wall):
        N = xy.shape[1]
        z = torch.cat([xy, d_wall[..., None], alpha[:, None].expand(-1, N, -1), cond[:, None].expand(-1, N, -1)], -1)
        o = self.net(z)
        return PINNOutput(psi=o[..., 0], p=o[..., 1], fx=o[..., 2] if self.v2 else None,
                          fy=o[..., 3] if self.v2 else None)


def make_cfg(*overrides):
    return load_config(None, "smoke", list(overrides))


def make_nurbs(device="cpu", dtype=torch.float32):
    return NURBSAirfoil(k=8, M=40, device=device, dtype=dtype)


def thin_alpha(B, D=8, t=0.06):
    a = torch.zeros(B, D)
    a[:, 1:-1] = t * (1 + 0.1 * torch.arange(B)[:, None])
    return a


def synthetic_anchors(S=3, P=3000, seed=0, box=(-1.5, 4.0, -1.5, 1.5)):
    """Anchor dict per data_contracts: uniform-flow values at random fluid points; points on x=box[1] flagged edge."""
    g = torch.Generator().manual_seed(seed)
    nurbs = make_nurbs()
    alpha = thin_alpha(S)
    b = nurbs(alpha)
    xy = torch.stack([box[0] + (box[1] - box[0]) * torch.rand(S, P, generator=g),
                      box[2] + (box[3] - box[2]) * torch.rand(S, P, generator=g)], -1)
    edge = torch.rand(S, P, generator=g) < 0.2
    # edge points: left/right window edges (right edge exercises the pressure gauge)
    right = torch.rand(S, P, generator=g) < 0.5
    xy[..., 0] = torch.where(edge, torch.where(right, torch.tensor(box[1]), torch.tensor(box[0])), xy[..., 0])
    inside = points_in_polygon(xy[..., 0], xy[..., 1], b.x, b.y)
    xy[inside] = torch.tensor([box[0], box[2]])  # fluid-only: move body points to the corner (kept non-edge-free)
    edge = edge & ~inside
    u, v, p = torch.ones(S, P), torch.zeros(S, P), torch.zeros(S, P)  # uniform flow: exact NS, easy to fit
    return {"alpha": alpha, "mode": "symmetric", "re": torch.full((S,), RE), "aoa": torch.zeros(S), "xy": xy,
            "edge": edge, "u": u, "v": v, "p": p, "cd": torch.zeros(S), "cl": torch.zeros(S),
            "split": torch.zeros(S, dtype=torch.int8), "kind": torch.zeros(S, dtype=torch.int8), "meta": {}}
