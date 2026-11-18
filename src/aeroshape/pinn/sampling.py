"""Per-step training batch sampling (Plan/04_models/ns_loss_and_sampling.md). Random numbers come from a CPU
generator and are then moved, so batches are reproducible across devices."""
import math

import torch

from aeroshape.geometry.ibb_grid_mapper import points_in_polygon
from aeroshape.geometry.nurbs import rotate
from aeroshape.core.types import Boundary
from aeroshape.pinn.navier_stokes_loss import TrainBatch

Tensor = torch.Tensor
D_MIN, D_MAX, JITTER, OVERSAMPLE = 0.002, 0.5, 0.1, 4


def inside_body(xy: Tensor, boundary: Boundary) -> Tensor:
    """[B, N] bool, same even-odd test as the grid mapper."""
    return points_in_polygon(xy[..., 0], xy[..., 1], boundary.x, boundary.y)


def _rand(g, *shape, like):
    return torch.rand(*shape, generator=g).to(device=like.device, dtype=like.dtype)


def near_wall(boundary: Boundary, n: int, g, d_min: float = D_MIN, d_max: float = D_MAX):
    """Offset random surface points along the normal by d = d_min (d_max/d_min)^U (density ~ 1/d) plus a small
    tangential jitter. -> xy [B, n, 2], d [B, n] (before any rejection)."""
    B, E = boundary.x.shape
    idx = torch.randint(E, (B, n), generator=g).to(boundary.x.device)
    x, y, nx, ny = (t.gather(1, idx) for t in (boundary.x, boundary.y, boundary.nx, boundary.ny))
    d = d_min * (d_max / d_min) ** _rand(g, B, n, like=x)
    j = JITTER * d * torch.randn(B, n, generator=g).to(x)
    return torch.stack([x + nx * d - ny * j, y + ny * d + nx * j], -1), d


def _first_valid(cand: Tensor, n: int, box, boundary: Boundary) -> Tensor:
    """First n candidates per shape that are inside the window and outside the body."""
    x, y = cand[..., 0], cand[..., 1]
    bad = inside_body(cand, boundary) | (x < box[0]) | (x > box[1]) | (y < box[2]) | (y > box[3])
    order = torch.argsort(bad.to(torch.uint8), dim=1, stable=True)[:, :n]
    out = cand.gather(1, order[..., None].expand(-1, -1, 2))
    # shortcut: the (vanishingly rare) shortfall is filled with a window corner, upgrade if bodies get huge
    return torch.where(bad.gather(1, order)[..., None], cand.new_tensor([box[0], box[2]]), out)


def _surface_points(boundary: Boundary, n: int, g) -> Tensor:
    """Random arc-length positions on the loop, linearly interpolated (differentiable in alpha)."""
    x, y = boundary.x, boundary.y
    seg = torch.sqrt((x[:, 1:] - x[:, :-1]) ** 2 + (y[:, 1:] - y[:, :-1]) ** 2).clamp_min(1e-12)
    cum = torch.cat([seg.new_zeros(seg.shape[0], 1), seg.cumsum(1)], 1)
    t = _rand(g, x.shape[0], n, like=x) * cum[:, -1:].detach()
    j = (torch.searchsorted(cum.detach(), t.detach(), right=True) - 1).clamp(0, seg.shape[1] - 1)
    f = ((t - cum.gather(1, j)) / seg.gather(1, j)).clamp(0, 1)
    pick = lambda a: a.gather(1, j) + f * (a.gather(1, j + 1) - a.gather(1, j))
    return torch.stack([pick(x), pick(y)], -1)


def _subset(mask: Tensor, n: int, g) -> Tensor:
    """[B, n] random indices (no repeats) among True entries of mask [B, P] (CPU)."""
    if int(mask.sum(1).min()) < n:
        raise ValueError(f"anchors have fewer than {n} points of the requested kind")
    score = torch.rand(mask.shape, generator=g) + 2 * (~mask)
    return score.topk(n, largest=False).indices


def sample_batch(anchors: dict, nurbs, cfg, generator, device=None) -> TrainBatch:
    """Shapes drawn at random (with replacement) from the training split of `anchors`."""
    S, sm, g = cfg.train.shapes_per_step, cfg.sampling, generator
    dev = torch.device(device) if device is not None else nurbs.x_ctrl.device
    box = cfg.pinn.domain_box
    train = (anchors["split"] == 0).nonzero()[:, 0]
    pick = train[torch.randint(len(train), (S,), generator=g)]
    alpha = anchors["alpha"][pick].to(dev, nurbs.x_ctrl.dtype)
    aoa, re = anchors["aoa"][pick], anchors["re"][pick]
    cond = torch.stack([re.log(), torch.deg2rad(aoa).sin(), torch.deg2rad(aoa).cos()], -1).to(alpha)
    nb = nurbs(alpha)
    rows = [rotate(Boundary(*(t[i:i + 1] for t in (nb.x, nb.y, nb.nx, nb.ny, nb.ds))), float(a)) for i, a in enumerate(aoa)]
    bd = Boundary(*(torch.cat([getattr(r, k) for r in rows]) for k in ("x", "y", "nx", "ny", "ds")))

    n_wall = round(sm.n_collocation * sm.wall_fraction)
    n_uni = sm.n_collocation - n_wall
    lo, span = bd.x.new_tensor([box[0], box[2]]), bd.x.new_tensor([box[1] - box[0], box[3] - box[2]])
    uni = lo + span * _rand(g, S, OVERSAMPLE * n_uni, 2, like=bd.x)
    wall, _ = near_wall(bd, OVERSAMPLE * n_wall, g)
    xy_colloc = torch.cat([_first_valid(uni, n_uni, box, bd), _first_valid(wall, n_wall, box, bd)], 1)

    def gather(key, idx):
        t = anchors[key][pick]
        return t.gather(1, idx[..., None].expand(-1, -1, 2)) if t.dim() == 3 else t.gather(1, idx)

    edge = anchors["edge"][pick]
    near = anchors["near"][pick] & ~edge if "near" in anchors else None  # optional: points close to the wall (WP 15)
    n_near = round(sm.n_anchor * getattr(sm, "near_frac", 0.0)) if near is not None else 0
    ie = _subset(edge, sm.n_edge, g)
    ia = _subset(~edge, sm.n_anchor, g) if n_near == 0 else torch.cat(
        [_subset(near, n_near, g), _subset(~edge & ~near, sm.n_anchor - n_near, g)], 1)
    mv = lambda t: t.to(dev, alpha.dtype)
    return TrainBatch(
        alpha=alpha, cond=cond, boundary=bd, xy_colloc=xy_colloc, xy_surface=_surface_points(bd, sm.n_surface, g),
        xy_edge=mv(gather("xy", ie)), uv_edge=mv(torch.stack([gather("u", ie), gather("v", ie)], -1)),
        p_edge=mv(gather("p", ie)), xy_anchor=mv(gather("xy", ia)), u_anchor=mv(gather("u", ia)),
        v_anchor=mv(gather("v", ia)), p_anchor=mv(gather("p", ia)), kind=anchors["kind"][pick].to(dev),
        cd=anchors["cd"][pick].to(dev, alpha.dtype), cl=anchors["cl"][pick].to(dev, alpha.dtype),
        **{k: anchors[k][pick].to(dev, alpha.dtype) for k in ("gcd", "gcl") if k in anchors})
