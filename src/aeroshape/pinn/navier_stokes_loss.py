"""Hybrid physics + data loss (Plan/04_models/ns_loss_and_sampling.md, derivations 5.2)."""
import math
from dataclasses import dataclass

import torch

from aeroshape.core.types import Boundary, FlowFields
from aeroshape.pinn.embeddings import predict_flow
from aeroshape.pinn.mixing import sym_grad
from aeroshape.pinn.stress_integrator import surface_forces

Tensor = torch.Tensor
TIME_AVG_PDE, TIME_AVG_DATA = 0.1, 3.0  # v2 per-sample multipliers for kind == 1 (w_pde 0.1, w_data 30 vs 10)


@dataclass
class TrainBatch:
    alpha: Tensor  # [B, D]
    cond: Tensor  # [B, 3]
    boundary: Boundary
    xy_colloc: Tensor  # [B, Nc, 2]
    xy_surface: Tensor
    xy_edge: Tensor
    uv_edge: Tensor  # [B, Ne, 2]
    p_edge: Tensor  # [B, Ne]
    xy_anchor: Tensor
    u_anchor: Tensor
    v_anchor: Tensor
    p_anchor: Tensor
    kind: Tensor  # [B] 0 steady, 1 time-averaged
    cd: Tensor | None = None  # [B] LBM targets for the force term (loss.w_cd)
    cl: Tensor | None = None
    gcd: Tensor | None = None  # [B, D] adjoint dCD/dalpha, dCL/dalpha (loss.w_grad)
    gcl: Tensor | None = None


def ns_residuals(fields: FlowFields, nu: Tensor) -> tuple[Tensor, Tensor]:
    """r_x, r_y [B, N]; nu = 1/Re, scalar or [B, 1]."""
    d, u, v = fields.derivs, fields.u, fields.v
    rx = u * d["u_x"] + v * d["u_y"] + d["p_x"] - nu * (d["u_xx"] + d["u_yy"]) - d.get("fx", 0.0)
    ry = u * d["v_x"] + v * d["v_y"] + d["p_y"] - nu * (d["v_xx"] + d["v_yy"]) - d.get("fy", 0.0)
    return rx, ry


def pde_weight(step: int, cfg) -> float:
    """0 during warm-up, then a linear ramp over pde_ramp steps to loss.w_pde."""
    r = (step - cfg.train.warmup_data_only) / max(cfg.train.pde_ramp, 1)
    return cfg.loss.w_pde * min(max(r, 0.0), 1.0)


GRAD_EPS, GRAD_MIN = 1e-8, 1e-3  # a reference gradient with norm < GRAD_MIN (e.g. dCL/dalpha of a symmetric body) is skipped


def _grad_mismatch(gp: Tensor, ga: Tensor) -> Tensor:
    """mean over shapes of (1 - cos) + |gp - ga|^2 / (|ga|^2 + eps); shapes without a usable reference count as 0."""
    fin = torch.isfinite(ga).all(1)
    ga = torch.nan_to_num(ga)
    n2 = (ga ** 2).sum(1)
    ok = (fin & (n2.sqrt() > GRAD_MIN)).to(gp.dtype)
    cos = torch.nn.functional.cosine_similarity(gp, ga, dim=1)
    return (ok * ((1 - cos) + ((gp - ga) ** 2).sum(1) / (n2 + GRAD_EPS))).mean()


def hybrid_loss(model, nurbs, batch: TrainBatch, cfg, step: int, log: bool = False) -> tuple[Tensor, dict]:
    """-> (total, per-term floats). Floats (one host sync) only when log=True, else {}. `nurbs` is only used by the force term (w_cd > 0)."""
    L, v2 = cfg.loss, cfg.pinn.version == "v2"
    b = batch
    one = torch.ones_like(b.cond[:, 0])
    steady = (b.kind == 0).to(one.dtype)
    w_pde_b = steady + (1 - steady) * TIME_AVG_PDE if v2 else one
    w_data_b = L.w_data * (steady + (1 - steady) * TIME_AVG_DATA) if v2 else L.w_data * one
    zero = one.sum() * 0

    ns, ne = b.xy_surface.shape[1], b.xy_edge.shape[1]
    f = predict_flow(model, torch.cat([b.xy_surface, b.xy_edge, b.xy_anchor], 1), b.alpha, b.cond, b.boundary)
    (us, ue, ua), (vs, ve, va), (_, pe, pa) = [(t[:, :ns], t[:, ns:ns + ne], t[:, ns + ne:]) for t in (f.u, f.v, f.p)]

    wall = (us ** 2 + vs ** 2).mean(1)
    edge = ((ue - b.uv_edge[..., 0]) ** 2 + (ve - b.uv_edge[..., 1]) ** 2).mean(1)
    outlet = (b.xy_edge[..., 0] > cfg.pinn.domain_box[1] - 1e-4).to(pe.dtype)  # pressure gauge on the outlet edge
    n_out = outlet.sum(1).clamp_min(1)
    pin = ((pe * outlet).sum(1) / n_out - (b.p_edge * outlet).sum(1) / n_out) ** 2
    data = ((ua - b.u_anchor) ** 2 + (va - b.v_anchor) ** 2 + L.c_p * (pa - b.p_anchor) ** 2).mean(1)

    pde = force = zero
    wp = pde_weight(step, cfg)
    if wp > 0:  # skip the third-derivative graph entirely during warm-up
        fc = predict_flow(model, b.xy_colloc, b.alpha, b.cond, b.boundary, derivs=2)
        rx, ry = ns_residuals(fc, torch.exp(-b.cond[:, :1]))  # cond[:, 0] = log Re
        pde = (w_pde_b * (rx ** 2 + ry ** 2).mean(1)).mean()
        if v2 and "fx" in fc.derivs:
            force = (fc.derivs["fx"] ** 2 + fc.derivs["fy"] ** 2).mean()

    force_data = grad_data = zero
    if L.w_cd > 0 or L.w_grad > 0:  # G29: integrated CD, CL (same code path as Gate 3) against the LBM values
        alpha = b.alpha.detach().requires_grad_(L.w_grad > 0)
        cd, cl = surface_forces(model, nurbs, alpha, b.cond)
        force_data = ((cd - b.cd) ** 2 + (cl - b.cl) ** 2).mean()
        if L.w_grad > 0:  # G30: shape sensitivities against the adjoint-LBM gradients on the free entries
            free = nurbs.free_idx

            def gp(f):
                g = torch.autograd.grad(f.sum(), alpha, create_graph=True)[0]
                # shortcut: in cambered layout the references are the symmetric steady-run gradients (first k columns), so project; true cambered adjoints need a new branch
                return torch.cat([sym_grad(g), torch.zeros_like(g[:, g.shape[1] // 2:])], 1) if nurbs.mode == "cambered" else g
            grad_data = sum(w * _grad_mismatch(gp(f)[:, free], ga[:, free]) for f, ga, w in ((cd, b.gcd, 1.0), (cl, b.gcl, 0.5)))

    wall, edge, pin, data = wall.mean(), edge.mean(), pin.mean(), (w_data_b * data).mean() / L.w_data
    total = wp * pde + L.w_wall * wall + L.w_edge * edge + L.w_pin * pin + L.w_data * data + (L.w_force * force if v2 else 0) + (L.w_cd * force_data if L.w_cd > 0 else 0) + L.w_grad * grad_data
    if not log:
        return total, {}
    vals = torch.stack([total, pde, wall, edge, pin, data, force, force_data, grad_data]).detach().tolist()
    return total, dict(zip(["total", "pde", "wall", "edge", "pin", "data", "force", "force_data", "grad_data"], vals), w_pde=wp)
