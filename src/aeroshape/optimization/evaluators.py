"""Cost/gradient evaluators for the shape optimizer (Plan/05_optimization/kkt_optimizer.md).
evaluate(alpha, cond, weights) -> EvalResult with extras["I"] (combined cost) and extras["dI_dalpha"] ([D]);
value(alpha, cond, weights) -> (cd, cl) forward only, used for backtracking."""
import math
import time

import torch

from aeroshape.core.types import CostWeights, EvalResult
from aeroshape.lbm import adjoint_solver as A
from aeroshape.pinn.stress_integrator import pinn_value_and_grad, surface_forces


def combined(cd, cl, w: CostWeights):
    return cd + w.lam * cl + 0.5 * w.rho_al * (cl - w.cl_ref) ** 2


class LBMScoreEvaluator:
    """Forward solve only (re-scoring); no gradient. value() warm-starts from the last converged state."""

    def __init__(self, cfg, nurbs, N=None):
        self.cfg, self.nurbs, self.N, self._f = cfg, nurbs, N, None

    def value(self, alpha, cond, weights=None):
        cd, cl, self._f = A.lbm_value(alpha, self.nurbs, cond, self.cfg, self.N, self._f)
        return cd, cl

    def evaluate(self, alpha, cond, weights=None):
        cd, cl = self.value(alpha, cond, weights)
        z = torch.zeros(alpha.numel(), dtype=alpha.dtype, device=alpha.device)
        return EvalResult(cd, cl, z, z.clone(), 0.0, {} if weights is None else {"I": combined(cd, cl, weights)})


class AdjointLBMEvaluator(LBMScoreEvaluator):
    """Forward + ONE adjoint for CD + lam CL + rho/2 (CL - CL_ref)^2; forward and adjoint warm-start from the previous cycle."""

    _F = None

    def evaluate(self, alpha, cond, weights=None):
        weights = weights or CostWeights(0.0, 0.0, 0.0)
        # time-averaged adjoint is a fixed-horizon iteration about an unstable mean flow: warm-starting compounds the growth
        F0 = None if self.cfg.adjoint.mode == "time_averaged" else self._F
        r = A.lbm_value_and_grad(alpha, self.nurbs, cond, self.cfg, weights, self.N, self._f, F0)
        self._f, self._F = r.extras["f"], r.extras["F_star"]
        return r


class FDLBMEvaluator(LBMScoreEvaluator):
    """Fallback when the adjoint gradient is unreliable: central differences of the LBM (cd, cl) over the free control
    points. Cost per evaluate = 1 + 2 * n_free forward solves; perturbed solves warm-start from the base state."""

    def evaluate(self, alpha, cond, weights=None):
        weights = weights or CostWeights(0.0, 0.0, 0.0)
        t0 = time.perf_counter()
        a = alpha.detach().reshape(-1)
        cd, cl = self.value(a, cond, weights)
        t1, f0 = time.perf_counter(), self._f
        h = getattr(self.cfg.optimizer, "fd_delta", 5e-3)
        dcd, dcl = torch.zeros_like(a), torch.zeros_like(a)
        for k in [int(i) for i in self.nurbs.free_idx]:  # loop over design variables, not lattice nodes
            r = []
            for s in (1, -1):
                b = a.clone()
                b[k] += s * h
                r.append(A.lbm_value(b, self.nurbs, cond, self.cfg, self.N, f0)[:2])
            dcd[k], dcl[k] = [(r[0][i] - r[1][i]) / (2 * h) for i in (0, 1)]
        c = weights.lam + weights.rho_al * float(cl - weights.cl_ref)
        wall = time.perf_counter() - t0  # all 1 + 2 * n_free solves
        return EvalResult(cd, cl, dcd, dcl, wall, dict(I=combined(cd, cl, weights), dI_dalpha=dcd + c * dcl,
                                                       t_forward=t1 - t0, wall_s=wall))


class PINNEvaluator:
    def __init__(self, model, nurbs):
        self.model, self.nurbs = model, nurbs

    def _cond(self, alpha, cond):
        r = getattr(self.model, "cond_range", None)  # L1: {log_re: [lo, hi], aoa: [lo, hi]} of the training data, set by the loaders
        if r is not None:
            lre, aoa = math.log(cond.re), cond.aoa_deg
            if not (r["log_re"][0] - 1e-6 <= lre <= r["log_re"][1] + 1e-6 and r["aoa"][0] - 0.5 <= aoa <= r["aoa"][1] + 0.5):
                raise ValueError(f"condition Re {cond.re:g}, AoA {aoa:g} deg is outside the PINN's training range "
                                 f"Re {math.exp(r['log_re'][0]):g}-{math.exp(r['log_re'][1]):g}, AoA {r['aoa'][0]:g}-{r['aoa'][1]:g} deg")
        return cond.as_tensor(alpha.device, alpha.dtype)

    def value(self, alpha, cond, weights=None):
        with torch.enable_grad():
            cd, cl = surface_forces(self.model, self.nurbs, alpha.detach().reshape(1, -1), self._cond(alpha, cond).reshape(1, 3))
        return cd[0].detach(), cl[0].detach()

    def evaluate(self, alpha, cond, weights=None):
        weights = weights or CostWeights(0.0, 0.0, 0.0)
        r = pinn_value_and_grad(self.model, self.nurbs, alpha, self._cond(alpha, cond))
        c = weights.lam + weights.rho_al * float(r.cl - weights.cl_ref)
        r.extras.update(I=combined(r.cd, r.cl, weights), dI_dalpha=r.dcd_dalpha + c * r.dcl_dalpha)
        return r


class HybridEvaluator:
    """R1 fallback: descent direction from grad_eval (PINN), every value and accept/reject from value_eval (LBM)."""

    def __init__(self, grad_eval, value_eval):
        self.grad_eval, self.value_eval = grad_eval, value_eval

    def value(self, alpha, cond, weights=None):
        return self.value_eval.value(alpha, cond, weights)

    def evaluate(self, alpha, cond, weights=None):
        weights = weights or CostWeights(0.0, 0.0, 0.0)
        r = self.grad_eval.evaluate(alpha, cond, weights)
        cd, cl = self.value_eval.value(alpha, cond, weights)
        c = weights.lam + weights.rho_al * float(cl - weights.cl_ref)  # multiplier term at the LBM CL
        return EvalResult(cd, cl, r.dcd_dalpha, r.dcl_dalpha, r.wall_s,
                          dict(I=combined(cd, cl, weights), dI_dalpha=r.dcd_dalpha + c * r.dcl_dalpha))


def build_evaluator(cfg, name: str, nurbs, N=None, model=None):
    """name: adjoint_lbm | fd_lbm | mlp_pinn | transformer_pinn | lbm_score | hybrid_{mlp,transformer}_pinn. PINN names need a loaded `model`."""
    if name == "adjoint_lbm":
        return AdjointLBMEvaluator(cfg, nurbs, N)
    if name == "fd_lbm":
        return FDLBMEvaluator(cfg, nurbs, N)
    if name == "lbm_score":
        return LBMScoreEvaluator(cfg, nurbs, N)
    if name in ("mlp_pinn", "transformer_pinn"):
        assert model is not None, "PINN evaluators need a model (see --ckpt)"
        return PINNEvaluator(model, nurbs)
    if name in ("hybrid_mlp_pinn", "hybrid_transformer_pinn"):
        assert model is not None, "hybrid evaluators need a model (see --ckpt)"
        return HybridEvaluator(PINNEvaluator(model, nurbs), LBMScoreEvaluator(cfg, nurbs, N))
    raise ValueError(name)
