"""Cost/gradient evaluators for the shape optimizer (Plan/05_optimization/kkt_optimizer.md).
evaluate(alpha, cond, weights) -> EvalResult with extras["I"] (combined cost) and extras["dI_dalpha"] ([D]);
value(alpha, cond, weights) -> (cd, cl) forward only, used for backtracking."""
import math

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
        r = A.lbm_value_and_grad(alpha, self.nurbs, cond, self.cfg, weights, self.N, self._f, self._F)
        self._f, self._F = r.extras["f"], r.extras["F_star"]
        return r


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


def build_evaluator(cfg, name: str, nurbs, N=None, model=None):
    """name: adjoint_lbm | mlp_pinn | transformer_pinn | lbm_score. PINN names need a loaded `model`."""
    if name == "adjoint_lbm":
        return AdjointLBMEvaluator(cfg, nurbs, N)
    if name == "lbm_score":
        return LBMScoreEvaluator(cfg, nurbs, N)
    if name in ("mlp_pinn", "transformer_pinn"):
        assert model is not None, "PINN evaluators need a model (see --ckpt)"
        return PINNEvaluator(model, nurbs)
    raise ValueError(name)
