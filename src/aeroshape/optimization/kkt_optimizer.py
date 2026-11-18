"""Augmented-Lagrangian shape optimizer (Plan/05_optimization/kkt_optimizer.md, derivations 7.2-7.4)."""
import time

import torch

from aeroshape.core.types import CostWeights

VIOL_TOL = 1e-3


def train_box(anchors):
    """G37: per-coordinate (min, max) of alpha over the PINN's train split (split == 0) of a merged anchor dict."""
    a = anchors["alpha"][anchors["split"] == 0]
    return a.amin(0), a.amax(0)


class AugmentedLagrangianOptimizer:
    def __init__(self, nurbs, evaluator, cond, t_min, cl_ref, cfg, box=None):
        """box: optional (lo, hi) [D]; free coordinates of every trial step are clamped into it (pinned ends untouched)."""
        self.box = box
        o = cfg.optimizer
        self.nurbs, self.ev, self.cond, self.cl_ref, self.o = nurbs, evaluator, cond, cl_ref, o
        self.g_cap, self.cd_rtol = o.g_cap, o.cd_rtol
        self.t_min = t_min
        self.lam, self.mu, self.eta = 0.0, o.mu0, o.eta
        self.alpha, self.records, self.n_flagged = None, [], 0

    def _weights(self):  # no lift constraint -> the CL terms vanish
        return CostWeights(0.0, 0.0, 0.0) if self.cl_ref is None else CostWeights(self.lam, self.o.rho_al, self.cl_ref)

    def _penalty(self, alpha):
        """(mu/2) sum relu(t_min - t)^2 and its gradient on the free entries; max violation."""
        a = alpha.detach().reshape(1, -1).requires_grad_(True)
        v = torch.relu(self.t_min.to(a) - self.nurbs.thickness_at_ctrl(a))
        P = 0.5 * self.mu * (v ** 2).sum()
        (g,) = torch.autograd.grad(P, a)
        return P.detach(), g.reshape(-1) * self.nurbs.mask, float(v.detach().max())

    def step(self) -> dict:
        t0, o, w = time.perf_counter(), self.o, self._weights()
        a = self.alpha.reshape(-1)
        res = self.ev.evaluate(a, self.cond, w)
        P, gP, viol = self._penalty(a)
        L0 = float(res.extras["I"] + P)
        g = (res.extras["dI_dalpha"].reshape(-1) + gP) * self.nurbs.mask
        eta, nb, flagged = min(o.eta, 2 * self.eta), 0, False
        while True:
            trial = a - eta * g / max(1.0, eta * float(g.abs().max()) / self.g_cap)  # largest move <= g_cap
            if self.box is not None:
                lo, hi = (b.to(trial) for b in self.box)
                trial = torch.where(self.nurbs.mask > 0, torch.minimum(torch.maximum(trial, lo), hi), trial)
            cd, cl = self.ev.value(trial, self.cond, w)
            L1 = float(cd + (w.lam * cl + 0.5 * w.rho_al * (cl - w.cl_ref) ** 2 if self.cl_ref is not None else 0)
                       + self._penalty(trial)[0])
            if L1 < L0:
                break
            if nb == 4:
                flagged = True  # keep the smallest step
                break
            eta, nb = eta * o.backtrack, nb + 1
        self.eta = eta
        self.n_flagged = self.n_flagged + 1 if flagged else 0
        cl0 = float(res.cl)
        rec = dict(alpha=a.detach().cpu().clone(), cd=float(res.cd), cl=cl0, L=L0, viol=viol, kkt=float(g.norm()),
                   eta=eta, n_backtrack=nb, flagged=flagged, lam=self.lam, mu=self.mu,
                   t_eval=res.wall_s, t_cycle=time.perf_counter() - t0, **{k: float(res.extras[k]) for k in ("t_forward", "t_adjoint") if k in res.extras})
        if self.cl_ref is not None:
            self.lam += o.rho_al * (float(cl) - self.cl_ref)  # CL at the accepted point (free: computed for the line search)
        if viol > VIOL_TOL:
            self.mu = min(2 * self.mu, o.mu_max)
        self.alpha = trial.detach().reshape(1, -1)
        self.records.append(rec)
        return rec

    def _done(self):
        r, o = self.records, self.o
        c = r[-1]
        feasible = (self.cl_ref is None or abs(c["cl"] - self.cl_ref) < o.cl_tol) and c["viol"] < VIOL_TOL
        if len(r) >= 4 and self.cd_rtol > 0 and all(abs(r[-i]["cd"] - r[-i - 1]["cd"]) < self.cd_rtol * abs(r[-i - 1]["cd"]) for i in (1, 2, 3)):
            return "cd_stalled" if feasible else "stalled_infeasible"
        if self.n_flagged >= o.max_flagged:
            return "stalled"
        return "kkt" if c["kkt"] < o.kkt_tol and feasible else None

    def run(self, alpha0, max_cycles, log=None) -> dict:
        """-> history {alpha [C,D], cd, cl, viol, wall_s, + L, kkt, eta, ...} of the evaluated points; self.alpha is the final shape."""
        self.alpha, self.records, self.lam, self.mu, self.eta = alpha0.detach().reshape(1, -1).clone(), [], 0.0, self.o.mu0, self.o.eta
        self.n_flagged = 0
        stop = "max_cycles"
        for _ in range(max_cycles):
            rec = self.step()
            if log:
                log(rec)
            if (s := self._done()):
                stop = s
                break
        self.stop_reason = stop
        h = {k: [r[k] for r in self.records] for k in self.records[0] if k != "alpha"}
        h = {k: torch.tensor(v, dtype=torch.float64) for k, v in h.items()}
        h["alpha"] = torch.stack([r["alpha"] for r in self.records])
        h["wall_s"] = h.pop("t_cycle")
        return h
