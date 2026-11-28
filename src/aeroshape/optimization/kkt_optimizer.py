"""Augmented-Lagrangian shape optimizer (Plan/05_optimization/kkt_optimizer.md, derivations 7.2-7.4)."""
import os
import time

import torch

from aeroshape.core.types import CostWeights

VIOL_TOL = 1e-3


def train_box(anchors):
    """G37: per-coordinate (min, max) of alpha over the PINN's train split (split == 0) of a merged anchor dict."""
    a = anchors["alpha"][anchors["split"] == 0]
    return a.amin(0), a.amax(0)


def _warm_objs(ev):
    return [o for o in (ev, getattr(ev, "grad_eval", None), getattr(ev, "value_eval", None)) if o is not None]


def save_atomic(obj, path):
    """torch.save to a tmp file then os.replace, so `path` is never partial."""
    tmp = f"{path}.tmp"
    try:
        torch.save(obj, tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


NEW_KEYS = ("alpha_new", "cd_new", "cl_new", "viol_new", "L_new")  # the cycle's accepted trial point (absent when the step was rejected)


def best_point(records, cl_ref, cl_tol):
    """Best evaluated point over every record's start point and accepted trial point -> dict(alpha [1, D], cd, cycle (1-based),
    point 'start'|'accepted', feasible). Lowest CD among feasible points; if none is, lexicographic (viol, |cl - cl_ref|, cd)."""
    cands = []
    for i, r in enumerate(records):
        cands.append((r["alpha"], r["cd"], r["cl"], r["viol"], i + 1, "start"))
        if r.get("alpha_new") is not None:
            cands.append((r["alpha_new"], r["cd_new"], r["cl_new"], r["viol_new"], i + 1, "accepted"))
    ok = [c for c in cands if c[3] < VIOL_TOL and (cl_ref is None or abs(c[2] - cl_ref) < cl_tol)]
    c = min(ok, key=lambda c: c[1]) if ok else min(cands, key=lambda c: (c[3], abs(c[2] - cl_ref) if cl_ref is not None else 0, c[1]))
    return dict(alpha=c[0].reshape(1, -1), cd=c[1], cycle=c[4], point=c[5], feasible=bool(ok))


class AugmentedLagrangianOptimizer:
    def __init__(self, nurbs, evaluator, cond, t_min, cl_ref, cfg, box=None):
        """box: optional (lo, hi) [D]; free coordinates of every trial step are clamped into it (pinned ends untouched)."""
        self.box = box
        o = cfg.optimizer
        self.nurbs, self.ev, self.cond, self.cl_ref, self.o = nurbs, evaluator, cond, cl_ref, o
        self.g_cap, self.cd_rtol = o.g_cap, o.cd_rtol
        self.t_min = t_min
        self.lam, self.mu, self.eta, self.cap_scale = 0.0, o.mu0, o.eta, 1.0
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
        eta, nb, flagged, new = min(o.eta, 2 * self.eta), 0, False, {}
        finite = bool(torch.isfinite(g).all())
        warm = [(o_, {k: getattr(o_, k) for k in ("_f", "_F") if isinstance(getattr(o_, k, None), torch.Tensor)}) for o_ in _warm_objs(self.ev)]
        while finite:  # a NaN/Inf gradient: no trial solves, flag at once
            cap = self.g_cap * self.cap_scale * o.backtrack ** nb  # the cap shrinks with every retry, else a capped step repeats the same trial
            trial = a - eta * g / max(1.0, eta * float(g.abs().max()) / cap)  # largest move <= cap
            if self.box is not None:
                lo, hi = (b.to(trial) for b in self.box)
                trial = torch.where(self.nurbs.mask > 0, torch.minimum(torch.maximum(trial, lo), hi), trial)
            cd, cl = self.ev.value(trial, self.cond, w)
            Pt, _, vt = self._penalty(trial)
            L1 = float(cd + (w.lam * cl + 0.5 * w.rho_al * (cl - w.cl_ref) ** 2 if self.cl_ref is not None else 0) + Pt)
            if L1 < L0:
                new = dict(alpha_new=trial.detach().cpu().clone(), cd_new=float(cd), cl_new=float(cl), viol_new=vt, L_new=L1)
                break
            if nb == 4:  # never accept a worse point: stay, and put the warm start back to the solve at `a`
                flagged, trial = True, a.detach().clone()
                for o_, d in warm:
                    for k, v in d.items():
                        setattr(o_, k, v)
                break
            eta, nb = eta * o.backtrack, nb + 1
        if not finite:
            flagged, trial, eta = True, a.detach().clone(), self.eta
        else:  # flagged: keep shrinking from where the line search ended; accepted: recover
            self.cap_scale = self.cap_scale * o.backtrack ** (nb + 1) if flagged else min(1.0, 2 * self.cap_scale)
        self.eta = eta
        self.n_flagged = self.n_flagged + 1 if flagged else 0
        cl0 = float(res.cl)
        rec = dict(alpha=a.detach().cpu().clone(), cd=float(res.cd), cl=cl0, L=L0, viol=viol, kkt=float(g.norm()),
                   eta=eta, n_backtrack=nb, flagged=flagged, lam=self.lam, mu=self.mu, **new,
                   t_eval=res.wall_s, t_cycle=time.perf_counter() - t0, **{k: float(res.extras[k]) for k in ("t_forward", "t_adjoint") if k in res.extras})
        if self.cl_ref is not None:
            cl_acc = cl0 if flagged else float(cl)  # CL at the accepted point (free: computed for the line search)
            lim = o.lam_step_max  # one noisy CL must not move lam far
            self.lam += max(-lim, min(lim, o.rho_al * (cl_acc - self.cl_ref)))
        # raise mu only if infeasible and the violation is not already shrinking (<= 0.75x the previous cycle's; first cycle: no history -> raise)
        if viol > VIOL_TOL and not (self.records and viol <= 0.75 * self.records[-1]["viol"]):
            self.mu = min(2 * self.mu, o.mu_max)
        self.alpha = trial.detach().reshape(1, -1)
        self.records.append(rec)
        return rec

    @property
    def best_alpha(self):
        """Best point by best_point() (start and accepted-trial points of every record). [1, D]."""
        return best_point(self.records, self.cl_ref, self.o.cl_tol)["alpha"].to(self.nurbs.mask)

    def _done(self):
        r, o = self.records, self.o
        c = r[-1]
        feasible = (self.cl_ref is None or abs(c["cl"] - self.cl_ref) < o.cl_tol) and c["viol"] < VIOL_TOL
        if len(r) >= 4 and self.cd_rtol > 0 and all(abs(r[-i]["cd"] - r[-i - 1]["cd"]) < self.cd_rtol * abs(r[-i - 1]["cd"]) for i in (1, 2, 3)):
            return "cd_stalled" if feasible else "stalled_infeasible"
        if self.n_flagged >= o.max_flagged:
            return "stalled"
        return "kkt" if c["kkt"] < o.kkt_tol and feasible else None

    def save_state(self, path):
        """Atomic per-cycle checkpoint: everything run() needs to continue (warm starts saved on CPU)."""
        rng = dict(cpu=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
        warm = [{k: getattr(o, k).cpu() for k in ("_f", "_F") if isinstance(getattr(o, k, None), torch.Tensor)} for o in _warm_objs(self.ev)]
        save_atomic(dict(alpha=self.alpha.cpu(), lam=self.lam, mu=self.mu, eta=self.eta, cap_scale=self.cap_scale, n_flagged=self.n_flagged,
                         records=self.records, cycle=len(self.records), wall_s=sum(r["t_cycle"] for r in self.records),
                         rng=rng, warm=warm), path)

    def load_state(self, path):
        st = torch.load(path, map_location="cpu", weights_only=False)
        dev = self.nurbs.mask.device
        self.alpha, self.lam, self.mu, self.eta, self.n_flagged = st["alpha"].to(self.nurbs.mask), st["lam"], st["mu"], st["eta"], st["n_flagged"]
        self.records, self.wall_prior, self.cap_scale = st["records"], st["wall_s"], st.get("cap_scale", 1.0)
        torch.set_rng_state(st["rng"]["cpu"])
        if st["rng"]["cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(st["rng"]["cuda"])
        for o, w in zip(_warm_objs(self.ev), st["warm"]):
            for k, v in w.items():
                setattr(o, k, v.to(dev))

    def run(self, alpha0, max_cycles, log=None, state_path=None, resume=False) -> dict:
        """-> history {alpha [C,D], best_alpha [D], cd, cl, viol, wall_s, + L, kkt, eta, ...} of the evaluated points; self.alpha is the final shape.
        state_path: checkpoint after every cycle; resume=True continues from it if it exists (else starts fresh)."""
        self.wall_prior = 0.0
        if resume and state_path and os.path.exists(state_path):
            self.load_state(state_path)
            print(f"resumed at cycle {len(self.records) + 1}", flush=True)
        else:
            self.alpha, self.records, self.lam, self.mu, self.eta = alpha0.detach().reshape(1, -1).clone(), [], 0.0, self.o.mu0, self.o.eta
            self.cap_scale = 1.0
            self.n_flagged = 0
        stop = "max_cycles"
        if self.records and (s := self._done()):  # the checkpointed cycle may already have met a stop rule
            stop, max_cycles = s, 0
        for _ in range(max_cycles - len(self.records)):
            rec = self.step()
            if state_path:
                self.save_state(state_path)
            if log:
                log(rec)
            if (s := self._done()):
                stop = s
                break
        self.stop_reason = stop
        h = {k: [r[k] for r in self.records] for k in self.records[0] if k != "alpha" and k not in NEW_KEYS}
        h = {k: torch.tensor(v, dtype=torch.float64) for k, v in h.items()}
        h["alpha"] = torch.stack([r["alpha"] for r in self.records])
        h["best_alpha"] = self.best_alpha.reshape(-1).cpu()
        h["wall_s"] = h.pop("t_cycle")
        return h
