"""WP9: augmented-Lagrangian optimizer on analytic evaluators (Plan/05_optimization/kkt_optimizer.md)."""
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from aeroshape.core.config import Config
from aeroshape.core.types import Condition, EvalResult
from aeroshape.geometry.nurbs import NURBSAirfoil
from aeroshape.optimization.kkt_optimizer import AugmentedLagrangianOptimizer

F64 = torch.float64
ROOT = Path(__file__).parents[2]
COND = Condition(20, 0)
NURBS = NURBSAirfoil(k=8, dtype=F64)
D = NURBS.D
NOPEN = torch.full((8,), -10.0, dtype=F64)  # thickness penalty inactive


def make_cfg(**kw):
    o = dict(max_cycles=400, eta=0.05, backtrack=0.5, rho_al=10.0, mu0=10.0, mu_max=1e4, kkt_tol=1e-9, cl_tol=1e-9,
             g_cap=10.0, cd_rtol=0.0, max_flagged=2)
    o.update(kw)
    return Config({"optimizer": o})


class Analytic:
    """CD = ||a - a*||^2 (+ cd_lin . a), CL = w . a, on the free entries; gradient deliberately includes pinned entries."""

    def __init__(self, a_star, w, lin=None):
        self.a_star, self.w = a_star, w
        self.lin = torch.zeros(D, dtype=F64) if lin is None else lin

    def value(self, alpha, cond, weights):
        a = alpha.reshape(-1)
        return ((a - self.a_star) ** 2).sum() + self.lin @ a, self.w @ a

    def evaluate(self, alpha, cond, weights):
        a = alpha.reshape(-1)
        cd, cl = self.value(a, cond, weights)
        dcd, dcl = 2 * (a - self.a_star) + self.lin, self.w.clone()
        c = weights.lam + weights.rho_al * (cl - weights.cl_ref)
        return EvalResult(cd, cl, dcd, dcl, 0.0, dict(I=cd + weights.lam * cl + 0.5 * weights.rho_al * (cl - weights.cl_ref) ** 2,
                                                      dI_dalpha=dcd + c * dcl))


def vec(*v):
    out = torch.zeros(D, dtype=F64)
    out[1:1 + len(v)] = torch.tensor(v, dtype=F64)
    return out


def test_quadratic_with_linear_equality_hits_kkt_point():
    a_star, w, c = vec(0.3, -0.2, 0.1, 0.4, 0.2, -0.1), vec(1.0, 0.5, -0.3, 0.8, 0.2, 0.1), 0.25
    ev = Analytic(a_star, w)
    opt = AugmentedLagrangianOptimizer(NURBS, ev, COND, NOPEN, c, make_cfg())
    h = opt.run(torch.zeros(1, D, dtype=F64), 400)
    expected = a_star - (w @ a_star - c) / (w @ w) * w
    err = (opt.alpha.reshape(-1) - expected).abs().max()
    print(f"\nKKT err {err:.2e} after {len(h['cd'])} cycles")
    assert err < 1e-4
    assert abs(float(w @ opt.alpha.reshape(-1)) - c) < 1e-4


def test_thickness_active_ends_feasible():
    t_min = torch.zeros(8, dtype=F64)
    t_min[2:6] = 0.05
    ev = Analytic(torch.zeros(D, dtype=F64), torch.zeros(D, dtype=F64), lin=vec(0.02, 0.02, 0.02, 0.02, 0.02, 0.02))
    opt = AugmentedLagrangianOptimizer(NURBS, ev, COND, t_min, None, make_cfg(eta=0.05, kkt_tol=1e-6, max_cycles=300))
    h = opt.run(vec(0.06, 0.06, 0.06, 0.06, 0.06, 0.06)[None], 300)
    print(f"\nfinal violation {h['viol'][-1]:.2e}, mu {opt.mu:.1f}")
    assert h["viol"][-1] < 1e-3


def test_pinned_ends_never_move():
    ev = Analytic(vec(0.3, 0.3, 0.3, 0.3, 0.3, 0.3) + 1.0 * (torch.arange(D) % 7 == 0).to(F64), torch.ones(D, dtype=F64))
    opt = AugmentedLagrangianOptimizer(NURBS, ev, COND, NOPEN, 0.1, make_cfg(max_cycles=20))
    h = opt.run(torch.zeros(1, D, dtype=F64), 20)
    a = h["alpha"]
    assert (a[:, 0] == 0).all() and (a[:, -1] == 0).all() and (opt.alpha[0, [0, -1]] == 0).all()
    assert a[-1].abs().max() > 0  # but free entries did move


def test_backtracking_halves_eta_and_decreases_L():
    ev = Analytic(vec(0.3, 0.3, 0.3, 0.3, 0.3, 0.3), torch.zeros(D, dtype=F64))
    # eta = 0.9 on CD = |a|^2 has gradient 2(a - a*): step factor 1 - 1.8 = -0.8 is fine, so use eta = 1.2 (factor -1.4, overshoots)
    opt = AugmentedLagrangianOptimizer(NURBS, ev, COND, NOPEN, None, make_cfg(eta=1.2))
    opt.alpha = torch.zeros(1, D, dtype=F64)
    r = opt.step()
    assert r["eta"] < 1.2 and r["eta"] == pytest.approx(1.2 * 0.5 ** r["n_backtrack"]) and r["n_backtrack"] >= 1
    assert not r["flagged"]
    a1 = opt.alpha.reshape(-1)
    assert float(ev.value(a1, COND, None)[0]) < float(ev.value(torch.zeros(D, dtype=F64), COND, None)[0])


def test_backtracking_failure_accepts_smallest_step_flagged():
    class Ascent(Analytic):  # gradient points the wrong way, so no step ever decreases L
        def evaluate(self, alpha, cond, weights):
            r = super().evaluate(alpha, cond, weights)
            return EvalResult(r.cd, r.cl, -r.dcd_dalpha, r.dcl_dalpha, 0.0, dict(I=r.extras["I"], dI_dalpha=-r.extras["dI_dalpha"]))
    ev = Ascent(vec(0.3, 0.3, 0.3, 0.3, 0.3, 0.3), torch.zeros(D, dtype=F64))
    opt = AugmentedLagrangianOptimizer(NURBS, ev, COND, NOPEN, None, make_cfg(eta=0.1))
    opt.alpha = torch.zeros(1, D, dtype=F64)
    r = opt.step()
    assert r["flagged"] and r["eta"] == pytest.approx(0.1 * 0.5 ** 4)


def test_move_cap_allows_g_cap_per_cycle():  # H1: distance ~0.4 away, moves must reach g_cap and never exceed it
    g_cap = 0.02
    ev = Analytic(vec(*[0.4] * 6), torch.zeros(D, dtype=F64))
    opt = AugmentedLagrangianOptimizer(NURBS, ev, COND, NOPEN, None, make_cfg(g_cap=g_cap))
    h = opt.run(torch.zeros(1, D, dtype=F64), 5)
    a = torch.cat([torch.zeros(1, D, dtype=F64), h["alpha"].reshape(-1, D)[1:], opt.alpha], 0)
    moves = (a[1:] - a[:-1]).abs().amax(1)
    assert (moves <= g_cap * (1 + 1e-9)).all()
    assert float(opt.alpha.abs().max()) >= 5 * g_cap * 0.9 and not h["flagged"].any()


def test_stalled_after_max_flagged_cycles():  # M5
    class Ascent(Analytic):
        def evaluate(self, alpha, cond, weights):
            r = super().evaluate(alpha, cond, weights)
            return EvalResult(r.cd, r.cl, -r.dcd_dalpha, r.dcl_dalpha, 0.0, dict(I=r.extras["I"], dI_dalpha=-r.extras["dI_dalpha"]))
    ev = Ascent(vec(0.3, 0.3, 0.3, 0.3, 0.3, 0.3), torch.zeros(D, dtype=F64))
    opt = AugmentedLagrangianOptimizer(NURBS, ev, COND, NOPEN, None, make_cfg(max_flagged=2))
    h = opt.run(torch.zeros(1, D, dtype=F64), 10)
    assert opt.stop_reason == "stalled" and len(h["cd"]) == 2 and h["flagged"].all()


def test_cd_stalled_only_when_feasible():  # M7
    ev = Analytic(vec(0.3, 0.3, 0.3, 0.3, 0.3, 0.3), vec(1.0, 1.0))
    cfg = make_cfg(cd_rtol=10.0, eta=1e-6)  # CD "stalls" immediately
    opt = AugmentedLagrangianOptimizer(NURBS, ev, COND, NOPEN, 50.0, cfg)  # CL ref unreachable -> infeasible
    opt.run(torch.zeros(1, D, dtype=F64), 10)
    assert opt.stop_reason == "stalled_infeasible"
    opt = AugmentedLagrangianOptimizer(NURBS, ev, COND, NOPEN, None, cfg)  # no constraint -> feasible
    opt.run(torch.zeros(1, D, dtype=F64), 10)
    assert opt.stop_reason == "cd_stalled"


@pytest.mark.slow
def test_smoke_run_case1_via_script(tmp_path):
    cmd = [sys.executable, str(ROOT / "scripts" / "06_optimize_case.py"), "--case", "case1_circle_to_naca_re20.yaml",
           "--evaluator", "adjoint_lbm", "--tier", "smoke", "--device", "cpu", "--out", str(tmp_path),
           "--set", "lbm.domain_d=[10,5]"]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=900)
    print(r.stdout[-1500:])
    assert r.returncode == 0, r.stderr[-2000:]
    h = torch.load(tmp_path / "history.pt")
    print("cd per cycle", h["cd"].tolist(), "eta", h["eta"].tolist())
    assert len(h["cd"]) == 2 and h["cd"][1] <= h["cd"][0] + 1e-3
    res = json.loads((tmp_path / "results.json").read_text())
    assert "shape_distance_to_target" in res and (tmp_path / "log.csv").exists() and (tmp_path / "config.yaml").exists()


def test_box_clamps_free_coordinates_and_none_changes_nothing():  # G37
    a_star = vec(0.3, -0.2, 0.1, 0.4, 0.2, -0.1)
    lo, hi = torch.full((D,), -0.05, dtype=F64), torch.full((D,), 0.15, dtype=F64)
    lo[NURBS.mask == 0] = hi[NURBS.mask == 0] = 0.0

    def run(box):
        opt = AugmentedLagrangianOptimizer(NURBS, Analytic(a_star, vec(0.0)), COND, NOPEN, None, make_cfg(max_cycles=5), box)
        opt.run(torch.zeros(1, D, dtype=F64), 5)
        return opt.alpha.reshape(-1)

    a = run((lo, hi))
    assert (a >= lo - 1e-12).all() and (a <= hi + 1e-12).all()
    assert torch.isclose(a.max(), hi.max())  # the unconstrained step leaves the box, so it ends on the boundary
    assert (a[NURBS.mask == 0] == 0).all()
    free = run(None)
    assert free.max() > 0.15 and torch.equal(run(None), free)
