"""WP9: augmented-Lagrangian optimizer on analytic evaluators (Plan/05_optimization/kkt_optimizer.md)."""
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from aeroshape.core.config import Config
from aeroshape.core.types import Condition, CostWeights, EvalResult
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
             g_cap=10.0, cd_rtol=0.0, max_flagged=2, lam_step_max=0.5)
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


class Ascent(Analytic):  # gradient points the wrong way, so no step ever decreases L
    def evaluate(self, alpha, cond, weights):
        r = super().evaluate(alpha, cond, weights)
        return EvalResult(r.cd, r.cl, -r.dcd_dalpha, r.dcl_dalpha, 0.0, dict(I=r.extras["I"], dI_dalpha=-r.extras["dI_dalpha"]))


def test_failed_line_search_keeps_alpha_and_flags():
    ev = Ascent(vec(0.3, 0.3, 0.3, 0.3, 0.3, 0.3), torch.zeros(D, dtype=F64))
    opt = AugmentedLagrangianOptimizer(NURBS, ev, COND, NOPEN, None, make_cfg(eta=0.1))
    opt.alpha = torch.zeros(1, D, dtype=F64)
    r = opt.step()
    assert r["flagged"] and r["eta"] == pytest.approx(0.1 * 0.5 ** 4) and torch.equal(opt.alpha, torch.zeros(1, D, dtype=F64))


def test_failed_line_search_restores_warm_state():
    ev = Ascent(vec(0.3, 0.3, 0.3, 0.3, 0.3, 0.3), torch.zeros(D, dtype=F64))
    ev._f = torch.zeros(2, dtype=F64)
    orig = ev.value
    ev.value = lambda a, c, w: (setattr(ev, "_f", ev._f + 1), orig(a, c, w))[1]  # a trial solve overwrites the warm start
    opt = AugmentedLagrangianOptimizer(NURBS, ev, COND, NOPEN, None, make_cfg(eta=0.1))
    opt.alpha = torch.zeros(1, D, dtype=F64)
    assert opt.step()["flagged"] and float(ev._f.sum()) == 2.0  # evaluate() solved once (+1 each); the 5 rejected trials are undone


def test_backtracks_give_different_trial_points_under_huge_gradient():
    seen = []

    class Rec(Analytic):
        def value(self, alpha, cond, weights):
            seen.append(alpha.reshape(-1).clone())
            return super().value(alpha, cond, weights)
    ev = Rec(vec(*[1e6] * 6), torch.zeros(D, dtype=F64))
    ev.evaluate = lambda a, c, w, f=Analytic.evaluate.__get__(ev): (lambda r: EvalResult(
        r.cd, r.cl, r.dcd_dalpha, r.dcl_dalpha, 0.0, dict(I=-r.extras["I"], dI_dalpha=r.extras["dI_dalpha"])))(f(a, c, w))  # L0 too low: never accepts
    opt = AugmentedLagrangianOptimizer(NURBS, ev, COND, NOPEN, None, make_cfg(g_cap=0.02))
    opt.alpha = torch.zeros(1, D, dtype=F64)
    opt.step()
    moves = [float(t.abs().max()) for t in seen[1:]]  # seen[0] is evaluate()'s own value() at alpha
    assert len(moves) == 5 and all(moves[i + 1] < moves[i] * 0.99 for i in range(4)) and moves[0] == pytest.approx(0.02)


def test_accepted_L_never_increases_over_a_run():
    ev = Analytic(vec(0.3, -0.2, 0.1, 0.4, 0.2, -0.1), torch.zeros(D, dtype=F64))
    opt = AugmentedLagrangianOptimizer(NURBS, ev, COND, NOPEN, None, make_cfg(eta=0.9, max_flagged=50))  # no lam drift: L is a fixed function
    h = opt.run(torch.zeros(1, D, dtype=F64), 30)
    assert (h["L"][1:] <= h["L"][:-1] + 1e-12).all()


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


class Wrong(Analytic):
    """Grad evaluator: right gradient of the true CD, but values are garbage (falls with any move)."""

    def value(self, alpha, cond, weights):
        return -alpha.reshape(-1).abs().sum(), self.w @ alpha.reshape(-1)


def test_hybrid_uses_value_truth_and_grad_direction():
    from aeroshape.optimization.evaluators import HybridEvaluator
    a_star, w = vec(0.3, -0.2, 0.1), torch.zeros(D, dtype=F64)
    truth, bad = Analytic(a_star, w), Wrong(a_star, w)
    calls = []
    orig = truth.value
    truth.value = lambda a, c, wt: (calls.append(1), orig(a, c, wt))[1]
    ev = HybridEvaluator(bad, truth)
    r = ev.evaluate(torch.zeros(D, dtype=F64), COND, CostWeights(0.0, 0.0, 0.0))
    assert float(r.cd) == pytest.approx(float(orig(torch.zeros(D, dtype=F64), COND, None)[0]))  # LBM value, not grad_eval's
    assert torch.allclose(r.dcd_dalpha, bad.evaluate(torch.zeros(D, dtype=F64), COND, CostWeights(0.0, 0.0, 0.0)).dcd_dalpha)
    opt = AugmentedLagrangianOptimizer(NURBS, ev, COND, NOPEN, None, make_cfg(eta=5.0))  # eta too big: forces backtracking
    h = opt.run(torch.zeros(1, D, dtype=F64), 5)
    for a, cd in zip(h["alpha"], h["cd"]):
        assert float(cd) == pytest.approx(float(orig(a, COND, None)[0]))  # history is value-evaluator truth
    assert h["n_backtrack"].max() > 0  # a step the value evaluator scores worse was rejected
    assert (h["cd"][1:] <= h["cd"][:-1] + 1e-12).all()  # never accepts a worse step


def _toy_opt(**kw):
    a_star, w = vec(0.3, -0.2, 0.1, 0.4, 0.2, -0.1), vec(1.0, 0.5, -0.3, 0.8, 0.2, 0.1)
    return AugmentedLagrangianOptimizer(NURBS, Analytic(a_star, w), COND, NOPEN, 0.25, make_cfg(**kw))


def test_resume_matches_straight_run(tmp_path):
    a0, sp = torch.zeros(1, D, dtype=F64), tmp_path / "state.pt"
    A = _toy_opt()
    hA = A.run(a0, 6)
    _toy_opt().run(a0, 3, state_path=sp)  # "killed" after cycle 3
    assert sp.exists()
    B = _toy_opt()
    hB = B.run(a0, 6, state_path=sp, resume=True)
    assert len(hB["cd"]) == 6
    for k in hA:
        if k not in ("wall_s", "t_eval"):
            assert torch.allclose(hA[k], hB[k]), k
    assert (A.lam, A.mu, A.eta, A.n_flagged, A.cap_scale) == (B.lam, B.mu, B.eta, B.n_flagged, B.cap_scale) and torch.equal(A.alpha, B.alpha)
    assert B.stop_reason == A.stop_reason


def test_state_write_is_atomic(tmp_path, monkeypatch):
    from aeroshape.optimization import kkt_optimizer as K
    sp, real = tmp_path / "state.pt", torch.save
    opt = _toy_opt()
    opt.run(torch.zeros(1, D, dtype=F64), 2, state_path=sp)
    before = sp.read_bytes()

    def boom(obj, path):
        Path(path).write_bytes(b"partial")
        raise OSError("disk full")
    monkeypatch.setattr(K.torch, "save", boom)
    with pytest.raises(OSError):
        opt.save_state(sp)
    monkeypatch.setattr(K.torch, "save", real)
    assert sp.read_bytes() == before and [p.name for p in tmp_path.iterdir()] == ["state.pt"]


def test_warm_start_roundtrip(tmp_path):
    from aeroshape.optimization.evaluators import AdjointLBMEvaluator, HybridEvaluator
    ev = AdjointLBMEvaluator(Config({}), NURBS)
    ev._f, ev._F = torch.rand(2, 3, dtype=F64), torch.rand(2, 3, dtype=F64)
    hy = HybridEvaluator(object(), ev)
    o = AugmentedLagrangianOptimizer(NURBS, hy, COND, NOPEN, None, make_cfg())
    o.alpha = torch.zeros(1, D, dtype=F64)
    o.save_state(tmp_path / "s.pt")
    f, F = ev._f.clone(), ev._F.clone()
    ev._f = ev._F = None
    o.load_state(tmp_path / "s.pt")
    assert torch.equal(ev._f, f) and torch.equal(ev._F, F)


def _rec(alpha, cd, cl, viol, L):
    return dict(alpha=torch.full((D,), alpha, dtype=F64), cd=cd, cl=cl, L=L, viol=viol, t_cycle=0.0)


def test_best_alpha_is_lowest_cd_feasible_with_lowest_L_fallback(tmp_path):
    opt = _toy_opt(cl_tol=0.01)
    opt.alpha = torch.zeros(1, D, dtype=F64)
    opt.records = [_rec(1, 0.5, 0.25, 0.0, 9.0), _rec(2, 0.3, 0.25, 0.0, 8.0), _rec(3, 0.1, 0.4, 0.0, 7.0), _rec(4, 0.05, 0.25, 0.1, 1.0)]
    assert float(opt.best_alpha[0, 0]) == 2.0  # 3 breaks CL, 4 breaks thickness
    opt.records = [_rec(1, 0.5, 0.9, 0.0, 9.0), _rec(2, 0.3, 0.9, 0.0, 2.0)]
    assert float(opt.best_alpha[0, 0]) == 2.0  # none feasible -> smallest viol, |dCL|, then cd
    opt.records = [_rec(1, 0.5, 0.25, 0.0, 9.0), _rec(2, 0.3, 0.25, 0.0, 8.0)]
    opt.save_state(tmp_path / "s.pt")
    other = _toy_opt(cl_tol=0.01)
    other.load_state(tmp_path / "s.pt")
    assert torch.equal(other.best_alpha, opt.best_alpha)
    st = torch.load(tmp_path / "s.pt", weights_only=False)
    st.pop("best_alpha", None)
    torch.save(st, tmp_path / "old.pt")
    other.load_state(tmp_path / "old.pt")  # old state files still load
    assert float(other.best_alpha[0, 0]) == 2.0


def test_run_history_has_best_alpha_and_resume_keeps_it(tmp_path):
    a0, sp = torch.zeros(1, D, dtype=F64), tmp_path / "state.pt"
    hA = _toy_opt().run(a0, 6)
    _toy_opt().run(a0, 3, state_path=sp)
    B = _toy_opt()
    hB = B.run(a0, 6, state_path=sp, resume=True)
    assert torch.allclose(hA["best_alpha"], hB["best_alpha"]) and torch.equal(B.best_alpha.reshape(-1), hB["best_alpha"])


def test_mu_does_not_double_while_violation_shrinks():
    t_min = torch.zeros(8, dtype=F64)
    t_min[2:6] = 0.05
    opt = AugmentedLagrangianOptimizer(NURBS, Analytic(torch.zeros(D, dtype=F64), torch.zeros(D, dtype=F64)), COND, t_min, None, make_cfg())
    opt.alpha = torch.zeros(1, D, dtype=F64)
    opt.records = [dict(alpha=torch.zeros(D, dtype=F64), cd=0., cl=0., L=0., viol=0.1, kkt=1., eta=0.05, n_backtrack=0, flagged=False,
                        lam=0., mu=10., t_eval=0., t_cycle=0.)]
    opt.step()  # viol 0.05 -> 0.5x of previous: shrinking
    assert opt.mu == 10.0
    opt.records[-1]["viol"] = 0.0501
    opt.alpha = torch.zeros(1, D, dtype=F64)
    opt.step()  # same viol again: not shrinking -> double
    assert opt.mu == 20.0


def test_lam_step_is_clipped_by_lam_step_max_not_cl_tol():
    opt = _toy_opt(cl_tol=0.0, lam_step_max=1e-3)
    opt.run(torch.zeros(1, D, dtype=F64), 5)
    lam = [r["lam"] for r in opt.records] + [opt.lam]
    assert max(abs(b - a) for a, b in zip(lam, lam[1:])) <= 1e-3 * (1 + 1e-9)
    assert max(abs(b - a) for a, b in zip(lam, lam[1:])) > 0


def test_best_alpha_includes_last_accepted_trial_on_monotone_run():
    opt = AugmentedLagrangianOptimizer(NURBS, Analytic(vec(0.3, -0.2, 0.1, 0.4, 0.2, -0.1), torch.zeros(D, dtype=F64)), COND, NOPEN, None,
                                       make_cfg(eta=0.05, g_cap=0.01))
    h = opt.run(torch.zeros(1, D, dtype=F64), 4)
    assert not h["flagged"].any() and (h["cd"][1:] < h["cd"][:-1]).all()
    assert torch.equal(opt.best_alpha, opt.alpha)


def test_cap_scale_keeps_shrinking_after_flagged_cycle(tmp_path):
    seen = []

    class Rec(Analytic):
        def value(self, alpha, cond, weights):
            seen.append(alpha.reshape(-1).clone())
            return super().value(alpha, cond, weights)
    ev = Rec(vec(*[1e6] * 6), torch.zeros(D, dtype=F64))
    ev.evaluate = lambda a, c, w, f=Analytic.evaluate.__get__(ev): (lambda r: EvalResult(
        r.cd, r.cl, r.dcd_dalpha, r.dcl_dalpha, 0.0, dict(I=-r.extras["I"], dI_dalpha=r.extras["dI_dalpha"])))(f(a, c, w))  # never accepts
    opt = AugmentedLagrangianOptimizer(NURBS, ev, COND, NOPEN, None, make_cfg(g_cap=0.02))
    opt.alpha = torch.zeros(1, D, dtype=F64)
    opt.step()
    opt.step()
    moves = [float(t.abs().max()) for t in seen]  # per cycle: evaluate()'s value at alpha (0), then 5 trials
    assert moves[7] < moves[5] * 0.99 and opt.cap_scale < 0.5 ** 5
    opt.save_state(tmp_path / "s.pt")
    o2 = AugmentedLagrangianOptimizer(NURBS, ev, COND, NOPEN, None, make_cfg(g_cap=0.02))
    o2.load_state(tmp_path / "s.pt")
    assert o2.cap_scale == opt.cap_scale


def test_cap_scale_doubles_back_to_one_on_accept():
    opt = _toy_opt()
    opt.alpha, opt.cap_scale = torch.zeros(1, D, dtype=F64), 0.1
    opt.step()
    assert opt.cap_scale == pytest.approx(0.2) and not opt.records[-1]["flagged"]


def test_best_alpha_fallback_prefers_feasibility_over_L():
    opt = _toy_opt(cl_tol=0.01)
    opt.records = [_rec(1, 0.1, 0.25, 0.1, 0.1), _rec(2, 0.5, 0.9, 0.0, 100.0), _rec(3, 0.4, 0.3, 0.0, 200.0)]
    assert float(opt.best_alpha[0, 0]) == 3.0  # none feasible: smallest viol, then |cl - cl_ref|, then cd (L is not comparable)


def test_best_point_considers_accepted_trial():
    from aeroshape.optimization.kkt_optimizer import best_point
    r = _rec(1, 0.5, 0.25, 0.0, 9.0)
    r.update(alpha_new=torch.full((D,), 7.0, dtype=F64), cd_new=0.2, cl_new=0.25, viol_new=0.0, L_new=8.0)
    b = best_point([r], 0.25, 0.01)
    assert float(b["alpha"][0, 0]) == 7.0 and b["cycle"] == 1 and b["point"] == "accepted" and b["feasible"]


def test_nan_gradient_flags_without_trial_solves():
    class NanGrad(Analytic):
        def evaluate(self, alpha, cond, weights):
            r = super().evaluate(alpha, cond, weights)
            return EvalResult(r.cd, r.cl, r.dcd_dalpha, r.dcl_dalpha, 0.0, dict(I=r.extras["I"], dI_dalpha=r.extras["dI_dalpha"] * float("nan")))
    ev = NanGrad(vec(0.3, 0.3), torch.zeros(D, dtype=F64))
    calls = []
    orig = ev.value
    ev.value = lambda a, c, w: (calls.append(1), orig(a, c, w))[1]
    opt = AugmentedLagrangianOptimizer(NURBS, ev, COND, NOPEN, None, make_cfg())
    opt.alpha = torch.zeros(1, D, dtype=F64)
    r = opt.step()
    assert r["flagged"] and torch.equal(opt.alpha, torch.zeros(1, D, dtype=F64)) and len(calls) == 1  # only evaluate()'s own value()


def test_project_thickness_keeps_pinned_ends():
    from aeroshape.diffusion.guided_shape_diffusion import project_thickness
    t_min = torch.zeros(8, dtype=F64)
    t_min[2:6] = 0.05
    a0 = vec(0.01, 0.01, 0.01, 0.01, 0.01, 0.01)
    a = project_thickness(NURBS, a0, t_min)
    assert (a[NURBS.mask == 0] == a0[NURBS.mask == 0]).all()
    assert float(torch.relu(t_min - NURBS.thickness_at_ctrl(a)).max()) < 1e-6
    assert torch.equal(project_thickness(NURBS, a, t_min), a)  # no-op when feasible
