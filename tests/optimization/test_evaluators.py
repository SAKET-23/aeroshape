"""WP9: evaluator protocol conformance on a tiny setup."""
import inspect

import pytest
import torch

from aeroshape.core.config import load_config
from aeroshape.core.types import Condition, CostWeights, EvalResult
from aeroshape.geometry.nurbs import NURBSAirfoil
from aeroshape.geometry.reference_shapes import circle_alpha
from aeroshape.optimization.evaluators import AdjointLBMEvaluator, LBMScoreEvaluator, PINNEvaluator, build_evaluator
from aeroshape.pinn.mlp_pinn import MLPPINN
from aeroshape.pinn.transformer_pinn import GeometryCrossAttentionPINN

COND = Condition(20, 0)
W = CostWeights(0.1, 5.0, 0.0)
CFG = load_config(None, "smoke", ["lbm.domain_d=[10,5]", "lbm.max_convective_times=3", "adjoint.horizon_convective_times=1"])


def check(ev, nurbs, alpha, grad=True):
    r = ev.evaluate(alpha, COND, W)
    assert isinstance(r, EvalResult) and r.cd.ndim == 0 and r.cl.ndim == 0
    cd, cl = ev.value(alpha, COND, W)
    assert abs(float(cd) - float(r.cd)) < 0.2 * abs(float(r.cd)) + 1e-3  # warm/cold forward differ only by convergence
    if grad:
        I, g = r.extras["I"], r.extras["dI_dalpha"]
        assert g.numel() == nurbs.D and torch.isfinite(g).all()
        assert float(I) == pytest.approx(float(r.cd + W.lam * r.cl + 0.5 * W.rho_al * (r.cl - W.cl_ref) ** 2), rel=1e-4)
    return r


def test_adjoint_lbm_evaluator():
    n = NURBSAirfoil(k=8, dtype=torch.float64)
    check(build_evaluator(CFG, "adjoint_lbm", n, N=10), n, circle_alpha(n))


def test_lbm_score_evaluator():
    n = NURBSAirfoil(k=8, dtype=torch.float64)
    ev = LBMScoreEvaluator(CFG, n, N=10)
    r = ev.evaluate(circle_alpha(n), COND, W)
    assert r.cd.ndim == 0 and "dI_dalpha" not in r.extras


@pytest.mark.parametrize("kind", ["mlp", "transformer"])
def test_pinn_evaluator(kind):
    n = NURBSAirfoil(k=8)
    m = (MLPPINN(8, hidden=16, layers=2, fourier=8) if kind == "mlp"
         else GeometryCrossAttentionPINN(n, d_model=16, heads=2, n_self=1, n_cross=1, fourier=8))
    ev = PINNEvaluator(m, n)
    r = ev.evaluate(circle_alpha(n).float(), COND, W)
    g = r.extras["dI_dalpha"]
    assert g.numel() == 8 and torch.isfinite(g).all()
    c = W.lam + W.rho_al * float(r.cl - W.cl_ref)
    assert torch.allclose(g, r.dcd_dalpha + c * r.dcl_dalpha, atol=1e-6)
    assert ev.value(circle_alpha(n).float(), COND, W)[0].ndim == 0


def test_adjoint_evaluator_warm_start(monkeypatch):
    from aeroshape.lbm import d2q9_solver as S
    n = NURBSAirfoil(k=8, dtype=torch.float64)
    cfg = load_config(None, "smoke", ["lbm.domain_d=[10,5]", "lbm.max_convective_times=60", "lbm.steady_tol=1.0e-4"])
    ev, a, calls, orig = build_evaluator(cfg, "adjoint_lbm", n, N=10), circle_alpha(n), [0], S.step

    def counted(*x, **k):
        calls[0] += 1
        return orig(*x, **k)

    monkeypatch.setattr(S, "step", counted)
    ev.evaluate(a, COND, W)
    first, calls[0] = calls[0], 0
    r = ev.evaluate(a, COND, W)
    print(f"steps cycle1 {first} cycle2 {calls[0]}")
    assert calls[0] < first and torch.isfinite(r.extras["dI_dalpha"]).all()


def test_evaluators_match_protocol_signatures():  # M3
    from aeroshape.core.types import Evaluator
    n = NURBSAirfoil(k=8)
    m = MLPPINN(8, hidden=16, layers=2, fourier=8)
    evs = [AdjointLBMEvaluator(CFG, n, N=10), LBMScoreEvaluator(CFG, n, N=10), PINNEvaluator(m, n)]
    for name in ("evaluate", "value"):
        want = list(inspect.signature(getattr(Evaluator, name)).parameters)[1:]
        assert want == ["alpha", "cond", "weights"]
        for ev in evs:
            ps = list(inspect.signature(getattr(ev, name)).parameters.values())
            assert [p.name for p in ps] == want and ps[2].default is None, (type(ev).__name__, name)


def test_pinn_evaluator_rejects_out_of_range_condition():  # L1
    n = NURBSAirfoil(k=8, dtype=torch.float64)
    m = MLPPINN(n.D, 8, 2, 4, 1.0, False).double()
    m.cond_range = {"log_re": [torch.log(torch.tensor(20.0)).item(), torch.log(torch.tensor(40.0)).item()], "aoa": [0.0, 0.0]}
    ev = PINNEvaluator(m, n)
    ev.value(circle_alpha(n), Condition(30, 0))  # inside
    for bad in (Condition(1000, 0), Condition(20, 10)):
        with pytest.raises(ValueError, match="training range"):
            ev.value(circle_alpha(n), bad)


def test_build_evaluator_hybrid():
    from aeroshape.optimization.evaluators import HybridEvaluator
    n = NURBSAirfoil(k=8)
    m = MLPPINN(8, hidden=16, layers=2, fourier=8)
    ev = build_evaluator(CFG, "hybrid_mlp_pinn", n, 10, m)
    assert isinstance(ev, HybridEvaluator) and isinstance(ev.grad_eval, PINNEvaluator) and isinstance(ev.value_eval, LBMScoreEvaluator)
    assert isinstance(build_evaluator(CFG, "hybrid_transformer_pinn", n, 10, m), HybridEvaluator)


@pytest.mark.parametrize("mode,warm", [("steady", True), ("time_averaged", False)])
def test_adjoint_warm_start_only_in_steady(monkeypatch, mode, warm):
    from types import SimpleNamespace
    from aeroshape.optimization import evaluators as E
    seen = []

    def fake(alpha, nurbs, cond, cfg, weights, N, f0, F0):
        seen.append(F0)
        return SimpleNamespace(extras={"f": "f", "F_star": "F"})
    monkeypatch.setattr(E.A, "lbm_value_and_grad", fake)
    cfg = load_config(None, "smoke", ["adjoint.mode=" + mode])
    ev = AdjointLBMEvaluator(cfg, None, N=10)
    ev.evaluate(None, COND, W), ev.evaluate(None, COND, W)
    assert seen == [None, "F" if warm else None]


def test_fd_lbm_evaluator_matches_analytic(monkeypatch):
    from aeroshape.optimization.evaluators import FDLBMEvaluator, combined
    from aeroshape.lbm import adjoint_solver as A
    n = NURBSAirfoil(k=8, dtype=torch.float64)
    a = circle_alpha(n).reshape(-1)
    w1, v1 = torch.randn(a.numel(), dtype=a.dtype), torch.randn(a.numel(), dtype=a.dtype)
    fn = lambda x: ((w1 * x**2).sum(), (v1 * x).sum())
    monkeypatch.setattr(A, "lbm_value", lambda x, *_: (*fn(x.reshape(-1)), None))
    ev = build_evaluator(CFG, "fd_lbm", n, N=10)
    assert isinstance(ev, FDLBMEvaluator)
    r = ev.evaluate(a, COND, W)
    x = a.clone().requires_grad_(True)
    ref, = torch.autograd.grad(combined(*fn(x), W), x)
    free = torch.zeros_like(a, dtype=torch.bool)
    free[[int(i) for i in n.free_idx]] = True
    assert r.wall_s >= r.extras["t_forward"] and r.extras["wall_s"] == r.wall_s
    g = r.extras["dI_dalpha"]
    assert torch.allclose(g[free], ref[free], atol=1e-6) and (g[~free] == 0).all()
