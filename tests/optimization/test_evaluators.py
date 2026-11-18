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
