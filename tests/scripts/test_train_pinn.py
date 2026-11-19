"""WP 15: micro-batch equivalence (F4) and a smoke-tier end-to-end run of scripts/03_train_transformer_pinn.py."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

from aeroshape.core.types import Condition
from aeroshape.optimization.evaluators import PINNEvaluator
from aeroshape.pinn.sampling import sample_batch
from tests.pinn import TinyMLP, make_cfg, make_nurbs, synthetic_anchors

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("train03", ROOT / "scripts" / "03_train_transformer_pinn.py")
train03 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train03)


def test_microbatch_gradient_equals_full_batch():
    torch.manual_seed(0)
    cfg, nurbs = make_cfg("train.shapes_per_step=4"), make_nurbs(dtype=torch.float64)
    batch = sample_batch(synthetic_anchors(S=4), nurbs, cfg, torch.Generator().manual_seed(1))
    m = TinyMLP(8).double()

    def grads(micro):
        m.zero_grad()
        tot, _ = train03.accumulate(m, batch, cfg, step=10_000, micro=micro)  # PDE term active
        return float(tot), torch.cat([p.grad.flatten() for p in m.parameters()])

    t4, g4 = grads(4)
    t1, g1 = grads(1)
    t3, g3 = grads(3)  # uneven last chunk
    assert g4.abs().max() > 0
    assert t1 == pytest.approx(t4, rel=1e-10) and t3 == pytest.approx(t4, rel=1e-10)
    assert torch.allclose(g1, g4, rtol=1e-8, atol=1e-12) and torch.allclose(g3, g4, rtol=1e-8, atol=1e-12)


@pytest.mark.slow
@pytest.mark.skipif(not (ROOT / train03.ANCHORS).exists(), reason="needs data/lbm_anchors/dev/merged.pt")
def test_smoke_run_writes_ckpt_and_gate3(tmp_path):
    out = tmp_path / "run"
    sys.argv = ["x", "--model", "mlp", "--tier", "smoke", "--device", "cpu", "--out", str(out), "--anchors", str(ROOT / train03.ANCHORS),
                "--adjoint-tier", "smoke", "--adjoint-cache", str(tmp_path / "adj.pt"), "--grad-shapes", "1",
                "-o", "train.adam_steps=6", "-o", "train.warmup_data_only=2", "-o", "train.pde_ramp=2",
                "-o", "train.lbfgs_steps=2", "-o", "train.eval_every=3", "-o", "sampling.n_collocation=100"]
    train03.main()
    g = json.loads((out / "gate3.json").read_text())
    assert all(map(lambda v: v == v, [g["holdout_vel_l2"], g["cd_rel_err_mean"], g["grad_cosine_free_mean"], g["grad_cosine_full_mean"], g["latency_single_cd_grad_s"]]))
    ck = torch.load(out / "ckpt" / "best.pt", weights_only=False)
    assert {"model", "config", "norm", "step", "metrics"} <= set(ck) and (out / "results.json").exists() and (out / "log.csv").exists()
    assert all(t.device.type == "cpu" for t in ck["model"].values())
    from aeroshape.core.config import Config
    cfg = Config(ck["config"])
    nurbs = make_nurbs()
    model = train03.build_model(cfg, "mlp", nurbs)
    model.load_state_dict(ck["model"])
    r = PINNEvaluator(model.eval(), nurbs).evaluate(torch.zeros(8).index_fill_(0, torch.arange(1, 7), 0.06), Condition(20.0, 0.0))
    assert torch.isfinite(r.cd) and torch.isfinite(r.extras["dI_dalpha"]).all()


@pytest.mark.skipif(not (ROOT / train03.ANCHORS).exists(), reason="needs data/lbm_anchors/dev/merged.pt")
def test_stress_integral_of_anchor_field_matches_lbm_cd():
    """Integrator conventions (p units, 1/Re viscous scale, outward normals, frame) checked against the LBM anchors:
    traction of the linearly interpolated anchor field (probes 0.015 / 0.03 outside the wall) gives the LBM CD within 20%."""
    import numpy as np
    from scipy.interpolate import LinearNDInterpolator
    from aeroshape.core.io import load_ckpt
    from aeroshape.geometry.nurbs import NURBSAirfoil
    from aeroshape.pinn.stress_integrator import _rotated
    A = load_ckpt(ROOT / train03.ANCHORS, "cpu")
    nurbs = NURBSAirfoil(k=8, M=160)
    alpha, cond, xy, u, v, p, cd = train03._flat_holdout(A, "cpu", torch.float32)
    for i in range(2):  # blunt holdout shapes; thin bodies need a finer wall probe than the anchors give
        b = _rotated(nurbs, alpha[i:i + 1], cond[i:i + 1], None, None)
        x, y = b.x[0].numpy(), b.y[0].numpy()
        L = np.hypot(np.diff(x), np.diff(y))
        nx, ny = np.diff(y) / L, -np.diff(x) / L
        mx, my = (x[1:] + x[:-1]) / 2, (y[1:] + y[:-1]) / 2
        f = [LinearNDInterpolator(xy[i].numpy(), q[i].numpy()) for q in (u, v, p)]
        h = 0.015
        at = lambda k: [g(mx + nx * k * h, my + ny * k * h) for g in f]
        (u1, v1, p1), (u2, v2, _) = at(1), at(2)
        dun, dvn = (4 * u1 - u2) / (2 * h), (4 * v1 - v2) / (2 * h)  # no slip: u(0) = 0
        tx = (2 * dun * nx * nx + (dun * ny + dvn * nx) * ny) / 20.0  # grad u = dU/dn (x) n, nu = 1/Re
        got = 2 * np.sum((-p1 * nx + tx) * L)
        assert got == pytest.approx(cd[i].item(), rel=0.2)


@pytest.mark.parametrize("kind", ["mlp", "transformer"])
def test_optimize_script_loads_ckpt_with_non_default_config(kind, tmp_path):
    from aeroshape.core.config import load_config
    from aeroshape.core.io import save_ckpt
    spec6 = importlib.util.spec_from_file_location("opt06", ROOT / "scripts" / "06_optimize_case.py")
    opt06 = importlib.util.module_from_spec(spec6)
    spec6.loader.exec_module(opt06)
    cfg = load_config(None, "smoke", ["pinn.fourier_scale=0.25", "pinn.mlp_hidden=32", "pinn.d_model=32"])
    nurbs = make_nurbs()
    m = train03.build_model(cfg, kind, nurbs)
    save_ckpt(tmp_path / "c.pt", {"model": m.state_dict(), "config": cfg.to_dict()})
    default = load_config(None, "dev", [])  # script config differs (fourier_scale 2.0, d_model 128)
    m2 = opt06.load_model(default, nurbs, kind + "_pinn", tmp_path / "c.pt", "cpu")
    assert torch.equal(m2.ff.B, m.ff.B)


def _force_batch():
    cfg, nurbs = make_cfg("train.shapes_per_step=4", "loss.w_cd=1.0"), make_nurbs(dtype=torch.float64)
    batch = sample_batch(synthetic_anchors(S=4), nurbs, cfg, torch.Generator().manual_seed(1))
    return cfg, nurbs, batch, TinyMLP(8).double()


def test_force_term_zero_at_target():
    from aeroshape.pinn.stress_integrator import surface_forces
    cfg, nurbs, b, m = _force_batch()
    cd, cl = surface_forces(m, nurbs, b.alpha, b.cond)
    b.cd, b.cl = cd.detach(), cl.detach()
    _, d = train03.hybrid_loss(m, nurbs, b, cfg, step=0, log=True)
    assert d["force_data"] == pytest.approx(0, abs=1e-20)
    b.cd = b.cd + 1.0
    assert train03.hybrid_loss(m, nurbs, b, cfg, step=0, log=True)[1]["force_data"] == pytest.approx(1.0, rel=1e-9)  # CL matches


def test_force_term_gradient_matches_finite_difference():
    cfg, nurbs, b, m = _force_batch()
    b.cd, b.cl = torch.full_like(b.cd, 0.7), torch.full_like(b.cl, 0.1)
    f = lambda: train03.hybrid_loss(m, nurbs, b, cfg, step=0)[0]
    params = [m.net[0].weight, m.net[4].bias]
    idx = [(0, 0), (0,)]
    f().backward()
    for p, i in zip(params, idx):
        g, eps = p.grad[i].item(), 1e-6
        def shifted(d):
            with torch.no_grad():
                p[i] += d
            return f().item()
        hi, lo = shifted(eps), shifted(-2 * eps)
        shifted(eps)
        assert g == pytest.approx((hi - lo) / (2 * eps), rel=1e-5, abs=1e-9)


def test_microbatch_equivalence_with_force_term():
    cfg, nurbs, b, m = _force_batch()
    b.cd, b.cl = torch.full_like(b.cd, 0.7), torch.full_like(b.cl, 0.1)
    out = []
    for micro in (4, 1):
        m.zero_grad()
        tot, _ = train03.accumulate(m, b, cfg, step=10_000, micro=micro, nurbs=nurbs)
        out.append((float(tot), torch.cat([p.grad.flatten() for p in m.parameters()])))
    assert out[0][0] == pytest.approx(out[1][0], rel=1e-10)
    assert torch.allclose(out[0][1], out[1][1], rtol=1e-8, atol=1e-12)


def _grad_batch(w_grad=1.0):
    cfg, nurbs = make_cfg("train.shapes_per_step=4", f"loss.w_grad={w_grad}", "loss.w_cd=1.0"), make_nurbs(dtype=torch.float64)
    A = synthetic_anchors(S=4)
    g = torch.Generator().manual_seed(5)
    A["gcd"], A["gcl"] = torch.randn(4, 8, generator=g), torch.randn(4, 8, generator=g)
    b = sample_batch(A, nurbs, cfg, torch.Generator().manual_seed(1))
    return cfg, nurbs, b, TinyMLP(8).double()


def test_grad_term_zero_at_target_and_skips_missing_reference():
    cfg, nurbs, b, m = _grad_batch()
    from aeroshape.pinn.stress_integrator import surface_forces
    a = b.alpha.clone().requires_grad_(True)
    cd, cl = surface_forces(m, nurbs, a, b.cond)
    b.gcd = torch.autograd.grad(cd.sum(), a, retain_graph=True)[0].detach()
    b.gcl = torch.autograd.grad(cl.sum(), a)[0].detach()
    assert train03.hybrid_loss(m, nurbs, b, cfg, step=0, log=True)[1]["grad_data"] == pytest.approx(0, abs=1e-9)
    b.gcd = torch.full_like(b.gcd, float("nan"))
    b.gcl = torch.zeros_like(b.gcl)  # unusable references contribute 0, not NaN
    assert train03.hybrid_loss(m, nurbs, b, cfg, step=0, log=True)[1]["grad_data"] == 0.0


def test_grad_term_gradient_matches_finite_difference():
    cfg, nurbs, b, m = _grad_batch()
    f = lambda: train03.hybrid_loss(m, nurbs, b, cfg, step=0)[0]
    f().backward()
    p, i = m.net[2].weight, (1, 2)
    g, eps = p.grad[i].item(), 1e-6

    def shifted(d):
        with torch.no_grad():
            p[i] += d
        return f().item()
    hi, lo = shifted(eps), shifted(-2 * eps)
    assert g == pytest.approx((hi - lo) / (2 * eps), rel=1e-5, abs=1e-9)


def test_microbatch_equivalence_with_grad_term():
    cfg, nurbs, b, m = _grad_batch()
    out = []
    for micro in (4, 1):
        m.zero_grad()
        tot, _ = train03.accumulate(m, b, cfg, step=10_000, micro=micro, nurbs=nurbs)
        out.append((float(tot), torch.cat([p.grad.flatten() for p in m.parameters()])))
    assert out[0][0] == pytest.approx(out[1][0], rel=1e-10)
    assert torch.allclose(out[0][1], out[1][1], rtol=1e-8, atol=1e-12)


def test_adjoint_cache_recomputed_on_settings_change(tmp_path, monkeypatch):  # L3
    from types import SimpleNamespace

    from aeroshape.core.config import load_config
    calls = []

    def fake(alpha, nurbs, cond, cfg, w, N):
        calls.append(1)
        return SimpleNamespace(extras={"dI_dalpha": torch.zeros(8)}, cd=1.0, wall_s=0.0)

    monkeypatch.setattr(train03, "lbm_value_and_grad", fake)
    A = synthetic_anchors(S=2)
    A["split"][:] = 1
    A["edge"][:] = False
    nurbs, cache = make_nurbs(), tmp_path / "adj.pt"
    cfg = lambda tol: load_config(None, "smoke", [f"lbm.steady_tol={tol}"])
    train03._adjoint_grads(nurbs, A, cfg(1e-6), 2, "smoke", cache, "cpu")
    assert len(calls) == 2
    train03._adjoint_grads(nurbs, A, cfg(1e-6), 2, "smoke", cache, "cpu")
    assert len(calls) == 2  # same settings: reused
    train03._adjoint_grads(nurbs, A, cfg(1e-5), 2, "smoke", cache, "cpu")
    assert len(calls) == 4  # steady_tol changed: recomputed


def test_load_for_eval_rejects_wrong_kind(tmp_path):  # L5
    from aeroshape.core.io import save_ckpt
    save_ckpt(tmp_path / "c.pt", {"model": {}, "config": make_cfg().to_dict(), "metrics": {"kind": "mlp"}})
    with pytest.raises(SystemExit, match="mlp"):
        train03.load_for_eval(tmp_path / "c.pt", "transformer", "cpu")


def test_attach_adjoint_grads_mixed_layout(tmp_path):
    """Steady symmetric gradients land in the first k columns of a cambered row; time-averaged rows stay NaN."""
    A = {"alpha": torch.zeros(3, 16), "split": torch.zeros(3, dtype=torch.int8), "kind": torch.tensor([0, 0, 1], dtype=torch.int8),
         "meta": {"runs": [{"id": "a"}, {"id": "b"}, {"id": "t"}]}}
    done = {k: dict(dcd=torch.full((8,), v), dcl=torch.full((8,), -v)) for k, v in (("a", 1.0), ("b", 2.0))}
    torch.save(done, tmp_path / "c.pt")
    train03.attach_adjoint_grads(A, tmp_path / "c.pt")
    assert A["gcd"][1, :8].eq(2.0).all() and A["gcd"][1, 8:].eq(0).all() and A["gcl"][0, :8].eq(-1.0).all()
    assert torch.isnan(A["gcd"][2]).all()
