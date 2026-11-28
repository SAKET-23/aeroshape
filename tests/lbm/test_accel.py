"""L3: opt-in forward-loop accelerations (compile, CUDA graph) match eager."""
import warnings

import pytest
import torch

from aeroshape.lbm import d2q9_solver as S
from aeroshape.lbm.aero_forces import force_fn
from tests.lbm.test_warm_transient import CFG, setup

L = CFG.lbm


def _run(mode, **kw):
    p, g = setup()
    f0 = S.init_state(p, g, torch.float32, perturb=0.1)
    return S.run(f0, g, p, force_fn=force_fn, mode=mode, record_every=20, outlet=L.outlet, n_steps=400,
                 max_convective_times=100, average_periods=1, transient=3, **kw)


def _same(a, b, tol=1e-4):
    assert torch.allclose(a.state.f, b.state.f, atol=tol, rtol=tol)
    assert abs(float(a.cd) - float(b.cd)) < tol * 10 and abs(float(a.cl) - float(b.cl)) < tol * 10
    assert a.state.step == b.state.step


@pytest.mark.parametrize("mode", ["steady", "unsteady"])
def test_compile_matches_eager(mode):
    with pytest.warns(RuntimeWarning) as w:
        warnings.warn("sentinel", RuntimeWarning)  # keeps pytest.warns from failing when compile works
        c = _run(mode, compile=True)
    if any("torch.compile failed" in str(x.message) for x in w):
        pytest.skip("inductor unavailable (no working C++ toolchain); eager fallback used")
    _same(c, _run(mode))


def test_cuda_graph_noop_on_cpu():
    a, b = _run("unsteady", cuda_graph=True, graph_steps=10), _run("unsteady")
    assert torch.equal(a.state.f, b.state.f)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_graph_matches_eager_on_gpu():
    from aeroshape.core.types import Condition
    from aeroshape.geometry.ibb_grid_mapper import map_to_grid
    from aeroshape.lbm.lattice import make_params
    from tests.lbm.test_warm_transient import _circle
    dev = torch.device("cuda")
    p = make_params(CFG, Condition(100, 0), 12, device=dev, dtype=torch.float32)
    b = _circle()
    b = type(b)(*[t.to(dev) for t in b])
    g = map_to_grid(b, p, body_offset_d=L.body_offset_d)
    res = [S.run(S.init_state(p, g, torch.float32, perturb=0.1), g, p, force_fn=force_fn, mode="unsteady", record_every=20,
                 outlet=L.outlet, n_steps=400, average_periods=1, transient=3, **kw)
           for kw in ({}, {"cuda_graph": True, "graph_steps": 10})]
    _same(*res, tol=1e-3)


def test_forward_flags_match_default_on_cpu():
    from aeroshape.core.config import load_config
    from aeroshape.lbm import adjoint_solver as AS
    from tests.lbm.test_warm_transient import OV
    p, g = setup()
    on = load_config(None, "dev", OV + ["lbm.cuda_graph=true", "lbm.compile=true"])
    a, b = (AS._forward(g, p, c, "time_averaged") for c in (on, CFG))
    assert torch.equal(a[1], b[1]) and torch.equal(a[2], b[2]) and torch.equal(a[3], b[3])


def test_compile_failure_is_cached(monkeypatch):
    monkeypatch.setattr(S, "_COMPILE_FAILED", False)
    monkeypatch.setattr(S.torch, "compile", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.warns(RuntimeWarning, match="torch.compile failed"):
        _run("steady", compile=True)
    assert S._COMPILE_FAILED
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a second warning would raise
        _run("steady", compile=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_graph_steps_clamped_to_record_gap():
    from aeroshape.core.types import Condition
    from aeroshape.geometry.ibb_grid_mapper import map_to_grid
    from aeroshape.lbm.lattice import make_params
    from tests.lbm.test_warm_transient import _circle
    dev = torch.device("cuda")
    p = make_params(CFG, Condition(100, 0), 12, device=dev, dtype=torch.float32)
    b = _circle()
    g = map_to_grid(type(b)(*[t.to(dev) for t in b]), p, body_offset_d=L.body_offset_d)
    n0 = S.GRAPH_REPLAYS
    S.run(S.init_state(p, g, torch.float32), g, p, force_fn=force_fn, mode="steady", record_every=50, outlet=L.outlet,
          n_steps=200, cuda_graph=True, graph_steps=50)
    assert S.GRAPH_REPLAYS > n0


def test_compiled_cache_holds_one_entry(monkeypatch):
    monkeypatch.setattr(torch, "compile", lambda fn: fn)
    S._COMPILED.clear()
    p, g1 = setup()
    _, g2 = setup()
    S._compiled_step(g1, p, "x")
    S._compiled_step(g2, p, "x")
    assert len(S._COMPILED) == 1
    S._COMPILED.clear()


def test_cuda_graph_record_every_1_terminates_on_cpu():
    p, g = setup()
    f0 = S.init_state(p, g, torch.float32, perturb=0.1)
    r = S.run(f0, g, p, force_fn=force_fn, mode="steady", record_every=1, outlet=L.outlet, n_steps=20, max_convective_times=100,
              cuda_graph=True)
    assert r.state.step == 20
