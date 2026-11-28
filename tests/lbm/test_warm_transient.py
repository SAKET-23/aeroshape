"""L2: a warm-started unsteady run needs only a short transient (cylinder Re 100, small CPU float32 grid)."""
import warnings
from functools import lru_cache

import torch

from aeroshape.core.config import load_config
from aeroshape.core.types import Boundary, Condition
from aeroshape.geometry.ibb_grid_mapper import map_to_grid
from aeroshape.lbm import adjoint_solver as AS
from aeroshape.lbm import d2q9_solver as S
from aeroshape.lbm.aero_forces import force_fn
from aeroshape.lbm.lattice import make_params
import math

OV = ["lbm.domain_d=[14,7]", "lbm.body_offset_d=4", "lbm.collision=trt", "lbm.outlet=extrapolate", "lbm.sponge.enabled=true",
      "lbm.perturb=0.1", "lbm.trt_magic=0.001", "lbm.record_every=20", "lbm.max_convective_times=100", "lbm.transient=30", "lbm.transient_warm=5"]
CFG = load_config(None, "dev", OV)
N = 12


def _circle(r=0.5, n=2048):
    th = (torch.arange(n + 1, dtype=torch.float64) + 0.5) * (2 * math.pi / n)
    x, y = (r * torch.cos(th))[None].float(), (r * torch.sin(th))[None].float()
    z = torch.zeros_like(x)
    return Boundary(x, y, z, z, z)


@lru_cache(maxsize=None)
def setup():
    p = make_params(CFG, Condition(100, 0), N, dtype=torch.float32)
    return p, map_to_grid(_circle(), p, body_offset_d=CFG.lbm.body_offset_d)


def run(f0, transient):
    p, g = setup()
    L = CFG.lbm
    return S.run(f0, g, p, force_fn=force_fn, mode="unsteady", record_every=L.record_every, outlet=L.outlet,
                 max_convective_times=L.max_convective_times, average_periods=L.average_periods, transient=transient)


@lru_cache(maxsize=None)
def cold():
    p, g = setup()
    return run(S.init_state(p, g, torch.float32, perturb=0.1), CFG.lbm.transient)


def test_warm_run_matches_cold_and_is_shorter():
    p, g = setup()
    c = cold()
    assert c.period_steps, "cylinder at Re 100 must shed on this grid"
    f0 = c.state.f + (S.init_state(p, g, torch.float32, perturb=0.1) - S.init_state(p, g, torch.float32))  # re-kick
    w = run(f0, CFG.lbm.transient_warm)
    assert w.period_steps
    assert abs(float(w.cd) - float(c.cd)) / float(c.cd) < 0.005
    assert w.state.step < c.state.step


def test_warm_fallback_without_period_still_returns(monkeypatch):
    p, g = setup()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = S.run(S.init_state(p, g, torch.float32), g, p, force_fn=force_fn, mode="unsteady", record_every=20,
                  outlet="extrapolate", max_convective_times=2, transient=1, window_convective_times=1)
    assert r.converged and torch.isfinite(r.cd)  # no shedding without a kick: window-mean fallback
