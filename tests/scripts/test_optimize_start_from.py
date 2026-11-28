import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).parents[2]
spec = importlib.util.spec_from_file_location("opt06c", ROOT / "scripts" / "06_optimize_case.py")
opt06 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(opt06)


def test_best_saved_picks_feasible_lowest_cd(tmp_path):
    rec = lambda v, cd, viol: dict(alpha=torch.full((4,), float(v), dtype=torch.float64), cd=cd, cl=0.5, viol=viol, L=cd + viol)
    torch.save({"records": [rec(1, 0.01, 0.5), rec(2, 0.03, 0.0), rec(3, 0.05, 0.0)]}, tmp_path / "s.pt")
    opt = SimpleNamespace(cl_ref=None, o=SimpleNamespace(cl_tol=1.0), nurbs=SimpleNamespace(mask=torch.zeros(1, dtype=torch.float64)))
    alpha, cycle, cd = opt06.best_saved(tmp_path / "s.pt", opt)
    assert cycle == 2 and cd == 0.03 and alpha.shape == (1, 4) and float(alpha[0, 0]) == 2.0


def test_resume_precedence(tmp_path):
    assert opt06.use_start_from("x.pt", False, tmp_path)
    assert opt06.use_start_from("x.pt", True, tmp_path)  # no state.pt yet
    (tmp_path / "state.pt").write_bytes(b"")
    assert not opt06.use_start_from("x.pt", True, tmp_path)
    assert opt06.use_start_from("x.pt", False, tmp_path)
    assert not opt06.use_start_from(None, False, tmp_path)


def test_best_saved_considers_accepted_trial_point(tmp_path):
    r = dict(alpha=torch.full((4,), 1.0, dtype=torch.float64), cd=0.05, cl=0.5, viol=0.0, L=0.05,
             alpha_new=torch.full((4,), 9.0, dtype=torch.float64), cd_new=0.02, cl_new=0.5, viol_new=0.0, L_new=0.02)
    torch.save({"records": [r]}, tmp_path / "s.pt")
    opt = SimpleNamespace(cl_ref=None, o=SimpleNamespace(cl_tol=1.0), nurbs=SimpleNamespace(mask=torch.zeros(1, dtype=torch.float64)))
    alpha, cycle, cd = opt06.best_saved(tmp_path / "s.pt", opt)
    assert cycle == 1 and cd == 0.02 and float(alpha[0, 0]) == 9.0


def test_fd_lbm_takes_the_lbm_only_path(tmp_path, monkeypatch):
    class Stop(Exception):
        pass

    seen = {}

    def fake_build(cfg, name, nurbs, N, model):
        seen.update(name=name, model=model, dtype=nurbs.mask.dtype)
        raise Stop

    monkeypatch.setattr(opt06, "lbm_cl_ref", lambda *a, **k: 0.5)
    monkeypatch.setattr(opt06, "build_evaluator", fake_build)
    try:
        opt06.main(["--case", str(ROOT / "configs/case1_circle_to_naca_re20.yaml"), "--evaluator", "fd_lbm", "--tier", "smoke",
                    "--device", "cpu", "--set", "optimizer.box=null", "--out", str(tmp_path)])
    except Stop:
        pass
    assert seen == dict(name="fd_lbm", model=None, dtype=torch.float64)


@pytest.mark.parametrize("ev_name,evaluate_called", [("fd_lbm", False), ("adjoint_lbm", True)])
def test_warmup_skips_evaluate_for_fd_lbm(tmp_path, monkeypatch, ev_name, evaluate_called):
    class Stop(Exception):
        pass

    built = []

    class Ev:
        def __init__(self):
            self.calls = []
            built.append(self)

        def evaluate(self, *a):
            self.calls.append("evaluate")

        def value(self, *a):
            self.calls.append("value")

    class FakeOpt:
        records = []

        def __init__(self, *a, **k):
            pass

        def _weights(self):
            return None

        def run(self, *a, **k):
            raise Stop

    monkeypatch.setattr(opt06, "lbm_cl_ref", lambda *a, **k: 0.5)
    monkeypatch.setattr(opt06, "load_model", lambda *a, **k: None)
    monkeypatch.setattr(opt06, "build_evaluator", lambda *a, **k: Ev())
    monkeypatch.setattr(opt06, "AugmentedLagrangianOptimizer", FakeOpt)
    with pytest.raises(Stop):
        opt06.main(["--case", str(ROOT / "configs/case1_circle_to_naca_re20.yaml"), "--evaluator", ev_name, "--tier", "smoke",
                    "--device", "cpu", "--set", "optimizer.box=null", "--out", str(tmp_path)])
    warm = built[-1]
    assert ("evaluate" in warm.calls) == evaluate_called and "value" in warm.calls
