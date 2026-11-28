import importlib.util
import json
import math
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).parents[2]
spec = importlib.util.spec_from_file_location("chk09", ROOT / "scripts" / "09_check_re1000_gradient.py")
chk = importlib.util.module_from_spec(spec)
spec.loader.exec_module(chk)

TINY = ["lbm.cells_per_d_high_re=16", "lbm.max_convective_times=3", "lbm.transient=1", "lbm.window_convective_times=1",
        "adjoint.horizon_convective_times=1", "adjoint.tol=1.0e-3"]


@pytest.mark.slow  # ~60-80 s on a loaded CPU
def test_tiny_run_writes_finite_cosine(tmp_path):
    out = tmp_path / "d/g.json"
    argv = ["--device", "cpu", "--shapes", "case3_start", "--idx", "4", "--horizons", "1", "--out", str(out),
            *sum((["--set", s] for s in TINY), [])]
    chk.main(argv)
    row = json.loads(out.read_text())["shapes"]["case3_start"]["horizons"]["1.0"]
    assert all(math.isfinite(row[k]) for k in ("cosine", "sign_match", "norm_ratio"))


def test_compare_float32_inputs():  # stand-in for MPS (no float64): inputs stay float32, math runs on CPU float64
    v = torch.tensor([1.0, -2.0, 3.0])
    r = chk.compare(v, 2 * v)
    assert abs(r["cosine"] - 1) < 1e-6 and r["sign_match"] == 1.0 and abs(r["norm_ratio"] - 0.5) < 1e-6
