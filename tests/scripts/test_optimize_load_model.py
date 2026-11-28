import importlib.util
from pathlib import Path

import pytest
import torch

from aeroshape.core.config import load_config
from aeroshape.core.io import save_ckpt
from aeroshape.geometry.nurbs import NURBSAirfoil

ROOT = Path(__file__).parents[2]
spec = importlib.util.spec_from_file_location("opt06b", ROOT / "scripts" / "06_optimize_case.py")
opt06 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(opt06)


def test_missing_ckpt_is_a_clear_error():  # L6
    cfg = load_config(None, "smoke", [])
    with pytest.raises(SystemExit, match="--ckpt"):
        opt06.load_model(cfg, NURBSAirfoil(k=8), "mlp_pinn", None, "cpu")


def test_geometry_mismatch_raises(tmp_path):  # L6
    cfg = load_config(None, "smoke", [])
    save_ckpt(tmp_path / "c.pt", {"model": {}, "config": {**cfg.to_dict(), "geometry": {"k": cfg.geometry.k + 1, "degree": cfg.geometry.degree, "M": cfg.geometry.M}}})
    with pytest.raises(ValueError, match="geometry"):
        opt06.load_model(cfg, NURBSAirfoil(k=8), "mlp_pinn", tmp_path / "c.pt", "cpu")


def test_hybrid_smoke_run(tmp_path):
    import json
    sel = ROOT / "runs/pinn_selected.json"
    anchors = ROOT / "data/lbm_anchors/dev/merged.pt"
    if not (sel.exists() and anchors.exists()):
        pytest.skip("needs the selected transformer checkpoint and anchors")
    ck = ROOT / json.loads(sel.read_text())["transformer"]["checkpoint"]
    if not ck.exists():
        pytest.skip("transformer checkpoint file absent")
    opt06.main(["--case", str(ROOT / "configs/case1_circle_to_naca_re20.yaml"), "--evaluator", "hybrid_transformer_pinn", "--tier", "smoke",
                "--device", "cpu", "--ckpt", str(ck), "--box-anchors", str(anchors), "--out", str(tmp_path),
                "--set", "optimizer.max_cycles=1", "--set", "lbm.max_convective_times=3"])
    res = json.loads((tmp_path / "results.json").read_text())
    assert res["evaluator"] == "hybrid_transformer_pinn" and res["cycles"] == 1
