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
