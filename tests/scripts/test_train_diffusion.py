"""WP 17: scripts/04_train_diffusion_model.py (helpers + a micro end-to-end run on CPU with the selected PINN)."""
import importlib.util
import json
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("train04", ROOT / "scripts" / "04_train_diffusion_model.py")
train04 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train04)


def test_envelope_screen_drops_outside_shapes():
    lo, hi = torch.zeros(3), torch.ones(3)
    a = torch.tensor([[0.5, 0.5, 0.5], [1.2, 0.5, 0.5], [0.0, 1.0, 0.2], [0.5, -0.1, 0.5]])
    assert train04.in_envelope(a, lo, hi).tolist() == [True, False, True, False]


def test_ema_update_moves_towards_model():
    m, e = torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)
    m.weight.data.fill_(1.0), e.weight.data.fill_(0.0)
    train04.ema_update(e, m, 0.9)
    assert torch.allclose(e.weight, torch.full((2, 2), 0.1))


@pytest.mark.skipif(not (ROOT / "runs/pinn_selected.json").exists(), reason="needs the selected PINN checkpoint")
def test_micro_run_writes_results(tmp_path):
    ov = ["diffusion.n_shapes=24", "diffusion.steps=6", "diffusion.batch=8", "diffusion.sample_steps=4", "diffusion.layers=1",
          "diffusion.d_model=32"]
    rc = train04.main(["--tier", "smoke", "--device", "cpu", "--out", str(tmp_path), "--n-lbm", "0", "--n-per-target", "2",
                       "--n-unguided", "4", "--n-targets", "2", "--grid", "0,1:0,2", *sum((["-o", o] for o in ov), [])])
    r = json.loads((tmp_path / "results.json").read_text())
    assert rc == 0 and (tmp_path / "ckpt" / "diffusion.pt").exists() and (tmp_path / "fig" / "samples.png").exists() and (tmp_path / "fig" / "denoise.gif").exists()
    assert r["data"]["n_kept"] + r["data"]["n_dropped_envelope"] >= 24 and len(r["train"]["loss_curve"]) > 0
    assert {"J_guided", "J_unguided", "pass"} <= set(r["acceptance"]["heldout_tmin"]) and set(r["acceptance"]) == {"heldout_tmin", "beyond_range"}


def test_nn_dist_matches_shape_distance():
    from aeroshape.geometry.nurbs import NURBSAirfoil
    from aeroshape.geometry.reference_shapes import circle_alpha, naca4_alpha, shape_distance
    ref = NURBSAirfoil(k=8, dtype=torch.float64)
    c, n = circle_alpha(ref), naca4_alpha(ref, "0012")
    d = train04.nn_dist(ref, c, torch.cat([n, c * 0.9]))
    assert torch.allclose(d, torch.minimum(shape_distance(ref, c, n), shape_distance(ref, c, c * 0.9)))


def test_accept_picks_valid_row_over_bigger_cut_with_low_validity():
    row = lambda g, c, J, v: {"guide_scale": g, "cfg_scale": c, "J": J, "validity": v}
    dc = type("C", (), {"guide_scale": 1.0, "cfg_scale": 2.0})()
    un = {"validity": 1.0}
    rows = [row(0.0, 0.0, 1.0, 1.0), row(0.0, 4.0, 10.0, 1.0), row(1.0, 4.0, 1.0, 0.7), row(3.0, 0.0, 0.5, 1.0)]
    a = train04.accept(rows, un, dc)
    assert a["best_pair"] == {"guide_scale": 3.0, "cfg_scale": 0.0} and a["pass"]
    assert train04.accept([rows[0], rows[1], rows[2]], un, dc)["best_pair"] is None
    assert not train04.accept([rows[0], rows[1], rows[2]], un, dc)["pass"]
