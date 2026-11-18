"""The smoke script's full stage list at micro size, plus compare_summary."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


MICRO = ["train.adam_steps=6", "train.warmup_data_only=2", "train.pde_ramp=2", "train.lbfgs_steps=2", "train.eval_every=3",
         "sampling.n_collocation=60", "diffusion.n_shapes=20", "diffusion.steps=10", "diffusion.sample_steps=4",
         "optimizer.max_cycles=1", "adjoint.tol=1.0e-3", "lbm.max_convective_times=3"]


@pytest.mark.slow
def test_all_stages_run_at_micro_size(tmp_path):
    smoke = _load("smoke", "00_smoke_e2e.py")
    rc = smoke.main(["--device", "cpu", "--out", str(tmp_path), "--min-loss-drop", "0", *sum((["-o", o] for o in MICRO), [])])
    s = json.loads((tmp_path / "summary.json").read_text())
    assert {k: v.get("error") for k, v in s["stages"].items() if v["status"] != "ok"} == {}
    assert rc == 0 and len(s["stages"]) == len(smoke.STAGES)
    assert (tmp_path / "results.md").read_text().count("\n") == 5  # header, rule, 3 evaluator rows


def test_failed_stage_gives_nonzero_exit(tmp_path, monkeypatch):
    smoke = _load("smoke", "00_smoke_e2e.py")
    monkeypatch.setitem(smoke.STAGES, 1, ("geometry", lambda ctx: 1 / 0))
    assert smoke.main(["--device", "cpu", "--out", str(tmp_path), "--stages", "1"]) == 1
    assert json.loads((tmp_path / "summary.json").read_text())["stages"]["1_geometry"]["status"] == "fail"


def test_compare_summary():
    cmp = _load("cmp", "compare_summary.py")
    ref = {"stages": {"2_lbm_forward": {"status": "ok", "cd": 2.0},
                      "5_pinn_train": {"status": "ok", "mlp_final_loss": 10.0, "transformer_final_loss": 10.0}}}
    assert cmp.compare(ref, ref) == []
    off = json.loads(json.dumps(ref))
    off["stages"]["2_lbm_forward"]["cd"] = 2.001
    off["stages"]["5_pinn_train"]["status"] = "fail"
    bad = cmp.compare(off, ref)
    assert len(bad) == 2 and any("lbm_forward.cd" in b for b in bad)
