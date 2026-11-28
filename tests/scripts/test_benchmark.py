import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("bench05", ROOT / "scripts" / "05_run_4way_benchmark.py")
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)

HAVE_MODELS = (ROOT / "runs/pinn_selected.json").exists() and (ROOT / "runs/diffusion_dev/ckpt/diffusion.pt").exists()
MICRO = ["optimizer.max_cycles=1", "adjoint.tol=1.0e-3", "lbm.max_convective_times=3", "diffusion.sample_steps=4"]


def _fake_runs(tmp_path, monkeypatch, with_ckpt):
    (tmp_path / "runs/pinn_selected.json").parent.mkdir(parents=True)
    sel = {k: {"checkpoint": f"{k}.pt", "wall_s": i + 1.0} for i, k in enumerate(["mlp", "transformer", "mlp_v2", "transformer_v2"])}
    (tmp_path / "runs/pinn_selected.json").write_text(json.dumps(sel))
    d = tmp_path / "runs/diffusion_v2_dev"
    (d / "ckpt").mkdir(parents=True)
    (d / "results.json").write_text(json.dumps({"train": {"wall_s": 7.0}, "chosen_pair": {"guide_scale": 3.0, "cfg_scale": 0.0},
                                              "acceptance": {"heldout_tmin": {"best_pair": {"guide_scale": 2.0, "cfg_scale": 1.0}}}}))
    if with_ckpt:
        (d / "ckpt/diffusion.pt").write_bytes(b"x")
    monkeypatch.setattr(bench, "ROOT", tmp_path)


def test_cases_2_3_use_v2_models(tmp_path, monkeypatch):
    _fake_runs(tmp_path, monkeypatch, True)
    m = bench.models_for(2, bench.argparse.Namespace(diffusion_ckpt="unused"))
    assert m["mlp_pinn"].name == "mlp_v2.pt" and m["transformer_pinn"].name == "transformer_v2.pt"
    assert m["diffusion"].parts[-3] == "diffusion_v2_dev" and m["train_s"] == {"mlp_pinn": 3.0, "transformer_pinn": 4.0, "diffusion_only": 7.0}
    assert m["diffusion_pair"] == {"guide_scale": 3.0, "cfg_scale": 0.0}  # chosen_pair wins over the differing heldout_tmin pair
    rj = tmp_path / "runs/diffusion_v2_dev/results.json"
    res = json.loads(rj.read_text())
    del res["chosen_pair"]
    res["acceptance"]["beyond_range"] = {"best_pair": {"guide_scale": 0.5, "cfg_scale": 2.0}}
    rj.write_text(json.dumps(res))
    assert bench.models_for(2, bench.argparse.Namespace(diffusion_ckpt="unused"))["diffusion_pair"] == {"guide_scale": 0.5, "cfg_scale": 2.0}


def test_cases_2_3_missing_diffusion_ckpt(tmp_path, monkeypatch):
    _fake_runs(tmp_path, monkeypatch, False)
    with pytest.raises(SystemExit, match=r"cambered diffusion checkpoint not found.*diffusion_v2_dev"):
        bench.models_for(3, bench.argparse.Namespace(diffusion_ckpt="unused"))


def test_box_dim_guard(tmp_path):
    import torch
    from aeroshape.core.config import load_config
    f = tmp_path / "box8.pt"
    torch.save({"alpha": torch.zeros(3, 8), "split": torch.zeros(3)}, f)
    with pytest.raises(ValueError, match="D=16"):
        bench.check_box_dim(str(f), load_config("case2_circle_to_rae_re1000.yaml", "smoke", []))
    bench.check_box_dim(str(f), load_config("case1_circle_to_naca_re20.yaml", "smoke", []))  # symmetric D=8 passes


def test_hybrid_flag_off_by_default():
    assert bench.HYBRID not in bench.METHODS and bench.HYBRID in bench.RUNS
    assert bench.RUNS[bench.HYBRID][1] == "pinn_train" and bench.up_front(bench.HYBRID, {"transformer_pinn": 5.0}) == 5.0


def test_row_and_table():
    r = dict(cd_final_lbm=2.0, cl_final_lbm=0.01, wall_s=3.0, peak_mem_bytes=2_000_000, cycles=5, stop_reason="x", cd_own_last=2.1,
             cl_own_last=0.0, viol_final=0.0, cl_ref=None, shape_distance_to_target=0.1)
    row = bench.row("mlp_pinn", r, {"cd_first": 2.5}, 0.01, {"mlp_pinn": 100.0, "transformer_pinn": 200.0, "diffusion_only": 50.0})
    assert row["drag_reduction"] == pytest.approx(0.2) and row["up_front_s"] == 100.0 and row["cl_err"] == pytest.approx(0.01)
    md = Path(ROOT / "runs" / "pytest_bench_table.md")
    bench.write_md({"tier": "t", "device": "cpu", "cases": {1: [row]}}, md)
    text = md.read_text()
    md.unlink()
    assert "MLP-PINN" in text and "20.00%" in text and "## Summary" in text


@pytest.mark.skipif(not HAVE_MODELS, reason="needs the selected PINN and diffusion checkpoints")
def test_smoke_run_all_methods_and_resume(tmp_path):
    argv = ["--tier", "smoke", "--device", "cpu", "--cases", "1", "--out", str(tmp_path), *sum((["-o", o] for o in MICRO), [])]
    assert bench.main(argv) == 0
    res = json.loads((tmp_path / "results.json").read_text())
    rows = res["cases"]["1"]
    assert [r["method"] for r in rows] == bench.METHODS and not res["failures"]
    assert rows[0]["wall_s"] > 0 and rows[1]["latency_s"] > 0 and rows[3]["iterations"] == 4
    assert json.loads((tmp_path / "case1_mlp_pinn" / "results.json").read_text())["box"] == "pinn_train"
    assert json.loads((tmp_path / "case1_mlp_pinn_nobox" / "results.json").read_text())["box"] is None
    assert (tmp_path / "fig" / "shapes_case1.png").exists() and (tmp_path / "fig" / "cd_vs_wallclock_case1.png").exists()
    assert (tmp_path / "results.md").exists() and (tmp_path / "histories" / "case1_mlp_pinn.pt").exists()
    t0 = (tmp_path / "case1_adjoint_lbm" / "results.json").stat().st_mtime
    assert bench.main(argv + ["--resume"]) == 0  # nothing re-run
    assert (tmp_path / "case1_adjoint_lbm" / "results.json").stat().st_mtime == t0


def test_optimizer_config_values_are_numbers():  # '1.0e4' (no exponent sign) loads as a string in PyYAML
    from aeroshape.core.config import load_config
    o = load_config(None, "smoke", []).optimizer.to_dict()
    assert all(isinstance(v, (int, float)) for k, v in o.items() if k not in ("evaluator", "box")), o


def test_resume_args_for_partial_subrun(tmp_path):
    assert bench.resume_args(True, tmp_path) == []  # nothing yet: fresh start
    (tmp_path / "state.pt").write_bytes(b"x")
    assert bench.resume_args(True, tmp_path) == ["--resume"] and bench.resume_args(False, tmp_path) == []
    (tmp_path / "results.json").write_text("{}")
    assert bench.resume_args(True, tmp_path) == []
