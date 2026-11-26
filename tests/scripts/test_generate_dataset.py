import importlib.util
from pathlib import Path

import pytest
import torch

from aeroshape.core.config import load_config
from aeroshape.core.io import check_anchors
from aeroshape.geometry.nurbs import NURBSAirfoil
from aeroshape.pinn.sampling import sample_batch

ROOT = Path(__file__).parents[2]
spec = importlib.util.spec_from_file_location("gen", ROOT / "scripts" / "02_generate_lbm_dataset.py")
gen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gen)

# tiny/fast: trivially loose steady tolerance so runs "converge" in a few hundred steps
FAST = ["lbm.steady_tol=0.5", "dataset.n_holdout=1", "dataset.points_per_shape=1500", "dataset.edge_points=500"]


def args(out, *extra):
    return ["--set", "v1", "--tier", "smoke", "--device", "cpu", "--out", str(out), "--n-re40", "1",
            *sum((["-o", o] for o in FAST), []), *extra]


def test_shards_are_disjoint_and_complete():
    runs = gen.run_list(load_config(None, "dev", []), n_re40=10)
    assert len(runs) == 40
    parts = [gen.shard(runs, i, 3) for i in range(3)]
    flat = [r for p in parts for r in p]
    assert sorted(flat) == sorted(runs) and len(set(flat)) == len(runs)


def test_v2_run_list_and_holdout():
    cfg = load_config(None, "dev", [])
    runs = gen.run_list(cfg, dset="v2")
    assert len(runs) == 60 and {r[1] for r in runs} == {1000} and {r[2] for r in runs} == {0, 10}
    assert sorted({r[0] for r in runs if r[0] >= cfg.dataset.n_shapes - cfg.dataset.n_holdout}) == list(range(25, 30))


# tiny CPU grid, Re 100 (Re 1000 is unstable at this N); v2 settings otherwise
V2_TINY = ["lbm.cells_per_d_high_re=10", "lbm.domain_d=[8,4]", "lbm.body_offset_d=2", "lbm.transient=2",
           "lbm.max_convective_times=4", "lbm.average_periods=1", "dataset.points_per_shape=300", "dataset.edge_points=100",
           "dataset.n_holdout=1", "lbm.sponge.width_d=1", "lbm.window_convective_times=2"]


def v2_tiny(tmp_path, monkeypatch, extra=()):
    monkeypatch.setattr(gen, "run_list", lambda cfg, n=0, dset="v1": [(0, 100, 10)])
    gen.main(["--set", "v2", "--tier", "smoke", "--device", "cpu", "--out", str(tmp_path),
              *sum((["-o", o] for o in V2_TINY + list(extra)), [])])
    return torch.load(tmp_path / "s00_re100_aoa10.pt", weights_only=False)


def test_v2_tiny_run_writes_contract_keys(tmp_path, monkeypatch):
    d = v2_tiny(tmp_path, monkeypatch)
    assert d["mode"] == "cambered" and d["alpha"].shape == (16,)
    assert d["meta"]["mode_used"] in ("period_mean", "window_mean") and "period" in d["meta"]
    assert d["meta"]["lbm_params"]["collision"] == "trt" and d["meta"]["lbm_params"]["trt_magic"] == 0.001
    assert torch.isfinite(d["cd"]) and torch.isfinite(d["u"]).all() and d["aoa"] == 10


def test_v2_no_period_uses_window_mean(tmp_path, monkeypatch):
    d = v2_tiny(tmp_path, monkeypatch)  # transient 2 of 4 convective times: too short to see 4 shedding periods
    m = d["meta"]
    assert m["period"] is None and m["mode_used"] == "window_mean" and m["st"] is None
    assert m["window_ct"] == 2 and m["converged"] and m["cl_rms"] >= 0 and torch.isfinite(d["u"]).all()
    assert m["steps"] == 6 * round(10 / (0.1 * 3 ** -0.5))  # limit (4 K) + window (2 K)


def test_redo_mode_reruns_only_flagged(tmp_path, monkeypatch):
    cfg = load_config(None, "smoke", gen.V2_CASE + V2_TINY)
    want, f = gen.lbm_signature(cfg, 10), tmp_path / "a.pt"
    meta = _fake_meta(N=10, domain_d=[8, 4], body_offset_d=2, collision="trt", outlet="extrapolate", trt_magic=0.001)
    meta["meta"]["mode_used"] = "last_state"
    torch.save(meta, f)
    assert gen.converged_on_disk(f, want) and not gen.converged_on_disk(f, want, "last_state")
    meta["meta"]["mode_used"] = "window_mean"
    torch.save(meta, f)
    assert gen.converged_on_disk(f, want, "last_state")
    d = v2_tiny(tmp_path, monkeypatch)  # end to end: redo replaces a stored last_state file
    f = tmp_path / "s00_re100_aoa10.pt"
    d["meta"]["mode_used"] = "last_state"
    torch.save(d, f)
    monkeypatch.setattr(gen, "run_list", lambda cfg, n=0, dset="v1": [(0, 100, 10)])
    gen.main(["--set", "v2", "--tier", "smoke", "--device", "cpu", "--out", str(tmp_path), "--redo-mode", "last_state",
              *sum((["-o", o] for o in V2_TINY), [])])
    assert torch.load(f, weights_only=False)["meta"]["mode_used"] == "window_mean"


def test_generate_resume_merge_check_and_sample(tmp_path, capsys):
    gen.main(args(tmp_path))
    files = sorted(tmp_path.glob("s*_re*_aoa*.pt"))
    assert len(files) == 4  # 3 shapes at Re 20 + 1 at Re 40
    mt = {f: f.stat().st_mtime_ns for f in files}
    gen.main(args(tmp_path))  # all converged -> skipped
    assert {f: f.stat().st_mtime_ns for f in files} == mt
    assert "skip" in capsys.readouterr().out

    gen.main(args(tmp_path, "--merge"))
    d = torch.load(tmp_path / "merged.pt", weights_only=False)
    assert check_anchors(d) == [] and d["xy"].shape[0] == 4
    assert d["split"].tolist() == [0, 0, 1, 0] or d["split"].sum() == 1  # last shape holdout
    assert d["edge"].sum(1).min() == 500 and (~d["edge"]).sum(1).min() == 1500
    with pytest.raises(SystemExit):  # smoke-size data cannot pass every gate; the report must still print every row
        gen.main(args(tmp_path, "--check"))
    assert "NURBS circle" in capsys.readouterr().out
    cfg = load_config(None, "smoke", [])
    # smoke sampling wants 1000 interior / 400 edge anchors, all available
    b = sample_batch(d, NURBSAirfoil(k=8, M=40), cfg, torch.Generator().manual_seed(0))
    assert b.xy_anchor.shape[1] == cfg.sampling.n_anchor and b.xy_edge.shape[1] == cfg.sampling.n_edge
    # window containment
    box = cfg.dataset.window_d
    assert d["xy"][..., 0].min() >= box[0] - 1e-5 and d["xy"][..., 1].max() <= box[3] + 1e-5


def _fake_meta(**over):
    lp = dict(N=20, domain_d=[20, 10], body_offset_d=5, collision="bgk", outlet="fixed", mach=0.1)
    lp.update(over)
    return dict(meta=dict(converged=True, cells_per_d=lp["N"], lbm_params=lp))


def test_converged_on_disk_requires_matching_config(tmp_path):  # M9a
    cfg = load_config(None, "dev", [])
    want = gen.lbm_signature(cfg, cfg.lbm.cells_per_d)
    f = tmp_path / "a.pt"
    torch.save(_fake_meta(), f)
    assert gen.converged_on_disk(f, want)
    for over in (dict(domain_d=[40, 20]), dict(collision="trt"), dict(N=10), dict(body_offset_d=10)):
        torch.save(_fake_meta(**over), f)
        assert not gen.converged_on_disk(f, want), over


def test_warm_start_only_from_converged_finite(tmp_path, monkeypatch):  # M9b
    from types import SimpleNamespace as NS
    cfg = load_config(None, "smoke", ["dataset.n_shapes=4", "dataset.n_holdout=1"])
    seen, outs = [], iter([(False, 1.0), (True, float("nan")), (True, 1.0), (True, 1.0)])

    def solve(cfg, nurbs, alpha, re, aoa, N, device, dtype, f0=None):
        seen.append(f0)
        conv, v = next(outs)
        p = NS(tau=1.0, u_lb=0.1, N=N, Nx=1, Ny=1, collision="bgk")
        return dict(cd=1.0, cl=0.0, f=torch.full((1,), v), geom=None, params=p, boundary=None, converged=conv, steps=1,
                    mode_used="steady", period=None, st=None, cl_rms=None, window_ct=None)
    monkeypatch.setattr(gen, "solve", solve)
    monkeypatch.setattr(gen, "sample_points", lambda *a: {})
    monkeypatch.setattr(gen, "save_ckpt", lambda path, d, dev: Path(path).touch())
    a = NS(out=str(tmp_path), n_re40=0, shard=(0, 1), tier="smoke")
    gen.generate(a, cfg, torch.device("cpu"), torch.float64)
    assert seen[:3] == [None] * 3 and seen[3] is not None  # unconverged / non-finite never warm-start


def test_circle_row_runs_pipeline():  # M8
    cfg = load_config(None, "smoke", FAST)
    ok, msg = gen.circle_row(gen.argparse.Namespace(device="cpu"), cfg)
    assert isinstance(ok, bool) and "CD=" in msg


def test_thin_shapes_respect_thickness_window_and_cover_it():
    import torch
    from aeroshape.geometry.nurbs import NURBSAirfoil
    nurbs = NURBSAirfoil(k=8, dtype=torch.float64)
    a = gen.thin_shapes(nurbs, 6, seed=3, lo=0.1, hi=0.3, pool=300)
    t = gen.thickness(nurbs, a)
    assert a.shape == (6, 8) and bool(((t >= 0.1) & (t <= 0.3)).all())
    assert t.max() - t.min() > 0.05  # spread over the window, not clustered
    cfg = load_config(None, "smoke", [])
    ids = [r[0] for r in gen.run_list(cfg, dset="v1b")]
    assert min(ids) == gen.V1B_BASE and all(r[1] == 20 for r in gen.run_list(cfg, dset="v1b"))


def test_v2b_list_covers_circle_and_is_valid():  # G38(b); no LBM
    nurbs = NURBSAirfoil(k=8, mode="cambered", dtype=torch.float64)
    from aeroshape.geometry.reference_shapes import circle_alpha
    a = gen.v2b_shapes(nurbs)  # raises on an invalid shape or the exact circle
    assert a.shape == (24, 16)
    c = circle_alpha(nurbs)[0]
    assert bool(((a.amin(0) <= c + 1e-12) & (a.amax(0) >= c - 1e-12)).all())  # v2b alone covers the circle
    cfg = load_config(None, "dev", [])
    runs = gen.run_list(cfg, dset="v2b")
    assert len(runs) == 24 and [r[2] for r in runs].count(10) == 4 and all(r[1] == 1000 and r[0] >= gen.V2B_BASE for r in runs)
