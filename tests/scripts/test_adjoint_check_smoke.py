import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parents[2]


def test_adjoint_check_smoke(tmp_path):
    cmd = [sys.executable, str(ROOT / "scripts" / "01_validate_lbm_cylinder.py"), "--adjoint-check", "--tier", "smoke",
           "--device", "cpu", "--out", str(tmp_path),
           "--set", "validate.adjoint_k=6", "--set", "validate.adjoint_N=10", "--set", "lbm.domain_d=[8,4]",
           "--set", "lbm.body_offset_d=2", "--set", "validate.adjoint_steady_tol=1.0e-9",
           "--set", "validate.adjoint_max_convective_times=60", "--set", "validate.noise_points=1"]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    res = json.loads((tmp_path / "results.json").read_text())
    assert res["n_free"] == 4 and len(res["g_adj"]) == 4 and len(res["g_fd"]) == 4
    assert sorted(res["included"] + res["excluded"]) == [1, 2, 3, 4]
    assert res["rel_l2_error"] >= 0 and res["wall_adjoint_s"] > 0 and res["wall_fd_s"] > 0 and "fd_noise_rel" in res
    assert (tmp_path / "fig" / "adjoint_vs_fd.png").exists() and (tmp_path / "fig" / "adjoint_sensitivity_field.png").exists()
    assert 0 <= res["excluded_adjoint_norm_fraction"] <= 1 and "base_converged" in res and "n_unconverged_fd_solves" in res
    r = subprocess.run(cmd + ["--excluded-onesided"], cwd=ROOT, capture_output=True, text=True, timeout=600)  # reuses results.json
    assert r.returncode == 0, r.stderr[-2000:]
    res2 = json.loads((tmp_path / "results.json").read_text())
    assert res2["included"] == res["included"] and res2["rel_l2_error"] == res["rel_l2_error"]  # central part untouched
    assert [x["index"] for x in res2["excluded_onesided"]] == res["excluded"] and "excluded_rel_l2_onesided" in res2
