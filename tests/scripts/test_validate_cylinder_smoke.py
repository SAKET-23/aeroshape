import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parents[2]


def test_smoke_runs_and_writes_outputs(tmp_path):
    cmd = [sys.executable, str(ROOT / "scripts" / "01_validate_lbm_cylinder.py"), "--tier", "smoke", "--device", "cpu",
           "--re", "20", "100", "--out", str(tmp_path),
           "--set", "validate.n_steps=300", "--set", "lbm.record_every=10", "--set", "validate.snapshot_t=[0.5]"]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
    res = json.loads((tmp_path / "results.json").read_text())
    assert {"20", "100"} <= set(res["cases"]) and "cd" in res["cases"]["20"]
    assert (tmp_path / "table1.md").exists() and (tmp_path / "fig" / "cd_cl_history_re20.png").exists()
