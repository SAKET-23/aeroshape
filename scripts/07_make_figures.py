"""Regenerate every available figure into runs/figures/ from existing results (no solver runs; PINN forward passes on CPU).
uv run python scripts/07_make_figures.py"""
import importlib.util
import json
import os
import shutil
from pathlib import Path

os.environ["AEROSHAPE_DEVICE"] = "cpu"
import numpy as np  # noqa: E402
import torch  # noqa: E402

from aeroshape.core.types import Condition  # noqa: E402
from aeroshape.pinn.embeddings import wall_distance  # noqa: E402
from aeroshape.pinn.stress_integrator import _rotated  # noqa: E402
from aeroshape.visualization import plots  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "figures"


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def copy_existing():
    """Figures already written by scripts 01/04/05 are collected (copied), not redrawn."""
    n = 0
    for f in sorted((ROOT / "runs").glob("*/*/fig/*")) + sorted((ROOT / "runs").glob("*/fig/*")) + sorted((ROOT / "runs/benchmark/dev/fig").glob("*")):
        if f.suffix in (".png", ".gif") and "runs/figures" not in str(f):
            rel = f.relative_to(ROOT / "runs").parts
            dst = OUT / "collected" / "_".join(p for p in rel if p != "fig")
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(f, dst)
            n += 1
    for t in (ROOT / "runs/validate_cylinder").glob("*/table1.md"):
        shutil.copy(t, OUT / "collected" / f"table1_{t.parent.name}.md")
    print(f"collected {n} existing figures")


def pinn_figures(sel):
    t03 = _load("train03", "03_train_transformer_pinn.py")
    model, nurbs = t03.load_for_eval(ROOT / sel["transformer"]["checkpoint"], "transformer", torch.device("cpu"))
    A = torch.load(ROOT / "data/lbm_anchors/dev_test/merged.pt", weights_only=False)
    for i in (0, 5):
        alpha, re, aoa = A["alpha"][i:i + 1].float(), A["re"][i:i + 1].float(), A["aoa"][i:i + 1].float()
        cond = Condition(float(re), float(aoa)).as_tensor(dtype=torch.float32)[None]
        b = _rotated(nurbs, alpha, cond, None, None)
        xy = A["xy"][i:i + 1].float()
        with torch.no_grad():
            out = model(xy, alpha, cond, wall_distance(xy, b), return_attn=True)
        with torch.enable_grad():
            from aeroshape.pinn.embeddings import predict_flow
            f = predict_flow(model, xy, alpha, cond, b)
        mag_p, mag_l = f.u.hypot(f.v)[0].detach(), A["u"][i].hypot(A["v"][i])
        body = (b.x[0].numpy(), b.y[0].numpy())
        plots.plot_field_comparison(xy[0], mag_l, mag_p, OUT / f"pinn_vs_lbm_speed_test{i}.png", f"|u| test shape {i}", body)
        plots.plot_attention(xy[0].numpy(), out.attn.mean(1)[0].detach().numpy(), nurbs.x_ctrl.numpy(), OUT / f"attention_test{i}.png",
                             f"test shape {i}", body)
    print("wrote PINN field comparison + attention figures")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    copy_existing()
    sel = json.load(open(ROOT / "runs/pinn_selected.json"))
    pinn_figures(sel)
    logs = {}
    for name in ("mlp", "transformer"):
        p = ROOT / sel[name]["checkpoint"]
        csv = p.parent.parent / "log.csv"
        if csv.exists():
            d = np.genfromtxt(csv, delimiter=",", names=True)
            logs[name] = {k: d[k] for k in d.dtype.names}
    plots.plot_training_log(logs, OUT / "training_loss.png", keys=("total", "data", "pde", "vel_l2"))
    res = json.load(open(ROOT / "runs/benchmark/dev/results.json"))["cases"]["1"]
    plots.plot_benchmark_bars(res, OUT / "benchmark_bars_case1.png")
    print("wrote training loss + benchmark bars ->", OUT)


if __name__ == "__main__":
    main()
