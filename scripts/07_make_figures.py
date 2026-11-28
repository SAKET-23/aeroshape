"""Regenerate every available figure into runs/figures/ from existing results (no solver runs; PINN forward passes on CPU).
uv run python scripts/07_make_figures.py"""
import argparse
import importlib.util
import json
import os
import shutil
from types import SimpleNamespace
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


def state_overlay(spec):
    """spec 'CASE:STATE_PT': start / target / last / best-feasible shapes of a Re 1000 optimizer checkpoint -> overlay_<case>.png."""
    import matplotlib.pyplot as plt
    from aeroshape.core.config import load_config
    from aeroshape.geometry.nurbs import NURBSAirfoil
    from aeroshape.optimization.kkt_optimizer import AugmentedLagrangianOptimizer
    case, pt = spec.split(":", 1)
    b5, o6 = _load("bench05", "05_run_4way_benchmark.py"), _load("opt06", "06_optimize_case.py")
    cfg = load_config(b5.CASES[int(case)], "dev", [])
    g = cfg.geometry
    nurbs = NURBSAirfoil(k=g.k, degree=g.degree, M=g.M, mode=g.mode, dtype=torch.float32)
    rec = torch.load(ROOT / pt, weights_only=False)["records"]
    # cl_ref: newest cached LBM reference for this case's AoA (written by 06's lbm_cl_ref); no LBM run here
    refs = sorted((ROOT / "runs" / "refs").glob(f"rae2822_re{cfg.case.re:g}_aoa{cfg.case.aoa:g}_*.json"), key=os.path.getmtime)
    cl_ref = json.loads(refs[-1].read_text())["cl"] if refs and cfg.case.cl_ref == "rae2822@lbm" else cfg.case.cl_ref
    a_best = AugmentedLagrangianOptimizer.best_alpha.fget(SimpleNamespace(records=rec, cl_ref=cl_ref, o=cfg.optimizer, nurbs=nurbs))
    best = next(r for r in rec if torch.equal(r["alpha"].reshape(1, -1).to(a_best), a_best))
    feas = best["viol"] < 1e-3 and (cl_ref is None or abs(best["cl"] - cl_ref) < cfg.optimizer.cl_tol)
    items = [("start", rec[0]["alpha"], "k", "0.55", 1, "--", 1.0), ("RAE2822 target", o6.shape(nurbs, "rae2822"), None, "k", 5, "--", 2.5),
             ("last", rec[-1]["alpha"], rec[-1], "C0", 3, "-", 1.4), ("best feasible" if feas else "best L", best["alpha"], best, "C3", 4, "-", 1.4)]
    fig, ax = plt.subplots(figsize=(9, 4))
    for name, al, r, col, z, ls, lw in items:
        r = rec[0] if name == "start" else r
        lab = name + (f"  CD {r['cd']:.3f}  CL {r['cl']:.3f}" if r else "")
        x, yu, yl = (t.detach().numpy() for t in nurbs.surface_y(al.reshape(1, -1).float()))
        ax.plot(x, yu[0], color=col, ls=ls, lw=lw, zorder=z, label=lab)
        ax.plot(x, yl[0], color=col, ls=ls, lw=lw, zorder=z)
    ax.margins(y=0.15)
    ax.set(aspect="equal", title=f"Case {case}: start, last, best feasible, target RAE2822")
    ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.savefig(OUT / f"overlay_{case}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / f"overlay_{case}.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state-overlay", help="CASE:STATE_PT[,CASE:STATE_PT...]; only draws the overlays")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.state_overlay:
        for spec in a.state_overlay.split(","):
            state_overlay(spec)
        return
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
