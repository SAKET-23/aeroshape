"""4-way benchmark (Plan/08_benchmark/four_way_benchmark.md): LBM adjoint vs MLP-PINN vs Transformer-PINN vs guided diffusion.
uv run python scripts/05_run_4way_benchmark.py --tier dev --device mps --cases 1 [--resume]
Each method x case is its own sub-run folder runs/benchmark/<tier>/case<n>_<method>/ (scripts/06_optimize_case.py for 1-3).
Every final shape is re-scored with the LBM (G16); wall-clock excludes training, which is reported as up-front cost."""
import argparse
import importlib.util
import json
import math
import os
import shutil
import statistics
import traceback
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from aeroshape.core.config import Config, load_config  # noqa: E402
from aeroshape.core.device import get_device  # noqa: E402
from aeroshape.core.io import load_ckpt  # noqa: E402
from aeroshape.core.seeding import seed_everything  # noqa: E402
from aeroshape.core.timing import Timer, peak_memory, reset_peak  # noqa: E402
from aeroshape.core.types import Condition  # noqa: E402
from aeroshape.diffusion.guided_shape_diffusion import GuidedSampler  # noqa: E402
from aeroshape.diffusion.schedule import CosineSchedule  # noqa: E402
from aeroshape.geometry.nurbs import NURBSAirfoil  # noqa: E402
from aeroshape.geometry.reference_shapes import shape_distance  # noqa: E402
from aeroshape.optimization.evaluators import LBMScoreEvaluator, PINNEvaluator  # noqa: E402
from aeroshape.visualization.plots import plot_denoise_gif  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CASES = {1: "case1_circle_to_naca_re20.yaml", 2: "case2_circle_to_rae_re1000.yaml", 3: "case3_naca_to_rae_aoa10.yaml"}
# run key -> (evaluator, optimizer.box); G37: boxed PINN rows are the headline, "_nobox" rows keep the surrogate-exploitation evidence
RUNS = {"adjoint_lbm": ("adjoint_lbm", None), "fd_lbm": ("fd_lbm", None), "mlp_pinn": ("mlp_pinn", "pinn_train"), "transformer_pinn": ("transformer_pinn", "pinn_train"),
        "mlp_pinn_nobox": ("mlp_pinn", None), "transformer_pinn_nobox": ("transformer_pinn", None)}
METHODS = ["adjoint_lbm", "mlp_pinn", "transformer_pinn", "diffusion", "mlp_pinn_nobox", "transformer_pinn_nobox"]
HYBRID = "hybrid_transformer_pinn"  # --hybrid only: PINN gradient, LBM line search (evaluator lives in optimization/evaluators.py)
RUNS[HYBRID] = (HYBRID, "pinn_train")
BOX_ANCHORS = {1: {"mlp_pinn": "data/lbm_anchors/dev/merged_v1v1b.pt",  # v1b MLP: 80 runs (pinn_selected.json)
                   "transformer_pinn": "data/lbm_anchors/dev/merged.pt"},  # v1c Transformer: 110 runs
               2: dict.fromkeys(("mlp_pinn", "transformer_pinn"), "data/lbm_anchors/dev_v2/merged_v2v2b.pt")}
BOX_ANCHORS[3] = BOX_ANCHORS[2]
TARGET_NAME = {1: "NACA0012", 2: "RAE2822", 3: "RAE2822"}
CAMBERED_DIFFUSION = "runs/diffusion_v2_dev/ckpt/diffusion.pt"
DRIVERS = ("adjoint_lbm", "fd_lbm")  # method 1; G42: fd_lbm replaces the adjoint for --fd-cases
LABEL = {"adjoint_lbm": "LBM adjoint", "fd_lbm": "LBM finite-difference gradient", "mlp_pinn": "MLP-PINN", "transformer_pinn": "Transformer-PINN",
         "diffusion": "Guided diffusion", "mlp_pinn_nobox": "MLP-PINN (no envelope)", "transformer_pinn_nobox": "Transformer-PINN (no envelope)",
         HYBRID: "Hybrid Transformer-PINN (LBM line search)"}
N_SAMPLES = 8


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def resume_args(resume, d):
    """Sub-run with a per-cycle state.pt but no results.json continues (06 --resume) instead of restarting."""
    return ["--resume"] if resume and (d / "state.pt").exists() and not (d / "results.json").exists() else []


def _abs(p):
    return Path(p) if Path(p).is_absolute() else ROOT / p


def models_for(case, a):
    """Checkpoints, training times and (cases 2-3) the accepted diffusion guidance pair for the case."""
    sel = json.loads((ROOT / "runs/pinn_selected.json").read_text())
    if case == 1:
        dck, mk, tk = _abs(a.diffusion_ckpt), ("mlp", "transformer"), None
    else:
        if "mlp_v2" not in sel or "transformer_v2" not in sel:
            raise SystemExit(f"case {case} needs PINN v2: runs/pinn_selected.json lacks mlp_v2 / transformer_v2")
        dck, mk = _abs(CAMBERED_DIFFUSION), ("mlp_v2", "transformer_v2")
    if not dck.exists():
        raise SystemExit(f"{'cambered ' if case > 1 else ''}diffusion checkpoint not found: {dck}")
    dres = json.loads((dck.parents[1] / "results.json").read_text())
    # G36: the best valid pair on the beyond-range targets (04 writes it as chosen_pair)
    pair = (dres.get("chosen_pair") or ((dres.get("acceptance") or {}).get("beyond_range") or {}).get("best_pair")) if case > 1 else None
    return {"mlp_pinn": _abs(sel[mk[0]]["checkpoint"]), "transformer_pinn": _abs(sel[mk[1]]["checkpoint"]), "diffusion": dck, "diffusion_pair": pair,
            "train_s": {"mlp_pinn": sel[mk[0]]["wall_s"], "transformer_pinn": sel[mk[1]]["wall_s"], "diffusion_only": dres["train"]["wall_s"]}}


def check_box_dim(anchor_file, cfg):
    """The box anchors' alpha width must equal the case's NURBS D (8 symmetric, 16 cambered)."""
    g = cfg.geometry
    D = NURBSAirfoil(k=g.k, degree=g.degree, M=g.M, mode=g.mode, dtype=torch.float32).D
    w = load_ckpt(_abs(anchor_file), "cpu")["alpha"].shape[1]
    if w != D:
        raise ValueError(f"box anchors {anchor_file} have alpha width {w}, but the {g.mode} case needs D={D}")


def up_front(method, train_s):
    """Training wall-clock (s) behind the method; diffusion also needs the Transformer-PINN for guidance. Data generation not included."""
    return 0.0 if method in DRIVERS else train_s["transformer_pinn"] + train_s["diffusion_only"] if method == "diffusion" \
        else train_s["transformer_pinn" if method == HYBRID else method.removesuffix("_nobox")]


def latency(model, nurbs, alpha, cond, reps=20):
    """Median wall-clock of one PINN CD + CL + gradient evaluation (after warm-up)."""
    ev = PINNEvaluator(model, nurbs)
    for _ in range(3):
        ev.evaluate(alpha, cond)
    ts = []
    for _ in range(reps):
        with Timer(alpha.device) as t:
            ev.evaluate(alpha, cond)
        ts.append(t.elapsed)
    return statistics.median(ts)


def run_diffusion(cfg, opt06, a, out, cd_target, cl_ref, models, device):
    """One guided pass (batch of N_SAMPLES, best by PINN J), re-scored with the LBM. Returns the results dict."""
    c, dc = cfg.case, cfg.diffusion
    g = cfg.geometry
    nurbs = NURBSAirfoil(k=g.k, degree=g.degree, M=g.M, mode=g.mode, device=device, dtype=torch.float32)
    pinn = opt06.load_model(cfg, nurbs, "transformer_pinn", models["transformer_pinn"], device)
    ck = load_ckpt(models["diffusion"], device)
    ema = _load("train04", "04_train_diffusion_model.py").build_denoiser(Config(ck["config"]), nurbs, device).eval()
    ema.load_state_dict(ck["model"])
    sampler = GuidedSampler(ema, CosineSchedule(Config(ck["config"]).diffusion.T), nurbs, pinn, tuple(x.to(device) for x in ck["norm"]),
                            {"w_L": dc.w_lift, "mu": dc.mu_thickness})
    cond = torch.tensor([[cl_ref or 0.0, cd_target, math.log(c.re), math.radians(c.aoa)]] * N_SAMPLES, device=device)
    t_min = opt06.case_t_min(nurbs, c.t_min).to(device).expand(N_SAMPLES, -1)
    kw = dict(cfg_scale=dc.cfg_scale, guide_scale=dc.guide_scale, guide_clip=dc.guide_clip)
    sampler.sample(cond, t_min, n_steps=min(5, dc.sample_steps), generator=torch.Generator().manual_seed(0), **kw)  # warm-up
    reset_peak(device)
    with Timer(device) as tm:
        s = sampler.sample(cond, t_min, n_steps=dc.sample_steps, generator=torch.Generator().manual_seed(cfg.seed), **kw)
        with torch.enable_grad():
            J = sampler.objective(s.alpha.detach(), cond, t_min).detach()
        best = int(J.argmin())
    alpha = s.alpha[best:best + 1].detach()
    with torch.enable_grad():
        own_cd, own_cl = (float(x[0].detach()) for x in sampler.physics_fn(alpha, cond[:1]))
    N = cfg.lbm.cells_per_d_high_re if c.re > 40 else None
    cd, cl = LBMScoreEvaluator(cfg, nurbs, N).value(alpha.reshape(-1), Condition(c.re, c.aoa))
    ref64 = NURBSAirfoil(k=g.k, M=g.M, mode=g.mode, dtype=torch.float64)
    res = dict(case=c.to_dict(), evaluator="diffusion", cycles=dc.sample_steps, n_samples=N_SAMPLES, best_sample=best, wall_s=tm.elapsed,
               peak_mem_bytes=peak_memory(device), cd_target=cd_target, cd_final_lbm=float(cd), cl_final_lbm=float(cl), cl_ref=cl_ref,
               cd_own_last=own_cd, cl_own_last=own_cl, J_best=float(J[best]), J_all=J.tolist(), stop_reason="one_pass",
               viol_final=float(torch.relu(t_min[0] - nurbs.thickness_at_ctrl(alpha).reshape(-1)).max()),
               shape_distance_to_target=float(target_distance(opt06, ref64, nurbs, alpha, c.target)),
               alpha_final=alpha.reshape(-1).tolist(), guide_scale=dc.guide_scale, cfg_scale=dc.cfg_scale)
    (out / "results.json").write_text(json.dumps(res, indent=2))
    return res


def target_distance(opt06, ref64, nurbs, alpha, target):
    """RMS y-distance of alpha [1, D] to the fitted reference; keep [1, D] (cambered _split indexes a[:, :k])."""
    return shape_distance(ref64, alpha.cpu().double(), opt06.shape(nurbs, target).cpu().double())


def row(method, r, r1, lat, train_s):
    """Metrics table row from a sub-run's results.json; drag reduction uses method 1's LBM CD of the start shape."""
    cd0 = r1["cd_first"]
    cl_ref = r["cl_ref"]
    return dict(method=method, cd_lbm=r["cd_final_lbm"], cl_lbm=r["cl_final_lbm"], drag_reduction=(cd0 - r["cd_final_lbm"]) / cd0,
                wall_s=r["wall_s"], up_front_s=up_front(method, train_s), peak_mem_bytes=r["peak_mem_bytes"], iterations=r["cycles"],
                stop_reason=r["stop_reason"], cd_own=r["cd_own_last"], cl_own=r["cl_own_last"], viol=r["viol_final"],
                cl_err=abs(r["cl_final_lbm"] - (cl_ref or 0.0)), dist_to_target=r["shape_distance_to_target"], latency_s=lat)


def fmt(v, f="{:.4g}"):
    return "-" if v is None else f.format(v)


def write_md(results, path, target_names=None):
    cols = [("method", "Method", "{}"), ("cd_lbm", "CD (LBM)", "{:.4f}"), ("cl_lbm", "CL (LBM)", "{:.4f}"),
            ("drag_reduction", "Drag red.", "{:.2%}"), ("wall_s", "Wall (s)", "{:.1f}"), ("up_front_s", "Up-front (s)", "{:.0f}"),
            ("iterations", "Iters", "{}"), ("viol", "Max t viol.", "{:.2e}"), ("cl_err", "abs(CL-CL_ref)", "{:.2e}"),
            ("dist_to_target", "Dist to target", "{:.4f}"), ("latency_s", "Latency (s)", "{:.4g}"), ("cd_own", "CD (own, last)", "{:.4f}"),
            ("peak_mem_bytes", "Peak mem (MB)", None)]
    lines = []
    for case, rows in results["cases"].items():
        cols[-4] = (cols[-4][0], f"Dist to {(target_names or {}).get(str(case), 'NACA0012')}", cols[-4][2])
        lines += [f"## Case {case} (tier {results['tier']}, device {results['device']})", "",
                  "| " + " | ".join(c[1] for c in cols) + " |", "|" + "---|" * len(cols)]
        for r in rows:
            lines.append("| " + " | ".join(fmt(r[k] / 1e6, "{:.0f}") if f is None else fmt(r[k], f) if k != "method" else LABEL[r[k]]
                                           for k, _, f in cols) + " |")
        lines.append("")
    lines += ["## Summary", "", "| Case | Method | CD (LBM) | Drag red. | Wall (s) |", "|---|---|---|---|---|"]
    lines += [f"| {c} | {LABEL[r['method']]} | {r['cd_lbm']:.4f} | {r['drag_reduction']:.2%} | {r['wall_s']:.1f} |"
              for c, rows in results["cases"].items() for r in rows]
    lines += ["", "Wall-clock excludes training and includes the line-search value evaluations; diffusion wall = one batch of "
              f"{N_SAMPLES} samples + best-of selection. CD/CL (own) is the last point the method evaluated itself (methods 1-3: one cycle "
              "behind the final shape). Up-front excludes data generation. Latency: one PINN CD + CL + gradient evaluation."]
    path.write_text("\n".join(lines) + "\n")


def plot_shapes(path, case, shapes, nurbs):
    """shapes: {name: alpha [D]}; start thin grey dashed, target thick black dashed on top, legend outside the axes."""
    fig, ax = plt.subplots(figsize=(9, 4))
    for name, al in shapes.items():
        x, yu, yl = (t.detach().cpu().numpy() for t in nurbs.surface_y(al.reshape(1, -1).to(nurbs.x_ctrl)))
        kw = (dict(color="0.55", ls="--", lw=1, zorder=2) if name == "start" else dict(color="k", ls="--", lw=2.5, zorder=5) if name == "target"
              else dict(lw=1.4, zorder=3))
        ln, = ax.plot(x, yu[0], label=LABEL.get(name, name), **kw)
        ax.plot(x, yl[0], **{**kw, "color": ln.get_color()})
    ax.margins(y=0.15)
    ax.set(aspect="equal", title=f"Case {case}: start, final shapes, target {TARGET_NAME[case]}")
    ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def figures(out, case, c, rows, shapes, hist, nurbs):
    """shapes: {name: alpha [D]}; hist: {method: history dict}."""
    (out / "fig").mkdir(exist_ok=True)
    plot_shapes(out / "fig" / f"shapes_case{case}.png", case, shapes, nurbs)
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    for r in rows:
        ax.scatter(r["wall_s"], r["cd_lbm"], label=LABEL[r["method"]], zorder=3)
    drv = next((k for k in DRIVERS if k in hist), None)
    if drv:  # method 1 trace: its own (LBM) CD per cycle vs cumulative wall-clock
        ax.plot(hist[drv]["wall_s"].cumsum(0), hist[drv]["cd"], color="gray", lw=0.8, label=f"{LABEL[drv]} trace")
    ax.set(xscale="log", xlabel="wall-clock to final shape (s)", ylabel="CD (LBM-scored)", title=f"Case {case}")
    ax.legend(fontsize=7)
    fig.savefig(out / "fig" / f"cd_vs_wallclock_case{case}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    for m, h in hist.items():
        plot_denoise_gif(nurbs, h["alpha"], out / "fig" / f"morph_case{case}_{m}.gif", label="cycle")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="dev", choices=["smoke", "dev", "full"])
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    ap.add_argument("--cases", type=int, nargs="+", default=[1], choices=[1, 2, 3])
    ap.add_argument("--fd-cases", type=int, nargs="*", default=[], choices=[1, 2, 3], help="cases whose LBM driver (method 1) is fd_lbm, not adjoint_lbm (G42)")
    ap.add_argument("--hybrid", action="store_true", help=f"also run {HYBRID} (PINN gradient, LBM line search)")
    ap.add_argument("--resume", action="store_true", help="skip sub-runs that already have a results.json; continue ones with a state.pt from their last cycle")
    ap.add_argument("--out", default=None)
    ap.add_argument("--diffusion-ckpt", default="runs/diffusion_dev/ckpt/diffusion.pt")
    ap.add_argument("-o", "--set", action="append", default=[], dest="sets", help="config override passed to every sub-run")
    a = ap.parse_args(argv)
    out = Path(a.out or ROOT / "runs/benchmark" / a.tier)
    opt06 = _load("opt06", "06_optimize_case.py")
    device = get_device(a.device)
    models = {c: models_for(c, a) for c in a.cases}  # fail before any work
    all_runs = {k: v for k, v in RUNS.items() if k != HYBRID or a.hybrid}
    all_methods = METHODS + [HYBRID] * a.hybrid
    results = {"tier": a.tier, "device": device.type, "overrides": a.sets, "cases": {}, "failures": {}}
    for case in a.cases:
        m, case_file = models[case], CASES[case]
        pair = m["diffusion_pair"]
        gsets = [f"diffusion.guide_scale={pair['guide_scale']}", f"diffusion.cfg_scale={pair['cfg_scale']}"] if pair else []
        if case > 1:
            print(f"case{case} diffusion guidance: {'04 results.json chosen_pair / beyond_range' if pair else 'config fallback'}", flush=True)
        cfg = load_config(case_file, a.tier, gsets + a.sets)  # user -o wins
        c = cfg.case
        g = cfg.geometry
        drv = "fd_lbm" if case in a.fd_cases else "adjoint_lbm"
        runs = {k: v for k, v in all_runs.items() if k == drv or k not in DRIVERS}
        methods = [drv if k == "adjoint_lbm" else k for k in all_methods]
        done, hist, lat = {}, {}, {}
        seed_everything(cfg.seed)
        for meth, (ev, box) in runs.items():
            d = out / f"case{case}_{meth}"
            if not (a.resume and (d / "results.json").exists()):
                cycles = cfg.optimizer.max_cycles * (1 if ev in DRIVERS else 10)
                args = resume_args(a.resume, d) + ["--case", case_file, "--evaluator", ev, "--tier", a.tier, "--device", a.device, "--out", str(d),
                        *sum((["--set", s] for s in a.sets), []), "--set", f"optimizer.max_cycles={cycles}"]
                ck = "transformer_pinn" if ev == HYBRID else ev
                args += ["--ckpt", str(m[ck])] if ev not in DRIVERS else []
                try:
                    if box:
                        check_box_dim(BOX_ANCHORS[min(case, 2)][ck], cfg)
                        args += ["--set", f"optimizer.box={box}", "--box-anchors", BOX_ANCHORS[min(case, 2)][ck]]
                    opt06.main(args)
                except (Exception, SystemExit):
                    results["failures"][f"case{case}_{meth}"] = traceback.format_exc()
                    print(f"FAILED case{case}_{meth}\n{traceback.format_exc()}", flush=True)
                    continue
            done[meth] = json.loads((d / "results.json").read_text())
            hist[meth] = torch.load(d / "history.pt")
            (out / "histories").mkdir(exist_ok=True)
            shutil.copy(d / "history.pt", out / "histories" / f"case{case}_{meth}.pt")
        nurbs = NURBSAirfoil(k=g.k, degree=g.degree, M=g.M, mode=g.mode, device=device, dtype=torch.float32)
        for meth in (k for k in runs if k != drv):
            if meth in done:
                d = out / f"case{case}_{meth}"
                if not (a.resume and (d / "latency.json").exists()):
                    model = opt06.load_model(cfg, nurbs, "transformer_pinn" if meth == HYBRID else RUNS[meth][0], m["transformer_pinn" if meth == HYBRID else RUNS[meth][0]], device)
                    (d / "latency.json").write_text(json.dumps(
                        {"latency_s": latency(model, nurbs, opt06.shape(nurbs, c.start).reshape(1, -1), Condition(c.re, c.aoa))}))
                lat[meth] = json.loads((d / "latency.json").read_text())["latency_s"]
        if drv in done:
            d = out / f"case{case}_diffusion"
            try:
                if not (a.resume and (d / "results.json").exists()):
                    d.mkdir(parents=True, exist_ok=True)
                    run_diffusion(cfg, opt06, a, d, done[drv]["cd_final_lbm"], done[drv]["cl_ref"], m, device)
                done["diffusion"] = json.loads((d / "results.json").read_text())
            except (Exception, SystemExit):
                results["failures"][f"case{case}_diffusion"] = traceback.format_exc()
                print(f"FAILED case{case}_diffusion\n{traceback.format_exc()}", flush=True)
        else:
            results["failures"][f"case{case}_diffusion"] = "needs method 1's final CD as the target"
        if drv not in done:
            continue
        rows = [row(k, done[k], done[drv], lat.get(k), m["train_s"]) for k in methods if k in done]
        results["cases"][case] = rows
        shapes = {"start": opt06.shape(nurbs, c.start), "target": opt06.shape(nurbs, c.target),
                  **{k: torch.tensor(done[k]["alpha_final"]) for k in done}}
        figures(out, case, c, rows, shapes, hist, nurbs)
        results[f"case{case}_target_cd"] = done[drv]["cd_final_lbm"]
    (out / "results.json").write_text(json.dumps(results, indent=2))
    write_md(results, out / "results.md", {str(c): TARGET_NAME[c] for c in a.cases})
    print("wrote", out / "results.json")
    return 1 if results["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
