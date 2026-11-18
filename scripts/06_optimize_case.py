"""Constrained shape optimization of one test case (Plan/05_optimization/kkt_optimizer.md).
uv run python scripts/06_optimize_case.py --case case1_circle_to_naca_re20.yaml --evaluator adjoint_lbm --tier dev --device mps
Writes runs/<name>/{config.yaml,log.csv,history.pt,results.json}."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import torch  # noqa: E402

from aeroshape.core.config import Config, dump_config, load_config  # noqa: E402
from aeroshape.core.device import device_info, get_device  # noqa: E402
from aeroshape.core.io import load_ckpt  # noqa: E402
from aeroshape.core.seeding import seed_everything  # noqa: E402
from aeroshape.core.timing import Timer, peak_memory, reset_peak  # noqa: E402
from aeroshape.core.types import Condition  # noqa: E402
from aeroshape.geometry.nurbs import NURBSAirfoil  # noqa: E402
from aeroshape.geometry.reference_shapes import (circle_alpha, naca4_alpha, rae2822_alpha,  # noqa: E402
                                                 shape_distance)
from aeroshape.optimization.evaluators import LBMScoreEvaluator, build_evaluator  # noqa: E402
from aeroshape.optimization.kkt_optimizer import AugmentedLagrangianOptimizer, train_box  # noqa: E402
from aeroshape.pinn.mlp_pinn import MLPPINN  # noqa: E402
from aeroshape.pinn.transformer_pinn import GeometryCrossAttentionPINN  # noqa: E402

REFS = Path("runs/refs")


def shape(nurbs, name):
    """Fitted reference shape on the optimizer's device/dtype (the fit itself runs on a CPU float64 twin, see main)."""
    ref = NURBSAirfoil(k=nurbs.k, M=nurbs.M, mode=nurbs.mode, dtype=torch.float64)
    return (circle_alpha(ref) if name == "circle" else rae2822_alpha(ref) if name == "rae2822"
            else naca4_alpha(ref, name[4:])).to(nurbs.mask)


def lbm_cl_ref(cfg, nurbs, cond, N, name, device):
    """LBM CL of the fitted `name` at the case condition, cached in runs/refs/ (keyed by the LBM settings)."""
    key = hashlib.md5(json.dumps([cfg.lbm.to_dict(), cfg.adjoint.to_dict(), cfg.geometry.to_dict(), name, cond.re, cond.aoa_deg, N,
                                  str(device)], sort_keys=True).encode()).hexdigest()[:10]
    f = REFS / f"{name}_re{cond.re:g}_aoa{cond.aoa_deg:g}_{key}.json"
    if not f.exists():
        cd, cl = LBMScoreEvaluator(cfg, nurbs, N).value(shape(nurbs, name), cond)
        REFS.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps({"cd": float(cd), "cl": float(cl)}))
    return json.loads(f.read_text())["cl"]


def case_t_min(nurbs, t_min_name):
    """Per-control-point floor of the case: thickness of the named shape, zero at the LE / TE entries."""
    t_min = torch.relu(nurbs.thickness_at_ctrl(shape(nurbs, t_min_name))).reshape(-1).detach()
    t_min[[0, 1, -1]] = 0  # thickness is identically 0 at the LE / TE control points
    return t_min


def load_model(cfg, nurbs, kind, ckpt, device):
    if ckpt is None:
        raise SystemExit(f"evaluator {kind} needs --ckpt <checkpoint.pt>")
    ck = load_ckpt(ckpt, device)
    gk = ck.get("config", {}).get("geometry", {})
    bad = {k: (gk[k], getattr(cfg.geometry, k)) for k in ("k", "degree", "M") if k in gk and gk[k] != getattr(cfg.geometry, k)}
    if bad:
        raise ValueError(f"checkpoint geometry != case geometry (ckpt, case): {bad}")
    P = Config(ck["config"]).pinn if "config" in ck else cfg.pinn  # architecture from the checkpoint's own config
    v2 = P.version == "v2"
    m = (MLPPINN(nurbs.D, P.mlp_hidden, P.mlp_layers, P.fourier_features, P.fourier_scale, v2) if kind == "mlp_pinn" else
         GeometryCrossAttentionPINN(nurbs, P.d_model, P.heads, P.self_attn_layers, P.cross_attn_layers, v2, P.fourier_features,
                                    P.fourier_scale))
    m.load_state_dict(next(ck[k] for k in ("model", "model_state", "state_dict") if k in ck))
    m.cond_range = ck.get("train_range")  # L1: PINNEvaluator refuses conditions outside it
    if m.cond_range is None:
        print(f"WARNING: {ckpt} has no stored training range; Re/AoA are not range-checked", flush=True)
    return m.to(device).eval()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--evaluator", default=None, choices=["adjoint_lbm", "mlp_pinn", "transformer_pinn"])
    ap.add_argument("--tier", default="dev", choices=["smoke", "dev", "full"])
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--set", action="append", default=[], dest="sets")
    ap.add_argument("--out", default=None)
    ap.add_argument("--box-anchors", default="data/lbm_anchors/dev/merged.pt",
                    help="anchor file whose train split defines the optimizer.box=pinn_train envelope (the data the PINN was trained on)")
    ap.add_argument("--ckpt", default=None, help="PINN checkpoint (required for the PINN evaluators)")
    a = ap.parse_args(argv)

    cfg = load_config(a.case, a.tier, a.sets)
    ev_name = a.evaluator or cfg.optimizer.evaluator
    out = Path(a.out or f"runs/{Path(a.case).stem}_{ev_name}_{a.tier}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(dump_config(cfg))
    seed_everything(a.seed if a.seed is not None else cfg.seed)
    device = get_device(a.device)
    pinn = ev_name != "adjoint_lbm"
    dtype = torch.float64 if device.type == "cpu" and not pinn else torch.float32  # float64 only on CPU
    c = cfg.case
    cond = Condition(c.re, c.aoa)
    N = cfg.lbm.cells_per_d_high_re if c.re > 40 else None
    g = cfg.geometry
    nurbs = NURBSAirfoil(k=g.k, degree=g.degree, M=g.M, mode=g.mode, device=device, dtype=dtype)

    t_min = case_t_min(nurbs, c.t_min)
    cl_ref = lbm_cl_ref(cfg, nurbs, cond, N, "rae2822", device) if c.cl_ref == "rae2822@lbm" else c.cl_ref
    model = load_model(cfg, nurbs, ev_name, a.ckpt, device) if pinn else None
    if cfg.optimizer.box not in (None, "pinn_train"):
        raise SystemExit(f"optimizer.box must be null or pinn_train, got {cfg.optimizer.box!r}")
    box = tuple(b.to(device, dtype) for b in train_box(load_ckpt(a.box_anchors, "cpu"))) if cfg.optimizer.box else None
    opt = AugmentedLagrangianOptimizer(nurbs, build_evaluator(cfg, ev_name, nurbs, N, model), cond, t_min, cl_ref, cfg, box)

    ref64 = NURBSAirfoil(k=g.k, M=g.M, mode=g.mode, dtype=torch.float64)
    rows = []

    def log(r):
        rows.append({k: v for k, v in r.items() if k != "alpha"})
        print(f"cycle {len(rows)}: cd {r['cd']:.5f} cl {r['cl']:.5f} L {r['L']:.5f} viol {r['viol']:.2e} kkt {r['kkt']:.2e} "
              f"eta {r['eta']:.4g} bt {r['n_backtrack']}{' FLAG' if r['flagged'] else ''} wall {r['t_cycle']:.1f}s", flush=True)

    # warm-up on a throwaway evaluator (MPS shader compile / CUDA context) so the timed run does not pay it and keeps a cold warm-start
    warm = build_evaluator(cfg, ev_name, nurbs, N, model)
    warm.evaluate(shape(nurbs, c.start), cond, opt._weights())
    warm.value(shape(nurbs, c.start), cond, opt._weights())
    reset_peak(device)
    with Timer(device) as tm:
        h = opt.run(shape(nurbs, c.start), cfg.optimizer.max_cycles, log)
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k != "t_cycle", k))
    with open(out / "log.csv", "w", newline="") as f:
        w = csv.DictWriter(f, ["cycle"] + keys)
        w.writeheader()
        w.writerows({"cycle": i + 1, **r} for i, r in enumerate(rows))
    torch.save({k: v.cpu() for k, v in h.items()}, out / "history.pt")

    # re-score the final shape with the LBM (forward only), whatever evaluator drove the optimization
    cd, cl = LBMScoreEvaluator(cfg, nurbs, N).value(opt.alpha, cond)
    target = shape(nurbs, c.target)
    res = dict(case=c.to_dict(), evaluator=ev_name, box=cfg.optimizer.box, tier=a.tier, cycles=len(rows), wall_s=tm.elapsed, peak_mem_bytes=peak_memory(device),
               cd_own_last=float(h["cd"][-1]), cl_own_last=float(h["cl"][-1]), stop_reason=opt.stop_reason,
               cd_first=float(h["cd"][0]), cd_final_lbm=float(cd), cl_final_lbm=float(cl), cl_ref=cl_ref,
               viol_final=float(torch.relu(t_min.to(nurbs.mask) - nurbs.thickness_at_ctrl(opt.alpha).reshape(-1)).max()),
               shape_distance_start=float(shape_distance(ref64, shape(nurbs, c.start).cpu().double(), target.cpu().double())),
               shape_distance_to_target=float(shape_distance(ref64, opt.alpha.cpu().double(), target.cpu().double())),
               flagged_cycles=int(h["flagged"].sum()), alpha_final=opt.alpha.reshape(-1).tolist(), device_info=device_info(device),
               config_hash=hashlib.md5(dump_config(cfg).encode()).hexdigest()[:10])
    (out / "results.json").write_text(json.dumps(res, indent=2))
    print("wrote", out / "results.json")


if __name__ == "__main__":
    main()
