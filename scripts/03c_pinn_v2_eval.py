"""PINN v2 (Re 1000, time-averaged) evaluation per AoA on the v2 holdout, plus the time-averaged-adjoint gradient reference.
uv run python scripts/03c_pinn_v2_eval.py --adjoint-ref --device cpu            # reference dCD/dalpha for 3 holdout runs (slow, cached)
uv run python scripts/03c_pinn_v2_eval.py --model mlp --ckpt runs/pinn_mlp_v2_dev_full/ckpt/best.pt --device mps [--select mlp_v2]"""
import argparse
import importlib.util
import json
import statistics
from pathlib import Path

import torch

from aeroshape.core.config import load_config
from aeroshape.core.device import get_device
from aeroshape.core.io import load_ckpt
from aeroshape.core.timing import Timer
from aeroshape.core.types import Condition
from aeroshape.geometry.nurbs import NURBSAirfoil
from aeroshape.lbm.adjoint_solver import lbm_value_and_grad
from aeroshape.optimization.evaluators import PINNEvaluator
from aeroshape.pinn.stress_integrator import surface_forces

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("train03", ROOT / "scripts" / "03_train_transformer_pinn.py")
t3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t3)
ANCHORS, REF = "data/lbm_anchors/dev_v2/merged.pt", Path("runs/adjoint_cache/holdout_v2_ta.pt")
CASE, N_ADJ = "case2_circle_to_rae_re1000.yaml", 48
GRAD_RUNS = (0, 3, 5)  # holdout positions: s25@AoA0, s26@AoA10, s27@AoA10 (3 distinct shapes)


def adjoint_ref(A, device):
    cfg = load_config(CASE, "dev", [])
    nurbs = NURBSAirfoil(k=8, mode="cambered", device=device, dtype=torch.float32)
    idx = (A["split"] == 1).nonzero()[:, 0]
    ids = [A["meta"]["runs"][int(i)]["id"] for i in idx]
    done = torch.load(REF, weights_only=False) if REF.exists() else {}
    for j in GRAD_RUNS:
        i = int(idx[j])
        if ids[j] in done:
            continue
        r = lbm_value_and_grad(A["alpha"][i].to(device), nurbs, Condition(float(A["re"][i]), float(A["aoa"][i])), cfg, None, N_ADJ)
        done[ids[j]] = dict(dcd=r.dcd_dalpha.cpu(), dcl=r.dcl_dalpha.cpu(), cd=float(r.cd), cl=float(r.cl), wall_s=r.wall_s,
                            t_forward=r.extras["t_forward"], t_adjoint=r.extras["t_adjoint"], matvecs=r.extras["adjoint_matvecs"])
        REF.parent.mkdir(parents=True, exist_ok=True)
        torch.save(done, REF)
        print(f"adjoint {ids[j]}: cd {r.cd:.4f} (anchor {A['cd'][i]:.4f}) wall {r.wall_s:.0f}s fwd {done[ids[j]]['t_forward']:.0f}s adj {done[ids[j]]['t_adjoint']:.0f}s", flush=True)


def evaluate(A, model, nurbs, device):
    idx = (A["split"] == 1).nonzero()[:, 0]
    ids = [A["meta"]["runs"][int(i)]["id"] for i in idx]
    h = t3.holdout_eval(model, nurbs, A, device)
    alpha, cond, *_ = t3._flat_holdout(A, device, nurbs.x_ctrl.dtype)
    cd, cl = (x.detach().cpu() for x in surface_forces(model, nurbs, alpha, cond))
    cd0, cl0, aoa = A["cd"][idx], A["cl"][idx], A["aoa"][idx]
    rows = [dict(id=ids[k], aoa=float(aoa[k]), vel_l2=h["vel_l2_per_shape"][k], p_l2=h["p_l2_per_shape"][k], cd_pinn=float(cd[k]), cd_lbm=float(cd0[k]),
                 cd_err=float((cd[k] - cd0[k]).abs() / cd0[k].abs()), cl_pinn=float(cl[k]), cl_lbm=float(cl0[k]), cl_abs_err=float((cl[k] - cl0[k]).abs()))
            for k in range(len(ids))]
    per_aoa = {}
    for a in (0.0, 10.0):
        r = [x for x in rows if x["aoa"] == a]
        per_aoa[f"aoa{a:g}"] = {k: statistics.mean(x[k] for x in r) for k in ("vel_l2", "p_l2", "cd_err", "cl_abs_err")} | {
            "n": len(r), "cl_lbm_abs_mean": statistics.mean(abs(x["cl_lbm"]) for x in r), "cl_rel_err_mean": statistics.mean(x["cl_abs_err"] / abs(x["cl_lbm"]) for x in r)}
    out = dict(per_aoa=per_aoa, rows=rows, holdout_vel_l2=h["vel_l2"], device=str(device))
    if REF.exists():  # gradient cosine vs the time-averaged LBM adjoint, free entries (the design variables)
        ref, ev, free, cs = torch.load(REF, weights_only=False), PINNEvaluator(model, nurbs), nurbs.free_idx.cpu(), []
        for j, rid in enumerate(ids):
            if rid not in ref:
                continue
            r = ev.evaluate(alpha[j], Condition(float(torch.exp(cond[j, 0])), float(torch.rad2deg(torch.atan2(cond[j, 1], cond[j, 2])))))
            g, ga = r.dcd_dalpha.cpu(), ref[rid]["dcd"]
            cs.append(dict(id=rid, cos_cd=float(torch.nn.functional.cosine_similarity(g[free], ga[free], dim=0)),
                           cd_adjoint_run=ref[rid]["cd"], cd_pinn=float(r.cd)))
        out["grad"] = dict(per_run=cs, cos_cd_mean=statistics.mean(c["cos_cd"] for c in cs), adjoint_wall_s={k: v["wall_s"] for k, v in ref.items() if k != "__cfg__"})
    c0 = Condition(1000.0, 0.0)
    ev = PINNEvaluator(model, nurbs)
    for _ in range(3):
        ev.evaluate(alpha[0], c0)
    lat = []
    for _ in range(20):
        with Timer(device) as t:
            ev.evaluate(alpha[0], c0)
        lat.append(t.elapsed)
    out["latency_single_cd_grad_s"] = statistics.median(lat)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adjoint-ref", action="store_true")
    ap.add_argument("--model", choices=["mlp", "transformer"])
    ap.add_argument("--ckpt")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=None)
    ap.add_argument("--select", default=None, help="key to add to runs/pinn_selected.json (e.g. mlp_v2)")
    a = ap.parse_args()
    device, A = get_device(a.device), load_ckpt(ANCHORS, "cpu")
    if a.adjoint_ref:
        return adjoint_ref(A, device)
    model, nurbs = t3.load_for_eval(a.ckpt, a.model, device, A)
    res = evaluate(A, model, nurbs, device)
    out = Path(a.out or Path(a.ckpt).parents[1] / "eval_v2.json")
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}, indent=2))
    if a.select:
        p = Path("runs/pinn_selected.json")
        sel = json.loads(p.read_text())
        tr = json.loads((Path(a.ckpt).parents[1] / "results.json").read_text())
        sel[a.select] = {"checkpoint": a.ckpt, "best_step": tr["best_step"], "wall_s": tr["wall_s"], "fourier_scale": tr["fourier_scale"],
                         "data": "v2: 50 time-averaged Re 1000 runs (cambered) + 105 steady Re 20 runs, w_grad 0 on time-averaged", "eval_v2": str(out)}
        p.write_text(json.dumps(sel, indent=2))


if __name__ == "__main__":
    main()
