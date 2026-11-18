"""WP 17: physics-guided shape diffusion (Plan/06_training/diffusion_training.md), symmetric k = 8, Re 20, AoA 0 only.
uv run python scripts/04_train_diffusion_model.py --tier dev --device mps --out runs/diffusion_dev
Writes <out>/{config.yaml,results.json,ckpt/diffusion.pt,fig/} and data/diffusion_shapes/<tier>_symmetric.pt.
Scope: the selected PINN (v1c) is trained at Re 20-40, AoA 0 on symmetric shapes, so cambered / Re 1000 waits for PINN v2."""
import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import torch  # noqa: E402

from aeroshape.core.config import dump_config, load_config  # noqa: E402
from aeroshape.core.device import device_info, get_device  # noqa: E402
from aeroshape.core.io import load_ckpt, save_ckpt  # noqa: E402
from aeroshape.core.seeding import seed_everything  # noqa: E402
from aeroshape.core.timing import Timer  # noqa: E402
from aeroshape.core.types import Condition  # noqa: E402
from aeroshape.diffusion.denoiser import ShapeDenoiser  # noqa: E402
from aeroshape.diffusion.guided_shape_diffusion import GuidedSampler, diffusion_loss  # noqa: E402
from aeroshape.diffusion.schedule import CosineSchedule  # noqa: E402
from aeroshape.geometry.nurbs import NURBSAirfoil  # noqa: E402
from aeroshape.geometry.reference_shapes import circle_alpha, is_valid, lhs_shapes, naca4_alpha, perturb  # noqa: E402
from aeroshape.optimization.evaluators import LBMScoreEvaluator  # noqa: E402
from aeroshape.pinn.stress_integrator import surface_forces  # noqa: E402
from aeroshape.visualization.plots import plot_denoise_gif, plot_guidance_sweep, plot_shapes  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
COND = Condition(20.0, 0.0)
ANCHORS = "data/lbm_anchors/dev/merged.pt"
LABEL_CHUNK = 128


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def in_envelope(a, lo, hi):
    """[S] bool: every coordinate inside the per-coordinate [lo, hi] box of the PINN's training shapes."""
    return ((a >= lo) & (a <= hi)).all(1)


def ema_update(ema, model, decay):
    with torch.no_grad():
        for pe, pm in zip(ema.parameters(), model.parameters()):
            pe.lerp_(pm, 1 - decay)


def _forces(pinn, nurbs, alpha, pc):
    cd, cl = surface_forces(pinn, nurbs, alpha, pc)
    return cd.detach(), cl.detach()


def pinn_labels(pinn, nurbs, alpha):
    """-> cond [S, 4] = [CL, CD, log Re, AoA rad] from the frozen PINN, batched."""
    pc = COND.as_tensor(alpha.device).repeat(alpha.shape[0], 1)
    out = []
    for i in range(0, alpha.shape[0], LABEL_CHUNK):
        out.append(_forces(pinn, nurbs, alpha[i:i + LABEL_CHUNK], pc[i:i + LABEL_CHUNK]))  # graph freed on return (MPS memory cliff)
    cd, cl = torch.cat([o[0] for o in out]), torch.cat([o[1] for o in out])
    return torch.stack([cl, cd, pc[:, 0], torch.zeros_like(cd)], 1)


def random_blends(chk, ref, m, seed):
    """[m, D] circle <-> {NACA0012, NACA0024, ellipse t in 0.3..1.1} blends with random weight w in 0.05..0.95 + smooth perturbation (G35)."""
    g = torch.Generator().manual_seed(seed)
    circ = circle_alpha(ref)
    tgt = torch.cat([naca4_alpha(ref, "0012"), naca4_alpha(ref, "0024"), *(chk._ellipse(ref, float(t)) for t in torch.linspace(0.3, 1.1, 9))])
    w = 0.05 + 0.9 * torch.rand(m, 1, generator=g, dtype=torch.float64)
    return perturb((1 - w) * circ + w * tgt[torch.randint(0, tgt.shape[0], (m,), generator=g)], m, 0.015, 0.3, seed + 1, ref)


def make_shapes(cfg, A, ref):
    """G35: mix of the 'diffusion' LHS family, v1c generators (03b v1c_shapes at many seeds) and random blends, screened by is_valid and
    to the PINN's training alpha box (coordinate-wise min/max of its train split)."""
    chk = _load("chk", "03b_pinn_checks.py")
    tr = A["alpha"][A["split"] == 0].double()
    lo, hi = tr.min(0).values, tr.max(0).values
    n = cfg.diffusion.n_shapes
    parts = [lhs_shapes(ref, n, cfg.seed, family="diffusion").double(),
             torch.cat([chk.v1c_shapes(ref, seed=cfg.seed + 11 * k) for k in range(max(1, math.ceil(n / 60)))]).double(),
             random_blends(chk, ref, n, cfg.seed)]
    cand = torch.cat(parts)
    cand = cand[torch.randperm(cand.shape[0], generator=torch.Generator().manual_seed(cfg.seed))]
    valid = is_valid(ref, cand)
    ok = valid & in_envelope(cand, lo, hi)
    keep = ok & (ok.cumsum(0) <= n)
    scanned = int(keep.nonzero()[-1]) + 1
    return cand[keep], {"n_candidates_scanned": scanned, "n_kept": int(keep.sum()), "n_dropped_invalid": int((~valid[:scanned]).sum()),
                        "n_dropped_envelope": int((valid & ~ok)[:scanned].sum()), "alpha_lo": lo.tolist(), "alpha_hi": hi.tolist(),
                        "sources": [p.shape[0] for p in parts]}


def nn_dist(ref, A, B, skip_self=False):
    """[len(A)] RMS y-distance (chord units, = shape_distance) from each A shape to its nearest B shape (skip_self: A is a subset of B)."""
    f = lambda a: torch.cat([t for t in ref.surface_y(a)[1:]], 1)
    fa, fb = f(A), f(B)
    d = torch.cdist(fa, fb) / math.sqrt(fa.shape[1])  # fa has 2M columns: sqrt(mean((du^2 + dl^2) / 2)) = ||.|| / sqrt(2M)
    return d.sort(1).values[:, int(skip_self)]


def lbm_check(cfg, alpha, cond, nurbs, n):
    """PINN vs LBM CD / CL on n random shapes (G13)."""
    if n == 0:
        return None
    idx = torch.randperm(alpha.shape[0], generator=torch.Generator().manual_seed(cfg.seed))[:n]
    rows = []
    for i in idx.tolist():
        cd, cl = LBMScoreEvaluator(cfg, nurbs, None).value(alpha[i:i + 1], COND)
        rows.append((float(cond[i, 1]), float(cd), float(cond[i, 0]), float(cl)))
    t = torch.tensor(rows, dtype=torch.float64)
    rel = ((t[:, 0] - t[:, 1]).abs() / t[:, 1].abs())
    return {"n": n, "cd_rel_err_mean": float(rel.mean()), "cd_rel_err_max": float(rel.max()), "cd_rel_err_median": float(rel.median()),
            "cl_abs_err_mean": float((t[:, 2] - t[:, 3]).abs().mean()), "cl_abs_err_max": float((t[:, 2] - t[:, 3]).abs().max()),
            "cd_pinn": t[:, 0].tolist(), "cd_lbm": t[:, 1].tolist(), "cl_pinn": t[:, 2].tolist(), "cl_lbm": t[:, 3].tolist(),
            "biased_gt_10pct": bool(rel.mean() > 0.10)}


def build_denoiser(cfg, nurbs, device):
    dc = cfg.diffusion
    return ShapeDenoiser(nurbs.n_free, nurbs.x_ctrl[nurbs.free_idx % nurbs.k], (nurbs.free_idx // nurbs.k).long(), d=dc.d_model,
                         layers=dc.layers, heads=dc.heads, T=dc.T).to(device)


def free_mps_cache(device):
    if device.type == "mps":
        torch.mps.empty_cache()  # LBM / training leave a large cache; guided sampling then hits the MPS memory cliff


def train(cfg, nurbs, free, cond, device, out):
    dc = cfg.diffusion
    sch = CosineSchedule(dc.T)
    mean, std = free.mean(0), free.std(0).clamp_min(1e-3)
    z0 = ((free - mean) / std).to(device)
    cond = cond.to(device)
    dn = build_denoiser(cfg, nurbs, device)
    dn.cond_mean.copy_(cond.mean(0)), dn.cond_std.copy_(cond.std(0).clamp_min(1e-3))
    ema = copy.deepcopy(dn).eval()
    opt = torch.optim.AdamW(dn.parameters(), dc.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, dc.steps, eta_min=1e-6)
    g, losses = torch.Generator().manual_seed(cfg.seed), []
    with Timer(device) as tm:
        for step in range(dc.steps):
            idx = torch.randint(0, z0.shape[0], (min(dc.batch, z0.shape[0]),), generator=g).to(device)
            loss = diffusion_loss(dn, sch, z0[idx], cond[idx], dc.cond_drop, g)
            opt.zero_grad(), loss.backward(), opt.step(), sched.step()
            ema_update(ema, dn, min(dc.ema, (1 + step) / (10 + step)))  # warm-up so short runs are not dominated by the init
            losses.append(loss.detach())
            if (step + 1) % max(dc.steps // 10, 1) == 0:
                print(f"step {step + 1}/{dc.steps} loss {float(loss.detach()):.4f}", flush=True)
    losses = [float(x) for x in losses]  # one host sync at the end
    assert all(map(math.isfinite, losses)), "diffusion loss not finite"
    k = max(dc.steps // 10, 1)
    return ema, sch, (mean, std), {"wall_s": tm.elapsed, "loss_first": losses[0], "loss_final_mean_last10pct": sum(losses[-k:]) / k,
                                  "loss_curve": losses[::k]}


def evaluate(cfg, a, sampler, nurbs, ref, sets, train_alpha, device):
    """Sweep guide_scale x cfg_scale on each named set {name: (targets [n, 4], t_min [n, k])}; every metric is measured with the frozen PINN."""
    dc, rows = cfg.diffusion, []

    def run(gs, cs, cond, tm, seed=1):
        s = sampler.sample(cond, tm, n_steps=dc.sample_steps, cfg_scale=cs, guide_scale=gs, guide_clip=dc.guide_clip,
                           generator=torch.Generator().manual_seed(seed))
        al = s.alpha.detach()
        with torch.enable_grad():
            cd, cl = sampler.physics_fn(al, cond)
            J = sampler.objective(al, cond, tm).detach()
        valid = is_valid(ref, al.cpu().double())
        slack = (nurbs.thickness_at_ctrl(al) - tm).min(1).values.cpu()
        return al, J.cpu(), cd.detach().cpu(), cl.detach().cpu(), valid, slack

    keep = {}
    for name, (targets, tmin) in sets.items():
        cond, tm = (x.repeat_interleave(a.n_per_target, 0).to(device) for x in (targets, tmin))
        for gs in a.guide_scales:
            for cs in a.cfg_scales:
                al, J, cd, cl, valid, slack = run(gs, cs, cond, tm)
                keep[(name, gs, cs)] = al
                rows.append({"set": name, "guide_scale": gs, "cfg_scale": cs, "J": float(J.mean()), "J_median": float(J.median()),
                             "validity": float(valid.float().mean()), "constraint_satisfied": float((slack >= -1e-6).float().mean()),
                             "cd_abs_err": float((cd - cond[:, 1].cpu()).abs().mean()), "cl_abs_err": float((cl - cond[:, 0].cpu()).abs().mean())})
                print(rows[-1], flush=True)
    # unguided validity / smoothness / memorisation (G35: vs the data's own spacing) on n_unguided samples at held-out targets
    t0, m0 = next(iter(sets.values()))
    idx = torch.arange(a.n_unguided) % t0.shape[0]
    al_u, _, _, _, valid_u, _ = run(0.0, dc.cfg_scale, t0[idx].to(device), m0[idx].to(device), seed=2)
    d_s = nn_dist(ref, al_u.cpu().double(), train_alpha)
    sub = train_alpha[torch.randperm(train_alpha.shape[0], generator=torch.Generator().manual_seed(0))[:512]]
    d_t = nn_dist(ref, sub, train_alpha, skip_self=True)
    d2 = (al_u.reshape(al_u.shape[0], -1, nurbs.k).diff(dim=2, n=2)).abs().amax((1, 2))
    return rows, keep, {"n": a.n_unguided, "validity": float(valid_u.float().mean()), "median_sample_to_train": float(d_s.median()),
                        "median_train_to_train": float(d_t.median()), "memorization_ratio": float(d_s.median() / d_t.median()),
                        "memorization_pass": bool(d_s.median() >= 0.5 * d_t.median()), "old_threshold_0.005_pass": bool(d_s.median() > 0.005),
                        "max_second_diff": float(d2.max())}, al_u


def accept(rows, unguided, dc):
    """Guided vs unguided mean J at the same cfg_scale; spec: >= 30% lower, validity >= 95% unguided / >= 90% guided."""
    base = {r["cfg_scale"]: r for r in rows if r["guide_scale"] == 0}
    guided = [r for r in rows if r["guide_scale"] > 0 and r["cfg_scale"] in base and r["validity"] >= 0.90]  # G36: only valid rows compete
    best = min(guided, key=lambda r: r["J"] / base[r["cfg_scale"]]["J"]) if guided else None
    dflt = next((r for r in guided if r["guide_scale"] == dc.guide_scale and r["cfg_scale"] == dc.cfg_scale), None)
    imp = lambda r: 1 - r["J"] / base[r["cfg_scale"]]["J"]
    return {"J_unguided": base[best["cfg_scale"]]["J"] if best else None, "J_guided": best and best["J"],
            "best_pair": best and {"guide_scale": best["guide_scale"], "cfg_scale": best["cfg_scale"]},
            "J_reduction": best and imp(best), "default_pair_J_reduction": dflt and imp(dflt),
            "validity_unguided": unguided["validity"], "validity_guided_best": best and best["validity"],
            "pass": bool(best and imp(best) >= 0.30 and unguided["validity"] >= 0.95 and best["validity"] >= 0.90)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="dev", choices=["smoke", "dev", "full"])
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    ap.add_argument("--out", default=None)
    ap.add_argument("-o", "--set", action="append", default=[], dest="sets")
    ap.add_argument("--anchors", default=ANCHORS)
    ap.add_argument("--pinn-ckpt", default=None, help="default: runs/pinn_selected.json[transformer][checkpoint]")
    ap.add_argument("--eval-from", default=None, help="run dir with ckpt/diffusion.pt + results.json: skip LBM check and training")
    ap.add_argument("--n-lbm", type=int, default=20, help="LBM spot-check shapes (G13); 0 skips")
    ap.add_argument("--n-targets", type=int, default=8)
    ap.add_argument("--n-per-target", type=int, default=8)
    ap.add_argument("--n-unguided", type=int, default=64)
    ap.add_argument("--grid", default="0,0.3,1,3:0,1,2,4", help="guide scales : cfg scales")
    a = ap.parse_args(argv)
    gs, cs = (list(map(float, s.split(","))) for s in a.grid.split(":"))
    a.guide_scales, a.cfg_scales = gs, cs

    cfg = load_config(None, a.tier, a.sets)
    seed_everything(cfg.seed)
    device = get_device(a.device)
    out = Path(a.out or ROOT / f"runs/diffusion_{a.tier}")
    (out / "fig").mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(dump_config(cfg))
    ck_path = a.pinn_ckpt or json.loads((ROOT / "runs/pinn_selected.json").read_text())["transformer"]["checkpoint"]
    ck_path = Path(ck_path) if Path(ck_path).is_absolute() else ROOT / ck_path
    A = load_ckpt(ROOT / a.anchors if not Path(a.anchors).is_absolute() else a.anchors, "cpu")
    pinn, nurbs = _load("train03", "03_train_transformer_pinn.py").load_for_eval(ck_path, "transformer", device, A)
    ref = NURBSAirfoil(k=8, dtype=torch.float64)
    res = {"device_info": device_info(device), "tier": a.tier, "scope": "symmetric k=8, Re 20, AoA 0 only (PINN v1c range); "
           "cambered / Re 1000 waits for PINN v2", "pinn_ckpt": str(ck_path), "overrides": a.sets}

    alpha, res["data"] = make_shapes(cfg, A, ref)
    alpha = alpha.float().to(device)
    cond = pinn_labels(pinn, nurbs, alpha).detach()
    free = nurbs.full_to_free(alpha)
    res["data"].update(cd_range=[float(cond[:, 1].min()), float(cond[:, 1].max())], cl_range=[float(cond[:, 0].min()), float(cond[:, 0].max())],
                       cl_abs_max=float(cond[:, 0].abs().max()))
    d = ROOT / "data/diffusion_shapes"
    d.mkdir(parents=True, exist_ok=True)
    torch.save({"alpha": alpha.cpu(), "cond": cond.cpu(), "label_source": "transformer_pinn",
                "pinn_ckpt": hashlib.md5(ck_path.read_bytes()).hexdigest()}, d / f"{a.tier}_symmetric.pt")
    prev = json.loads((Path(a.eval_from) / "results.json").read_text()) if a.eval_from else None
    res["lbm_check"] = prev["lbm_check"] if prev else lbm_check(cfg, alpha, cond, nurbs, a.n_lbm)
    free_mps_cache(device)
    print("lbm_check:", {k: v for k, v in (res["lbm_check"] or {}).items() if not isinstance(v, list)}, flush=True)

    # hold out ~10% of the shapes: their (CL, CD) are the guidance targets
    perm = torch.randperm(alpha.shape[0], generator=torch.Generator().manual_seed(cfg.seed))
    n_ho = max(a.n_targets, alpha.shape[0] // 10)
    ho, tr = perm[:n_ho], perm[n_ho:]
    if prev:
        ck = load_ckpt(Path(a.eval_from) / "ckpt" / "diffusion.pt", device)
        ema, sch, norm, res["train"] = build_denoiser(cfg, nurbs, device).eval(), CosineSchedule(cfg.diffusion.T), tuple(x.to(device) for x in ck["norm"]), prev["train"]
        ema.load_state_dict(ck["model"])
    else:
        ema, sch, norm, res["train"] = train(cfg, nurbs, free[tr], cond[tr], device, out)
    save_ckpt(out / "ckpt" / "diffusion.pt", {"model": ema.state_dict(), "norm": norm, "config": cfg.to_dict(), "pinn_ckpt": str(ck_path)}, str(device))
    free_mps_cache(device)

    dc = cfg.diffusion
    sampler = GuidedSampler(ema, sch, nurbs, pinn, norm, {"w_L": dc.w_lift, "mu": dc.mu_thickness})
    ho_cond = cond[ho]
    ho_t = ho_cond[ho_cond[:, 1].argsort()][torch.linspace(0, n_ho - 1, a.n_targets).round().long()]  # spread over the CD range
    # G35 set 1: per held-out target a floor 15% above the median unguided thickness at that target (ends free), so unguided samples tend to violate it
    u = sampler.sample(ho_t.repeat_interleave(a.n_per_target, 0), torch.zeros(a.n_targets * a.n_per_target, nurbs.k, device=device), n_steps=dc.sample_steps,
                       cfg_scale=dc.cfg_scale, guide_scale=0.0, guide_clip=dc.guide_clip, generator=torch.Generator().manual_seed(3)).alpha.detach()
    ho_tm = 1.15 * nurbs.thickness_at_ctrl(u).reshape(a.n_targets, a.n_per_target, -1).median(1).values
    ho_tm[:, [0, 1, -1]] = 0
    # G35 set 2: CD 5-15% beyond the data range (CL = median), with the plain half-NACA0012 floor
    lo_hi = cond[:, 1].min(), cond[:, 1].max()
    cd_out = torch.stack([lo_hi[1] * 1.05, lo_hi[1] * 1.15, lo_hi[0] * 0.95, lo_hi[0] * 0.85])
    base_tm = 0.5 * torch.relu(nurbs.thickness_at_ctrl(naca4_alpha(ref, "0012").reshape(1, -1).float().to(device)))
    base_tm[:, [0, 1, -1]] = 0
    out_t = torch.stack([cond[:, 0].median().expand(4), cd_out, cond[0, 2].expand(4), cond[0, 3].expand(4)], 1)
    sets = {"heldout_tmin": (ho_t, ho_tm), "beyond_range": (out_t, base_tm.expand(4, -1))}
    rows, keep, res["unguided"], al_u = evaluate(cfg, a, sampler, nurbs, ref, sets, alpha[tr].cpu().double(), device)
    res["targets"], res["sweep"] = {k: v[0].tolist() for k, v in sets.items()}, rows
    res["t_min_heldout"] = ho_tm.tolist()
    res["acceptance"] = {k: accept([r for r in rows if r["set"] == k], res["unguided"], dc) for k in sets}
    res["data"]["thickness_range"] = [float(nurbs.thickness_at_ctrl(alpha).max(1).values.min()), float(nurbs.thickness_at_ctrl(alpha).max(1).values.max())]
    b = res["acceptance"]["heldout_tmin"]["best_pair"]
    plot_shapes(nurbs, {"train": alpha[tr], "unguided": al_u, **({"guided (best, heldout_tmin)": keep[("heldout_tmin", b["guide_scale"], b["cfg_scale"])]} if b else {})},
                out / "fig" / "samples.png")
    plot_guidance_sweep([r for r in rows if r["set"] == "heldout_tmin"], out / "fig" / "guidance_sweep.png")
    g, c = (b["guide_scale"], b["cfg_scale"]) if b else (dc.guide_scale, dc.cfg_scale)
    traj = sampler.sample(ho_t[:1], ho_tm[:1], n_steps=dc.sample_steps, cfg_scale=c, guide_scale=g, guide_clip=dc.guide_clip,
                          generator=torch.Generator().manual_seed(1), record=True).z_trajectory[:, 0]
    plot_denoise_gif(nurbs, sampler.to_alpha(traj).detach(), out / "fig" / "denoise.gif")
    (out / "results.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: res[k] for k in ("data", "unguided", "acceptance")} | {"train": {k: v for k, v in res["train"].items() if k != "loss_curve"}}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
